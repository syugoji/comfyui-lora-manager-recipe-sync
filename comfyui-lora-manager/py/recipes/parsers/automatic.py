"""Parser for Automatic1111 metadata format."""

import re
import os
import json
import logging
from typing import Dict, Any
from ..base import RecipeMetadataParser
from ..constants import GEN_PARAM_KEYS
from ...services.metadata_service import get_default_metadata_provider

logger = logging.getLogger(__name__)

class AutomaticMetadataParser(RecipeMetadataParser):
    """Parser for Automatic1111 metadata format"""
    
    METADATA_MARKER = r"Steps: \d+"
    
    # Regular expressions for extracting specific metadata
    HASHES_REGEX = r', Hashes:\s*({[^}]+})'
    LORA_HASHES_REGEX = r', Lora hashes:\s*"([^"]+)"'
    TI_HASHES_REGEX = r', TI hashes:\s*"([^"]+)"'
    CIVITAI_RESOURCES_REGEX = r', Civitai resources:\s*(\[\{.*?\}\])'
    CIVITAI_METADATA_REGEX = r', Civitai metadata:\s*(\{.*?\})'
    EXTRANETS_REGEX = r'<(lora|hypernet):([^:]+):(-?[0-9.]+)>'
    MODEL_HASH_PATTERN = r'Model hash: ([a-zA-Z0-9]+)'
    MODEL_NAME_PATTERN = r'Model: ([^,]+)'
    VAE_HASH_PATTERN = r'VAE hash: ([a-zA-Z0-9]+)'
    ADDNET_INDEX_PATTERN = r'AddNet Module (\d+):'

    @staticmethod
    def _safe_float(value: Any, default: float = 1.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_addnet_entries(params_section: str) -> list[Dict[str, Any]]:
        """Extract legacy Additional Networks entries from A1111 metadata.

        Additional Networks stores LoRAs outside the prompt as numbered
        ``AddNet Module/Model/Weight`` fields.  ComfyUI's native A1111 importer
        ignores those fields, so preserving them as structured recipe LoRAs is
        required for deterministic replay.
        """
        entries: list[Dict[str, Any]] = []
        indexes = sorted(
            {int(value) for value in re.findall(AutomaticMetadataParser.ADDNET_INDEX_PATTERN, params_section)}
        )
        for index in indexes:
            module_match = re.search(
                rf'AddNet Module {index}:\s*([^,]+)', params_section, re.IGNORECASE
            )
            model_match = re.search(
                rf'AddNet Model {index}:\s*([^,]+)', params_section, re.IGNORECASE
            )
            if not module_match or not model_match:
                continue

            module = module_match.group(1).strip().lower()
            if module not in {"lora", "locon", "lycoris", "hypernet"}:
                continue

            raw_model = model_match.group(1).strip()
            model_parts = re.match(r'^(.*?)\s*\(([0-9a-fA-F]{8,})\)\s*$', raw_model)
            if model_parts:
                model_name = model_parts.group(1).strip()
                model_hash = model_parts.group(2).lower()
            else:
                model_name = raw_model
                model_hash = ""
            if not model_name:
                continue

            def read_weight(kind: str, default: float) -> float:
                match = re.search(
                    rf'AddNet Weight {kind} {index}:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))',
                    params_section,
                    re.IGNORECASE,
                )
                if not match:
                    return default
                try:
                    return round(float(match.group(1)), 4)
                except (TypeError, ValueError):
                    return default

            strength_model = read_weight("A", 1.0)
            strength_clip = read_weight("B", strength_model)
            entries.append(
                {
                    "name": model_name,
                    "file_name": model_name,
                    "type": module,
                    "weight": strength_model,
                    "strength": strength_model,
                    "strength_model": strength_model,
                    "strength_clip": strength_clip,
                    "hash": model_hash,
                    "existsLocally": False,
                    "localPath": None,
                    "thumbnailUrl": "/loras_static/images/no-preview.png",
                    "baseModel": "",
                    "size": 0,
                    "downloadUrl": "",
                    "isDeleted": False,
                    "metadataSource": "a1111_addnet",
                }
            )
        return entries
    
    def is_metadata_matching(self, user_comment: str) -> bool:
        """Check if the user comment matches the Automatic1111 format"""
        return re.search(self.METADATA_MARKER, user_comment) is not None
    
    async def parse_metadata(self, user_comment: str, recipe_scanner=None, civitai_client=None) -> Dict[str, Any]:
        """Parse metadata from Automatic1111 format"""
        try:
            # Get metadata provider instead of using civitai_client directly
            metadata_provider = await get_default_metadata_provider()
            
            # Split on Negative prompt if it exists
            if "Negative prompt:" in user_comment:
                parts = user_comment.split('Negative prompt:', 1)
                prompt = parts[0].strip()
                negative_and_params = parts[1] if len(parts) > 1 else ""
            else:
                # No negative prompt section
                param_start = re.search(self.METADATA_MARKER, user_comment)
                if param_start:
                    prompt = user_comment[:param_start.start()].strip()
                    negative_and_params = user_comment[param_start.start():]
                else:
                    prompt = user_comment.strip()
                    negative_and_params = ""
            
            # Initialize metadata
            metadata = {
                "prompt": prompt,
                "loras": []
            }
            
            # Extract negative prompt and parameters
            if negative_and_params:
                # If we split on "Negative prompt:", check for params section
                if "Negative prompt:" in user_comment:
                    param_start = re.search(r'Steps: ', negative_and_params)
                    if param_start:
                        neg_prompt = negative_and_params[:param_start.start()].strip()
                        metadata["negative_prompt"] = neg_prompt
                        params_section = negative_and_params[param_start.start():]
                    else:
                        metadata["negative_prompt"] = negative_and_params.strip()
                        params_section = ""
                else:
                    # No negative prompt, entire section is params
                    params_section = negative_and_params
                
                # Extract generation parameters
                if params_section:
                    # Parse legacy sd-webui-additional-networks fields before
                    # generic parameter filtering discards them.
                    metadata["additional_networks"] = self._parse_addnet_entries(
                        params_section
                    )

                    # Extract Civitai resources
                    civitai_resources_match = re.search(self.CIVITAI_RESOURCES_REGEX, params_section)
                    if civitai_resources_match:
                        try:
                            civitai_resources = json.loads(civitai_resources_match.group(1))
                            metadata["civitai_resources"] = civitai_resources
                            params_section = params_section.replace(civitai_resources_match.group(0), '')
                        except json.JSONDecodeError:
                            logger.error("Error parsing Civitai resources JSON")
                    
                    # Extract Hashes
                    hashes_match = re.search(self.HASHES_REGEX, params_section)
                    if hashes_match:
                        try:
                            hashes = json.loads(hashes_match.group(1))
                            # Process hash keys
                            processed_hashes = {}
                            for key, value in hashes.items():
                                # Convert Model: or LORA: prefix to lowercase if present
                                if ':' in key:
                                    prefix, name = key.split(':', 1)
                                    prefix = prefix.lower()
                                else:
                                    prefix = ''
                                    name = key

                                # Clean up the name part
                                if '/' in name:
                                    name = name.split('/')[-1]  # Get last part after /
                                if '.safetensors' in name:
                                    name = name.split('.safetensors')[0]  # Remove .safetensors
                                
                                # Reconstruct the key
                                new_key = f"{prefix}:{name}" if prefix else name
                                processed_hashes[new_key] = value

                            metadata["hashes"] = processed_hashes
                            # Remove hashes from params section to not interfere with other parsing
                            params_section = params_section.replace(hashes_match.group(0), '')
                        except json.JSONDecodeError:
                            logger.error("Error parsing hashes JSON")
                    
                    # Pick up model hash from parsed hashes if available
                    if "hashes" in metadata and not metadata.get("model_hash"):
                        model_hash_from_hashes = metadata["hashes"].get("model")
                        if model_hash_from_hashes:
                            metadata["model_hash"] = model_hash_from_hashes
                    
                    # Extract Lora hashes in alternative format.
                    # Run unconditionally (not just as fallback) so that
                    # non-empty hashes from Lora hashes fill in the gaps left
                    # by empty values in the Hashes JSON dict.  Some WebUI
                    # builds write real hash values only to Lora hashes and
                    # leave the Hashes JSON values empty.
                    lora_hashes_match = re.search(self.LORA_HASHES_REGEX, params_section)
                    if lora_hashes_match:
                        try:
                            lora_hashes_str = lora_hashes_match.group(1)
                            lora_hash_entries = lora_hashes_str.split(', ')

                            # Parse each lora hash entry (format: "name: hash")
                            for entry in lora_hash_entries:
                                if ': ' in entry:
                                    lora_name, lora_hash = entry.split(': ', 1)
                                    lora_hash = lora_hash.strip()
                                    if not lora_hash:
                                        # Skip entries without a hash value
                                        continue
                                    # Initialize hashes dict if it doesn't exist
                                    if "hashes" not in metadata:
                                        metadata["hashes"] = {}
                                    # Add as lora type in the same format as
                                    # regular hashes.  Only override an
                                    # existing entry if its value is empty
                                    # (Lora hashes is the more reliable
                                    # source when Hashes JSON has blanks).
                                    key = f"lora:{lora_name}"
                                    existing = metadata["hashes"].get(key, "")
                                    if not existing:
                                        metadata["hashes"][key] = lora_hash

                            # Remove lora hashes from params section
                            params_section = params_section.replace(lora_hashes_match.group(0), '')
                        except Exception as e:
                            logger.error(f"Error parsing Lora hashes: {e}")

                    # A1111 writes Textual Inversion usage separately from
                    # LoRAs. Preserve those hashes as downloadable embeddings.
                    ti_hashes_match = re.search(self.TI_HASHES_REGEX, params_section)
                    if ti_hashes_match:
                        embedding_hashes = metadata.setdefault("embedding_hashes", {})
                        for entry in ti_hashes_match.group(1).split(", "):
                            if ": " not in entry:
                                continue
                            embedding_name, embedding_hash = entry.split(": ", 1)
                            if embedding_name.strip() and embedding_hash.strip():
                                embedding_hashes[embedding_name.strip()] = embedding_hash.strip()
                        params_section = params_section.replace(ti_hashes_match.group(0), "")

                    # Extract checkpoint model hash/name when provided outside Civitai resources
                    model_hash_match = re.search(self.MODEL_HASH_PATTERN, params_section)
                    if model_hash_match:
                        metadata["model_hash"] = model_hash_match.group(1).strip()
                        params_section = params_section.replace(model_hash_match.group(0), '')

                    model_name_match = re.search(self.MODEL_NAME_PATTERN, params_section)
                    if model_name_match:
                        metadata["model_name"] = model_name_match.group(1).strip()
                        params_section = params_section.replace(model_name_match.group(0), '')
                    
                    # Extract basic parameters
                    param_pattern = r'([A-Za-z\s]+): ([^,]+)'
                    params = re.findall(param_pattern, params_section)
                    gen_params = {}
                    
                    for key, value in params:
                        clean_key = key.strip().lower().replace(' ', '_')
                        
                        # Skip if not in recognized gen param keys
                        if clean_key not in GEN_PARAM_KEYS:
                            continue
                            
                        # Convert numeric values
                        if clean_key in ['steps', 'seed']:
                            try:
                                gen_params[clean_key] = int(value.strip())
                            except ValueError:
                                gen_params[clean_key] = value.strip()
                        elif clean_key in ['cfg_scale']:
                            try:
                                gen_params[clean_key] = float(value.strip())
                            except ValueError:
                                gen_params[clean_key] = value.strip()
                        else:
                            gen_params[clean_key] = value.strip()
                    
                    # Extract size if available and add to gen_params if a recognized key
                    size_match = re.search(r'Size: (\d+)x(\d+)', params_section)
                    if size_match and 'size' in GEN_PARAM_KEYS:
                        width, height = size_match.groups()
                        gen_params['size'] = f"{width}x{height}"
                    
                    # Add prompt and negative_prompt to gen_params if they're in GEN_PARAM_KEYS
                    if 'prompt' in GEN_PARAM_KEYS and 'prompt' in metadata:
                        gen_params['prompt'] = metadata['prompt']
                    if 'negative_prompt' in GEN_PARAM_KEYS and 'negative_prompt' in metadata:
                        gen_params['negative_prompt'] = metadata['negative_prompt']
                    
                    metadata["gen_params"] = gen_params
            
            # Extract LoRA and checkpoint information 
            loras = []
            embeddings = []
            base_model_counts = {}
            checkpoint = None
            
            # First use Civitai resources if available (more reliable source)
            if metadata.get("civitai_resources"):
                for resource in metadata.get("civitai_resources", []):
                    # --- Added: Parse 'air' field if present ---
                    air = resource.get("air")
                    if air:
                        # Format: urn:air:sdxl:lora:civitai:1221007@1375651
                        # Or: urn:air:sdxl:checkpoint:civitai:623891@2019115
                        air_pattern = r"urn:air:[^:]+:(?P<type>[^:]+):civitai:(?P<modelId>\d+)@(?P<modelVersionId>\d+)"
                        air_match = re.match(air_pattern, air)
                        if air_match:
                            air_type = air_match.group("type")
                            air_modelId = int(air_match.group("modelId"))
                            air_modelVersionId = int(air_match.group("modelVersionId"))
                            # checkpoint/lycoris/lora/hypernet
                            resource["type"] = air_type
                            resource["modelId"] = air_modelId
                            resource["modelVersionId"] = air_modelVersionId
                    # --- End added ---

                    resource_type = str(resource.get("type") or "").lower().replace("_", "").replace(" ", "")

                    if resource_type == "checkpoint" and resource.get("modelVersionId"):
                        version_id = resource.get("modelVersionId")
                        version_id_str = str(version_id)
                        checkpoint_entry = {
                            'id': version_id,
                            'modelId': resource.get("modelId", 0),
                            'name': resource.get("modelName", "Unknown Checkpoint"),
                            'version': resource.get("modelVersionName", resource.get("versionName", "")),
                            'type': resource.get("type", "checkpoint"),
                            'existsLocally': False,
                            'localPath': None,
                            'file_name': resource.get("modelName", ""),
                            'hash': resource.get("hash", "") or "",
                            'thumbnailUrl': '/loras_static/images/no-preview.png',
                            'baseModel': '',
                            'size': 0,
                            'downloadUrl': '',
                            'isDeleted': False
                        }

                        if metadata_provider:
                            try:
                                civitai_info = await metadata_provider.get_model_version_info(version_id_str)
                                checkpoint_entry = await self.populate_checkpoint_from_civitai(
                                    checkpoint_entry,
                                    civitai_info
                                )
                            except Exception as e:
                                logger.error(
                                    "Error fetching Civitai info for checkpoint version %s: %s",
                                    version_id,
                                    e,
                                )

                        # Prefer the first checkpoint found
                        if checkpoint_entry.get("baseModel"):
                            base_model_value = checkpoint_entry["baseModel"]
                            base_model_counts[base_model_value] = base_model_counts.get(base_model_value, 0) + 1

                        if checkpoint is None:
                            checkpoint = checkpoint_entry

                        continue

                    if resource_type in {"textualinversion", "embedding"} and resource.get("modelVersionId"):
                        embedding_entry = {
                            'id': resource.get("modelVersionId", 0),
                            'modelId': resource.get("modelId", 0),
                            'name': resource.get("modelName", "Unknown Embedding"),
                            'version': resource.get("modelVersionName", resource.get("versionName", "")),
                            'type': 'embedding',
                            'hash': resource.get("hash", "") or "",
                            'existsLocally': False,
                            'localPath': None,
                            'file_name': resource.get("modelName", ""),
                            'thumbnailUrl': '/loras_static/images/no-preview.png',
                            'baseModel': '',
                            'size': 0,
                            'downloadUrl': '',
                            'isDeleted': False,
                        }
                        if metadata_provider:
                            try:
                                civitai_info = await metadata_provider.get_model_version_info(
                                    resource.get("modelVersionId")
                                )
                                populated_embedding = await self.populate_embedding_from_civitai(
                                    embedding_entry, civitai_info, recipe_scanner
                                )
                                if populated_embedding is None:
                                    continue
                                embedding_entry = populated_embedding
                            except Exception as e:
                                logger.error("Error fetching Civitai embedding %s: %s", embedding_entry['name'], e)
                        embeddings.append(embedding_entry)
                        continue

                    if resource_type in ["lora", "lycoris", "hypernet"] and resource.get("modelVersionId"):
                        # Initialize lora entry
                        lora_entry = {
                            'id': resource.get("modelVersionId", 0),
                            'modelId': resource.get("modelId", 0),
                            'name': resource.get("modelName", "Unknown LoRA"),
                            'version': resource.get("modelVersionName", resource.get("versionName", "")),
                            'type': resource.get("type", "lora"),
                            'weight': round(self._safe_float(resource.get("weight")), 2),
                            'existsLocally': False,
                            'thumbnailUrl': '/loras_static/images/no-preview.png',
                            'baseModel': '',
                            'size': 0,
                            'downloadUrl': '',
                            'isDeleted': False
                        }
                        
                        # Get additional info from Civitai
                        if metadata_provider:
                            try:
                                civitai_info = await metadata_provider.get_model_version_info(resource.get("modelVersionId"))
                                populated_entry = await self.populate_lora_from_civitai(
                                    lora_entry,
                                    civitai_info,
                                    recipe_scanner,
                                    base_model_counts
                                )
                                if populated_entry is None:
                                    continue  # Skip invalid LoRA types
                                lora_entry = populated_entry
                            except Exception as e:
                                logger.error(f"Error fetching Civitai info for LoRA {lora_entry['name']}: {e}")
                        
                        loras.append(lora_entry)

            # Hashes JSON may use embed:/embedding:/ti: prefixes, while
            # classic A1111 uses the dedicated TI hashes field.
            embedding_hashes = dict(metadata.get("embedding_hashes") or {})
            for hash_key, hash_value in (metadata.get("hashes") or {}).items():
                normalized_key = str(hash_key)
                if ":" not in normalized_key:
                    continue
                prefix, embedding_name = normalized_key.split(":", 1)
                if prefix.lower() in {"embed", "embedding", "ti", "textualinversion"}:
                    embedding_hashes.setdefault(embedding_name, hash_value)

            existing_embedding_hashes = {
                str(item.get("hash") or "").lower() for item in embeddings
            }
            for embedding_name, embedding_hash in embedding_hashes.items():
                normalized_hash = str(embedding_hash or "").lower()
                if not normalized_hash or normalized_hash in existing_embedding_hashes:
                    continue
                embedding_entry = {
                    'name': embedding_name,
                    'type': 'embedding',
                    'hash': normalized_hash,
                    'existsLocally': False,
                    'localPath': None,
                    'file_name': embedding_name,
                    'thumbnailUrl': '/loras_static/images/no-preview.png',
                    'baseModel': '',
                    'size': 0,
                    'downloadUrl': '',
                    'isDeleted': False,
                }
                if metadata_provider:
                    try:
                        civitai_info = await metadata_provider.get_model_by_hash(normalized_hash)
                        populated_embedding = await self.populate_embedding_from_civitai(
                            embedding_entry,
                            civitai_info,
                            recipe_scanner,
                            normalized_hash,
                        )
                        if populated_embedding is None:
                            continue
                        embedding_entry = populated_embedding
                    except Exception as e:
                        logger.error("Error resolving embedding %s: %s", embedding_name, e)
                if embedding_entry.get("isDeleted"):
                    # The image proves this embedding was used. A short
                    # AutoV3 hash may not resolve directly even when page data
                    # can still supply a downloadable version ID later.
                    embedding_entry["isDeleted"] = False
                    embedding_entry["unresolved"] = True
                embeddings.append(embedding_entry)
                existing_embedding_hashes.add(normalized_hash)

            known_embedding_names = {
                re.sub(
                    r"[^a-z0-9]+",
                    "",
                    str(item.get("name") or item.get("file_name") or "").lower(),
                )
                for item in embeddings
            }
            for prompt_text in (prompt, metadata.get("negative_prompt")):
                if not isinstance(prompt_text, str):
                    continue
                for embedding_name in re.findall(
                    r'(?i)(?:embedding|embed):([^\s,;()\[\]{}]+)',
                    prompt_text,
                ):
                    normalized_name = embedding_name.strip()
                    normalized_key = re.sub(r"[^a-z0-9]+", "", normalized_name.lower())
                    if not normalized_name or normalized_key in known_embedding_names:
                        continue
                    embeddings.append({
                        'name': normalized_name,
                        'file_name': normalized_name,
                        'type': 'embedding',
                        'hash': '',
                        'existsLocally': False,
                        'localPath': None,
                        'isDeleted': False,
                        'unresolved': True,
                    })
                    known_embedding_names.add(normalized_key)
            
            # Fallback checkpoint parsing from generic "Model" and "Model hash" fields
            if checkpoint is None:
                model_hash = metadata.get("model_hash")
                if not model_hash and metadata.get("hashes"):
                    model_hash = metadata["hashes"].get("model")

                model_name = metadata.get("model_name")
                file_name = ""
                if model_name:
                    cleaned_name = re.split(r"[\\\\/]", model_name)[-1]
                    file_name = os.path.splitext(cleaned_name)[0]

                if model_hash or model_name:
                    checkpoint_entry = {
                        'id': 0,
                        'modelId': 0,
                        'name': model_name or "Unknown Checkpoint",
                        'version': '',
                        'type': 'checkpoint',
                        'hash': model_hash or "",
                        'existsLocally': False,
                        'localPath': None,
                        'file_name': file_name,
                        'thumbnailUrl': '/loras_static/images/no-preview.png',
                        'baseModel': '',
                        'size': 0,
                        'downloadUrl': '',
                        'isDeleted': False
                    }

                    if metadata_provider and model_hash:
                        try:
                            civitai_info = await metadata_provider.get_model_by_hash(model_hash)
                            checkpoint_entry = await self.populate_checkpoint_from_civitai(
                                checkpoint_entry,
                                civitai_info
                            )
                        except Exception as e:
                            logger.error(f"Error fetching Civitai info for checkpoint hash {model_hash}: {e}")

                    if checkpoint_entry.get("baseModel"):
                        base_model_value = checkpoint_entry["baseModel"]
                        base_model_counts[base_model_value] = base_model_counts.get(base_model_value, 0) + 1

                    checkpoint = checkpoint_entry

            # If no LoRAs from Civitai resources, extract from metadata["hashes"]
            if not loras or len(loras) == 0:
                # Extract lora weights from extranet tags in prompt (for later use)
                lora_weights = {}
                lora_matches = re.findall(self.EXTRANETS_REGEX, prompt)
                for lora_type, lora_name, lora_weight in lora_matches:
                    key = f"{lora_type}:{lora_name}"
                    lora_weights[key] = round(float(lora_weight), 2)
                
                # Use hashes from metadata as the primary source
                if metadata.get("hashes"):
                    for hash_key, lora_hash in metadata.get("hashes", {}).items():
                        # Only process lora or hypernet types
                        if not hash_key.startswith(("lora:", "hypernet:")):
                            continue
                        
                        # Skip entries without a hash value — they can't be
                        # resolved via CivitAI and would only produce a
                        # useless "Deleted" entry in the recipe.
                        if not lora_hash:
                            continue
                            
                        lora_type, lora_name = hash_key.split(':', 1)
                        
                        # Get weight from extranet tags if available, else default to 1.0
                        weight = lora_weights.get(hash_key, 1.0)
                        
                        # Initialize lora entry
                        lora_entry = {
                            'name': lora_name,
                            'type': lora_type,  # 'lora' or 'hypernet'
                            'weight': weight,
                            'hash': lora_hash,
                            'existsLocally': False,
                            'localPath': None,
                            'file_name': lora_name,
                            'thumbnailUrl': '/loras_static/images/no-preview.png',
                            'baseModel': '',
                            'size': 0,
                            'downloadUrl': '',
                            'isDeleted': False
                        }
                        
                        # Try to get info from Civitai
                        if metadata_provider:
                            try:
                                civitai_info = await self.get_lora_metadata_by_hash(
                                    metadata_provider, lora_hash
                                )
                                
                                populated_entry = await self.populate_lora_from_civitai(
                                    lora_entry, 
                                    civitai_info, 
                                    recipe_scanner,
                                    base_model_counts,
                                    lora_hash
                                )
                                if populated_entry is None:
                                    continue  # Skip invalid LoRA types
                                lora_entry = populated_entry
                            except Exception as e:
                                logger.error(f"Error fetching Civitai info for LoRA {lora_name}: {e}")
                        
                        loras.append(lora_entry)

            # Explicit AddNet entries describe the networks actually applied
            # by A1111.  They are therefore authoritative over page-associated
            # CivitAI resources.  Resolve each hash for IDs/local paths, while
            # preserving the original A/B strengths.
            addnet_entries = metadata.get("additional_networks") or []
            if addnet_entries:
                resolved_addnet = []
                for addnet_entry in addnet_entries:
                    lora_entry = dict(addnet_entry)
                    lora_hash = str(lora_entry.get("hash") or "").lower()
                    if metadata_provider and lora_hash:
                        try:
                            civitai_info = await self.get_lora_metadata_by_hash(
                                metadata_provider, lora_hash
                            )
                            populated_entry = await self.populate_lora_from_civitai(
                                lora_entry,
                                civitai_info,
                                recipe_scanner,
                                base_model_counts,
                                lora_hash,
                            )
                            if populated_entry is not None:
                                lora_entry = populated_entry
                        except Exception as e:
                            logger.error(
                                "Error resolving AddNet LoRA %s: %s",
                                lora_entry.get("name"),
                                e,
                            )
                    if lora_entry.get("isDeleted"):
                        # The image proves this network participated in the
                        # original generation. Keep an unresolved loader node
                        # even when CivitAI no longer exposes the file.
                        lora_entry["isDeleted"] = False
                        lora_entry["unresolved"] = True
                    resolved_addnet.append(lora_entry)
                if resolved_addnet:
                    loras = resolved_addnet
                
            # Try to get base model from resources or make educated guess
            base_model = None
            if checkpoint and checkpoint.get("baseModel"):
                base_model = checkpoint.get("baseModel")
            elif base_model_counts:
                # Use the most common base model from the loras
                base_model = max(base_model_counts.items(), key=lambda x: x[1])[0]
            
            # Prepare final result structure
            # Make sure gen_params only contains recognized keys
            filtered_gen_params = {}
            for key in GEN_PARAM_KEYS:
                if key in metadata.get("gen_params", {}):
                    filtered_gen_params[key] = metadata["gen_params"][key]
            
            result = {
                'base_model': base_model,
                'loras': loras,
                'embeddings': embeddings,
                'gen_params': filtered_gen_params,
                'from_automatic_metadata': True
            }

            if checkpoint:
                result['checkpoint'] = checkpoint
                result['model'] = checkpoint
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing Automatic1111 metadata: {e}", exc_info=True)
            return {"error": str(e), "loras": []}
