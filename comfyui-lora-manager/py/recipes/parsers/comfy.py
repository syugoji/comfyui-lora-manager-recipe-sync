"""Parser for ComfyUI metadata format."""

import re
import json
import logging
from typing import Dict, Any
from ..base import RecipeMetadataParser
from ..constants import GEN_PARAM_KEYS
from ...config import config
from ...services.metadata_service import get_default_metadata_provider

logger = logging.getLogger(__name__)

class ComfyMetadataParser(RecipeMetadataParser):
    """Parser for Civitai ComfyUI metadata JSON format"""
    
    METADATA_MARKER = r"class_type"

    # Stock ComfyUI loaders: exactly one LoRA per node, held in `inputs.lora_name`.
    LORA_NODE_TYPES = ('LoraLoader', 'LoraLoaderModelOnly')

    # This repository's own nodes: many LoRAs per node, held in a `loras` widget
    # (and/or a `<lora:name:strength>` text widget). Keep in sync with the NAME
    # attributes registered in NODE_CLASS_MAPPINGS (see py/nodes/).
    LORA_MANAGER_NODE_TYPES = (
        'Lora Loader (LoraManager)',
        'LoRA Text Loader (LoraManager)',
        'Lora Stacker (LoraManager)',
        'WanVideo Lora Select (LoraManager)',
        'WanVideo Lora Select From Text (LoraManager)',
    )

    LORA_SYNTAX_PATTERN = re.compile(r'<lora:([^:>]+):([-+]?[\d.]+)')

    def is_metadata_matching(self, user_comment: str) -> bool:
        """Check if the user comment matches the ComfyUI metadata format"""
        try:
            data = json.loads(user_comment)
            # Check if it contains class_type nodes typical of ComfyUI workflow
            return isinstance(data, dict) and any(isinstance(v, dict) and 'class_type' in v for v in data.values())
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    async def _get_local_model_info(recipe_scanner, scanner_attr: str, raw_name: str):
        """Resolve a local file style name (e.g. 'folder\\model.safetensors') via a scanner cache."""
        if not recipe_scanner or not isinstance(raw_name, str) or not raw_name.strip():
            return None
        scanner = getattr(recipe_scanner, scanner_attr, None)
        getter = getattr(scanner, 'get_model_info_by_name', None)
        if not callable(getter):
            return None
        try:
            return await getter(raw_name)
        except Exception as e:
            logger.error(f"Error resolving local model name '{raw_name}': {e}")
            return None

    @classmethod
    def _iter_lora_manager_entries(cls, node: Dict[str, Any]):
        """Yield (lora_name, weight) pairs from a *(LoraManager) node.

        These nodes bundle several LoRAs into one node, so the stock `lora_name`
        handling does not apply. Two widget shapes exist — ``{'__value__': [...]}``
        and a bare list; ``py/nodes/utils.py::get_loras_list`` is the source of truth
        for that format. Entries switched off in the UI (``active: false``) did not
        affect the image and are skipped. When the structured widget is absent
        (text-only loader variants) the ``<lora:name:strength>`` text widget is used.
        """
        inputs = node.get('inputs')
        if not isinstance(inputs, dict):
            return

        loras_data = inputs.get('loras')
        if isinstance(loras_data, dict) and '__value__' in loras_data:
            entries = loras_data['__value__']
        elif isinstance(loras_data, list):
            entries = loras_data
        else:
            entries = None

        emitted = False
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict) or entry.get('active') is False:
                    continue
                name = entry.get('name')
                if not isinstance(name, str) or not name.strip():
                    continue
                try:
                    weight = float(entry.get('strength', 1.0))
                except (TypeError, ValueError):
                    weight = 1.0
                emitted = True
                yield name.strip(), weight

        if emitted:
            return

        text = inputs.get('text')
        if isinstance(text, str):
            for name, raw_weight in cls.LORA_SYNTAX_PATTERN.findall(text):
                if not name.strip():
                    continue
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    weight = 1.0
                yield name.strip(), weight

    @staticmethod
    def _entry_from_local_info(item: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Build a recipe resource entry from a scanner cache item (locally resolved model)."""
        civitai_info = item.get('civitai') or {}
        if not isinstance(civitai_info, dict):
            civitai_info = {}
        thumbnail = '/loras_static/images/no-preview.png'
        if item.get('preview_url'):
            try:
                thumbnail = config.get_preview_static_url(item['preview_url'])
            except Exception:
                pass
        return {
            'id': civitai_info.get('id', 0),
            'modelId': civitai_info.get('modelId', 0),
            'name': item.get('model_name') or item.get('file_name', ''),
            'version': civitai_info.get('name', ''),
            'type': model_type,
            'existsLocally': True,
            'localPath': item.get('file_path'),
            'file_name': item.get('file_name', ''),
            'hash': (item.get('sha256') or '').lower(),
            'thumbnailUrl': thumbnail,
            'baseModel': item.get('base_model', ''),
            'size': item.get('size', 0),
            'downloadUrl': civitai_info.get('downloadUrl', ''),
            'isDeleted': False,
        }
    
    async def parse_metadata(self, user_comment: str, recipe_scanner=None, civitai_client=None) -> Dict[str, Any]:
        """Parse metadata from Civitai ComfyUI metadata format"""
        try:
            # Get metadata provider instead of using civitai_client directly
            metadata_provider = await get_default_metadata_provider()
            
            data = json.loads(user_comment)
            loras = []
            
            # Collect (name, weight) pairs from every node type that can carry LoRAs.
            # Stock loaders hold one each; the (LoraManager) nodes hold many per node.
            lora_specs = []
            for node in data.values():
                if not isinstance(node, dict):
                    continue
                class_type = node.get('class_type')
                if class_type in self.LORA_NODE_TYPES:
                    inputs = node.get('inputs')
                    if not isinstance(inputs, dict) or 'lora_name' not in inputs:
                        continue
                    lora_specs.append((inputs.get('lora_name', ''), inputs.get('strength_model', 1.0)))
                elif class_type in self.LORA_MANAGER_NODE_TYPES:
                    lora_specs.extend(self._iter_lora_manager_entries(node))

            # Process each collected LoRA
            for lora_name, weight in lora_specs:
                # Parse the URN to extract model ID and version ID
                # Format: "urn:air:sdxl:lora:civitai:1107767@1253442"
                lora_id_match = re.search(r'civitai:(\d+)@(\d+)', lora_name)
                if not lora_id_match:
                    # Local ComfyUI outputs store a file path (e.g. 'folder\\model.safetensors')
                    # instead of a civitai URN — resolve it against the local lora cache
                    local_info = await self._get_local_model_info(recipe_scanner, '_lora_scanner', lora_name)
                    if local_info:
                        lora_entry = self._entry_from_local_info(local_info, 'lora')
                        lora_entry['weight'] = weight
                        loras.append(lora_entry)
                    else:
                        logger.warning(f"Could not resolve LoRA name to a local model: {lora_name}")
                    continue

                model_id = lora_id_match.group(1)
                model_version_id = lora_id_match.group(2)

                # Initialize lora entry with default values
                lora_entry = {
                    'id': model_version_id,
                    'modelId': model_id,
                    'name': f"Lora {model_id}",  # Default name
                    'version': '',
                    'type': 'lora',
                    'weight': weight,
                    'existsLocally': False,
                    'localPath': None,
                    'file_name': '',
                    'hash': '',
                    'thumbnailUrl': '/loras_static/images/no-preview.png',
                    'baseModel': '',
                    'size': 0,
                    'downloadUrl': '',
                    'isDeleted': False
                }
                
                # Get additional info from Civitai if metadata provider is available
                if metadata_provider:
                    try:
                        civitai_info_tuple = await metadata_provider.get_model_version_info(model_version_id)
                        # Populate lora entry with Civitai info
                        populated_entry = await self.populate_lora_from_civitai(
                            lora_entry, 
                            civitai_info_tuple, 
                            recipe_scanner
                        )
                        if populated_entry is None:
                            continue  # Skip invalid LoRA types
                        lora_entry = populated_entry
                    except Exception as e:
                        logger.error(f"Error fetching Civitai info for LoRA: {e}")
                
                loras.append(lora_entry)
            
            # Find checkpoint info
            checkpoint_nodes = {k: v for k, v in data.items() if isinstance(v, dict) and v.get('class_type') == 'CheckpointLoaderSimple'}
            checkpoint = None
            checkpoint_id = None
            checkpoint_version_id = None
            
            if checkpoint_nodes:
                # Get the first checkpoint node
                checkpoint_node = next(iter(checkpoint_nodes.values()))
                if 'inputs' in checkpoint_node and 'ckpt_name' in checkpoint_node['inputs']:
                    checkpoint_name = checkpoint_node['inputs']['ckpt_name']
                    # Parse checkpoint URN
                    checkpoint_match = re.search(r'civitai:(\d+)@(\d+)', checkpoint_name)
                    if not checkpoint_match:
                        # Local ComfyUI outputs store a file path instead of a civitai URN
                        local_info = await self._get_local_model_info(recipe_scanner, '_checkpoint_scanner', checkpoint_name)
                        if local_info:
                            checkpoint = self._entry_from_local_info(local_info, 'checkpoint')
                        else:
                            logger.warning(f"Could not resolve checkpoint name to a local model: {checkpoint_name}")
                    else:
                        checkpoint_id = checkpoint_match.group(1)
                        checkpoint_version_id = checkpoint_match.group(2)
                        checkpoint = {
                            'id': checkpoint_version_id,
                            'modelId': checkpoint_id,
                            'name': f"Checkpoint {checkpoint_id}",
                            'version': '',
                            'type': 'checkpoint'
                        }
                        
                        # Get additional checkpoint info from Civitai
                        if metadata_provider:
                            try:
                                civitai_info_tuple = await metadata_provider.get_model_version_info(checkpoint_version_id)
                                civitai_info, _ = civitai_info_tuple if isinstance(civitai_info_tuple, tuple) else (civitai_info_tuple, None)
                                # Populate checkpoint with Civitai info
                                checkpoint = await self.populate_checkpoint_from_civitai(checkpoint, civitai_info)
                            except Exception as e:
                                logger.error(f"Error fetching Civitai info for checkpoint: {e}")
            
            # Extract generation parameters
            gen_params = {}
            
            # First try to get from extraMetadata
            if 'extraMetadata' in data:
                try:
                    # extraMetadata is a JSON string that needs to be parsed
                    extra_metadata = json.loads(data['extraMetadata'])
                    
                    # Map fields from extraMetadata to our standard format
                    mapping = {
                        'prompt': 'prompt',
                        'negativePrompt': 'negative_prompt',
                        'steps': 'steps',
                        'sampler': 'sampler',
                        'cfgScale': 'cfg_scale',
                        'seed': 'seed'
                    }
                    
                    for src_key, dest_key in mapping.items():
                        if src_key in extra_metadata:
                            gen_params[dest_key] = extra_metadata[src_key]
                    
                    # If size info is available, format as "width x height"
                    if 'width' in extra_metadata and 'height' in extra_metadata:
                        gen_params['size'] = f"{extra_metadata['width']}x{extra_metadata['height']}"
                    
                except Exception as e:
                    logger.error(f"Error parsing extraMetadata: {e}")
            
            # If extraMetadata doesn't have all the info, try to get from nodes
            if not gen_params or len(gen_params) < 3:  # At least we want prompt, negative_prompt, and steps
                # Find positive prompt node
                positive_nodes = {k: v for k, v in data.items() if isinstance(v, dict) and 
                                v.get('class_type', '').endswith('CLIPTextEncode') and 
                                v.get('_meta', {}).get('title') == 'Positive'}
                
                if positive_nodes:
                    positive_node = next(iter(positive_nodes.values()))
                    if 'inputs' in positive_node and 'text' in positive_node['inputs']:
                        gen_params['prompt'] = positive_node['inputs']['text']
                
                # Find negative prompt node
                negative_nodes = {k: v for k, v in data.items() if isinstance(v, dict) and 
                                v.get('class_type', '').endswith('CLIPTextEncode') and 
                                v.get('_meta', {}).get('title') == 'Negative'}
                
                if negative_nodes:
                    negative_node = next(iter(negative_nodes.values()))
                    if 'inputs' in negative_node and 'text' in negative_node['inputs']:
                        gen_params['negative_prompt'] = negative_node['inputs']['text']
                
                # Find KSampler node for other parameters
                ksampler_nodes = {k: v for k, v in data.items() if isinstance(v, dict) and v.get('class_type') == 'KSampler'}
                
                if ksampler_nodes:
                    ksampler_node = next(iter(ksampler_nodes.values()))
                    if 'inputs' in ksampler_node:
                        inputs = ksampler_node['inputs']
                        if 'sampler_name' in inputs:
                            gen_params['sampler'] = inputs['sampler_name']
                        if 'steps' in inputs:
                            gen_params['steps'] = inputs['steps']
                        if 'cfg' in inputs:
                            gen_params['cfg_scale'] = inputs['cfg']
                        if 'seed' in inputs:
                            gen_params['seed'] = inputs['seed']
            
            # Determine base model from loras info
            base_model = None
            if loras:
                # Use the most common base model from loras
                base_models = [lora['baseModel'] for lora in loras if lora.get('baseModel')]
                if base_models:
                    from collections import Counter
                    base_model_counts = Counter(base_models)
                    base_model = base_model_counts.most_common(1)[0][0]

            embeddings = []
            embedding_names = set()
            for prompt_text in (
                gen_params.get('prompt'),
                gen_params.get('negative_prompt'),
            ):
                if not isinstance(prompt_text, str):
                    continue
                for embedding_name in re.findall(
                    r'(?i)(?:embedding|embed):([^\s,;()\[\]{}]+)',
                    prompt_text,
                ):
                    normalized_name = embedding_name.strip()
                    key = normalized_name.lower()
                    if not normalized_name or key in embedding_names:
                        continue
                    embedding_names.add(key)
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
            
            return {
                'base_model': base_model,
                'loras': loras,
                'embeddings': embeddings,
                'checkpoint': checkpoint,
                'gen_params': gen_params,
                'from_comfy_metadata': True
            }
            
        except Exception as e:
            logger.error(f"Error parsing ComfyUI metadata: {e}", exc_info=True)
            return {"error": str(e), "loras": []}
