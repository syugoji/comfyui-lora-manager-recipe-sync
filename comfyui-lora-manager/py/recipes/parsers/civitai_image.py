"""Parser for Civitai image metadata format."""

import json
import logging
import re
from typing import Dict, Any, Union
from ..base import RecipeMetadataParser
from ..constants import GEN_PARAM_KEYS
from ...services.metadata_service import get_default_metadata_provider
from ...config import config

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 1.0) -> float:
    """Convert optional Civitai numeric fields without failing on null/blank values."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resource_kind(value: Any) -> str:
    normalized = str(value or "").lower().replace("_", "").replace(" ", "")
    if normalized in {"textualinversion", "embedding"}:
        return "embedding"
    if normalized in {"model", "checkpoint"}:
        return "checkpoint"
    if normalized in {"lora", "locon", "lycoris", "dora", "hypernet"}:
        return "lora"
    return normalized


class CivitaiApiMetadataParser(RecipeMetadataParser):
    """Parser for Civitai image metadata format"""

    def is_metadata_matching(self, metadata) -> bool:
        """Check if the metadata matches the Civitai image metadata format

        Args:
            metadata: The metadata from the image (dict)

        Returns:
            bool: True if this parser can handle the metadata
        """
        if not metadata or not isinstance(metadata, dict):
            return False

        def has_markers(payload: Dict[str, Any]) -> bool:
            # Check for common CivitAI image metadata fields
            civitai_image_fields = (
                "resources",
                "civitaiResources",
                "additionalResources",
                "hashes",
                "prompt",
                "negativePrompt",
                "steps",
                "sampler",
                "cfgScale",
                "seed",
                "width",
                "height",
                "Model",
                "Model hash",
                "modelVersionIds",
            )
            return any(key in payload for key in civitai_image_fields)

        # Check the main metadata object
        if has_markers(metadata):
            return True

        # Check for LoRA hash patterns
        hashes = metadata.get("hashes")
        if isinstance(hashes, dict) and any(
            str(key).lower().startswith("lora:") for key in hashes
        ):
            return True

        # Check nested meta object (common in CivitAI image responses)
        nested_meta = metadata.get("meta")
        if isinstance(nested_meta, dict):
            if has_markers(nested_meta):
                return True

            # Also check for LoRA hash patterns in nested meta
            hashes = nested_meta.get("hashes")
            if isinstance(hashes, dict) and any(
                str(key).lower().startswith("lora:") for key in hashes
            ):
                return True

        return False

    async def parse_metadata(  # type: ignore[override]
        self, user_comment, recipe_scanner=None, civitai_client=None,
        local_cache: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Parse metadata from Civitai image format

        Args:
            user_comment: The metadata from the image (dict)
            recipe_scanner: Optional recipe scanner service
            civitai_client: Optional Civitai API client (deprecated, use metadata_provider instead)
            local_cache: Optional dict mapping sha256/autov3 hash → scanner cache item.
                         When provided, matching models skip CivitAI API calls.

        Returns:
            Dict containing parsed recipe data
        """
        metadata: Dict[str, Any] = user_comment  # type: ignore[assignment]
        metadata = user_comment
        try:
            # Get metadata provider instead of using civitai_client directly
            metadata_provider = await get_default_metadata_provider()

            # Civitai image responses may wrap the actual metadata inside a "meta" key
            if (
                isinstance(metadata, dict)
                and "meta" in metadata
                and isinstance(metadata["meta"], dict)
            ):
                inner_meta = metadata["meta"]
                if any(
                    key in inner_meta
                    for key in (
                        "resources",
                        "civitaiResources",
                        "additionalResources",
                        "hashes",
                        "prompt",
                        "negativePrompt",
                    )
                ):
                    metadata = inner_meta

            # Initialize result structure
            result = {
                "base_model": None,
                "loras": [],
                "embeddings": [],
                "model": None,
                "gen_params": {},
                "from_civitai_image": True,
            }

            # Track already added LoRAs to prevent duplicates
            added_loras = {}  # key: model_version_id or hash, value: index in result["loras"]
            added_embeddings = set()

            # Extract hash information from hashes field for LoRA matching
            lora_hashes = {}
            embedding_hashes = {}
            if "hashes" in metadata and isinstance(metadata["hashes"], dict):
                for key, hash_value in metadata["hashes"].items():
                    key_str = str(key)
                    if key_str.lower().startswith("lora:"):
                        lora_name = key_str.split(":", 1)[1]
                        lora_hashes[lora_name] = hash_value
                    elif ":" in key_str:
                        prefix, embedding_name = key_str.split(":", 1)
                        if _resource_kind(prefix) == "embedding":
                            embedding_hashes[embedding_name] = hash_value

            # Extract prompt and negative prompt
            if "prompt" in metadata:
                result["gen_params"]["prompt"] = metadata["prompt"]

            if "negativePrompt" in metadata:
                result["gen_params"]["negative_prompt"] = metadata["negativePrompt"]

            # Extract other generation parameters
            param_mapping = {
                "steps": "steps",
                "sampler": "sampler",
                "cfgScale": "cfg_scale",
                "seed": "seed",
                "Size": "size",
                "clipSkip": "clip_skip",
                "Schedule type": "scheduler",
                "scheduler": "scheduler",
                "Denoising strength": "denoising_strength",
                "Model": "model",
                "VAE": "vae",
                "Hires upscale": "hires_upscale",
                "Hires resize": "hires_resize",
                "Hires steps": "hires_steps",
                "Hires upscaler": "hires_upscaler",
                "Hires CFG Scale": "hires_cfg_scale",
            }

            for civitai_key, our_key in param_mapping.items():
                if civitai_key in metadata and our_key in GEN_PARAM_KEYS:
                    result["gen_params"][our_key] = metadata[civitai_key]

            # Extract base model information - directly if available
            if "baseModel" in metadata:
                result["base_model"] = metadata["baseModel"]
            elif "Model hash" in metadata and metadata_provider:
                model_hash = metadata["Model hash"]
                model_info, error = await metadata_provider.get_model_by_hash(
                    model_hash
                )
                if model_info:
                    result["base_model"] = model_info.get("baseModel", "")
            elif "Model" in metadata and isinstance(metadata.get("resources"), list):
                # Try to find base model in resources
                for resource in metadata.get("resources", []):
                    if resource.get("type") == "model" and resource.get(
                        "name"
                    ) == metadata.get("Model"):
                        # This is likely the checkpoint model
                        if metadata_provider and resource.get("hash"):
                            (
                                model_info,
                                error,
                            ) = await metadata_provider.get_model_by_hash(
                                resource.get("hash")
                            )
                            if model_info:
                                result["base_model"] = model_info.get("baseModel", "")

            base_model_counts = {}

            # Process standard resources array
            if "resources" in metadata and isinstance(metadata["resources"], list):
                for resource in metadata["resources"]:
                    resource_type = _resource_kind(resource.get("type", "lora"))

                    # Track resources with type "model" — these are checkpoint models.
                    # The resources array is the most reliable source for checkpoint
                    # identification because it has an explicit type field and hash,
                    # unlike modelVersionIds which is a flat list with no type info.
                    if resource_type == "checkpoint":
                        checkpoint_entry = {
                            "id": 0,
                            "modelId": 0,
                            "name": resource.get("name", "Unknown Model"),
                            "version": "",
                            "type": resource.get("type", "model"),
                            "existsLocally": False,
                            "localPath": None,
                            "file_name": resource.get("name", ""),
                            "hash": resource.get("hash", "") or "",
                            "thumbnailUrl": "/loras_static/images/no-preview.png",
                            "baseModel": "",
                            "size": 0,
                            "downloadUrl": "",
                            "isDeleted": False,
                        }

                        # Try to look up base model from the checkpoint hash
                        cp_hash = checkpoint_entry.get("hash")
                        if cp_hash and metadata_provider:
                            local_cached = local_cache.get(cp_hash) if local_cache else None
                            if local_cached:
                                self._populate_entry_from_cache(
                                    checkpoint_entry, local_cached
                                )
                                bm = checkpoint_entry.get("baseModel", "")
                                if bm and not result["base_model"]:
                                    result["base_model"] = bm
                            else:
                                try:
                                    civitai_info = (
                                        await metadata_provider.get_model_by_hash(
                                            cp_hash
                                        )
                                    )
                                    civitai_data, error_msg = (
                                        (civitai_info, None)
                                        if not isinstance(civitai_info, tuple)
                                        else civitai_info
                                    )
                                    if civitai_data and error_msg != "Model not found":
                                        if 'model' in civitai_data and 'name' in civitai_data['model']:
                                            checkpoint_entry['name'] = civitai_data['model']['name']
                                        checkpoint_entry['id'] = civitai_data.get('id', 0)
                                        checkpoint_entry['modelId'] = civitai_data.get('modelId', 0)
                                        if 'name' in civitai_data:
                                            checkpoint_entry['version'] = civitai_data['name']
                                        base_model = civitai_data.get('baseModel', '')
                                        if base_model:
                                            checkpoint_entry['baseModel'] = base_model
                                            if not result['base_model']:
                                                result['base_model'] = base_model
                                except Exception as e:
                                    logger.error(
                                        f"Error fetching checkpoint info for hash "
                                        f"{cp_hash}: {e}"
                                    )

                        if result["model"] is None:
                            result["model"] = checkpoint_entry
                        continue

                    if resource_type == "embedding":
                        embedding_hash = str(resource.get("hash") or "").lower()
                        version_id = resource.get("modelVersionId")
                        embedding_key = str(
                            version_id
                            or embedding_hash
                            or resource.get("name")
                            or ""
                        ).lower()
                        if not embedding_key or embedding_key in added_embeddings:
                            continue
                        embedding_entry = {
                            "id": version_id or 0,
                            "modelId": resource.get("modelId", 0),
                            "name": resource.get("name", "Unknown Embedding"),
                            "version": resource.get("modelVersionName", ""),
                            "type": "embedding",
                            "hash": embedding_hash,
                            "existsLocally": False,
                            "localPath": None,
                            "file_name": resource.get("name", "Unknown Embedding"),
                            "thumbnailUrl": "/loras_static/images/no-preview.png",
                            "baseModel": "",
                            "size": 0,
                            "downloadUrl": "",
                            "isDeleted": False,
                        }
                        if metadata_provider and (embedding_hash or version_id):
                            try:
                                civitai_info = (
                                    await metadata_provider.get_model_by_hash(embedding_hash)
                                    if embedding_hash
                                    else await metadata_provider.get_model_version_info(str(version_id))
                                )
                                populated_embedding = await self.populate_embedding_from_civitai(
                                    embedding_entry,
                                    civitai_info,
                                    recipe_scanner,
                                    embedding_hash or None,
                                )
                                if populated_embedding is None:
                                    continue
                                embedding_entry = populated_embedding
                            except Exception as e:
                                logger.error(
                                    "Error fetching embedding %s: %s",
                                    embedding_entry["name"],
                                    e,
                                )
                        result["embeddings"].append(embedding_entry)
                        added_embeddings.add(embedding_key)
                        if embedding_hash:
                            added_embeddings.add(embedding_hash)
                        continue

                    # Modified to process resources without a type field as potential LoRAs
                    if resource_type == "lora":
                        lora_hash = resource.get("hash", "")

                        # Try to get hash from the hashes field if not present in resource
                        if not lora_hash and resource.get("name"):
                            lora_hash = lora_hashes.get(resource["name"], "")

                        # Skip LoRAs without proper identification (hash or modelVersionId)
                        if not lora_hash and not resource.get("modelVersionId"):
                            logger.debug(
                                f"Skipping LoRA resource '{resource.get('name', 'Unknown')}' - no hash or modelVersionId"
                            )
                            continue

                        # Skip if we've already added this LoRA by hash
                        if lora_hash and lora_hash in added_loras:
                            continue

                        lora_entry = {
                            "name": resource.get("name", "Unknown LoRA"),
                            "type": "lora",
                            "weight": _safe_float(resource.get("weight")),
                            "hash": lora_hash,
                            "existsLocally": False,
                            "localPath": None,
                            "file_name": resource.get("name", "Unknown"),
                            "thumbnailUrl": "/loras_static/images/no-preview.png",
                            "baseModel": "",
                            "size": 0,
                            "downloadUrl": "",
                            "isDeleted": False,
                        }

                        # Try to get info from Civitai if hash is available
                        if lora_hash and metadata_provider:
                            local_cached = local_cache.get(lora_hash) if local_cache else None
                            if local_cached:
                                self._populate_entry_from_cache(
                                    lora_entry, local_cached
                                )
                                # Track by version ID for deduplication
                                if lora_entry.get("id"):
                                    added_loras[str(lora_entry["id"])] = len(
                                        result["loras"]
                                    )
                            else:
                                try:
                                    civitai_info = (
                                        await self.get_lora_metadata_by_hash(
                                            metadata_provider, lora_hash
                                        )
                                    )

                                    populated_entry = await self.populate_lora_from_civitai(
                                        lora_entry,
                                        civitai_info,
                                        recipe_scanner,
                                        base_model_counts,
                                        lora_hash,
                                    )

                                    if populated_entry is None:
                                        continue  # Skip invalid LoRA types

                                    lora_entry = populated_entry

                                    # If we have a version ID from Civitai, track it for deduplication
                                    if "id" in lora_entry and lora_entry["id"]:
                                        added_loras[str(lora_entry["id"])] = len(
                                            result["loras"]
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Error fetching Civitai info for LoRA hash {lora_entry['hash']}: {e}"
                                    )

                        # Track by hash if we have it
                        if lora_hash:
                            added_loras[lora_hash] = len(result["loras"])

                        result["loras"].append(lora_entry)

            # Process civitaiResources array
            if "civitaiResources" in metadata and isinstance(
                metadata["civitaiResources"], list
            ):
                for resource in metadata["civitaiResources"]:
                    # Get resource type and identifier
                    resource_type = _resource_kind(resource.get("type"))
                    version_id = str(resource.get("modelVersionId", ""))

                    if resource_type == "checkpoint":
                        checkpoint_entry = {
                            "id": resource.get("modelVersionId", 0),
                            "modelId": resource.get("modelId", 0),
                            "name": resource.get("modelName", "Unknown Checkpoint"),
                            "version": resource.get("modelVersionName", ""),
                            "type": resource.get("type", "checkpoint"),
                            "existsLocally": False,
                            "localPath": None,
                            "file_name": resource.get("modelName", ""),
                            "hash": resource.get("hash", "") or "",
                            "thumbnailUrl": "/loras_static/images/no-preview.png",
                            "baseModel": "",
                            "size": 0,
                            "downloadUrl": "",
                            "isDeleted": False,
                        }

                        if version_id and metadata_provider:
                            try:
                                civitai_info = (
                                    await metadata_provider.get_model_version_info(
                                        version_id
                                    )
                                )

                                checkpoint_entry = (
                                    await self.populate_checkpoint_from_civitai(
                                        checkpoint_entry, civitai_info
                                    )
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error fetching Civitai info for checkpoint version {version_id}: {e}"
                                )

                        if result["model"] is None:
                            result["model"] = checkpoint_entry

                        continue

                    if resource_type == "embedding":
                        embedding_key = str(
                            version_id
                            or resource.get("hash")
                            or resource.get("modelName")
                            or ""
                        ).lower()
                        if not embedding_key or embedding_key in added_embeddings:
                            continue
                        embedding_entry = {
                            "id": resource.get("modelVersionId", 0),
                            "modelId": resource.get("modelId", 0),
                            "name": resource.get("modelName", "Unknown Embedding"),
                            "version": resource.get("modelVersionName", ""),
                            "type": "embedding",
                            "hash": str(resource.get("hash") or "").lower(),
                            "existsLocally": False,
                            "localPath": None,
                            "file_name": resource.get("modelName", ""),
                            "thumbnailUrl": "/loras_static/images/no-preview.png",
                            "baseModel": "",
                            "size": 0,
                            "downloadUrl": "",
                            "isDeleted": False,
                        }
                        if version_id and metadata_provider:
                            try:
                                civitai_info = await metadata_provider.get_model_version_info(
                                    version_id
                                )
                                populated_embedding = await self.populate_embedding_from_civitai(
                                    embedding_entry, civitai_info, recipe_scanner
                                )
                                if populated_embedding is None:
                                    continue
                                embedding_entry = populated_embedding
                            except Exception as e:
                                logger.error(
                                    "Error fetching Civitai embedding version %s: %s",
                                    version_id,
                                    e,
                                )
                        result["embeddings"].append(embedding_entry)
                        added_embeddings.add(embedding_key)
                        continue

                    # Skip if we've already added this LoRA
                    if version_id and version_id in added_loras:
                        continue

                    # Initialize lora entry
                    lora_entry = {
                        "id": resource.get("modelVersionId", 0),
                        "modelId": resource.get("modelId", 0),
                        "name": resource.get("modelName", "Unknown LoRA"),
                        "version": resource.get("modelVersionName", ""),
                        "type": resource.get("type", "lora"),
                        "weight": round(_safe_float(resource.get("weight")), 2),
                        "existsLocally": False,
                        "thumbnailUrl": "/loras_static/images/no-preview.png",
                        "baseModel": "",
                        "size": 0,
                        "downloadUrl": "",
                        "isDeleted": False,
                    }

                    # Try to get info from Civitai if modelVersionId is available
                    if version_id and metadata_provider:
                        try:
                            # Use get_model_version_info instead of get_model_version
                            civitai_info = (
                                await metadata_provider.get_model_version_info(
                                    version_id
                                )
                            )

                            populated_entry = await self.populate_lora_from_civitai(
                                lora_entry,
                                civitai_info,
                                recipe_scanner,
                                base_model_counts,
                            )

                            if populated_entry is None:
                                continue  # Skip invalid LoRA types

                            lora_entry = populated_entry
                        except Exception as e:
                            logger.error(
                                f"Error fetching Civitai info for model version {version_id}: {e}"
                            )

                    # Track this LoRA in our deduplication dict
                    if version_id:
                        added_loras[version_id] = len(result["loras"])

                    result["loras"].append(lora_entry)

            # Process additionalResources array
            if "additionalResources" in metadata and isinstance(
                metadata["additionalResources"], list
            ):
                for resource in metadata["additionalResources"]:
                    resource_kind = _resource_kind(resource.get("type", "lora"))
                    if resource_kind not in {"lora", "embedding"}:
                        continue

                    lora_type = resource.get("type", "lora")
                    name = resource.get("name", "")

                    # Extract ID from URN format if available
                    version_id = None
                    if name and "civitai:" in name:
                        parts = name.split("@")
                        if len(parts) > 1:
                            version_id = parts[1]

                            # Skip if we've already added this LoRA
                            if version_id in added_loras:
                                continue

                    if resource_kind == "embedding":
                        embedding_key = str(
                            version_id or resource.get("hash") or name or ""
                        ).lower()
                        if not embedding_key or embedding_key in added_embeddings:
                            continue
                        embedding_entry = {
                            "id": version_id or resource.get("modelVersionId", 0),
                            "modelId": resource.get("modelId", 0),
                            "name": resource.get("modelName") or name or "Unknown Embedding",
                            "version": resource.get("modelVersionName", ""),
                            "type": "embedding",
                            "hash": str(resource.get("hash") or "").lower(),
                            "existsLocally": False,
                            "localPath": None,
                            "file_name": name,
                            "thumbnailUrl": "/loras_static/images/no-preview.png",
                            "baseModel": "",
                            "size": 0,
                            "downloadUrl": "",
                            "isDeleted": False,
                        }
                        if version_id and metadata_provider:
                            try:
                                civitai_info = await metadata_provider.get_model_version_info(
                                    version_id
                                )
                                populated_embedding = await self.populate_embedding_from_civitai(
                                    embedding_entry, civitai_info, recipe_scanner
                                )
                                if populated_embedding is None:
                                    continue
                                embedding_entry = populated_embedding
                            except Exception as e:
                                logger.error(
                                    "Error fetching additional embedding %s: %s",
                                    name,
                                    e,
                                )
                        result["embeddings"].append(embedding_entry)
                        added_embeddings.add(embedding_key)
                        continue

                    lora_entry = {
                        "name": name,
                        "type": lora_type,
                        "weight": _safe_float(resource.get("strength")),
                        "hash": "",
                        "existsLocally": False,
                        "localPath": None,
                        "file_name": name,
                        "thumbnailUrl": "/loras_static/images/no-preview.png",
                        "baseModel": "",
                        "size": 0,
                        "downloadUrl": "",
                        "isDeleted": False,
                    }

                    # If we have a version ID and metadata provider, try to get more info
                    if version_id and metadata_provider:
                        try:
                            # Use get_model_version_info with the version ID
                            civitai_info = (
                                await metadata_provider.get_model_version_info(
                                    version_id
                                )
                            )

                            populated_entry = await self.populate_lora_from_civitai(
                                lora_entry,
                                civitai_info,
                                recipe_scanner,
                                base_model_counts,
                            )

                            if populated_entry is None:
                                continue  # Skip invalid LoRA types

                            lora_entry = populated_entry

                            # Track this LoRA for deduplication
                            if version_id:
                                added_loras[version_id] = len(result["loras"])
                        except Exception as e:
                            logger.error(
                                f"Error fetching Civitai info for model ID {version_id}: {e}"
                            )

                    result["loras"].append(lora_entry)

            # Process modelVersionIds from Civitai image API.
            # These are version IDs returned at root level of the API response.
            # When resources or civitaiResources are already present in metadata
            # (which they are when ?withMeta=true is passed), those sections have
            # complete hash/type information — modelVersionIds is a fallback for
            # when meta is null and only the flat ID list is available. Skipping
            # it here avoids duplicates: the same file hash often resolves to
            # different version IDs via hash lookup (resources) vs the original
            # version ID in modelVersionIds, and both paths would create entries.
            if (
                "modelVersionIds" in metadata
                and isinstance(metadata["modelVersionIds"], list)
            ):

                for version_id in metadata["modelVersionIds"]:
                    version_id_str = str(version_id)

                    # Skip if we've already added this LoRA by version ID
                    if version_id_str in added_loras:
                        continue

                    # Skip if this version ID is already the recipe's checkpoint
                    # (resolved earlier from embedded resources/Model hash,
                    # avoiding a duplicate CivitAI API call).
                    existing_model = result.get("model")
                    if existing_model and str(existing_model.get("id")) == version_id_str:
                        continue

                    # Initialize lora entry with version ID
                    lora_entry = {
                        "id": version_id,
                        "modelId": 0,
                        "name": "Unknown LoRA",
                        "version": "",
                        "type": "lora",
                        "weight": 1.0,
                        "existsLocally": False,
                        "thumbnailUrl": "/loras_static/images/no-preview.png",
                        "baseModel": "",
                        "size": 0,
                        "downloadUrl": "",
                        "isDeleted": False,
                    }

                    # Fetch model info from Civitai
                    if metadata_provider and version_id_str:
                        try:
                            civitai_info = (
                                await metadata_provider.get_model_version_info(
                                    version_id_str
                                )
                            )

                            populated_entry = await self.populate_lora_from_civitai(
                                lora_entry,
                                civitai_info,
                                recipe_scanner,
                                base_model_counts,
                            )

                            if populated_entry is None:
                                civitai_data = (
                                    civitai_info[0]
                                    if isinstance(civitai_info, tuple)
                                    else civitai_info
                                )
                                civitai_model_type = _resource_kind(
                                    ((civitai_data or {}).get("model") or {}).get("type")
                                )
                                if civitai_model_type == "embedding":
                                    embedding_entry = {
                                        "id": version_id,
                                        "modelId": 0,
                                        "name": "Unknown Embedding",
                                        "version": "",
                                        "type": "embedding",
                                        "hash": "",
                                        "existsLocally": False,
                                        "localPath": None,
                                        "file_name": "",
                                        "thumbnailUrl": "/loras_static/images/no-preview.png",
                                        "baseModel": "",
                                        "size": 0,
                                        "downloadUrl": "",
                                        "isDeleted": False,
                                    }
                                    populated_embedding = await self.populate_embedding_from_civitai(
                                        embedding_entry, civitai_info, recipe_scanner
                                    )
                                    if populated_embedding is not None:
                                        embedding_hash = str(
                                            populated_embedding.get("hash") or ""
                                        ).lower()
                                        embedding_key = str(version_id)
                                        if (
                                            embedding_key not in added_embeddings
                                            and embedding_hash not in added_embeddings
                                        ):
                                            result["embeddings"].append(populated_embedding)
                                            added_embeddings.add(embedding_key)
                                            if embedding_hash:
                                                added_embeddings.add(embedding_hash)
                                    continue

                                # Not a LoRA — try as checkpoint (only if we
                                # don't already have one).  Reuses the same
                                # civitai_info from the API call above so no
                                # extra query is made.
                                if result["model"] is None:
                                    checkpoint_entry = {
                                        "id": version_id,
                                        "modelId": 0,
                                        "name": "Unknown Model",
                                        "version": "",
                                        "type": "checkpoint",
                                        "existsLocally": False,
                                        "localPath": None,
                                        "file_name": "",
                                        "hash": "",
                                        "thumbnailUrl": (
                                            "/loras_static/images/no-preview.png"
                                        ),
                                        "baseModel": "",
                                        "size": 0,
                                        "downloadUrl": "",
                                        "isDeleted": False,
                                    }
                                    cp_populated = await (
                                        self.populate_checkpoint_from_civitai(
                                            checkpoint_entry, civitai_info
                                        )
                                    )
                                    if cp_populated.get("modelId"):
                                        result["model"] = cp_populated
                                continue  # Not a LoRA, don't add to loras

                            lora_entry = populated_entry

                            populated_hash = str(lora_entry.get("hash") or "").lower()
                            if populated_hash and populated_hash in added_loras:
                                continue

                        except Exception as e:
                            logger.error(
                                f"Error fetching Civitai info for model version {version_id}: {e}"
                            )

                    # Track this LoRA for deduplication
                    if version_id_str:
                        added_loras[version_id_str] = len(result["loras"])

                    result["loras"].append(lora_entry)

            # If we found LoRA hashes in the metadata but haven't already
            # populated entries for them, fall back to creating LoRAs from
            # the hashes section. Some Civitai image responses only include
            # LoRA information here without explicit resources entries.
            for lora_name, lora_hash in lora_hashes.items():
                if not lora_hash:
                    continue

                # Skip LoRAs we've already added via resources or other fields
                if lora_hash in added_loras:
                    continue

                lora_entry = {
                    "name": lora_name,
                    "type": "lora",
                    "weight": 1.0,
                    "hash": lora_hash,
                    "existsLocally": False,
                    "localPath": None,
                    "file_name": lora_name,
                    "thumbnailUrl": "/loras_static/images/no-preview.png",
                    "baseModel": "",
                    "size": 0,
                    "downloadUrl": "",
                    "isDeleted": False,
                }

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
                            lora_hash,
                        )

                        if populated_entry is None:
                            continue

                        lora_entry = populated_entry

                        if "id" in lora_entry and lora_entry["id"]:
                            added_loras[str(lora_entry["id"])] = len(result["loras"])
                    except Exception as e:
                        logger.error(
                            f"Error fetching Civitai info for LoRA hash {lora_hash}: {e}"
                        )

                added_loras[lora_hash] = len(result["loras"])
                result["loras"].append(lora_entry)

            for embedding_name, embedding_hash in embedding_hashes.items():
                normalized_hash = str(embedding_hash or "").lower()
                if not normalized_hash or normalized_hash in added_embeddings:
                    continue
                embedding_entry = {
                    "name": embedding_name,
                    "type": "embedding",
                    "hash": normalized_hash,
                    "existsLocally": False,
                    "localPath": None,
                    "file_name": embedding_name,
                    "thumbnailUrl": "/loras_static/images/no-preview.png",
                    "baseModel": "",
                    "size": 0,
                    "downloadUrl": "",
                    "isDeleted": False,
                }
                if metadata_provider:
                    try:
                        civitai_info = await metadata_provider.get_model_by_hash(
                            normalized_hash
                        )
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
                        logger.error(
                            "Error resolving embedding hash %s: %s",
                            normalized_hash,
                            e,
                        )
                if embedding_entry.get("isDeleted"):
                    embedding_entry["isDeleted"] = False
                    embedding_entry["unresolved"] = True
                result["embeddings"].append(embedding_entry)
                added_embeddings.add(normalized_hash)

            # Check for LoRA info in the format "Lora_0 Model hash", "Lora_0 Model name", etc.
            lora_index = 0
            while (
                f"Lora_{lora_index} Model hash" in metadata
                and f"Lora_{lora_index} Model name" in metadata
            ):
                lora_hash = metadata[f"Lora_{lora_index} Model hash"]
                lora_name = metadata[f"Lora_{lora_index} Model name"]
                lora_strength_model = float(
                    metadata.get(f"Lora_{lora_index} Strength model", 1.0)
                )

                # Skip if we've already added this LoRA by hash
                if lora_hash and lora_hash in added_loras:
                    lora_index += 1
                    continue

                lora_entry = {
                    "name": lora_name,
                    "type": "lora",
                    "weight": lora_strength_model,
                    "hash": lora_hash,
                    "existsLocally": False,
                    "localPath": None,
                    "file_name": lora_name,
                    "thumbnailUrl": "/loras_static/images/no-preview.png",
                    "baseModel": "",
                    "size": 0,
                    "downloadUrl": "",
                    "isDeleted": False,
                }

                # Try to get info from Civitai if hash is available
                if lora_entry["hash"] and metadata_provider:
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

                        if populated_entry is None:
                            lora_index += 1
                            continue  # Skip invalid LoRA types

                        lora_entry = populated_entry

                        # If we have a version ID from Civitai, track it for deduplication
                        if "id" in lora_entry and lora_entry["id"]:
                            added_loras[str(lora_entry["id"])] = len(result["loras"])
                    except Exception as e:
                        logger.error(
                            f"Error fetching Civitai info for LoRA hash {lora_entry['hash']}: {e}"
                        )

                # Track by hash if we have it
                if lora_hash:
                    added_loras[lora_hash] = len(result["loras"])

                result["loras"].append(lora_entry)

                lora_index += 1

            # If base model wasn't found earlier, use the most common one from LoRAs
            if not result["base_model"] and base_model_counts:
                result["base_model"] = max(
                    base_model_counts.items(), key=lambda x: x[1]
                )[0]

            # Generation Data frequently has only a display model name. Keep
            # it as an unresolved checkpoint instead of silently dropping it.
            if result["model"] is None and metadata.get("Model"):
                model_name = str(metadata["Model"]).strip()
                result["model"] = {
                    "id": 0,
                    "modelId": 0,
                    "name": model_name,
                    "version": "",
                    "type": "checkpoint",
                    "existsLocally": False,
                    "localPath": None,
                    "file_name": model_name,
                    "hash": str(metadata.get("Model hash") or "").lower(),
                    "thumbnailUrl": "/loras_static/images/no-preview.png",
                    "baseModel": result.get("base_model") or "",
                    "size": 0,
                    "downloadUrl": "",
                    "isDeleted": False,
                    "unresolved": True,
                }

            known_embedding_names = {
                re.sub(
                    r"[^a-z0-9]+",
                    "",
                    str(item.get("name") or item.get("file_name") or "").lower(),
                )
                for item in result["embeddings"]
            }
            for prompt_text in (
                result["gen_params"].get("prompt"),
                result["gen_params"].get("negative_prompt"),
            ):
                if not isinstance(prompt_text, str):
                    continue
                for embedding_name in re.findall(
                    r"(?i)(?:embedding|embed):([^\s,;()\[\]{}]+)",
                    prompt_text,
                ):
                    normalized_name = embedding_name.strip()
                    normalized_key = re.sub(
                        r"[^a-z0-9]+", "", normalized_name.lower()
                    )
                    if (
                        not normalized_name
                        or normalized_key in known_embedding_names
                    ):
                        continue
                    result["embeddings"].append(
                        {
                            "name": normalized_name,
                            "file_name": normalized_name,
                            "type": "embedding",
                            "hash": "",
                            "existsLocally": False,
                            "localPath": None,
                            "isDeleted": False,
                            "unresolved": True,
                        }
                    )
                    known_embedding_names.add(normalized_key)

            return result

        except Exception as e:
            logger.error(f"Error parsing Civitai image metadata: {e}", exc_info=True)
            return {"error": str(e), "loras": []}

    @staticmethod
    def _populate_entry_from_cache(
        entry: dict[str, Any],
        cache_item: dict[str, Any],
    ) -> None:
        """Fill a lora/checkpoint entry from a scanner cache item.

        Avoids CivitAI API calls for models that exist locally.
        Mirrors the population logic in
        ``RecipeMetadataParser.populate_lora_from_civitai()`` but operates
        entirely on cached data.
        """
        civ = cache_item.get("civitai") or {}
        if isinstance(civ, dict):
            if civ.get("id") is not None:
                entry["id"] = civ["id"]
            if civ.get("modelId") is not None:
                entry["modelId"] = civ["modelId"]
            if civ.get("name"):
                entry["version"] = civ["name"]
            cached_name = cache_item.get("model_name")
            if cached_name:
                entry["name"] = cached_name
        entry["existsLocally"] = True
        local_path = cache_item.get("file_path")
        if local_path:
            entry["localPath"] = local_path
        sha256 = cache_item.get("sha256")
        if sha256:
            entry["hash"] = sha256
        if "preview_url" in cache_item:
            entry["thumbnailUrl"] = config.get_preview_static_url(
                cache_item["preview_url"]
            )
        base_model = cache_item.get("base_model", "")
        if base_model:
            entry["baseModel"] = base_model
