"""Services responsible for recipe metadata analysis."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from ...utils.utils import calculate_recipe_fingerprint
from ...utils.civitai_utils import extract_civitai_image_id, rewrite_preview_url
from ...utils.generation_metadata import (
    build_a1111_parameters,
    json_safe_copy,
    normalize_metadata_text,
)
from ...recipes.enrichment import RecipeEnricher
from .errors import (
    RecipeDownloadError,
    RecipeNotFoundError,
    RecipeServiceError,
    RecipeValidationError,
)


@dataclass(frozen=True)
class AnalysisResult:
    """Return payload from analysis operations."""

    payload: dict[str, Any]
    status: int = 200


class RecipeAnalysisService:
    """Extract recipe metadata from various image sources."""

    GENERATION_SOURCE_POLICY = "embedded-first-v1"

    def __init__(
        self,
        *,
        exif_utils,
        recipe_parser_factory,
        downloader_factory: Callable[[], Any],
        metadata_collector: Optional[Callable[[], Any]] = None,
        metadata_processor_cls: Optional[type] = None,
        metadata_registry_cls: Optional[type] = None,
        standalone_mode: bool = False,
        logger,
    ) -> None:
        self._exif_utils = exif_utils
        self._recipe_parser_factory = recipe_parser_factory
        self._downloader_factory = downloader_factory
        self._metadata_collector = metadata_collector
        self._metadata_processor_cls = metadata_processor_cls
        self._metadata_registry_cls = metadata_registry_cls
        self._standalone_mode = standalone_mode
        self._logger = logger

    @staticmethod
    def _has_value(value: Any, *, key: str = "") -> bool:
        if value is None or value == "" or value == [] or value == {}:
            return False
        if key in {"id", "modelId", "modelVersionId"} and (
            value == 0 or value == "0"
        ):
            return False
        return True

    @classmethod
    def merge_generation_params(
        cls,
        embedded: dict[str, Any] | None,
        api: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge generation fields with original-image metadata authoritative."""
        merged = dict(api or {})
        for key, value in (embedded or {}).items():
            if cls._has_value(value, key=key) or key not in merged:
                merged[key] = value
        return merged

    @staticmethod
    def _normalized_resource_names(resource: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for key in ("file_name", "filename", "name", "modelName"):
            value = resource.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            basename = value.replace("\\", "/").rsplit("/", 1)[-1]
            normalized = os.path.splitext(basename)[0].lower()
            normalized = "".join(char for char in normalized if char.isalnum())
            if normalized:
                names.add(normalized)
        return names

    @classmethod
    def _resources_match(
        cls, embedded: dict[str, Any], api: dict[str, Any]
    ) -> bool:
        embedded_id = embedded.get("modelVersionId") or embedded.get("id")
        api_id = api.get("modelVersionId") or api.get("id")
        if embedded_id not in (None, 0, "0") and api_id not in (None, 0, "0"):
            if str(embedded_id) == str(api_id):
                return True

        embedded_hash = str(embedded.get("hash") or "").lower()
        api_hash = str(api.get("hash") or "").lower()
        if len(embedded_hash) >= 8 and len(api_hash) >= 8:
            if embedded_hash.startswith(api_hash) or api_hash.startswith(embedded_hash):
                return True

        return bool(
            cls._normalized_resource_names(embedded)
            & cls._normalized_resource_names(api)
        )

    @classmethod
    def _merge_resource_record(
        cls, primary: dict[str, Any], supplement: dict[str, Any] | None
    ) -> dict[str, Any]:
        merged = dict(supplement or {})
        for key, value in primary.items():
            if cls._has_value(value, key=key) or key not in merged:
                merged[key] = value
        return merged

    @classmethod
    def merge_lora_resources(
        cls,
        embedded: list[dict[str, Any]] | None,
        api: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Use image-declared LoRAs; let API data enrich only matching entries.

        CivitAI's resource list can include a model associated with the page but
        absent from the actual A1111/Comfy workflow.  Therefore unmatched API
        resources must not become loader nodes when the image explicitly lists
        its own networks.
        """
        embedded_items = [dict(item) for item in (embedded or []) if isinstance(item, dict)]
        api_items = [dict(item) for item in (api or []) if isinstance(item, dict)]
        if not embedded_items:
            return api_items

        merged_items: list[dict[str, Any]] = []
        for embedded_item in embedded_items:
            api_match = next(
                (
                    api_item
                    for api_item in api_items
                    if cls._resources_match(embedded_item, api_item)
                ),
                None,
            )
            merged_items.append(cls._merge_resource_record(embedded_item, api_match))
        return merged_items

    @classmethod
    def merge_embedding_resources(
        cls,
        embedded: list[dict[str, Any]] | None,
        api: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Apply the same embedded-authoritative rule to textual inversions."""
        embedded_items = [
            dict(item) for item in (embedded or []) if isinstance(item, dict)
        ]
        api_items = [dict(item) for item in (api or []) if isinstance(item, dict)]
        if not embedded_items:
            return api_items

        merged_items: list[dict[str, Any]] = []
        for embedded_item in embedded_items:
            api_match = next(
                (
                    api_item
                    for api_item in api_items
                    if cls._resources_match(embedded_item, api_item)
                ),
                None,
            )
            merged = cls._merge_resource_record(embedded_item, api_match)
            # A short A1111 AutoV3 hash may be insufficient for a direct API
            # lookup.  Generation Data can still resolve that same embedding
            # by version ID; in that case it is a valid download target.
            if api_match and (
                api_match.get("id") or api_match.get("modelVersionId")
            ):
                merged["isDeleted"] = False
                merged["unresolved"] = False
            merged_items.append(merged)
        return merged_items

    @classmethod
    def apply_generation_source(cls, payload: dict[str, Any]) -> str:
        if payload.get("comfy_prompt") or payload.get("comfy_workflow"):
            source = "embedded_comfy"
        elif isinstance(payload.get("a1111_parameters"), str) and payload[
            "a1111_parameters"
        ].strip():
            source = "embedded_a1111"
        else:
            gen_params = payload.get("gen_params") or {}
            prompt = gen_params.get("prompt") or gen_params.get("positivePrompt")
            source = "civitai_generation_data" if prompt else "reconstructed"
        payload["generation_source"] = source
        payload["generation_source_policy"] = cls.GENERATION_SOURCE_POLICY
        return source

    async def analyze_uploaded_image(
        self,
        *,
        image_bytes: bytes | None,
        recipe_scanner,
    ) -> AnalysisResult:
        """Analyze an uploaded image payload."""

        if not image_bytes:
            raise RecipeValidationError("No image data provided")

        temp_path = self._write_temp_file(image_bytes)
        try:
            metadata = self._exif_utils.extract_image_metadata(temp_path)
            if not metadata:
                return AnalysisResult(
                    {"error": "No metadata found in this image", "loras": []}
                )

            return await self._parse_metadata(
                metadata,
                recipe_scanner=recipe_scanner,
                image_path=None,
                include_image_base64=False,
            )
        finally:
            self._safe_cleanup(temp_path)

    async def analyze_remote_image(
        self,
        *,
        url: str | None,
        recipe_scanner,
        civitai_client,
    ) -> AnalysisResult:
        """Analyze an image accessible via URL, including Civitai integration."""

        if not url:
            raise RecipeValidationError("No URL provided")

        if civitai_client is None:
            raise RecipeServiceError("Civitai client unavailable")

        temp_path = None
        metadata: Optional[dict[str, Any]] = None
        generation_data: Optional[dict[str, Any]] = None
        is_video = False
        extension = ".jpg"  # Default

        try:
            civitai_image_id = extract_civitai_image_id(url)
            if civitai_image_id:
                image_info = await civitai_client.get_image_info(
                    civitai_image_id, source_url=url
                )
                if not image_info:
                    raise RecipeDownloadError(
                        "Failed to fetch image information from Civitai"
                    )

                generation_data_getter = getattr(
                    civitai_client, "get_image_generation_data", None
                )
                if callable(generation_data_getter):
                    try:
                        generation_data = await generation_data_getter(
                            civitai_image_id
                        )
                    except Exception as exc:
                        # The public image response and downloaded image are
                        # still useful fallbacks when the page-only endpoint is
                        # unavailable or rate limited.
                        self._logger.debug(
                            "Civitai page generation data unavailable for %s: %s",
                            civitai_image_id,
                            exc,
                        )

                image_url = image_info.get("url")
                if not image_url:
                    raise RecipeDownloadError("No image URL found in Civitai response")

                is_video = image_info.get("type") == "video"

                # Use optimized preview URLs if possible
                rewritten_url, _ = rewrite_preview_url(
                    image_url, media_type=image_info.get("type")
                )
                if rewritten_url:
                    image_url = rewritten_url

                if is_video:
                    # Extract extension from URL
                    url_path = image_url.split("?")[0].split("#")[0]
                    extension = os.path.splitext(url_path)[1].lower() or ".mp4"
                else:
                    extension = ".jpg"

                temp_path = self._create_temp_path(suffix=extension)
                await self._download_image(image_url, temp_path)

                metadata = image_info.get("meta") if "meta" in image_info else None
                if (
                    isinstance(metadata, dict)
                    and "meta" in metadata
                    and isinstance(metadata["meta"], dict)
                ):
                    metadata = metadata["meta"]

                # Civitai's page-level Generation Data is independent from the
                # metadata embedded in the downloaded file and from /api/v1.
                # Merge it here so prompt/steps/resources remain available even
                # when the JPEG/PNG has no EXIF payload.
                if isinstance(generation_data, dict):
                    page_meta = generation_data.get("meta")
                    merged_metadata: dict[str, Any] = (
                        dict(metadata) if isinstance(metadata, dict) else {}
                    )
                    if isinstance(page_meta, dict):
                        merged_metadata.update(page_meta)

                    page_resources = generation_data.get("resources")
                    if isinstance(page_resources, list) and page_resources:
                        existing_resources = merged_metadata.get(
                            "civitaiResources"
                        )
                        combined_resources = (
                            list(existing_resources)
                            if isinstance(existing_resources, list)
                            else []
                        )
                        existing_ids = {
                            str(resource.get("modelVersionId"))
                            for resource in combined_resources
                            if isinstance(resource, dict)
                            and resource.get("modelVersionId") is not None
                        }
                        for resource in page_resources:
                            if not isinstance(resource, dict):
                                continue
                            version_id = resource.get("modelVersionId")
                            if (
                                version_id is not None
                                and str(version_id) in existing_ids
                            ):
                                continue
                            combined_resources.append(resource)
                            if version_id is not None:
                                existing_ids.add(str(version_id))
                        merged_metadata["civitaiResources"] = combined_resources

                    if merged_metadata:
                        metadata = merged_metadata

                # Include modelVersionIds from root level if available.
                # CivitAI API returns modelVersionIds at root level, not in meta.
                # When meta is null (None), create a minimal dict so downstream
                # parsers can still discover LoRAs and checkpoints.
                model_version_ids = image_info.get("modelVersionIds")
                if model_version_ids:
                    if isinstance(metadata, dict):
                        metadata["modelVersionIds"] = model_version_ids
                    else:
                        metadata = {"modelVersionIds": model_version_ids}

                # Inject browsingLevel (canonical integer) so the recipe's
                # preview_nsfw_level can be set, enabling proper NSFW blur
                # of the preview image.  Fall back to nsfwLevel (string)
                # when browsingLevel is absent.
                if isinstance(metadata, dict):
                    browsing_level = image_info.get("browsingLevel")
                    nsfw_level_str = image_info.get("nsfwLevel")
                    if isinstance(browsing_level, int) and browsing_level > 0:
                        metadata["browsingLevel"] = browsing_level
                    elif (
                        isinstance(nsfw_level_str, str)
                        and nsfw_level_str
                        in (
                            "PG", "PG13", "R", "X", "XXX", "Blocked",
                        )
                    ):
                        from ...utils.constants import NSFW_LEVELS

                        metadata["browsingLevel"] = NSFW_LEVELS.get(
                            nsfw_level_str, 0
                        )

                # Validate that metadata contains meaningful recipe fields
                # If not, treat as None to trigger EXIF extraction from downloaded image
                if isinstance(metadata, dict) and not self._has_recipe_fields(metadata):
                    self._logger.debug(
                        "Civitai API metadata lacks recipe fields, will extract from EXIF"
                    )
                    metadata = None
            else:
                # Basic extension detection for non-Civitai URLs
                url_path = url.split("?")[0].split("#")[0]
                extension = os.path.splitext(url_path)[1].lower()
                if extension in [".mp4", ".webm"]:
                    is_video = True
                else:
                    extension = ".jpg"

                temp_path = self._create_temp_path(suffix=extension)
                await self._download_image(url, temp_path)

            # Always extract EXIF from the downloaded image for generation
            # params (prompt, negative prompt, sampler, steps, etc.).
            # Previously this was gated on ``metadata is None``, but that
            # skipped EXIF entirely when API metadata (modelVersionIds,
            # browsingLevel) is present, losing all generation parameters.
            exif_metadata = None
            exif_metadata_fields = None
            if not is_video:
                extract_fields = getattr(self._exif_utils, "extract_image_metadata_fields", None)
                if extract_fields:
                    exif_metadata_fields = await asyncio.to_thread(extract_fields, temp_path)
                    exif_metadata = (
                        exif_metadata_fields.get("parameters")
                        or exif_metadata_fields.get("prompt")
                        or exif_metadata_fields.get("workflow")
                    )
                else:
                    exif_metadata = await asyncio.to_thread(
                        self._exif_utils.extract_image_metadata, temp_path
                    )

            # Fallback: try the original (non-optimized) image for EXIF data
            if not exif_metadata and civitai_image_id and image_info:
                original_url = image_info.get("url")
                if original_url:
                    self._logger.debug(
                        "Optimized image lacks embedded metadata, "
                        "falling back to original image: %s",
                        original_url,
                    )
                    orig_temp_path = self._create_temp_path(suffix=".png")
                    try:
                        await self._download_image(original_url, orig_temp_path)
                        extract_fields = getattr(self._exif_utils, "extract_image_metadata_fields", None)
                        if extract_fields:
                            exif_metadata_fields = await asyncio.to_thread(extract_fields, orig_temp_path)
                            exif_metadata = (
                                exif_metadata_fields.get("parameters")
                                or exif_metadata_fields.get("prompt")
                                or exif_metadata_fields.get("workflow")
                            )
                        else:
                            exif_metadata = await asyncio.to_thread(
                                self._exif_utils.extract_image_metadata,
                                orig_temp_path,
                            )
                    finally:
                        self._safe_cleanup(orig_temp_path)

            # Parse EXIF data (typically a string like parameters/prompt/workflow)
            # and API metadata (dict with modelVersionIds, browsingLevel) separately,
            # then merge: API loras/checkpoint override, EXIF gen_params fill in gaps.
            # This mirrors the two-pass approach in _do_import_from_url.
            exif_parsed_result = None
            if isinstance(exif_metadata, str):
                exif_parser = self._recipe_parser_factory.create_parser(exif_metadata)
                if exif_parser:
                    exif_data = await exif_parser.parse_metadata(
                        exif_metadata, recipe_scanner=recipe_scanner,
                    )
                    if exif_data and not exif_data.get("error"):
                        exif_parsed_result = exif_data

            # Merge API metadata (dict) with EXIF data (if dict) for the
            # CivitaiApiMetadataParser.  If EXIF data is a string it was
            # parsed above — don't try to merge a string into a dict.
            merged = {}
            if isinstance(exif_metadata, dict):
                merged.update(exif_metadata)
            if isinstance(metadata, dict):
                merged.update(metadata)

            result = await self._parse_metadata(
                merged,
                recipe_scanner=recipe_scanner,
                image_path=temp_path,
                include_image_base64=True,
                is_video=is_video,
                extension=extension,
            )

            # Merge the two independent sources deterministically.  The
            # original image describes what was actually generated; CivitAI's
            # API is supplemental for IDs and missing fields only.
            if exif_parsed_result and not result.payload.get("error"):
                exif_gp = exif_parsed_result.get("gen_params") or {}
                result_gp = result.payload.get("gen_params") or {}
                merged_gp = self.merge_generation_params(exif_gp, result_gp)
                if merged_gp:
                    result.payload["gen_params"] = merged_gp

                result.payload["loras"] = self.merge_lora_resources(
                    exif_parsed_result.get("loras") or [],
                    result.payload.get("loras") or [],
                )
                result.payload["embeddings"] = self.merge_embedding_resources(
                    exif_parsed_result.get("embeddings") or [],
                    result.payload.get("embeddings") or [],
                )

                embedded_checkpoint = (
                    exif_parsed_result.get("checkpoint")
                    or exif_parsed_result.get("model")
                )
                api_checkpoint = (
                    result.payload.get("checkpoint") or result.payload.get("model")
                )
                if isinstance(embedded_checkpoint, dict):
                    merged_checkpoint = self._merge_resource_record(
                        embedded_checkpoint,
                        api_checkpoint if isinstance(api_checkpoint, dict) else None,
                    )
                    result.payload["checkpoint"] = merged_checkpoint
                    result.payload["model"] = merged_checkpoint

                embedded_base_model = exif_parsed_result.get("base_model")
                if embedded_base_model:
                    result.payload["base_model"] = embedded_base_model

            if exif_metadata_fields and not result.payload.get("error"):
                self._attach_comfy_replay_metadata(result.payload, exif_metadata_fields)

            if isinstance(metadata, dict) and not result.payload.get("error"):
                self._attach_generation_data(result.payload, metadata)

            if civitai_image_id and image_info and not result.payload.get("error"):
                # Use the metadata dict we built (may contain modelVersionIds
                # and browsingLevel from the API root level).  Do NOT pass
                # image_info.get("meta") — it is null for images whose meta
                # lives at the root level only.  Also do NOT derive
                # model_version_id from modelVersionIds[0] — that array mixes
                # checkpoints, LoRAs, and other types without ordering
                # guarantees; the parser already resolved them correctly.
                recipe_for_enrich = {
                    "gen_params": result.payload.get("gen_params", {}),
                    "loras": result.payload.get("loras", []),
                    "embeddings": result.payload.get("embeddings", []),
                    "base_model": result.payload.get("base_model", "") or "",
                    "checkpoint": result.payload.get("checkpoint") or result.payload.get("model"),
                    "source_path": url,
                }

                await RecipeEnricher.enrich_recipe(
                    recipe=recipe_for_enrich,
                    civitai_client=civitai_client,
                    request_params=None,
                    prefetched_civitai_meta_raw=(
                        metadata if isinstance(metadata, dict) else None
                    ),
                    prefetched_model_version_id=None,
                )

                result.payload["gen_params"] = recipe_for_enrich["gen_params"]
                result.payload["embeddings"] = recipe_for_enrich.get(
                    "embeddings", result.payload.get("embeddings", [])
                )
                if recipe_for_enrich.get("checkpoint"):
                    result.payload["checkpoint"] = recipe_for_enrich["checkpoint"]
                if recipe_for_enrich.get("base_model"):
                    result.payload["base_model"] = recipe_for_enrich["base_model"]

                # Extract browsingLevel from our constructed metadata for NSFW blur
                if isinstance(metadata, dict):
                    bl = metadata.get("browsingLevel")
                    if isinstance(bl, int) and bl > 0:
                        result.payload["preview_nsfw_level"] = bl

            if not result.payload.get("error"):
                self.apply_generation_source(result.payload)

            return result
        finally:
            if temp_path:
                self._safe_cleanup(temp_path)

    async def analyze_local_image(
        self,
        *,
        file_path: str | None,
        recipe_scanner,
    ) -> AnalysisResult:
        """Analyze a file already present on disk."""

        if not file_path:
            raise RecipeValidationError("No file path provided")

        normalized_path = os.path.normpath(file_path.strip('"').strip("'"))
        if not os.path.isfile(normalized_path):
            raise RecipeNotFoundError("File not found")

        metadata_fields = None
        extract_fields = getattr(self._exif_utils, "extract_image_metadata_fields", None)
        if extract_fields:
            metadata_fields = await asyncio.to_thread(extract_fields, normalized_path)
            metadata = (
                metadata_fields.get("parameters")
                or metadata_fields.get("prompt")
                or metadata_fields.get("workflow")
            )
        else:
            metadata = await asyncio.to_thread(
                self._exif_utils.extract_image_metadata, normalized_path
            )
        if not metadata:
            return self._metadata_not_found_response(normalized_path)

        result = await self._parse_metadata(
            metadata,
            recipe_scanner=recipe_scanner,
            image_path=normalized_path,
            include_image_base64=True,
        )
        if metadata_fields and not result.payload.get("error"):
            self._attach_comfy_replay_metadata(result.payload, metadata_fields)
        if not result.payload.get("error"):
            self.apply_generation_source(result.payload)
        return result

    async def analyze_widget_metadata(self, *, recipe_scanner) -> AnalysisResult:
        """Analyse the most recent generation metadata for widget saves."""

        if self._metadata_collector is None or self._metadata_processor_cls is None:
            raise RecipeValidationError("Metadata collection not available")

        raw_metadata = self._metadata_collector()
        metadata_dict = self._metadata_processor_cls.to_dict(raw_metadata)
        if not metadata_dict:
            raise RecipeValidationError("No generation metadata found")

        latest_image = None
        if not self._standalone_mode and self._metadata_registry_cls is not None:
            metadata_registry = self._metadata_registry_cls()
            latest_image = metadata_registry.get_first_decoded_image()

        if latest_image is None:
            raise RecipeValidationError(
                "No recent images found to use for recipe. Try generating an image first."
            )

        image_bytes = self._convert_tensor_to_png_bytes(latest_image)
        if image_bytes is None:
            raise RecipeValidationError(
                "Cannot handle this data shape from metadata registry"
            )

        return AnalysisResult(
            {
                "metadata": metadata_dict,
                "image_bytes": image_bytes,
            }
        )

    # Internal helpers -------------------------------------------------

    @staticmethod
    def _attach_comfy_replay_metadata(payload: dict[str, Any], fields: dict[str, Any]) -> None:
        """Preserve graph data and A1111 parameters for future recipe replay."""
        for field_name, target_name in (
            ("prompt", "comfy_prompt"),
            ("workflow", "comfy_workflow"),
        ):
            raw_value = fields.get(field_name)
            if not raw_value:
                continue
            try:
                value = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                payload[target_name] = value

        # ComfyUI imports this text through its native A1111 converter when an
        # image only carries ``parameters`` instead of a Comfy graph. Keep the
        # original text rather than reducing it to the common gen_params so
        # hires-fix and LoRA nodes can be rebuilt in the same way as image drop.
        parameters = fields.get("parameters")
        if isinstance(parameters, str) and parameters.strip():
            payload["a1111_parameters"] = normalize_metadata_text(parameters)

    @staticmethod
    def _attach_generation_data(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Preserve full Generation Data and provide native-import fallback text."""
        safe_metadata = json_safe_copy(metadata)
        if isinstance(safe_metadata, dict) and safe_metadata:
            payload["generation_metadata"] = safe_metadata

        parameters = payload.get("a1111_parameters")
        if not (
            isinstance(parameters, str)
            and "\nSteps:" in parameters
            and "\nNegative prompt:" in parameters
        ):
            synthesized = build_a1111_parameters(metadata)
            if synthesized:
                payload["a1111_parameters"] = synthesized

    def _has_recipe_fields(self, metadata: dict[str, Any]) -> bool:
        """Check if metadata contains meaningful recipe-related fields."""
        recipe_fields = {
            "prompt",
            "negative_prompt",
            "resources",
            "hashes",
            "params",
            "generationData",
            "Workflow",
            "prompt_type",
            "positive",
            "negative",
            # modelVersionIds is injected at the root level by CivitAI's image
            # API when meta is null. It carries the version IDs of ALL models
            # (checkpoint + LoRAs) used to generate the image.
            "modelVersionIds",
        }
        return any(field in metadata for field in recipe_fields)

    async def _parse_metadata(
        self,
        metadata: dict[str, Any],
        *,
        recipe_scanner,
        image_path: Optional[str],
        include_image_base64: bool,
        is_video: bool = False,
        extension: str = ".jpg",
    ) -> AnalysisResult:
        parser = self._recipe_parser_factory.create_parser(metadata)
        if parser is None:
            # Provide more specific error message based on metadata source
            if not metadata:
                error_msg = "This image does not contain any generation metadata (prompt, models, or parameters)"
            else:
                error_msg = "No parser found for this image"
            payload = {"error": error_msg, "loras": []}
            if include_image_base64 and image_path:
                payload["image_base64"] = self._encode_file(image_path)
            payload["is_video"] = is_video
            payload["extension"] = extension
            return AnalysisResult(payload)

        result = await parser.parse_metadata(metadata, recipe_scanner=recipe_scanner)

        if include_image_base64 and image_path:
            result["image_base64"] = self._encode_file(image_path)

        result["is_video"] = is_video
        result["extension"] = extension

        if "error" in result and not result.get("loras"):
            return AnalysisResult(result)

        fingerprint = calculate_recipe_fingerprint(result.get("loras", []))
        result["fingerprint"] = fingerprint

        matching_recipes: list[str] = []
        if fingerprint:
            matching_recipes = await recipe_scanner.find_recipes_by_fingerprint(
                fingerprint
            )
        result["matching_recipes"] = matching_recipes

        return AnalysisResult(result)

    async def _download_image(self, url: str, temp_path: str) -> None:
        downloader = await self._downloader_factory()
        success, result = await downloader.download_file(url, temp_path, use_auth=False)
        if not success:
            raise RecipeDownloadError(f"Failed to download image from URL: {result}")

    def _metadata_not_found_response(self, path: str) -> AnalysisResult:
        payload: dict[str, Any] = {
            "error": "No metadata found in this image",
            "loras": [],
        }
        if os.path.exists(path):
            payload["image_base64"] = self._encode_file(path)
        return AnalysisResult(payload)

    def _write_temp_file(self, data: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(data)
            return temp_file.name

    def _create_temp_path(self, suffix: str = ".jpg") -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            return temp_file.name

    def _safe_cleanup(self, path: Optional[str]) -> None:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error("Error deleting temporary file: %s", exc)

    def _encode_file(self, path: str) -> str:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _convert_tensor_to_png_bytes(self, latest_image: Any) -> Optional[bytes]:
        try:
            if isinstance(latest_image, tuple):
                tensor_image = latest_image[0] if latest_image else None
                if tensor_image is None:
                    return None
            else:
                tensor_image = latest_image

            if hasattr(tensor_image, "shape"):
                self._logger.debug(
                    "Tensor shape: %s, dtype: %s",
                    tensor_image.shape,
                    getattr(tensor_image, "dtype", None),
                )

            import torch  # type: ignore[import-not-found]

            if isinstance(tensor_image, torch.Tensor):
                image_np = tensor_image.cpu().numpy()
            else:
                image_np = np.array(tensor_image)

            while len(image_np.shape) > 3:
                image_np = image_np[0]

            if image_np.dtype in (np.float32, np.float64) and image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)

            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                pil_image = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format="PNG")
                return img_byte_arr.getvalue()
        except Exception as exc:  # pragma: no cover - defensive logging path
            self._logger.error("Error processing image data: %s", exc, exc_info=True)
            return None

        return None
