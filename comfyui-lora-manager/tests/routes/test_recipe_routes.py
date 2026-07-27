"""Integration smoke tests for the recipe route stack."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from py.config import config
from py.routes import base_recipe_routes
from py.routes.handlers import recipe_handlers
from py.routes.recipe_routes import RecipeRoutes
from py.services.recipes import (
    PromptDraftError,
    RecipeNotFoundError,
    RecipeRevisionError,
    RecipeValidationError,
)
from py.services.service_registry import ServiceRegistry


@dataclass
class RecipeRouteHarness:
    """Container exposing the aiohttp client and stubbed collaborators."""

    client: TestClient
    scanner: "StubRecipeScanner"
    analysis: "StubAnalysisService"
    persistence: "StubPersistenceService"
    sharing: "StubSharingService"
    prompt_draft: "StubPromptDraftService"
    revision: "StubRevisionService"
    downloader: "StubDownloader"
    civitai: "StubCivitaiClient"
    tmp_dir: Path


class StubRecipeScanner:
    """Minimal scanner double with the surface used by the handlers."""

    def __init__(self, base_dir: Path) -> None:
        self.recipes_dir = str(base_dir / "recipes")
        self.listing_items: List[Dict[str, Any]] = []
        self.cached_raw: List[Dict[str, Any]] = []
        self.recipes: Dict[str, Dict[str, Any]] = {}
        self.removed: List[str] = []
        self.last_paginated_params: Dict[str, Any] | None = None
        self.lora_lookup: Dict[str, List[Dict[str, Any]]] = {}
        self.checkpoint_lookup: Dict[str, List[Dict[str, Any]]] = {}
        self.image_id_map_override: Dict[str, str] = {}

        async def _noop_get_cached_data(force_refresh: bool = False) -> None:  # noqa: ARG001 - signature mirrors real scanner
            return None

        self._lora_scanner = SimpleNamespace(  # mimic BaseRecipeRoutes expectations
            get_cached_data=_noop_get_cached_data,
            _hash_index=SimpleNamespace(_hash_to_path={}),
        )

    async def get_cached_data(self, force_refresh: bool = False) -> SimpleNamespace:  # noqa: ARG002 - flag unused by stub
        return SimpleNamespace(
            raw_data=list(self.cached_raw),
            image_id_map=dict(getattr(self, "image_id_map_override", {})),
        )

    async def get_paginated_data(self, **params: Any) -> Dict[str, Any]:
        self.last_paginated_params = params
        items = [dict(item) for item in self.listing_items]
        page = int(params.get("page", 1))
        page_size = int(params.get("page_size", 20))
        return {
            "items": items,
            "total": len(items),
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (len(items) + page_size - 1) // max(page_size, 1)),
        }

    async def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        return self.recipes.get(recipe_id)

    async def get_recipes_for_lora(self, lora_hash: str) -> List[Dict[str, Any]]:
        return list(self.lora_lookup.get(lora_hash.lower(), []))

    async def get_recipes_for_checkpoint(
        self, checkpoint_hash: str
    ) -> List[Dict[str, Any]]:
        return list(self.checkpoint_lookup.get(checkpoint_hash.lower(), []))

    async def get_recipe_json_path(self, recipe_id: str) -> Optional[str]:
        candidate = Path(self.recipes_dir) / f"{recipe_id}.recipe.json"
        return str(candidate) if candidate.exists() else None

    async def remove_recipe(self, recipe_id: str) -> None:
        self.removed.append(recipe_id)
        self.recipes.pop(recipe_id, None)


class StubAnalysisService:
    """Captures calls made by analysis routes while returning canned responses."""

    instances: List["StubAnalysisService"] = []

    def __init__(self, **_: Any) -> None:
        self.raise_for_uploaded: Optional[Exception] = None
        self.raise_for_remote: Optional[Exception] = None
        self.raise_for_local: Optional[Exception] = None
        self.upload_calls: List[bytes] = []
        self.remote_calls: List[Optional[str]] = []
        self.local_calls: List[Optional[str]] = []
        self.result = SimpleNamespace(payload={"loras": []}, status=200)
        self._recipe_parser_factory = None
        StubAnalysisService.instances.append(self)

    async def analyze_uploaded_image(
        self, *, image_bytes: bytes | None, recipe_scanner
    ) -> SimpleNamespace:  # noqa: D401 - mirrors real signature
        if self.raise_for_uploaded:
            raise self.raise_for_uploaded
        self.upload_calls.append(image_bytes or b"")
        return self.result

    async def analyze_remote_image(
        self, *, url: Optional[str], recipe_scanner, civitai_client
    ) -> SimpleNamespace:  # noqa: D401
        if self.raise_for_remote:
            raise self.raise_for_remote
        self.remote_calls.append(url)
        return self.result

    async def analyze_local_image(
        self, *, file_path: Optional[str], recipe_scanner
    ) -> SimpleNamespace:  # noqa: D401
        if self.raise_for_local:
            raise self.raise_for_local
        self.local_calls.append(file_path)
        return self.result

    async def analyze_widget_metadata(self, *, recipe_scanner) -> SimpleNamespace:
        return SimpleNamespace(payload={"metadata": {}, "image_bytes": b""}, status=200)


class StubPersistenceService:
    """Stub for persistence operations to avoid filesystem writes."""

    instances: List["StubPersistenceService"] = []

    def __init__(self, **_: Any) -> None:
        self.save_calls: List[Dict[str, Any]] = []
        self.delete_calls: List[str] = []
        self.move_calls: List[Dict[str, str]] = []
        self.update_calls: List[Dict[str, Any]] = []
        self.save_result = SimpleNamespace(
            payload={"success": True, "recipe_id": "stub-id"}, status=200
        )
        self.delete_result = SimpleNamespace(payload={"success": True}, status=200)
        StubPersistenceService.instances.append(self)

    async def save_recipe(
        self,
        *,
        recipe_scanner,
        image_bytes,
        image_base64,
        name,
        tags,
        metadata,
        extension=None,
    ) -> SimpleNamespace:  # noqa: D401
        self.save_calls.append(
            {
                "recipe_scanner": recipe_scanner,
                "image_bytes": image_bytes,
                "image_base64": image_base64,
                "name": name,
                "tags": list(tags),
                "metadata": metadata,
                "extension": extension,
            }
        )
        return self.save_result

    async def delete_recipe(self, *, recipe_scanner, recipe_id: str) -> SimpleNamespace:
        self.delete_calls.append(recipe_id)
        await recipe_scanner.remove_recipe(recipe_id)
        return self.delete_result

    async def move_recipe(
        self, *, recipe_scanner, recipe_id: str, target_path: str
    ) -> SimpleNamespace:  # noqa: D401
        self.move_calls.append({"recipe_id": recipe_id, "target_path": target_path})
        return SimpleNamespace(
            payload={
                "success": True,
                "recipe_id": recipe_id,
                "new_file_path": target_path,
            },
            status=200,
        )

    async def update_recipe(
        self, *, recipe_scanner, recipe_id: str, updates: Dict[str, Any]
    ) -> SimpleNamespace:
        self.update_calls.append(
            {
                "recipe_scanner": recipe_scanner,
                "recipe_id": recipe_id,
                "updates": updates,
            }
        )
        return SimpleNamespace(
            payload={"success": True, "recipe_id": recipe_id, "updates": updates},
            status=200,
        )

    async def reconnect_lora(
        self, *, recipe_scanner, recipe_id: str, lora_index: int, target_name: str
    ) -> SimpleNamespace:  # pragma: no cover
        return SimpleNamespace(payload={"success": True}, status=200)

    async def bulk_delete(
        self, *, recipe_scanner, recipe_ids: List[str]
    ) -> SimpleNamespace:  # pragma: no cover
        return SimpleNamespace(
            payload={"success": True, "deleted": recipe_ids}, status=200
        )

    async def save_recipe_from_widget(
        self, *, recipe_scanner, metadata: Dict[str, Any], image_bytes: bytes
    ) -> SimpleNamespace:  # pragma: no cover
        return SimpleNamespace(payload={"success": True}, status=200)


class StubSharingService:
    """Share service stub recording requests and returning canned responses."""

    instances: List["StubSharingService"] = []

    def __init__(self, *, ttl_seconds: int = 300, logger) -> None:  # noqa: ARG002 - ttl unused in stub
        self.share_calls: List[str] = []
        self.download_calls: List[str] = []
        self.share_result = SimpleNamespace(
            payload={
                "success": True,
                "download_url": "/share/stub",
                "filename": "recipe.png",
            },
            status=200,
        )
        self.download_info = SimpleNamespace(file_path="", download_filename="")
        StubSharingService.instances.append(self)

    async def share_recipe(self, *, recipe_scanner, recipe_id: str) -> SimpleNamespace:
        self.share_calls.append(recipe_id)
        return self.share_result

    async def prepare_download(
        self, *, recipe_scanner, recipe_id: str
    ) -> SimpleNamespace:
        self.download_calls.append(recipe_id)
        return self.download_info


class StubPromptDraftService:
    """Prompt drafting stub that proves the HTTP route never mutates recipes."""

    instances: List["StubPromptDraftService"] = []

    def __init__(self, **_: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.options: List[Dict[str, Any]] = []
        self.release_calls = 0
        self.result: Dict[str, Any] = {
            "schema": "lora-manager.prompt-draft",
            "version": 1,
            "prompt_source": "lm_studio",
            "proposed_prompt": "protected lora, generated scene",
        }
        self.error: Optional[PromptDraftError] = None
        StubPromptDraftService.instances.append(self)

    async def create_draft(
        self,
        recipe: Dict[str, Any],
        *,
        model: str,
        force_regenerate: bool,
    ) -> Dict[str, Any]:
        self.calls.append(recipe)
        self.options.append({"model": model, "force_regenerate": force_regenerate})
        if self.error:
            raise self.error
        return dict(self.result)

    async def release_managed_models(self) -> List[str]:
        self.release_calls += 1
        return ["lora-manager-qwen35-q6"]


class StubRevisionService:
    """Revision service stub for ETag and immutable adoption route contracts."""

    instances: List["StubRevisionService"] = []

    def __init__(self, **_: Any) -> None:
        self.etag = "e" * 64
        self.active_ids: set[str] = set()
        self.summary: Dict[str, Any] = {"active": False, "stale": False, "count": 0}
        self.adopt_calls: List[Dict[str, Any]] = []
        self.activate_calls: List[Dict[str, Any]] = []
        self.cleanup_calls: List[Dict[str, str]] = []
        self.adopt_error: Optional[RecipeRevisionError] = None
        StubRevisionService.instances.append(self)

    async def get_active_prompt_recipe_ids(self, recipes_dir):
        return set(self.active_ids)

    async def get_source_state(self, recipe):
        return {"etag": self.etag, "recipe_image_sha256": "i" * 64}

    async def get_summary(self, recipe_id, recipe_path, *, current_etag):
        return dict(self.summary)

    async def resolve_active_recipe(self, recipe, recipe_path, *, current_etag):
        result = dict(recipe)
        result["gen_params"] = {
            **(recipe.get("gen_params") or {}),
            "prompt": "active prompt",
            "seed": 99,
            "prompt_source": "lm_studio",
        }
        return result

    async def list_revisions(self, recipe_id, recipe_path, *, current_etag):
        return [{"revision_id": "rev-1", "active": True, "seed": 99}]

    async def adopt_revision(self, **kwargs):
        self.adopt_calls.append(kwargs)
        if self.adopt_error:
            raise self.adopt_error
        return {
            "created": True,
            "revision": {"revision_id": "rev-1"},
            "summary": {"active": True, "stale": False, "count": 1, "seed": 99},
        }

    async def activate_revision(self, recipe_id, recipe_path, *, current_etag, revision_id):
        self.activate_calls.append({"recipe_id": recipe_id, "revision_id": revision_id})
        return {"active": bool(revision_id), "stale": False, "count": 1}

    async def cleanup_recipe(self, recipe_id, recipe_path):
        self.cleanup_calls.append({"recipe_id": recipe_id, "recipe_path": str(recipe_path)})


class StubDownloader:
    """Downloader stub that writes deterministic bytes to requested locations."""

    def __init__(self) -> None:
        self.urls: List[str] = []

    async def download_file(self, url: str, destination: str, use_auth: bool = False):  # noqa: ARG002 - use_auth unused
        self.urls.append(url)
        Path(destination).write_bytes(b"imported-image")
        return True, destination


class StubCivitaiClient:
    """Stub for Civitai API client."""

    def __init__(self) -> None:
        self.image_info: Dict[str, Any] = {}

    async def get_image_info(
        self, image_id: str, source_url: str | None = None
    ) -> Optional[Dict[str, Any]]:
        return self.image_info.get(image_id)


@asynccontextmanager
async def recipe_harness(
    monkeypatch, tmp_path: Path
) -> AsyncIterator[RecipeRouteHarness]:
    """Context manager that yields a fully wired recipe route harness."""

    StubAnalysisService.instances.clear()
    StubPersistenceService.instances.clear()
    StubSharingService.instances.clear()
    StubPromptDraftService.instances.clear()
    StubRevisionService.instances.clear()

    scanner = StubRecipeScanner(tmp_path)
    civitai_client = StubCivitaiClient()

    async def fake_get_recipe_scanner():
        return scanner

    async def fake_get_civitai_client():
        return civitai_client

    downloader = StubDownloader()

    async def fake_get_downloader():
        return downloader

    monkeypatch.setattr(ServiceRegistry, "get_recipe_scanner", fake_get_recipe_scanner)
    monkeypatch.setattr(ServiceRegistry, "get_civitai_client", fake_get_civitai_client)
    monkeypatch.setattr(
        base_recipe_routes, "RecipeAnalysisService", StubAnalysisService
    )
    monkeypatch.setattr(
        base_recipe_routes, "RecipePersistenceService", StubPersistenceService
    )
    monkeypatch.setattr(base_recipe_routes, "RecipeSharingService", StubSharingService)
    monkeypatch.setattr(
        base_recipe_routes, "RecipePromptDraftService", StubPromptDraftService
    )
    monkeypatch.setattr(base_recipe_routes, "RecipeRevisionService", StubRevisionService)
    monkeypatch.setattr(base_recipe_routes, "get_downloader", fake_get_downloader)
    monkeypatch.setattr(config, "loras_roots", [str(tmp_path)], raising=False)

    app = web.Application()
    RecipeRoutes.setup_routes(app)

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()

    harness = RecipeRouteHarness(
        client=client,
        scanner=scanner,
        analysis=StubAnalysisService.instances[-1],
        persistence=StubPersistenceService.instances[-1],
        sharing=StubSharingService.instances[-1],
        prompt_draft=StubPromptDraftService.instances[-1],
        revision=StubRevisionService.instances[-1],
        downloader=downloader,
        civitai=civitai_client,
        tmp_dir=tmp_path,
    )

    try:
        yield harness
    finally:
        await client.close()
        StubAnalysisService.instances.clear()
        StubPersistenceService.instances.clear()
        StubSharingService.instances.clear()
        StubPromptDraftService.instances.clear()
        StubRevisionService.instances.clear()


def persist_stub_recipe(harness: RecipeRouteHarness, recipe: Dict[str, Any]) -> None:
    recipe_id = str(recipe["id"])
    harness.scanner.recipes[recipe_id] = recipe
    recipes_dir = Path(harness.scanner.recipes_dir)
    recipes_dir.mkdir(parents=True, exist_ok=True)
    (recipes_dir / f"{recipe_id}.recipe.json").write_text(
        json.dumps(recipe), encoding="utf-8"
    )


async def test_list_recipes_provides_file_urls(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        recipe_path = harness.tmp_dir / "recipes" / "demo.png"
        harness.scanner.listing_items = [
            {
                "id": "recipe-1",
                "file_path": str(recipe_path),
                "title": "Demo",
                "loras": [],
            }
        ]
        harness.scanner.cached_raw = list(harness.scanner.listing_items)

        response = await harness.client.get("/api/lm/recipes")
        payload = await response.json()

        assert response.status == 200
        assert payload["items"][0]["file_url"].endswith("demo.png")
        assert payload["items"][0]["loras"] == []


async def test_recipe_detail_attaches_replay_manifest_lazily(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        recipe = {
            "id": "recipe-1",
            "title": "Manifest Recipe",
            "loras": [{"file_name": "catalog-only.safetensors"}],
            "gen_params": {"prompt": "portrait"},
        }
        persist_stub_recipe(harness, recipe)
        harness.scanner.listing_items = [dict(recipe)]

        detail_response = await harness.client.get("/api/lm/recipe/recipe-1")
        detail = await detail_response.json()
        list_response = await harness.client.get("/api/lm/recipes")
        listing = await list_response.json()

        assert detail_response.status == 200
        assert detail["replay_manifest"]["schema"] == "lora-manager.replay-manifest"
        assert detail["replay_manifest"]["version"] == 1
        assert detail["replay_manifest"]["required_resources"] == []
        assert detail["source_etag"] == "e" * 64
        assert detail["revision_summary"]["active"] is False
        assert detail_response.headers["ETag"] == f'"{"e" * 64}"'
        assert "replay_manifest" not in listing["items"][0]


async def test_recipe_detail_active_variant_keeps_source_default_immutable(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        source = {
            "id": "recipe-active",
            "gen_params": {"prompt": "source prompt", "seed": 1},
            "loras": [],
        }
        persist_stub_recipe(harness, source)
        harness.revision.summary = {
            "active": True, "stale": False, "count": 1, "revision_id": "rev-1"
        }

        source_response = await harness.client.get("/api/lm/recipe/recipe-active")
        active_response = await harness.client.get(
            "/api/lm/recipe/recipe-active?variant=active"
        )
        source_payload = await source_response.json()
        active_payload = await active_response.json()

        assert source_payload["gen_params"]["prompt"] == "source prompt"
        assert active_payload["gen_params"]["prompt"] == "active prompt"
        assert active_payload["gen_params"]["seed"] == 99
        assert source["gen_params"]["prompt"] == "source prompt"


async def test_recipe_revision_routes_require_etag_and_never_use_metadata_update(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        source = {
            "id": "recipe-revision",
            "gen_params": {"prompt": "source prompt", "negative_prompt": "bad"},
            "loras": [],
        }
        persist_stub_recipe(harness, source)
        detail_response = await harness.client.get("/api/lm/recipe/recipe-revision")
        detail = await detail_response.json()
        manifest_hash = detail["replay_manifest"]["manifest_hash"]
        body = {
            "action": "adopt",
            "manifest_hash": manifest_hash,
            "draft": {"draft_hash": "d" * 64, "manifest_hash": manifest_hash},
            "candidate": {"prompt_id": "prompt", "seed": 99},
        }

        missing_match = await harness.client.post(
            "/api/lm/recipe/recipe-revision/revisions", json=body
        )
        adopted = await harness.client.post(
            "/api/lm/recipe/recipe-revision/revisions",
            json=body,
            headers={"If-Match": f'"{harness.revision.etag}"'},
        )
        adopted_payload = await adopted.json()
        listed = await harness.client.get(
            "/api/lm/recipe/recipe-revision/revisions"
        )
        listed_payload = await listed.json()
        deactivated = await harness.client.put(
            "/api/lm/recipe/recipe-revision/revisions/active",
            json={"action": "activate", "revision_id": None},
            headers={"If-Match": harness.revision.etag},
        )

        assert missing_match.status == 428
        assert adopted.status == 201
        assert adopted_payload["original_recipe_changed"] is False
        assert adopted_payload["revision_summary"]["active"] is True
        assert harness.revision.adopt_calls[0]["if_match"] == f'"{harness.revision.etag}"'
        assert listed.status == 200
        assert listed_payload["revisions"][0]["revision_id"] == "rev-1"
        assert deactivated.status == 200
        assert harness.revision.activate_calls[-1]["revision_id"] is None
        assert harness.persistence.update_calls == []


async def test_create_ai_prompt_draft_uses_current_manifest_without_saving(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        original = {
            "id": "recipe-draft",
            "title": "Draft Recipe",
            "gen_params": {"prompt": "portrait", "negative_prompt": "bad hands"},
            "loras": [],
        }
        persist_stub_recipe(harness, original)
        detail_response = await harness.client.get("/api/lm/recipe/recipe-draft")
        detail = await detail_response.json()

        response = await harness.client.post(
            "/api/lm/recipe/recipe-draft/ai-prompt-draft",
            json={
                "action": "draft",
                "manifest_hash": detail["replay_manifest"]["manifest_hash"],
            },
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["success"] is True
        assert payload["draft"]["prompt_source"] == "lm_studio"
        assert len(harness.prompt_draft.calls) == 1
        assert harness.prompt_draft.calls[0]["replay_manifest"]["manifest_hash"] == (
            detail["replay_manifest"]["manifest_hash"]
        )
        assert harness.prompt_draft.options == [
            {"model": "qwen35-q6", "force_regenerate": False}
        ]
        assert "replay_manifest" not in original
        assert harness.persistence.save_calls == []
        assert harness.persistence.update_calls == []


async def test_ai_prompt_draft_release_unloads_managed_model_without_recipe_write(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.post(
            "/api/lm/recipe/any-recipe/ai-prompt-draft",
            json={"action": "release"},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload == {
            "success": True,
            "released_models": ["lora-manager-qwen35-q6"],
            "recipe_changed": False,
        }
        assert harness.prompt_draft.release_calls == 1
        assert harness.prompt_draft.calls == []
        assert harness.persistence.save_calls == []
        assert harness.persistence.update_calls == []


async def test_create_ai_prompt_draft_rejects_invalid_or_stale_requests(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        persist_stub_recipe(harness, {
            "id": "recipe-draft",
            "gen_params": {"prompt": "portrait"},
            "loras": [],
        })

        wrong_type = await harness.client.post(
            "/api/lm/recipe/recipe-draft/ai-prompt-draft",
            data='{"action":"draft"}',
            headers={"Content-Type": "text/plain"},
        )
        wrong_type_payload = await wrong_type.json()
        bad_action = await harness.client.post(
            "/api/lm/recipe/recipe-draft/ai-prompt-draft",
            json={"action": "save", "manifest_hash": "0" * 64},
        )
        bad_action_payload = await bad_action.json()
        stale = await harness.client.post(
            "/api/lm/recipe/recipe-draft/ai-prompt-draft",
            json={"action": "draft", "manifest_hash": "0" * 64},
        )
        stale_payload = await stale.json()

        assert wrong_type.status == 415
        assert wrong_type_payload["recipe_changed"] is False
        assert bad_action.status == 400
        assert bad_action_payload["recipe_changed"] is False
        assert stale.status == 409
        assert stale_payload["error"] == "REPLAY_MANIFEST_CHANGED"
        assert stale_payload["recipe_changed"] is False
        assert harness.prompt_draft.calls == []


async def test_create_ai_prompt_draft_maps_service_error_without_saving(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        persist_stub_recipe(harness, {
            "id": "recipe-draft",
            "gen_params": {"prompt": "portrait"},
            "loras": [],
        })
        detail_response = await harness.client.get("/api/lm/recipe/recipe-draft")
        detail = await detail_response.json()
        harness.prompt_draft.error = PromptDraftError(
            "LM_STUDIO_TIMEOUT", "LM Studio timed out", 504
        )

        response = await harness.client.post(
            "/api/lm/recipe/recipe-draft/ai-prompt-draft",
            json={
                "action": "draft",
                "manifest_hash": detail["replay_manifest"]["manifest_hash"],
            },
        )
        payload = await response.json()

        assert response.status == 504
        assert payload["error"] == "LM_STUDIO_TIMEOUT"
        assert payload["recipe_changed"] is False
        assert harness.persistence.save_calls == []
        assert harness.persistence.update_calls == []


async def test_list_recipes_passes_checkpoint_hash_filter(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.get("/api/lm/recipes?checkpoint_hash=ckpt123")
        payload = await response.json()

        assert response.status == 200
        assert payload["items"] == []
        assert harness.scanner.last_paginated_params["checkpoint_hash"] == "ckpt123"


async def test_list_recipes_passes_excluded_favorite_and_prompt_status_filters(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.get(
            "/api/lm/recipes?favorite=false&prompt_status=missing"
        )

        assert response.status == 200
        filters = harness.scanner.last_paginated_params["filters"]
        assert filters["favorite"] is False
        assert filters["prompt_status"] == "missing"


async def test_get_recipes_for_checkpoint(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.scanner.checkpoint_lookup["abc123"] = [
            {"id": "recipe-1", "title": "Linked recipe"}
        ]

        response = await harness.client.get(
            "/api/lm/recipes/for-checkpoint?hash=ABC123"
        )
        payload = await response.json()

        assert response.status == 200
        assert payload == {
            "success": True,
            "recipes": [{"id": "recipe-1", "title": "Linked recipe"}],
        }


async def test_get_recipes_for_checkpoint_requires_hash(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.get("/api/lm/recipes/for-checkpoint")
        payload = await response.json()

        assert response.status == 400
        assert payload["success"] is False


async def test_save_and_delete_recipe_round_trip(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        form = FormData()
        form.add_field(
            "image", b"stub", filename="sample.png", content_type="image/png"
        )
        form.add_field("name", "Test Recipe")
        form.add_field("tags", json.dumps(["tag-a"]))
        form.add_field("metadata", json.dumps({"loras": []}))
        form.add_field("image_base64", "aW1hZ2U=")

        harness.persistence.save_result = SimpleNamespace(
            payload={"success": True, "recipe_id": "saved-id"},
            status=201,
        )

        save_response = await harness.client.post("/api/lm/recipes/save", data=form)
        save_payload = await save_response.json()

        assert save_response.status == 201
        assert save_payload["recipe_id"] == "saved-id"
        assert harness.persistence.save_calls[-1]["name"] == "Test Recipe"

        harness.persistence.delete_result = SimpleNamespace(
            payload={"success": True}, status=200
        )

        delete_response = await harness.client.delete("/api/lm/recipe/saved-id")
        delete_payload = await delete_response.json()

        assert delete_response.status == 200
        assert delete_payload["success"] is True
        assert harness.persistence.delete_calls == ["saved-id"]


async def test_move_recipe_invokes_persistence(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.post(
            "/api/lm/recipe/move",
            json={
                "recipe_id": "move-me",
                "target_path": str(tmp_path / "recipes" / "subdir"),
            },
        )

        payload = await response.json()
        assert response.status == 200
        assert payload["recipe_id"] == "move-me"
        assert harness.persistence.move_calls == [
            {
                "recipe_id": "move-me",
                "target_path": str(tmp_path / "recipes" / "subdir"),
            }
        ]


async def test_import_remote_recipe(monkeypatch, tmp_path: Path) -> None:
    provider_calls: list[str | int] = []

    class Provider:
        async def get_model_version_info(self, model_version_id):
            provider_calls.append(model_version_id)
            return {
                "baseModel": "Flux Provider",
                "model": {"type": "Checkpoint", "name": "Flux"},
            }, None

    async def fake_get_default_metadata_provider():
        return Provider()

    monkeypatch.setattr(
        "py.recipes.enrichment.get_default_metadata_provider",
        fake_get_default_metadata_provider,
    )

    async with recipe_harness(monkeypatch, tmp_path) as harness:
        resources = [
            {
                "type": "checkpoint",
                "modelId": 10,
                "modelVersionId": 33,
                "modelName": "Flux",
                "modelVersionName": "Dev",
            },
            {
                "type": "lora",
                "modelId": 20,
                "modelVersionId": 44,
                "modelName": "Painterly",
                "modelVersionName": "v2",
                "weight": 0.25,
            },
            {
                "type": "TextualInversion",
                "modelId": 30,
                "modelVersionId": 55,
                "modelName": "Easy Negative",
                "modelVersionName": "v1",
            },
        ]
        response = await harness.client.get(
            "/api/lm/recipes/import-remote",
            params={
                "image_url": "https://example.com/images/1",
                "name": "Remote Recipe",
                "resources": json.dumps(resources),
                "tags": "foo,bar",
                "base_model": "Flux",
                "source_path": "https://example.com/images/1",
                "gen_params": json.dumps({"prompt": "hello world", "cfg_scale": 7}),
            },
        )

        payload = await response.json()
        assert response.status == 200
        assert payload["success"] is True

        call = harness.persistence.save_calls[-1]
        assert call["name"] == "Remote Recipe"
        assert call["tags"] == ["foo", "bar"]
        metadata = call["metadata"]
        assert metadata["base_model"] == "Flux Provider"
        assert provider_calls == ["33"]
        assert metadata["checkpoint"]["modelVersionId"] == 33
        assert metadata["loras"][0]["weight"] == 0.25
        assert metadata["embeddings"][0]["id"] == 55
        assert metadata["embeddings"][0]["modelId"] == 30
        assert metadata["gen_params"]["prompt"] == "hello world"
        assert harness.downloader.urls == ["https://example.com/images/1"]


async def test_import_remote_recipe_falls_back_to_request_base_model(
    monkeypatch, tmp_path: Path
) -> None:
    provider_calls: list[str | int] = []

    class Provider:
        async def get_model_version_info(self, model_version_id):
            provider_calls.append(model_version_id)
            return {}, None

    async def fake_get_default_metadata_provider():
        return Provider()

    monkeypatch.setattr(
        "py.recipes.enrichment.get_default_metadata_provider",
        fake_get_default_metadata_provider,
    )

    async with recipe_harness(monkeypatch, tmp_path) as harness:
        resources = [
            {
                "type": "checkpoint",
                "modelId": 11,
                "modelVersionId": 77,
                "modelName": "Flux",
                "modelVersionName": "Dev",
            },
        ]
        response = await harness.client.get(
            "/api/lm/recipes/import-remote",
            params={
                "image_url": "https://example.com/images/1",
                "name": "Remote Recipe",
                "resources": json.dumps(resources),
                "tags": "foo,bar",
                "base_model": "Flux",
            },
        )

        payload = await response.json()
        assert response.status == 200
        assert payload["success"] is True

        metadata = harness.persistence.save_calls[-1]["metadata"]
        assert metadata["base_model"] == "Flux"
        assert provider_calls == ["77"]


async def test_update_recipe_accepts_gen_params(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        payload = {
            "gen_params": {
                "prompt": "updated prompt",
                "negative_prompt": "updated negative",
                "steps": 30,
            }
        }

        response = await harness.client.put(
            "/api/lm/recipe/recipe-42/update",
            json=payload,
        )
        data = await response.json()

        assert response.status == 200
        assert data["success"] is True
        assert harness.persistence.update_calls == [
            {
                "recipe_scanner": harness.scanner,
                "recipe_id": "recipe-42",
                "updates": payload,
            }
        ]


async def test_import_remote_video_recipe(monkeypatch, tmp_path: Path) -> None:
    async def fake_get_default_metadata_provider():
        return SimpleNamespace(get_model_version_info=lambda id: ({}, None))

    monkeypatch.setattr(
        "py.recipes.enrichment.get_default_metadata_provider",
        fake_get_default_metadata_provider,
    )

    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.civitai.image_info["12345"] = {
            "id": 12345,
            "url": "https://image.civitai.com/x/y/original=true/video.mp4",
            "type": "video",
        }

        response = await harness.client.get(
            "/api/lm/recipes/import-remote",
            params={
                "image_url": "https://civitai.com/images/12345",
                "name": "Video Recipe",
                "resources": json.dumps([]),
                "base_model": "Flux",
            },
        )

        payload = await response.json()
        assert response.status == 200
        assert payload["success"] is True

        # Verify downloader was called with rewritten URL
        assert "transcode=true" in harness.downloader.urls[0]

        # Verify persistence was called with correct extension
        call = harness.persistence.save_calls[-1]
        assert call["extension"] == ".mp4"


async def test_import_remote_recipe_supports_civitai_red(monkeypatch, tmp_path: Path) -> None:
    async def fake_get_default_metadata_provider():
        return SimpleNamespace(get_model_version_info=lambda id: ({}, None))

    monkeypatch.setattr(
        "py.recipes.enrichment.get_default_metadata_provider",
        fake_get_default_metadata_provider,
    )

    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.civitai.image_info["126920345"] = {
            "id": 126920345,
            "url": "https://image.civitai.com/x/y/original=true/sample.jpeg",
            "type": "image",
        }

        response = await harness.client.get(
            "/api/lm/recipes/import-remote",
            params={
                "image_url": "https://civitai.red/images/126920345",
                "name": "Red Recipe",
                "resources": json.dumps([]),
                "base_model": "Flux",
            },
        )

        payload = await response.json()
        assert response.status == 200
        assert payload["success"] is True
        assert harness.downloader.urls
        assert "width=450,optimized=true" in harness.downloader.urls[0]


async def test_analyze_remote_image_supports_civitai_red(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.analysis.result = SimpleNamespace(payload={"loras": []}, status=200)

        response = await harness.client.post(
            "/api/lm/recipes/analyze-image",
            json={"url": "https://civitai.red/images/126920345"},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload == {"loras": []}
        assert harness.analysis.remote_calls == [
            "https://civitai.red/images/126920345"
        ]


async def test_analyze_uploaded_image_error_path(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.analysis.raise_for_uploaded = RecipeValidationError(
            "No image data provided"
        )

        form = FormData()
        form.add_field("image", b"", filename="empty.png", content_type="image/png")

        response = await harness.client.post("/api/lm/recipes/analyze-image", data=form)
        payload = await response.json()

        assert response.status == 400
        assert payload["error"] == "No image data provided"
        assert payload["loras"] == []


async def test_share_and_download_recipe(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        recipe_id = "share-me"
        download_path = harness.tmp_dir / "recipes" / "share.png"
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_path.write_bytes(b"stub")

        harness.scanner.recipes[recipe_id] = {
            "id": recipe_id,
            "title": "Shared",
            "file_path": str(download_path),
        }

        harness.sharing.share_result = SimpleNamespace(
            payload={
                "success": True,
                "download_url": "/api/share",
                "filename": "share.png",
            },
            status=200,
        )
        harness.sharing.download_info = SimpleNamespace(
            file_path=str(download_path),
            download_filename="share.png",
        )

        share_response = await harness.client.get(f"/api/lm/recipe/{recipe_id}/share")
        share_payload = await share_response.json()

        assert share_response.status == 200
        assert share_payload["filename"] == "share.png"
        assert harness.sharing.share_calls == [recipe_id]

        download_response = await harness.client.get(
            f"/api/lm/recipe/{recipe_id}/share/download"
        )
        body = await download_response.read()

        assert download_response.status == 200
        assert (
            download_response.headers["Content-Disposition"]
            == 'attachment; filename="share.png"'
        )
        assert body == b"stub"

        download_path.unlink(missing_ok=True)


async def test_import_remote_recipe_merges_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    # 1. Mock Metadata Provider
    class Provider:
        async def get_model_version_info(self, model_version_id):
            return {"baseModel": "Flux Provider"}, None

    async def fake_get_default_metadata_provider():
        return Provider()

    monkeypatch.setattr(
        "py.recipes.enrichment.get_default_metadata_provider",
        fake_get_default_metadata_provider,
    )

    # 2. Mock ExifUtils to return some embedded metadata
    class MockExifUtils:
        @staticmethod
        def extract_image_metadata(path):
            return "Recipe metadata: " + json.dumps(
                {"gen_params": {"prompt": "from embedded", "seed": 123}}
            )

    monkeypatch.setattr(recipe_handlers, "ExifUtils", MockExifUtils)

    # 3. Mock Parser Factory for StubAnalysisService
    class MockParser:
        async def parse_metadata(self, raw, recipe_scanner=None):
            return json.loads(raw[len("Recipe metadata: ") :])

    class MockApiParser:
        async def parse_metadata(self, raw, recipe_scanner=None):
            return {"gen_params": raw, "loras": []}

    class MockFactory:
        def create_parser(self, raw):
            if isinstance(raw, str) and raw.startswith("Recipe metadata: "):
                return MockParser()
            if isinstance(raw, dict):
                return MockApiParser()
            return None

    # 4. Setup Harness and run test
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.analysis._recipe_parser_factory = MockFactory()

        # Civitai meta via image_info
        harness.civitai.image_info["1"] = {
            "id": 1,
            "url": "https://example.com/images/1.jpg",
            "meta": {"prompt": "from civitai", "cfg": 7.0},
        }

        resources = []
        response = await harness.client.get(
            "/api/lm/recipes/import-remote",
            params={
                "image_url": "https://civitai.com/images/1",
                "name": "Merged Recipe",
                "resources": json.dumps(resources),
                "gen_params": json.dumps({"prompt": "from request", "steps": 25}),
            },
        )

        payload = await response.json()
        assert response.status == 200

        call = harness.persistence.save_calls[-1]
        metadata = call["metadata"]
        gen_params = metadata["gen_params"]

        assert gen_params["seed"] == 123


async def test_get_recipe_syntax(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        recipe_id = "test-recipe-id"
        harness.scanner.recipes[recipe_id] = {
            "id": recipe_id,
            "title": "Syntax Test",
            "loras": [{"name": "lora1", "weight": 0.5}],
        }

        # Mock the method that handlers call
        async def fake_get_recipe_syntax_tokens(rid):
            if rid == recipe_id:
                return ["<lora:lora1:0.5>"]
            raise RecipeNotFoundError(f"Recipe {rid} not found")

        harness.scanner.get_recipe_syntax_tokens = fake_get_recipe_syntax_tokens

        response = await harness.client.get(f"/api/lm/recipe/{recipe_id}/syntax")
        payload = await response.json()

        assert response.status == 200
        assert payload["success"] is True
        assert payload["syntax"] == "<lora:lora1:0.5>"

        # Test error path
        response_404 = await harness.client.get("/api/lm/recipe/non-existent/syntax")
        assert response_404.status == 404


async def test_batch_import_start_success(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.post(
            "/api/lm/recipes/batch-import/start",
            json={
                "items": [
                    {"source": "https://example.com/image1.png"},
                    {"source": "https://example.com/image2.png"},
                ],
                "tags": ["batch", "import"],
                "skip_no_metadata": True,
            },
        )
        payload = await response.json()
        assert response.status == 200
        assert payload["success"] is True
        assert "operation_id" in payload


async def test_batch_import_start_empty_items(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.post(
            "/api/lm/recipes/batch-import/start",
            json={"items": [], "tags": []},
        )
        payload = await response.json()
        assert response.status == 400
        assert payload["success"] is False
        assert "No items provided" in payload["error"]


async def test_batch_import_start_missing_source(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.post(
            "/api/lm/recipes/batch-import/start",
            json={"items": [{"source": ""}]},
        )
        payload = await response.json()
        assert response.status == 400
        assert payload["success"] is False
        assert "source" in payload["error"].lower()


async def test_batch_import_start_already_running(monkeypatch, tmp_path: Path) -> None:
    import asyncio

    async with recipe_harness(monkeypatch, tmp_path) as harness:
        original_analyze = harness.analysis.analyze_remote_image

        async def slow_analyze(*, url, recipe_scanner, civitai_client):
            await asyncio.sleep(0.5)
            return await original_analyze(
                url=url, recipe_scanner=recipe_scanner, civitai_client=civitai_client
            )

        harness.analysis.analyze_remote_image = slow_analyze

        items = [{"source": f"https://example.com/image{i}.png"} for i in range(10)]

        response1 = await harness.client.post(
            "/api/lm/recipes/batch-import/start",
            json={"items": items},
        )
        assert response1.status == 200

        payload1 = await response1.json()
        assert payload1["success"] is True

        await asyncio.sleep(0.1)

        response2 = await harness.client.post(
            "/api/lm/recipes/batch-import/start",
            json={"items": [{"source": "https://example.com/other.png"}]},
        )
        payload2 = await response2.json()
        assert response2.status == 409
        assert "already in progress" in payload2["error"].lower()


async def test_batch_import_get_progress_not_found(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.get(
            "/api/lm/recipes/batch-import/progress",
            params={"operation_id": "nonexistent-id"},
        )
        payload = await response.json()
        assert response.status == 404
        assert payload["success"] is False


async def test_batch_import_get_progress_missing_id(
    monkeypatch, tmp_path: Path
) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.get("/api/lm/recipes/batch-import/progress")
        payload = await response.json()
        assert response.status == 400
        assert payload["success"] is False


async def test_batch_import_cancel_success(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        start_response = await harness.client.post(
            "/api/lm/recipes/batch-import/start",
            json={"items": [{"source": "https://example.com/image.png"}]},
        )
        start_payload = await start_response.json()
        operation_id = start_payload["operation_id"]

        cancel_response = await harness.client.post(
            "/api/lm/recipes/batch-import/cancel",
            json={"operation_id": operation_id},
        )
        cancel_payload = await cancel_response.json()
        assert cancel_response.status == 200
        assert cancel_payload["success"] is True


async def test_batch_import_cancel_not_found(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.post(
            "/api/lm/recipes/batch-import/cancel",
            json={"operation_id": "nonexistent-id"},
        )
        payload = await response.json()
        assert response.status == 404
        assert payload["success"] is False


async def test_batch_import_cancel_missing_id(monkeypatch, tmp_path: Path) -> None:
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.post(
            "/api/lm/recipes/batch-import/cancel",
            json={},
        )
        payload = await response.json()
        assert response.status == 400
        assert payload["success"] is False


async def test_check_image_exists_uses_image_id_map(monkeypatch, tmp_path: Path) -> None:
    """check_image_exists must use precomputed image_id_map instead of scanning raw_data."""
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.scanner.image_id_map_override = {
            "123": "recipe-alpha",
            "789": "recipe-gamma",
        }

        response = await harness.client.get(
            "/api/lm/recipes/check-image-exists",
            params={"image_ids": "123,456,789"},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["success"] is True
        assert payload["results"]["123"] == {
            "in_library": True,
            "recipe_id": "recipe-alpha",
        }
        assert payload["results"]["456"] == {
            "in_library": False,
            "recipe_id": None,
        }
        assert payload["results"]["789"] == {
            "in_library": True,
            "recipe_id": "recipe-gamma",
        }


async def test_check_image_exists_handles_empty_input(monkeypatch, tmp_path: Path) -> None:
    """Empty or non-numeric image_ids must return an empty results dict."""
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        response = await harness.client.get(
            "/api/lm/recipes/check-image-exists",
            params={"image_ids": ""},
        )
        payload = await response.json()
        assert response.status == 200
        assert payload["results"] == {}


async def test_import_from_url_detects_duplicate_via_image_id_map(
    monkeypatch, tmp_path: Path,
) -> None:
    """import_from_url must return already_exists when image_id is in image_id_map."""
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.scanner.cached_raw = [
            {"id": "existing-recipe", "title": "My Recipe"},
        ]
        harness.scanner.image_id_map_override = {
            "99999": "existing-recipe",
        }

        response = await harness.client.get(
            "/api/lm/recipes/import-from-url",
            params={"image_url": "https://civitai.com/images/99999"},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["already_exists"] is True
        assert payload["recipe_id"] == "existing-recipe"
        assert payload["name"] == "My Recipe"


async def test_import_from_url_proceeds_when_image_id_not_in_map(
    monkeypatch, tmp_path: Path,
) -> None:
    """When image_id is absent from image_id_map, import_from_url must proceed to import."""
    async with recipe_harness(monkeypatch, tmp_path) as harness:
        harness.scanner.image_id_map_override = {
            "111": "some-other-recipe",
        }
        harness.civitai.image_info["99999"] = {
            "id": 99999,
            "url": "https://image.civitai.com/x/y/original=true/sample.jpeg",
            "type": "image",
            "meta": {"prompt": "test"},
        }

        response = await harness.client.get(
            "/api/lm/recipes/import-from-url",
            params={"image_url": "https://civitai.com/images/99999"},
        )

        # The import may succeed or fail depending on downstream stubs,
        # but it must NOT return already_exists
        payload = await response.json()
        assert payload.get("already_exists") is not True
