import io
from pathlib import Path

import pytest
from PIL import Image

from py.services.recipes import prompt_draft_service as prompt_draft_module
from py.services.recipes.prompt_draft_service import PromptDraftError, RecipePromptDraftService
from py.services.recipes.replay_manifest_service import ReplayManifestService


def _manifest_entry(index: int, name: str, strength: float, *, trained_words=None):
    resource = {"file_name": name, "modelName": f"Model {name}"}
    if trained_words:
        resource["trainedWords"] = trained_words
    return {
        "requirement_id": f"recipe:{index}",
        "kind": "lora",
        "required": True,
        "resource": resource,
        "resolution": {"status": "recipe_match", "match": "model_version_id"},
        "expected": {"strength_model": strength, "strength_clip": strength},
        "evidence": [{"source": "a1111_civitai_resources"}],
    }


@pytest.mark.asyncio
async def test_prompt_draft_preserves_loras_trigger_and_negative(tmp_path: Path, monkeypatch):
    preview = tmp_path / "recipe.webp"
    Image.new("RGB", (480, 701), (240, 100, 180)).save(preview, "WEBP")
    manifest = {
        "schema": "lora-manager.replay-manifest",
        "version": 1,
        "manifest_hash": "manifest-4",
        "errors": [],
        "advisory_resources": [],
        "required_resources": [
            _manifest_entry(0, "748cmSDXL", 0.45),
            _manifest_entry(1, "NV_KawaiiTech_WM_IL_SH", 0.9),
            _manifest_entry(2, "ILLMythP0rtr4itStyle", 0.7),
            _manifest_entry(3, "tove-nikke-richy-v1_ixl", 1.0, trained_words=["tovrd"]),
        ],
    }
    recipe = {
        "id": "recipe-4",
        "title": "Four LoRAs",
        "file_path": str(preview),
        "checkpoint": {"file_name": "wai.safetensors", "baseModel": "Illustrious"},
        "gen_params": {
            "prompt": "masterpiece, <lora:tove-nikke-richy-v1_ixl:1>, tovrd, blue eyes",
            "negative_prompt": "bad hands, watermark",
        },
        "replay_manifest": manifest,
    }
    service = RecipePromptDraftService(
        replay_manifest_service=ReplayManifestService()
    )

    async def fake_model(profile):
        return profile.identifier

    async def fake_request(**kwargs):
        assert kwargs["prompt_payload"]["context"]["model"]["required_loras"]
        assert kwargs["image_data_url"].startswith("data:image/jpeg;base64,")
        return {
            "description": "A detailed pink portrait.",
            "scene_prompt": "blonde woman on an ornate pink sofa, side view, cinematic lighting",
        }

    monkeypatch.setattr(service, "_ensure_lm_studio_model", fake_model)
    monkeypatch.setattr(service, "_request_scene_prompt", fake_request)

    draft = await service.create_draft(recipe)

    assert draft["prompt_source"] == "lm_studio"
    assert draft["negative_prompt"] == "bad hands, watermark"
    assert draft["protected"]["negative_prompt_unchanged"] is True
    assert draft["protected"]["trigger_tokens"] == ["tovrd"]
    assert draft["proposed_prompt"].startswith(
        "<lora:748cmSDXL:0.45>, <lora:NV_KawaiiTech_WM_IL_SH:0.9>, "
        "<lora:ILLMythP0rtr4itStyle:0.7>, <lora:tove-nikke-richy-v1_ixl:1>, tovrd"
    )
    assert draft["image"]["preview_used"] is True
    assert draft["image"]["input_width"] == 480
    assert draft["image"]["warning"]
    assert draft["lm_studio"]["gpu"] == "off"


@pytest.mark.asyncio
async def test_promptless_draft_adds_primary_trained_word_for_each_required_lora(
    tmp_path: Path, monkeypatch
):
    preview = tmp_path / "promptless.webp"
    Image.new("RGB", (320, 480), "purple").save(preview, "WEBP")
    manifest = {
        "schema": "lora-manager.replay-manifest",
        "version": 1,
        "manifest_hash": "promptless-manifest",
        "errors": [],
        "advisory_resources": [],
        "required_resources": [
            _manifest_entry(0, "USNR_STYLE", 0.9, trained_words=["(usnr)", "optional outfit"]),
            _manifest_entry(1, "CHARACTER", 1.0, trained_words=["character_token", "blue dress"]),
        ],
    }
    recipe = {
        "id": "promptless",
        "file_path": str(preview),
        "gen_params": {"prompt": "", "negative_prompt": "bad hands"},
        "replay_manifest": manifest,
    }
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())

    async def fake_model(profile):
        return profile.identifier

    async def fake_request(**_kwargs):
        return {"description": "portrait", "scene_prompt": "detailed portrait"}

    monkeypatch.setattr(service, "_ensure_lm_studio_model", fake_model)
    monkeypatch.setattr(service, "_request_scene_prompt", fake_request)

    draft = await service.create_draft(recipe)

    assert draft["protected"]["trigger_tokens"] == ["(usnr)", "character_token"]
    assert draft["suggested_triggers"] == [
        "(usnr)", "optional outfit", "character_token", "blue dress"
    ]
    assert draft["proposed_prompt"].startswith(
        "<lora:USNR_STYLE:0.9>, <lora:CHARACTER:1>, (usnr), character_token"
    )


@pytest.mark.asyncio
async def test_prompt_draft_enriches_exact_local_triggers_and_preserves_primary_tokens(
    tmp_path: Path, monkeypatch
):
    preview = tmp_path / "local-triggers.webp"
    Image.new("RGB", (320, 480), "navy").save(preview, "WEBP")
    manifest = {
        "schema": "lora-manager.replay-manifest",
        "version": 1,
        "manifest_hash": "local-trigger-manifest",
        "errors": [],
        "advisory_resources": [],
        "required_resources": [
            {
                **_manifest_entry(0, "Mei_Misaki_ILXL", 0.3),
                "resource": {
                    "file_name": "Mei_Misaki_ILXL",
                    "hash": "a" * 64,
                },
            },
            {
                **_manifest_entry(1, "HighlegArmor-Persona3-Illustrious-V1", 0.9),
                "resource": {
                    "file_name": "HighlegArmor-Persona3-Illustrious-V1",
                    "modelVersionId": 1423104,
                },
            },
        ],
    }
    recipe = {
        "id": "local-triggers",
        "file_path": str(preview),
        "gen_params": {
            "prompt": "black hair, misaki mei, highleg armor, gold armor",
            "negative_prompt": "bad hands",
        },
        "replay_manifest": manifest,
    }

    class FakeCache:
        raw_data = [
            {
                "file_name": "Mei_Misaki_ILXL",
                "sha256": "a" * 64,
                "civitai": {
                    "id": 1076221,
                    "trainedWords": [
                        "misaki mei, short hair, black hair, red eyes, eyepatch"
                    ],
                },
            },
            {
                "file_name": "HighlegArmor-Persona3-Illustrious-V1",
                "sha256": "b" * 64,
                "civitai": {
                    "id": 1423104,
                    "trainedWords": [
                        "highleg armor, bikini armor, midriff cutout, side cutout"
                    ],
                },
            },
        ]

    class FakeScanner:
        async def get_cached_data(self):
            return FakeCache()

    service = RecipePromptDraftService(
        replay_manifest_service=ReplayManifestService(),
        lora_scanner_getter=lambda: FakeScanner(),
    )

    async def fake_model(profile):
        return profile.identifier

    async def fake_request(**_kwargs):
        return {"description": "portrait", "scene_prompt": "detailed portrait"}

    monkeypatch.setattr(service, "_ensure_lm_studio_model", fake_model)
    monkeypatch.setattr(service, "_request_scene_prompt", fake_request)

    draft = await service.create_draft(recipe)

    assert draft["protected"]["trigger_tokens"] == ["misaki mei", "highleg armor"]
    assert draft["suggested_triggers"] == [
        "misaki mei, short hair, black hair, red eyes, eyepatch",
        "highleg armor, bikini armor, midriff cutout, side cutout",
    ]
    assert draft["manifest_hash"] == "local-trigger-manifest"
    assert "trainedWords" not in manifest["required_resources"][0]["resource"]


@pytest.mark.parametrize(
    ("resource", "local_loras"),
    [
        (
            {"file_name": "same-name", "hash": "a" * 64},
            [
                {
                    "file_name": "same-name",
                    "sha256": "b" * 64,
                    "civitai": {"trainedWords": ["wrong"]},
                }
            ],
        ),
        (
            {"file_name": "same-name", "modelVersionId": 10},
            [
                {
                    "file_name": "same-name",
                    "civitai": {"id": 11, "trainedWords": ["wrong"]},
                }
            ],
        ),
        (
            {"file_name": "same-name"},
            [
                {"file_name": "same-name", "civitai": {"trainedWords": ["one"]}},
                {"file_name": "same-name", "civitai": {"trainedWords": ["two"]}},
            ],
        ),
    ],
)
def test_local_trigger_matching_fails_closed_on_identity_mismatch_or_ambiguity(
    resource, local_loras
):
    matches = RecipePromptDraftService._matching_local_loras(resource, local_loras)

    assert len(matches) != 1


def test_trained_word_does_not_match_inside_an_unrelated_prompt_word():
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())
    manifest = {
        "required_resources": [
            _manifest_entry(0, "style", 1.0, trained_words=["art"]),
        ]
    }

    protected = service._protected_prompt_parts(
        {"gen_params": {"prompt": "cartoon portrait"}}, manifest
    )

    assert protected["trigger_tokens"] == []
    assert protected["suggested_triggers"] == ["art"]


@pytest.mark.asyncio
async def test_prompt_draft_rejects_ai_protected_syntax(tmp_path: Path, monkeypatch):
    preview = tmp_path / "recipe.png"
    Image.new("RGB", (64, 64), "white").save(preview)
    recipe = {
        "id": "recipe-bad-output",
        "file_path": str(preview),
        "gen_params": {"prompt": "portrait", "negative_prompt": "bad"},
        "replay_manifest": {
            "schema": "lora-manager.replay-manifest",
            "version": 1,
            "manifest_hash": "none",
            "required_resources": [],
            "advisory_resources": [],
            "errors": [],
        },
    }
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())

    async def fake_model(profile):
        return profile.identifier

    async def fake_request(**_kwargs):
        return {"description": "bad", "scene_prompt": "portrait, <lora:injected:1>"}

    monkeypatch.setattr(service, "_ensure_lm_studio_model", fake_model)
    monkeypatch.setattr(service, "_request_scene_prompt", fake_request)

    with pytest.raises(PromptDraftError) as exc_info:
        await service.create_draft(recipe)

    assert exc_info.value.code == "LM_OUTPUT_PROTECTED_SYNTAX"


def test_prepare_image_prefers_local_original_over_preview(tmp_path: Path):
    original = tmp_path / "original.png"
    preview = tmp_path / "preview.webp"
    Image.new("RGB", (1200, 800), "blue").save(original)
    Image.new("RGB", (480, 320), "red").save(preview, "WEBP")
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())

    prepared = service._prepare_image(
        {"source_image_path": str(original), "file_path": str(preview)}
    )

    assert prepared.preview_used is False
    assert prepared.source_kind == "source_image"
    assert (prepared.input_width, prepared.input_height) == (1200, 800)


@pytest.mark.asyncio
async def test_prepare_best_image_prefers_civitai_original_over_local_preview(
    tmp_path: Path,
):
    preview = tmp_path / "preview.webp"
    Image.new("RGB", (480, 320), "red").save(preview, "WEBP")
    original_bytes = io.BytesIO()
    Image.new("RGB", (1400, 900), "blue").save(original_bytes, "PNG")

    class FakeCivitaiClient:
        async def get_image_info(self, image_id, source_url=None):
            assert image_id == "12345"
            assert source_url == "https://civitai.com/images/12345"
            return {"url": "https://image.civitai.com/x/original.png"}

    class FakeDownloader:
        async def download_to_memory(self, url, use_auth=False):
            assert url == "https://image.civitai.com/x/original.png"
            assert use_auth is False
            return True, original_bytes.getvalue(), {}

    async def downloader_factory():
        return FakeDownloader()

    service = RecipePromptDraftService(
        replay_manifest_service=ReplayManifestService(),
        civitai_client_getter=FakeCivitaiClient,
        downloader_factory=downloader_factory,
    )

    prepared = await service._prepare_best_image(
        {
            "id": "recipe-original",
            "source_path": "https://civitai.com/images/12345",
            "file_path": str(preview),
        }
    )

    assert prepared.preview_used is False
    assert prepared.source_kind == "civitai_original"
    assert (prepared.input_width, prepared.input_height) == (1400, 900)


@pytest.mark.asyncio
async def test_ensure_model_loads_selected_qwythos_with_two_hour_ttl(monkeypatch):
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())
    loaded_responses = [
        ["user-loaded-qwythos"],
        ["user-loaded-qwythos", "lora-manager-qwythos"],
    ]
    calls = []

    async def server_ready():
        return True

    async def loaded_models():
        return loaded_responses.pop(0)

    async def run_lms(*args, timeout):
        calls.append((args, timeout))
        return "loaded"

    monkeypatch.setattr(service, "_lm_server_ready", server_ready)
    monkeypatch.setattr(service, "_loaded_models", loaded_models)
    monkeypatch.setattr(service, "_run_lms", run_lms)

    profile = prompt_draft_module.LM_STUDIO_MODELS["qwythos-q5"]
    model_id = await service._ensure_lm_studio_model(profile)

    assert model_id == "lora-manager-qwythos"
    assert len(calls) == 1
    args, timeout = calls[0]
    assert args[:2] == ("load", "qwythos-9b-claude-mythos-5-1m@q5_k_m")
    assert args[args.index("--gpu") + 1] == "off"
    assert args[args.index("--context-length") + 1] == "8192"
    assert args[args.index("--ttl") + 1] == "7200"
    assert args[args.index("--identifier") + 1] == "lora-manager-qwythos"
    assert timeout == 300


@pytest.mark.asyncio
async def test_ensure_model_switch_unloads_only_other_managed_model(monkeypatch):
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())
    loaded_responses = [
        ["user-owned-model", "lora-manager-qwythos"],
        ["user-owned-model", "lora-manager-qwen35-q6"],
    ]
    calls = []

    async def server_ready():
        return True

    async def loaded_models():
        return loaded_responses.pop(0)

    async def run_lms(*args, timeout):
        calls.append((args, timeout))
        return "ok"

    monkeypatch.setattr(service, "_lm_server_ready", server_ready)
    monkeypatch.setattr(service, "_loaded_models", loaded_models)
    monkeypatch.setattr(service, "_run_lms", run_lms)

    profile = prompt_draft_module.LM_STUDIO_MODELS["qwen35-q6"]
    model_id = await service._ensure_lm_studio_model(profile)

    assert model_id == "lora-manager-qwen35-q6"
    assert calls[0] == (("unload", "lora-manager-qwythos"), 120)
    assert calls[1][0][:2] == ("load", "qwen/qwen3.5-9b")
    assert all("user-owned-model" not in call[0] for call in calls)


@pytest.mark.asyncio
async def test_release_managed_models_preserves_user_loaded_models(monkeypatch):
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())
    calls = []

    async def server_ready():
        return True

    async def loaded_models():
        return [
            "user-owned-model",
            "lora-manager-qwen35-q6",
            "lora-manager-qwythos",
        ]

    async def run_lms(*args, timeout):
        calls.append((args, timeout))
        return "ok"

    monkeypatch.setattr(service, "_lm_server_ready", server_ready)
    monkeypatch.setattr(service, "_loaded_models", loaded_models)
    monkeypatch.setattr(service, "_run_lms", run_lms)

    released = await service.release_managed_models()

    assert released == ["lora-manager-qwen35-q6", "lora-manager-qwythos"]
    assert calls == [
        (("unload", "lora-manager-qwen35-q6"), 120),
        (("unload", "lora-manager-qwythos"), 120),
    ]
    assert all("user-owned-model" not in args for args, _timeout in calls)


@pytest.mark.asyncio
async def test_prompt_draft_cache_reuses_result_and_force_regenerate_bypasses_it(
    tmp_path: Path, monkeypatch
):
    preview = tmp_path / "cache-source.png"
    Image.new("RGB", (96, 96), "teal").save(preview)
    recipe = {
        "id": "cache-recipe",
        "source_image_path": str(preview),
        "gen_params": {"prompt": "portrait", "negative_prompt": "bad hands"},
        "replay_manifest": {
            "schema": "lora-manager.replay-manifest",
            "version": 1,
            "manifest_hash": "c" * 64,
            "required_resources": [],
            "advisory_resources": [],
            "errors": [],
        },
    }
    service = RecipePromptDraftService(
        replay_manifest_service=ReplayManifestService(),
        cache_dir=tmp_path / "prompt-cache",
    )
    model_calls = []
    request_calls = []

    async def fake_model(profile):
        model_calls.append(profile.option)
        return profile.identifier

    async def fake_request(**_kwargs):
        request_calls.append(True)
        return {
            "description": f"description {len(request_calls)}",
            "scene_prompt": f"portrait, teal light, pass {len(request_calls)}",
        }

    monkeypatch.setattr(service, "_ensure_lm_studio_model", fake_model)
    monkeypatch.setattr(service, "_request_scene_prompt", fake_request)

    first = await service.create_draft(recipe, model="qwen35-q6")
    second = await service.create_draft(recipe, model="qwen35-q6")
    forced = await service.create_draft(
        recipe, model="qwen35-q6", force_regenerate=True
    )

    assert first["lm_studio"]["cache_hit"] is False
    assert second["lm_studio"]["cache_hit"] is True
    assert second["scene_prompt"] == first["scene_prompt"]
    assert forced["lm_studio"]["cache_hit"] is False
    assert forced["scene_prompt"] != first["scene_prompt"]
    assert model_calls == ["qwen35-q6", "qwen35-q6"]
    assert len(request_calls) == 2
    assert len(list((tmp_path / "prompt-cache").glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_prompt_draft_cache_is_separate_for_each_model(tmp_path: Path, monkeypatch):
    preview = tmp_path / "model-source.png"
    Image.new("RGB", (64, 64), "blue").save(preview)
    recipe = {
        "id": "model-cache",
        "source_image_path": str(preview),
        "gen_params": {"prompt": "portrait"},
        "replay_manifest": {
            "manifest_hash": "d" * 64,
            "required_resources": [],
            "advisory_resources": [],
            "errors": [],
        },
    }
    service = RecipePromptDraftService(
        replay_manifest_service=ReplayManifestService(),
        cache_dir=tmp_path / "model-cache",
    )

    async def fake_model(profile):
        return profile.identifier

    async def fake_request(**kwargs):
        return {
            "description": kwargs["model_id"],
            "scene_prompt": f"portrait, {kwargs['model_id']}",
        }

    monkeypatch.setattr(service, "_ensure_lm_studio_model", fake_model)
    monkeypatch.setattr(service, "_request_scene_prompt", fake_request)

    qwen = await service.create_draft(recipe, model="qwen35-q6")
    qwythos = await service.create_draft(recipe, model="qwythos-q5")

    assert qwen["lm_studio"]["model_option"] == "qwen35-q6"
    assert qwythos["lm_studio"]["model_option"] == "qwythos-q5"
    assert qwen["lm_studio"]["cache_key"] != qwythos["lm_studio"]["cache_key"]
    assert len(list((tmp_path / "model-cache").glob("*.json"))) == 2


@pytest.mark.asyncio
async def test_request_scene_prompt_accepts_qwythos_reasoning_content(monkeypatch):
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": (
                        '{"description":"visible description",'
                        '"scene_prompt":"portrait, pink lighting"}'
                    ),
                }
            }
        ]
    }

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def text(self):
            import json

            return json.dumps(response_payload)

    class FakeSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(prompt_draft_module.aiohttp, "ClientSession", FakeSession)
    service = RecipePromptDraftService(replay_manifest_service=ReplayManifestService())

    result = await service._request_scene_prompt(
        model_id="lora-manager-qwythos",
        image_data_url="data:image/jpeg;base64,AA==",
        prompt_payload={"task": "test"},
    )

    assert result == {
        "description": "visible description",
        "scene_prompt": "portrait, pink lighting",
    }
