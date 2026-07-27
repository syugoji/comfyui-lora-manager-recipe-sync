import pytest
import json
from py.recipes.parsers.comfy import ComfyMetadataParser

@pytest.mark.asyncio
async def test_parse_metadata_without_loras(monkeypatch):
    checkpoint_info = {
        "id": 2224012,
        "modelId": 1908679,
        "model": {"name": "SDXL Checkpoint", "type": "checkpoint"},
        "name": "v1.0",
        "images": [{"url": "https://image.civitai.com/checkpoints/original=true"}],
        "baseModel": "sdxl",
        "downloadUrl": "https://civitai.com/api/download/checkpoint",
    }

    async def fake_metadata_provider():
        class Provider:
            async def get_model_version_info(self, version_id):
                assert version_id == "2224012"
                return checkpoint_info, None
        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    parser = ComfyMetadataParser()

    # User provided metadata
    metadata_json = {
        "resource-stack": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "urn:air:sdxl:checkpoint:civitai:1908679@2224012"}
        },
        "6": {
            "class_type": "smZ CLIPTextEncode",
            "inputs": {"text": "Positive prompt content"},
            "_meta": {"title": "Positive"}
        },
        "7": {
            "class_type": "smZ CLIPTextEncode",
            "inputs": {"text": "Negative prompt content"},
            "_meta": {"title": "Negative"}
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "seed": 904124997,
                "steps": 35,
                "cfg": 6,
                "denoise": 0.1,
                "model": ["resource-stack", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["21", 0]
            },
            "_meta": {"title": "KSampler"}
        },
        "extraMetadata": json.dumps({
            "prompt": "One woman, (solo:1.3), ...",
            "negativePrompt": "embedding:EasyNegative, lowres, worst quality, ...",
            "steps": 35,
            "cfgScale": 6,
            "sampler": "euler_ancestral",
            "seed": 904124997,
            "width": 1024,
            "height": 1024
        })
    }

    result = await parser.parse_metadata(json.dumps(metadata_json))

    assert "error" not in result
    assert result["loras"] == []
    assert result["checkpoint"] is not None
    assert int(result["checkpoint"]["modelId"]) == 1908679
    assert int(result["checkpoint"]["id"]) == 2224012
    assert result["gen_params"]["prompt"] == "One woman, (solo:1.3), ..."
    assert result["gen_params"]["steps"] == 35
    assert result["gen_params"]["size"] == "1024x1024"
    assert result["embeddings"][0]["file_name"] == "EasyNegative"
    assert result["from_comfy_metadata"] is True

@pytest.mark.asyncio
async def test_parse_metadata_without_extra_metadata(monkeypatch):
    async def fake_metadata_provider():
        class Provider:
            async def get_model_version_info(self, version_id):
                return {"model": {"name": "Test"}, "id": version_id}, None
        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    parser = ComfyMetadataParser()

    metadata_json = {
        "node_1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "urn:air:sdxl:checkpoint:civitai:123@456"}
        }
    }

    result = await parser.parse_metadata(json.dumps(metadata_json))

    assert "error" not in result
    assert result["loras"] == []
    assert result["checkpoint"]["id"] == "456"


class _FakeModelScanner:
    """Scanner stub resolving local path names against a single cache item."""

    def __init__(self, item, expected_name):
        self.item = item
        self.expected_name = expected_name

    async def get_model_info_by_name(self, name):
        assert name == self.expected_name
        return self.item


class _FakeRecipeScanner:
    pass


@pytest.mark.asyncio
async def test_parse_metadata_with_local_path_names(monkeypatch):
    """Local ComfyUI outputs (file path style lora_name/ckpt_name) must not lose resources."""
    async def fake_metadata_provider():
        return None

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    lora_item = {
        "file_name": "my_style_lora",
        "file_path": "D:/models/loras/Illustrious/anime/my_style_lora.safetensors",
        "sha256": "ABCDEF123",
        "base_model": "Illustrious",
        "size": 12345,
        "model_name": "My Style LoRA",
        "civitai": {"id": 111, "modelId": 222, "name": "v2.0"},
    }
    ckpt_item = {
        "file_name": "wai_v14",
        "file_path": "D:/models/checkpoints/wai_v14.safetensors",
        "sha256": "FFEE99",
        "base_model": "Illustrious",
        "model_name": "WAI",
        "civitai": {"id": 333, "modelId": 444, "name": "v14"},
    }

    recipe_scanner = _FakeRecipeScanner()
    recipe_scanner._lora_scanner = _FakeModelScanner(
        lora_item, "Illustrious\\anime\\my_style_lora.safetensors"
    )
    recipe_scanner._checkpoint_scanner = _FakeModelScanner(
        ckpt_item, "wai_v14.safetensors"
    )

    metadata_json = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "wai_v14.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "Illustrious\\anime\\my_style_lora.safetensors",
                "strength_model": 0.8,
            },
        },
    }

    parser = ComfyMetadataParser()
    result = await parser.parse_metadata(
        json.dumps(metadata_json), recipe_scanner=recipe_scanner
    )

    assert "error" not in result
    assert len(result["loras"]) == 1
    lora = result["loras"][0]
    assert lora["existsLocally"] is True
    assert lora["localPath"] == lora_item["file_path"]
    assert lora["weight"] == 0.8
    assert lora["hash"] == "abcdef123"
    assert lora["name"] == "My Style LoRA"
    assert lora["id"] == 111
    assert lora["modelId"] == 222

    checkpoint = result["checkpoint"]
    assert checkpoint is not None
    assert checkpoint["type"] == "checkpoint"
    assert checkpoint["existsLocally"] is True
    assert checkpoint["id"] == 333
    assert checkpoint["name"] == "WAI"

    assert result["base_model"] == "Illustrious"


@pytest.mark.asyncio
async def test_parse_metadata_local_lora_model_only_node(monkeypatch):
    """LoraLoaderModelOnly nodes are also picked up for local path resolution."""
    async def fake_metadata_provider():
        return None

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    lora_item = {
        "file_name": "detailer",
        "file_path": "D:/models/loras/detailer.safetensors",
        "sha256": "AA11",
        "base_model": "SDXL 1.0",
        "civitai": {},
    }
    recipe_scanner = _FakeRecipeScanner()
    recipe_scanner._lora_scanner = _FakeModelScanner(lora_item, "detailer.safetensors")

    metadata_json = {
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": "detailer.safetensors", "strength_model": 1.0},
        }
    }

    parser = ComfyMetadataParser()
    result = await parser.parse_metadata(
        json.dumps(metadata_json), recipe_scanner=recipe_scanner
    )

    assert "error" not in result
    assert len(result["loras"]) == 1
    assert result["loras"][0]["file_name"] == "detailer"
    assert result["loras"][0]["name"] == "detailer"


class _FakeMultiScanner:
    """Scanner stub resolving several names against a name -> item mapping."""

    def __init__(self, items_by_name):
        self.items_by_name = items_by_name
        self.requested = []

    async def get_model_info_by_name(self, name):
        self.requested.append(name)
        return self.items_by_name.get(name)


def _lora_item(name):
    return {
        "file_name": name,
        "file_path": f"D:/models/loras/{name}.safetensors",
        "sha256": f"HASH{name}",
        "base_model": "Illustrious",
        "model_name": f"{name} model",
        "civitai": {"id": 1, "modelId": 2, "name": "v1"},
    }


@pytest.mark.asyncio
async def test_parse_metadata_lora_manager_node(monkeypatch):
    """`Lora Loader (LoraManager)` bundles many LoRAs in one node — none may be lost."""
    async def fake_metadata_provider():
        return None

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    recipe_scanner = _FakeRecipeScanner()
    recipe_scanner._lora_scanner = _FakeMultiScanner(
        {name: _lora_item(name) for name in ("first_lora", "second_lora", "disabled_lora")}
    )

    metadata_json = {
        "10": {
            "class_type": "Lora Loader (LoraManager)",
            "inputs": {
                "text": "<lora:first_lora:0.25> <lora:second_lora:0.90> <lora:disabled_lora:1.00>",
                "loras": {
                    "__value__": [
                        {"name": "first_lora", "strength": 0.25, "active": True, "clipStrength": 0.25},
                        {"name": "second_lora", "strength": 0.9, "active": True, "clipStrength": 0.9},
                        {"name": "disabled_lora", "strength": 1.0, "active": False, "clipStrength": 1.0},
                    ]
                },
            },
        }
    }

    parser = ComfyMetadataParser()
    result = await parser.parse_metadata(
        json.dumps(metadata_json), recipe_scanner=recipe_scanner
    )

    assert "error" not in result
    # The deactivated entry did not affect the image, so it must not be recorded.
    assert [l["file_name"] for l in result["loras"]] == ["first_lora", "second_lora"]
    assert [l["weight"] for l in result["loras"]] == [0.25, 0.9]
    assert all(l["existsLocally"] for l in result["loras"])
    # The structured widget wins: the text widget must not add duplicates.
    assert recipe_scanner._lora_scanner.requested == ["first_lora", "second_lora"]


@pytest.mark.asyncio
async def test_parse_metadata_lora_manager_old_list_format(monkeypatch):
    """The pre-`__value__` widget shape (a bare list) is still accepted."""
    async def fake_metadata_provider():
        return None

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    recipe_scanner = _FakeRecipeScanner()
    recipe_scanner._lora_scanner = _FakeMultiScanner({"legacy_lora": _lora_item("legacy_lora")})

    metadata_json = {
        "3": {
            "class_type": "Lora Stacker (LoraManager)",
            "inputs": {"loras": [{"name": "legacy_lora", "strength": 0.5, "active": True}]},
        }
    }

    parser = ComfyMetadataParser()
    result = await parser.parse_metadata(
        json.dumps(metadata_json), recipe_scanner=recipe_scanner
    )

    assert [l["file_name"] for l in result["loras"]] == ["legacy_lora"]
    assert result["loras"][0]["weight"] == 0.5


@pytest.mark.asyncio
async def test_parse_metadata_lora_manager_text_fallback(monkeypatch):
    """Text-only loader variants carry no `loras` widget — fall back to <lora:...> syntax."""
    async def fake_metadata_provider():
        return None

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    recipe_scanner = _FakeRecipeScanner()
    recipe_scanner._lora_scanner = _FakeMultiScanner(
        {"text_lora": _lora_item("text_lora"), "spaced name_lora": _lora_item("spaced name_lora")}
    )

    metadata_json = {
        "4": {
            "class_type": "LoRA Text Loader (LoraManager)",
            "inputs": {"text": "<lora:text_lora:0.75> <lora:spaced name_lora:1.20>"},
        }
    }

    parser = ComfyMetadataParser()
    result = await parser.parse_metadata(
        json.dumps(metadata_json), recipe_scanner=recipe_scanner
    )

    assert [l["file_name"] for l in result["loras"]] == ["text_lora", "spaced name_lora"]
    assert [l["weight"] for l in result["loras"]] == [0.75, 1.2]


@pytest.mark.asyncio
async def test_parse_metadata_local_name_unresolved_is_skipped(monkeypatch):
    """Unresolvable local names are skipped without raising an error."""
    async def fake_metadata_provider():
        return None

    monkeypatch.setattr(
        "py.recipes.parsers.comfy.get_default_metadata_provider",
        fake_metadata_provider,
    )

    class _MissScanner:
        async def get_model_info_by_name(self, name):
            return None

    recipe_scanner = _FakeRecipeScanner()
    recipe_scanner._lora_scanner = _MissScanner()
    recipe_scanner._checkpoint_scanner = _MissScanner()

    metadata_json = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "gone.safetensors"},
        },
        "2": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "gone_lora.safetensors", "strength_model": 0.5},
        },
    }

    parser = ComfyMetadataParser()
    result = await parser.parse_metadata(
        json.dumps(metadata_json), recipe_scanner=recipe_scanner
    )

    assert "error" not in result
    assert result["loras"] == []
    assert result["checkpoint"] is None
