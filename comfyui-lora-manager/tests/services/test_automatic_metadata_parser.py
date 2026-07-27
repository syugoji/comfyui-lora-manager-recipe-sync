import pytest

from py.recipes.parsers.automatic import AutomaticMetadataParser


@pytest.mark.asyncio
async def test_parse_metadata_extracts_textual_inversion_hashes(monkeypatch):
    embedding_info = {
        "id": 202,
        "modelId": 101,
        "model": {"name": "Easy Negative", "type": "TextualInversion"},
        "name": "v1",
        "baseModel": "SD 1.5",
        "files": [
            {
                "type": "Model",
                "primary": True,
                "name": "EasyNegative.safetensors",
                "hashes": {"SHA256": "AABBCCDD0011"},
            }
        ],
    }

    async def fake_metadata_provider():
        class Provider:
            async def get_model_by_hash(self, model_hash):
                assert model_hash == "aabbccdd0011"
                return embedding_info, None

        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.automatic.get_default_metadata_provider",
        fake_metadata_provider,
    )

    result = await AutomaticMetadataParser().parse_metadata(
        'portrait\nNegative prompt: EasyNegative\nSteps: 20, Sampler: Euler, '
        'CFG scale: 7, Seed: 1, TI hashes: "EasyNegative: aabbccdd0011"'
    )

    assert len(result["embeddings"]) == 1
    assert result["embeddings"][0]["id"] == 202
    assert result["embeddings"][0]["modelId"] == 101
    assert result["embeddings"][0]["hash"] == "aabbccdd0011"
    assert result["embeddings"][0]["isDeleted"] is False
    assert result["embeddings"][0]["unresolved"] is False


@pytest.mark.asyncio
async def test_parse_metadata_keeps_used_embedding_when_short_hash_is_unresolved(
    monkeypatch,
):
    async def fake_metadata_provider():
        class Provider:
            async def get_model_by_hash(self, _model_hash):
                return None, "Model not found"

        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.automatic.get_default_metadata_provider",
        fake_metadata_provider,
    )

    result = await AutomaticMetadataParser().parse_metadata(
        'portrait\nNegative prompt: lazyneg\nSteps: 20, TI hashes: "lazyneg: 5bbc32fdd8a0"'
    )

    assert result["embeddings"][0]["isDeleted"] is False
    assert result["embeddings"][0]["unresolved"] is True


@pytest.mark.asyncio
async def test_parse_metadata_extracts_legacy_addnet_loras_with_separate_strengths(
    monkeypatch,
):
    requested_hashes = []

    async def fake_metadata_provider():
        class Provider:
            async def get_model_by_hash(self, model_hash):
                requested_hashes.append(model_hash)
                return None, "Model not found"

        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.automatic.get_default_metadata_provider",
        fake_metadata_provider,
    )

    parser = AutomaticMetadataParser()
    metadata_text = (
        "ink painting portrait\n"
        "Negative prompt: low quality\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123, Size: 768x1792, "
        "Model hash: abc123, Model: baseModel, "
        "AddNet Enabled: True, "
        "AddNet Module 1: LoRA, AddNet Model 1: Moxin_Shukezouma11(494301de3d6e), "
        "AddNet Weight A 1: 0.7, AddNet Weight B 1: 0.6, "
        "AddNet Module 2: LoRA, AddNet Model 2: Moxin_10(17cd20c7b6ea), "
        "AddNet Weight A 2: 0.3, AddNet Weight B 2: 0.3, "
        "AddNet Module 3: LoRA, AddNet Model 3: chilloutmixss_xss10(9d82c7787e79), "
        "AddNet Weight A 3: 0.5, AddNet Weight B 3: 0.5, "
        "AddNet Module 4: LoRA, AddNet Model 4: firekeeperLoraFrom_fierkeeper16(acd58eb24484), "
        "AddNet Weight A 4: 0.8, AddNet Weight B 4: 0.8"
    )

    result = await parser.parse_metadata(metadata_text)

    loras = result["loras"]
    assert [lora["file_name"] for lora in loras] == [
        "Moxin_Shukezouma11",
        "Moxin_10",
        "chilloutmixss_xss10",
        "firekeeperLoraFrom_fierkeeper16",
    ]
    assert [lora["hash"] for lora in loras] == [
        "494301de3d6e",
        "17cd20c7b6ea",
        "9d82c7787e79",
        "acd58eb24484",
    ]
    assert loras[0]["strength_model"] == 0.7
    assert loras[0]["strength_clip"] == 0.6
    assert loras[0]["weight"] == 0.7
    assert all(lora["isDeleted"] is False for lora in loras)
    assert all(lora["unresolved"] is True for lora in loras)
    assert (
        "f79768ec7b9e4f615458e0ea645424af183ffc0ebf020caab994eebe4dc84f7d"
        in requested_hashes
    )
    assert (
        "acd58eb244848f0ffddc647435ddd983eec5ceeb10c70baa6ca333a089715b69"
        in requested_hashes
    )


@pytest.mark.asyncio
async def test_legacy_addnet_hash_falls_back_to_verified_version(monkeypatch):
    version_info = {
        "id": 20143,
        "modelId": 12597,
        "model": {"name": "墨心 MoXin", "type": "LORA"},
        "name": "疏可走马 Shukezouma 1.1",
        "baseModel": "SD 1.5",
        "downloadUrl": "https://civitai.com/api/download/models/20143",
        "files": [
            {
                "type": "Model",
                "primary": True,
                "name": "shukezouma_v1_1.safetensors",
                "hashes": {
                    "SHA256": "F79768EC7B9E4F615458E0EA645424AF183FFC0EBF020CAAB994EEBE4DC84F7D"
                },
            }
        ],
    }

    async def fake_metadata_provider():
        class Provider:
            async def get_model_by_hash(self, model_hash):
                assert model_hash == version_info["files"][0]["hashes"]["SHA256"].lower()
                return None, "Model not found"

            async def get_model_version_info(self, version_id):
                assert version_id == "20143"
                return version_info, None

        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.automatic.get_default_metadata_provider",
        fake_metadata_provider,
    )

    result = await AutomaticMetadataParser().parse_metadata(
        "ink painting\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 1, "
        "AddNet Enabled: True, AddNet Module 1: LoRA, "
        "AddNet Model 1: Moxin_Shukezouma11(494301de3d6e), "
        "AddNet Weight A 1: 0.7, AddNet Weight B 1: 0.6"
    )

    lora = result["loras"][0]
    assert lora["id"] == 20143
    assert lora["modelId"] == 12597
    assert lora["hash"] == version_info["files"][0]["hashes"]["SHA256"].lower()
    assert lora.get("unresolved") is not True


@pytest.mark.asyncio
async def test_parse_metadata_extracts_checkpoint_from_civitai_resources(monkeypatch):
    checkpoint_info = {
        "id": 2442439,
        "modelId": 123456,
        "model": {"name": "Z Image", "type": "checkpoint"},
        "name": "Turbo",
        "images": [{"url": "https://image.civitai.com/checkpoints/original=true"}],
        "baseModel": "sdxl",
        "downloadUrl": "https://civitai.com/api/download/checkpoint",
        "files": [
            {
                "type": "Model",
                "primary": True,
                "sizeKB": 2048,
                "name": "Z_Image_Turbo.safetensors",
                "hashes": {"SHA256": "ABC123FF"},
            }
        ],
    }

    async def fake_metadata_provider():
        class Provider:
            async def get_model_version_info(self, version_id):
                assert version_id == "2442439"
                return checkpoint_info, None

        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.automatic.get_default_metadata_provider",
        fake_metadata_provider,
    )

    parser = AutomaticMetadataParser()

    metadata_text = (
        "Negative space, fog, BLACK blue color GRADIENT BACKGROUND, a vintage car in the middle, "
        "FOG, and a silhouetted figure near the car, in the style of the Blade Runner movie "
        "Negative prompt: Steps: 23, Sampler: Undefined, CFG scale: 3.5, Seed: 1760020955, "
        "Size: 832x1216, Clip skip: 2, Created Date: 2025-11-28T09:18:43.5269343Z, "
        'Civitai resources: [{"type":"checkpoint","modelVersionId":2442439,"modelName":"Z Image","modelVersionName":"Turbo"}], '
        "Civitai metadata: {}"
    )

    result = await parser.parse_metadata(metadata_text)

    checkpoint = result.get("checkpoint")
    assert checkpoint is not None
    assert checkpoint["name"] == "Z Image"
    assert checkpoint["version"] == "Turbo"
    assert checkpoint["type"] == "checkpoint"
    assert checkpoint["modelId"] == 123456
    assert checkpoint["hash"] == "abc123ff"
    assert checkpoint["file_name"] == "Z_Image_Turbo"
    assert checkpoint["thumbnailUrl"].endswith("width=450,optimized=true")
    assert result["model"] == checkpoint
    assert result["base_model"] == "sdxl"
    assert result["loras"] == []


@pytest.mark.asyncio
async def test_parse_metadata_merges_lora_hashes_over_empty_hashes_json(monkeypatch):
    """When Hashes JSON has empty lora hashes but Lora hashes text field has
    real ones, the real hashes should be used and those LoRAs resolved
    correctly; entries with empty hashes in both sources should be skipped."""
    lora_version_info = {
        "id": 947620,
        "modelId": 98765,
        "model": {"name": "cfg_scale_boost", "type": "LORA"},
        "name": "v1",
        "images": [{"url": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/original=true"}],
        "baseModel": "illustrious",
        "downloadUrl": "https://civitai.com/api/download/models/947620",
        "files": [
            {
                "type": "Model",
                "primary": True,
                "sizeKB": 1024,
                "name": "cfg_scale_boost.safetensors",
                "hashes": {"SHA256": "4605b2de07"},
            }
        ],
    }

    async def fake_metadata_provider():
        class Provider:
            async def get_model_by_hash(self, model_hash):
                assert model_hash == "4605b2de07"
                return lora_version_info, None

            async def get_model_version_info(self, version_id):
                raise AssertionError("get_model_version_info should not be called")

        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.automatic.get_default_metadata_provider",
        fake_metadata_provider,
    )

    parser = AutomaticMetadataParser()

    metadata_text = (
        "a cyberpunk portrait <lora:cfg_scale_boost:0.6>\n"
        "Negative prompt: low quality\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123456, Size: 512x768, "
        "Model hash: abc123, Model: test.safetensors, "
        'Lora hashes: "cfg_scale_boost: 4605b2de07, EmptyLora: ", '
        'Hashes: {"model": "abc123", "lora:cfg_scale_boost": "", "lora:EmptyLora": "", "lora:UnusedLora": ""}'
    )

    result = await parser.parse_metadata(metadata_text)

    # cfg_scale_boost should be resolved (hash from Lora hashes overrode empty Hashes JSON)
    loras = result.get("loras", [])
    assert len(loras) == 1, f"Expected 1 LoRA, got {len(loras)}"
    lora = loras[0]
    assert lora["name"] == "cfg_scale_boost", f"Expected cfg_scale_boost, got {lora['name']}"
    assert lora["hash"] == "4605b2de07", f"Expected hash 4605b2de07, got {lora['hash']}"
    assert lora.get("isDeleted") in (None, False), f"LoRA should not be deleted"
    assert lora["weight"] == 0.6, f"Expected weight 0.6, got {lora['weight']}"

    # EmptyLora and UnusedLora should be skipped (no hash in either source)
    lora_names = [l["name"] for l in loras]
    assert "EmptyLora" not in lora_names, "EmptyLora should have been skipped"
    assert "UnusedLora" not in lora_names, "UnusedLora should have been skipped"


@pytest.mark.asyncio
async def test_parse_metadata_extracts_checkpoint_from_model_hash(monkeypatch):
    checkpoint_info = {
        "id": 98765,
        "modelId": 654321,
        "model": {"name": "Flux Illustrious", "type": "checkpoint"},
        "name": "v1",
        "images": [{"url": "https://image.civitai.com/checkpoints/original=true"}],
        "baseModel": "flux",
        "downloadUrl": "https://civitai.com/api/download/checkpoint",
        "files": [
            {
                "type": "Model",
                "primary": True,
                "sizeKB": 1024,
                "name": "FluxIllustrious_v1.safetensors",
                "hashes": {"SHA256": "C3688EE04C"},
            }
        ],
    }

    async def fake_metadata_provider():
        class Provider:
            async def get_model_by_hash(self, model_hash):
                assert model_hash == "c3688ee04c"
                return checkpoint_info, None

        return Provider()

    monkeypatch.setattr(
        "py.recipes.parsers.automatic.get_default_metadata_provider",
        fake_metadata_provider,
    )

    parser = AutomaticMetadataParser()

    metadata_text = (
        "A cyberpunk portrait with neon highlights.\n"
        "Negative prompt: low quality\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123456, Size: 832x1216, "
        "Model hash: c3688ee04c, Model: models/waiNSFWIllustrious_v110.safetensors"
    )

    result = await parser.parse_metadata(metadata_text)

    checkpoint = result.get("checkpoint")
    assert checkpoint is not None
    assert checkpoint["hash"] == "c3688ee04c"
    assert checkpoint["name"] == "Flux Illustrious"
    assert checkpoint["version"] == "v1"
    assert checkpoint["file_name"] == "FluxIllustrious_v1"
    assert result["model"] == checkpoint
    assert result["base_model"] == "flux"
    assert result["loras"] == []
