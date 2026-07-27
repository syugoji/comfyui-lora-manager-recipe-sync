from copy import deepcopy

from py.services.recipes.replay_manifest_service import ReplayManifestService


def _recipe_lora(name: str, version_id: int, strength: float = 1.0):
    return {
        "file_name": name,
        "modelVersionId": version_id,
        "strength": strength,
        "hash": f"{version_id:064x}",
        "inLibrary": True,
        "localPath": f"D:/models/loras/{name}.safetensors",
    }


def test_manifest_uses_all_explicit_a1111_loras_not_only_inline_tag():
    loras = [
        _recipe_lora("748cmSDXL", 1056404, 0.45),
        _recipe_lora("NV_KawaiiTech_WM_IL_SH", 1135769, 0.9),
        _recipe_lora("ILLMythP0rtr4itStyle", 1373674, 0.7),
        _recipe_lora("tove-nikke-richy-v1_ixl", 1809862, 1.0),
    ]
    resources = (
        '[{"type":"lora","weight":0.45,"modelVersionId":1056404},'
        '{"type":"lora","weight":0.9,"modelVersionId":1135769},'
        '{"type":"lora","weight":0.7,"modelVersionId":1373674},'
        '{"type":"lora","weight":1,"modelVersionId":1809862}]'
    )
    recipe = {
        "loras": loras,
        "gen_params": {
            "prompt": "<lora:tove-nikke-richy-v1_ixl:1> portrait",
        },
        "a1111_parameters": (
            "<lora:tove-nikke-richy-v1_ixl:1> portrait\n"
            f"Steps: 30, Civitai resources: {resources}, Civitai metadata: {{}}"
        ),
    }
    original = deepcopy(recipe)

    manifest = ReplayManifestService().build(recipe)

    assert recipe == original
    assert manifest["schema"] == "lora-manager.replay-manifest"
    assert manifest["version"] == 1
    assert manifest["errors"] == []
    assert [item["resource"]["file_name"] for item in manifest["required_resources"]] == [
        "748cmSDXL",
        "NV_KawaiiTech_WM_IL_SH",
        "ILLMythP0rtr4itStyle",
        "tove-nikke-richy-v1_ixl",
    ]
    assert [item["expected"]["strength_model"] for item in manifest["required_resources"]] == [
        0.45,
        0.9,
        0.7,
        1.0,
    ]
    assert [len(item["evidence"]) for item in manifest["required_resources"]] == [1, 1, 1, 2]


def test_manifest_keeps_catalog_only_lora_advisory_without_explicit_resources():
    recipe = {
        "loras": [
            _recipe_lora("actual_style", 1),
            _recipe_lora("catalog_only", 2),
        ],
        "gen_params": {"prompt": "<lora:actual_style:0.4> portrait"},
        "a1111_parameters": "<lora:actual_style:0.4> portrait\nSteps: 20",
    }

    manifest = ReplayManifestService().build(recipe)

    assert manifest["errors"] == []
    assert [item["resource"]["file_name"] for item in manifest["required_resources"]] == [
        "actual_style"
    ]
    assert manifest["required_resources"][0]["expected"]["strength_model"] == 0.4
    assert [item["resource"]["file_name"] for item in manifest["advisory_resources"]] == [
        "catalog_only"
    ]


def test_manifest_accepts_explicit_air_lora_identity_and_nested_json():
    recipe = {
        "loras": [_recipe_lora("air_style", 1375651, 0.6)],
        "gen_params": {"prompt": "portrait"},
        "a1111_parameters": (
            "portrait\nSteps: 20, Civitai resources: "
            '[{"air":"urn:air:sdxl:lora:civitai:1221007@1375651",'
            '"weight":0.6,"modelName":"Name, with comma",'
            '"meta":{"nested":{"ok":true}}}], Civitai metadata: {}'
        ),
    }

    manifest = ReplayManifestService().build(recipe)

    assert manifest["errors"] == []
    assert len(manifest["required_resources"]) == 1
    evidence = manifest["required_resources"][0]["evidence"][0]
    assert str(evidence["model_id"]) == "1221007"
    assert str(evidence["model_version_id"]) == "1375651"


def test_manifest_rejects_ambiguous_recipe_identity():
    recipe = {
        "loras": [
            {"file_name": "folder-a/style.safetensors", "modelName": "Same Style"},
            {"file_name": "folder-b/style.safetensors", "modelName": "Same Style"},
        ],
        "gen_params": {"prompt": "portrait"},
        "a1111_parameters": (
            "portrait\nSteps: 20, Civitai resources: "
            '[{"type":"lora","weight":0.7,"modelName":"Same Style"}]'
        ),
    }

    manifest = ReplayManifestService().build(recipe)

    assert any(error["code"] == "LORA_IDENTITY_AMBIGUOUS" for error in manifest["errors"])
    assert manifest["required_resources"] == []


def test_manifest_uses_only_reachable_enabled_embedded_lora_nodes():
    recipe = {
        "loras": [
            _recipe_lora("active_style", 1),
            _recipe_lora("disconnected_style", 2),
            _recipe_lora("bypassed_style", 3),
        ],
        "comfy": {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
            "2": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0], "clip": ["1", 1],
                    "lora_name": "active_style.safetensors",
                    "strength_model": 0.8, "strength_clip": 0.8,
                },
            },
            "3": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0], "clip": ["1", 1],
                    "lora_name": "disconnected_style.safetensors",
                    "strength_model": 0.7, "strength_clip": 0.7,
                },
            },
            "4": {
                "class_type": "LoraLoader", "mode": 4,
                "inputs": {
                    "model": ["1", 0], "clip": ["1", 1],
                    "lora_name": "bypassed_style.safetensors",
                    "strength_model": 0.6, "strength_clip": 0.6,
                },
            },
            "5": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        },
        "gen_params": {"prompt": "portrait"},
    }

    manifest = ReplayManifestService().build(recipe)

    assert manifest["errors"] == []
    assert [item["resource"]["file_name"] for item in manifest["required_resources"]] == [
        "active_style"
    ]
    assert {item["resource"]["file_name"] for item in manifest["advisory_resources"]} == {
        "disconnected_style",
        "bypassed_style",
    }


def test_manifest_recovers_chained_sd_lora_loaders_reaching_sd_prompt_saver():
    recipe = {
        "loras": [
            _recipe_lora(r"Illustrious\Mei_Misaki_ILXL", 10),
            _recipe_lora(r"Illustrious\HighlegArmor-Persona3-Illustrious-V1", 20),
            _recipe_lora("disconnected_style", 30),
            _recipe_lora("bypassed_style", 40),
        ],
        "comfy_prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "Hoseki_LustrousMix.safetensors"},
            },
            "30": {
                "class_type": "SDLoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": r"Illustrious\Mei_Misaki_ILXL.safetensors",
                    "strength_model": 0.3,
                    "strength_clip": 0.3,
                },
            },
            "40": {
                "class_type": "SDLoraLoader",
                "inputs": {
                    "model": ["30", 0],
                    "clip": ["30", 1],
                    "lora_name": r"Illustrious\HighlegArmor-Persona3-Illustrious-V1.safetensors",
                    "strength_model": 0.9,
                    "strength_clip": 0.9,
                },
            },
            "41": {
                "class_type": "SDLoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": "disconnected_style.safetensors",
                    "strength_model": 0.7,
                    "strength_clip": 0.7,
                },
            },
            "42": {
                "class_type": "SDLoraLoader",
                "mode": 4,
                "inputs": {
                    "model": ["40", 0],
                    "clip": ["40", 1],
                    "lora_name": "bypassed_style.safetensors",
                    "strength_model": 0.6,
                    "strength_clip": 0.6,
                },
            },
            "50": {
                "class_type": "KSampler",
                "inputs": {"model": ["40", 0], "latent_image": ["51", 0]},
            },
            "60": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["50", 0], "vae": ["1", 2]},
            },
            "31": {
                "class_type": "SDPromptSaver",
                "inputs": {"images": ["60", 0], "lora_name": ["40", 0]},
            },
        },
        "gen_params": {"prompt": "portrait"},
    }

    manifest = ReplayManifestService().build(recipe)

    assert manifest["errors"] == []
    assert [
        item["resource"]["file_name"] for item in manifest["required_resources"]
    ] == [
        r"Illustrious\Mei_Misaki_ILXL",
        r"Illustrious\HighlegArmor-Persona3-Illustrious-V1",
    ]
    assert [
        item["expected"]["strength_model"] for item in manifest["required_resources"]
    ] == [0.3, 0.9]
    assert {item["resource"]["file_name"] for item in manifest["advisory_resources"]} == {
        "disconnected_style",
        "bypassed_style",
    }


def test_manifest_blocks_hypernet_and_conflicting_air_identity():
    recipe = {
        "loras": [],
        "gen_params": {"prompt": "portrait"},
        "a1111_parameters": (
            "portrait\nSteps: 20, Civitai resources: ["
            '{"type":"hypernet","weight":0.5,"modelVersionId":1},'
            '{"type":"lora","air":"urn:air:sdxl:lycoris:civitai:2@3",'
            '"weight":0.7,"modelVersionId":4}]'
        ),
    }

    manifest = ReplayManifestService().build(recipe)

    codes = {error["code"] for error in manifest["errors"]}
    assert "UNSUPPORTED_REQUIRED_RESOURCE_TYPE" in codes
    assert "A1111_RESOURCE_IDENTITY_CONFLICT" in codes
