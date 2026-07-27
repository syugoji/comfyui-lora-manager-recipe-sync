import json
from pathlib import Path

import pytest
from PIL import Image

from py.services.recipes.revision_service import (
    RecipeRevisionError,
    RecipeRevisionService,
)


PROMPT_ID = "00000000-0000-4000-8000-000000000001"
DRAFT_HASH = "d" * 64
MANIFEST_HASH = "a" * 64


def make_recipe(tmp_path: Path):
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    preview = recipes_dir / "recipe.webp"
    Image.new("RGB", (128, 192), "purple").save(preview, "WEBP")
    recipe = {
        "id": "recipe-one",
        "title": "Original recipe",
        "file_path": str(preview),
        "base_model": "Illustrious",
        "gen_params": {
            "prompt": "source prompt",
            "negative_prompt": "bad hands",
            "seed": 10,
            "steps": 28,
            "sampler": "euler_ancestral",
        },
        "checkpoint": {"file_name": "model.safetensors", "hash": "c" * 64},
        "loras": [{"file_name": "style.safetensors", "strength": 0.9}],
        "comfy_prompt": {"1": {"class_type": "SaveImage", "inputs": {}}},
    }
    recipe_json = recipes_dir / "recipe-one.recipe.json"
    recipe_json.write_text(json.dumps(recipe), encoding="utf-8")
    return recipe, recipe_json, preview


def make_draft():
    return {
        "schema": "lora-manager.prompt-draft",
        "version": 1,
        "draft_hash": DRAFT_HASH,
        "manifest_hash": MANIFEST_HASH,
        "prompt_source": "lm_studio",
        "proposed_prompt": "<lora:style:0.9>, generated scene",
        "negative_prompt": "bad hands",
        "image": {"sha256": "e" * 64},
        "lm_studio": {
            "model": "qwythos-9b-claude-mythos-5-1m@q5_k_m",
            "loaded_identifier": "lora-manager-qwythos",
            "gpu": "off",
        },
    }


def make_candidate():
    return {
        "candidate_id": "recipe-one:1000:0",
        "prompt_id": PROMPT_ID,
        "output_node_id": "9",
        "image_index": 0,
        "seed": 42,
    }


def make_history(source_etag: str, *, filename="candidate.png", candidate_id="recipe-one:1000:0"):
    trial = {
        "schema": "lora-manager.recipe-trial",
        "version": 1,
        "recipe_id": "recipe-one",
        "source_etag": source_etag,
        "manifest_hash": MANIFEST_HASH,
        "draft_hash": DRAFT_HASH,
        "candidate_id": candidate_id,
        "seed": 42,
    }
    return {
        PROMPT_ID: {
            "prompt": [0, PROMPT_ID, {}, {"lora_manager_recipe_trial": trial}, []],
            "status": {"completed": True, "status_str": "success", "messages": []},
            "outputs": {
                "9": {
                    "images": [
                        {"filename": filename, "subfolder": "", "type": "output"}
                    ]
                }
            },
        }
    }


@pytest.mark.asyncio
async def test_adopt_revision_preserves_source_and_mechanical_recipe_data(tmp_path: Path):
    recipe, recipe_json, preview = make_recipe(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    Image.new("RGB", (256, 384), "pink").save(output_dir / "candidate.png")
    history = {}
    service = RecipeRevisionService(
        history_getter=lambda prompt_id: history,
        output_dir_getter=lambda: str(output_dir),
    )
    source_state = await service.get_source_state(recipe)
    history.update(make_history(source_state["etag"]))
    source_json_bytes = recipe_json.read_bytes()
    source_image_bytes = preview.read_bytes()

    result = await service.adopt_revision(
        recipe=recipe,
        recipe_json_path=recipe_json,
        replay_manifest={"manifest_hash": MANIFEST_HASH},
        if_match=f'"{source_state["etag"]}"',
        draft=make_draft(),
        candidate=make_candidate(),
    )

    assert result["created"] is True
    assert result["summary"]["active"] is True
    assert result["summary"]["prompt_source"] == "lm_studio"
    assert recipe_json.read_bytes() == source_json_bytes
    assert preview.read_bytes() == source_image_bytes
    active = await service.resolve_active_recipe(
        recipe, recipe_json, current_etag=source_state["etag"]
    )
    assert active["gen_params"]["prompt"] == "<lora:style:0.9>, generated scene"
    assert active["gen_params"]["negative_prompt"] == "bad hands"
    assert active["gen_params"]["seed"] == 42
    assert active["gen_params"]["prompt_source"] == "lm_studio"
    assert active["loras"] == recipe["loras"]
    assert active["checkpoint"] == recipe["checkpoint"]
    assert active["comfy_prompt"] == recipe["comfy_prompt"]
    assert Path(active["file_path"]).is_file()
    revision = result["revision"]
    assert revision["provenance"]["model"].startswith("qwythos-9b")
    assert revision["provenance"]["gpu"] == "off"


@pytest.mark.asyncio
async def test_adopt_revision_is_idempotent_for_same_verified_candidate(tmp_path: Path):
    recipe, recipe_json, _ = make_recipe(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    Image.new("RGB", (64, 64), "pink").save(output_dir / "candidate.png")
    history = {}
    service = RecipeRevisionService(
        history_getter=lambda prompt_id: history,
        output_dir_getter=lambda: str(output_dir),
    )
    source_state = await service.get_source_state(recipe)
    history.update(make_history(source_state["etag"]))
    kwargs = {
        "recipe": recipe,
        "recipe_json_path": recipe_json,
        "replay_manifest": {"manifest_hash": MANIFEST_HASH},
        "if_match": source_state["etag"],
        "draft": make_draft(),
        "candidate": make_candidate(),
    }

    first = await service.adopt_revision(**kwargs)
    second = await service.adopt_revision(**kwargs)

    assert first["created"] is True
    assert second["created"] is False
    revisions = await service.list_revisions(
        "recipe-one", recipe_json, current_etag=source_state["etag"]
    )
    assert len(revisions) == 1
    assert revisions[0]["active"] is True


@pytest.mark.asyncio
async def test_adopt_revision_rejects_stale_etag_without_writing(tmp_path: Path):
    recipe, recipe_json, _ = make_recipe(tmp_path)
    service = RecipeRevisionService(
        history_getter=lambda prompt_id: {},
        output_dir_getter=lambda: str(tmp_path / "output"),
    )

    with pytest.raises(RecipeRevisionError) as exc_info:
        await service.adopt_revision(
            recipe=recipe,
            recipe_json_path=recipe_json,
            replay_manifest={"manifest_hash": MANIFEST_HASH},
            if_match="0" * 64,
            draft=make_draft(),
            candidate=make_candidate(),
        )

    assert exc_info.value.code == "RECIPE_ETAG_CHANGED"
    assert not (recipe_json.parent / ".recipe-revisions").exists()


@pytest.mark.asyncio
async def test_adopt_revision_rejects_history_provenance_and_path_traversal(tmp_path: Path):
    recipe, recipe_json, _ = make_recipe(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (64, 64), "red").save(outside)
    history = {}
    service = RecipeRevisionService(
        history_getter=lambda prompt_id: history,
        output_dir_getter=lambda: str(output_dir),
    )
    source_state = await service.get_source_state(recipe)
    history.update(make_history(source_state["etag"], candidate_id="wrong"))

    with pytest.raises(RecipeRevisionError) as mismatch:
        await service.adopt_revision(
            recipe=recipe,
            recipe_json_path=recipe_json,
            replay_manifest={"manifest_hash": MANIFEST_HASH},
            if_match=source_state["etag"],
            draft=make_draft(),
            candidate=make_candidate(),
        )
    assert mismatch.value.code == "CANDIDATE_PROVENANCE_MISMATCH"

    history.clear()
    history.update(make_history(source_state["etag"], filename="../outside.png"))
    with pytest.raises(RecipeRevisionError) as traversal:
        await service.adopt_revision(
            recipe=recipe,
            recipe_json_path=recipe_json,
            replay_manifest={"manifest_hash": MANIFEST_HASH},
            if_match=source_state["etag"],
            draft=make_draft(),
            candidate=make_candidate(),
        )
    assert traversal.value.code == "CANDIDATE_IMAGE_INVALID"


@pytest.mark.asyncio
async def test_activate_revision_can_restore_original_variant(tmp_path: Path):
    recipe, recipe_json, _ = make_recipe(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    Image.new("RGB", (64, 64), "pink").save(output_dir / "candidate.png")
    history = {}
    service = RecipeRevisionService(
        history_getter=lambda prompt_id: history,
        output_dir_getter=lambda: str(output_dir),
    )
    source_state = await service.get_source_state(recipe)
    history.update(make_history(source_state["etag"]))
    adopted = await service.adopt_revision(
        recipe=recipe,
        recipe_json_path=recipe_json,
        replay_manifest={"manifest_hash": MANIFEST_HASH},
        if_match=source_state["etag"],
        draft=make_draft(),
        candidate=make_candidate(),
    )

    inactive = await service.activate_revision(
        "recipe-one", recipe_json, current_etag=source_state["etag"], revision_id=None
    )
    assert inactive["active"] is False
    original = await service.resolve_active_recipe(
        recipe, recipe_json, current_etag=source_state["etag"]
    )
    assert original["gen_params"]["prompt"] == "source prompt"
    active = await service.activate_revision(
        "recipe-one",
        recipe_json,
        current_etag=source_state["etag"],
        revision_id=adopted["revision"]["revision_id"],
    )
    assert active["active"] is True


@pytest.mark.asyncio
async def test_active_prompt_ids_include_nested_recipe_stores(tmp_path: Path):
    recipes_dir = tmp_path / "recipes"
    marker_dir = recipes_dir / "nested" / ".recipe-revisions" / "active"
    marker_dir.mkdir(parents=True)
    marker_dir.joinpath("valid.json").write_text(
        json.dumps(
            {
                "schema": "lora-manager.recipe-revision-active",
                "version": 1,
                "recipe_id": "nested-recipe",
                "revision_id": "revision-one",
            }
        ),
        encoding="utf-8",
    )
    marker_dir.joinpath("invalid.json").write_text(
        json.dumps({"schema": "other", "recipe_id": "ignored"}),
        encoding="utf-8",
    )

    service = RecipeRevisionService()

    assert await service.get_active_prompt_recipe_ids(recipes_dir) == {"nested-recipe"}


@pytest.mark.asyncio
async def test_activate_revision_rejects_non_revision_path(tmp_path: Path):
    recipe, recipe_json, _ = make_recipe(tmp_path)
    service = RecipeRevisionService()
    source_state = await service.get_source_state(recipe)

    with pytest.raises(RecipeRevisionError) as exc_info:
        await service.activate_revision(
            "recipe-one",
            recipe_json,
            current_etag=source_state["etag"],
            revision_id="../active/marker",
        )

    assert exc_info.value.code == "REVISION_NOT_FOUND"
