from py.services.recipe_scanner import sort_recipe_entries


def _recipe(title, checkpoint=None, loras=0, modified=0):
    return {
        "title": title,
        "modified": modified,
        "file_path": f"/recipes/{title}.recipe.json",
        "loras": [{} for _ in range(loras)],
        "checkpoint": checkpoint,
    }


def test_sort_by_checkpoint_groups_same_model_and_puts_missing_last():
    entries = [
        _recipe("r1", checkpoint={"name": "WAI v14"}),
        _recipe("r2", checkpoint=None),
        _recipe("r3", checkpoint={"name": "Animagine XL"}),
        _recipe("r4", checkpoint={"name": "WAI v14"}),
        _recipe("r5", checkpoint={"name": "Illustrious base"}),
    ]

    result = sort_recipe_entries(entries, "checkpoint:asc")

    assert [item["title"] for item in result] == ["r3", "r5", "r1", "r4", "r2"]


def test_sort_by_checkpoint_desc_keeps_missing_last():
    entries = [
        _recipe("r1", checkpoint=None),
        _recipe("r2", checkpoint={"name": "Animagine XL"}),
        _recipe("r3", checkpoint={"name": "WAI v14"}),
    ]

    result = sort_recipe_entries(entries, "checkpoint:desc")

    assert [item["title"] for item in result] == ["r3", "r2", "r1"]


def test_sort_by_checkpoint_falls_back_to_file_name():
    entries = [
        _recipe("r1", checkpoint={"file_name": "zzz_model"}),
        _recipe("r2", checkpoint={"name": "Animagine XL"}),
    ]

    result = sort_recipe_entries(entries, "checkpoint:asc")

    assert [item["title"] for item in result] == ["r2", "r1"]


def test_plain_date_sort_passes_through_unchanged():
    entries = [
        _recipe("r1", modified=3),
        _recipe("r2", modified=1),
        _recipe("r3", modified=2),
    ]

    result = sort_recipe_entries(entries, "date")

    assert [item["title"] for item in result] == ["r1", "r2", "r3"]


def test_loras_count_sort_still_works_after_refactor():
    entries = [
        _recipe("r1", loras=1),
        _recipe("r2", loras=3),
        _recipe("r3", loras=2),
    ]

    result = sort_recipe_entries(entries, "loras_count:desc")

    assert [item["title"] for item in result] == ["r2", "r3", "r1"]
