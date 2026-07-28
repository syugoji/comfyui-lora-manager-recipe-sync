"""Raindrop 同期サービス（別プロセス起動）のテスト。

秘匿値の扱いを重点的に見る。トークンは子プロセスの環境変数へ渡すだけで、
進捗スナップショットやログには一切現れてはならない。
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from py.services.raindrop_sync_service import (
    RaindropSyncConfigError,
    RaindropSyncService,
)


TOKEN = "test-raindrop-token-4f7a1c9e"


class FakeSettings:
    def __init__(self, values):
        self._values = dict(values)

    def get(self, key, default=None):
        return self._values.get(key, default)


def make_settings(**overrides):
    values = {
        "raindrop_token": TOKEN,
        "raindrop_collection_id": "12345678",
        "raindrop_sync_script_path": "",
        "raindrop_sync_comfy_base_url": "",
        "civitai_api_key": "",
        "recipes_path": "",
    }
    values.update(overrides)
    return FakeSettings(values)


def make_service(tmp_path, **overrides):
    return RaindropSyncService(
        settings_manager=make_settings(**overrides),
        recipes_dir_getter=lambda: str(tmp_path),
    )


# -- 設定の解決 -------------------------------------------------------


def test_default_script_path_points_at_distribution_sibling(tmp_path):
    service = make_service(tmp_path)
    resolved = service.resolve_script_path()
    assert resolved.name == "civitai_image_download.py"
    assert resolved.parent.name == "civitai-recipe-sync"


def test_missing_script_path_setting_is_reported(tmp_path):
    service = make_service(tmp_path, raindrop_sync_script_path=str(tmp_path / "nope.py"))
    with pytest.raises(RaindropSyncConfigError):
        service.resolve_script_path()


@pytest.mark.parametrize(
    "overrides",
    [
        {"raindrop_token": ""},
        {"raindrop_collection_id": ""},
    ],
)
def test_missing_credentials_are_reported(tmp_path, overrides):
    service = make_service(tmp_path, **overrides)
    with pytest.raises(RaindropSyncConfigError):
        service._build_environment()


def test_environment_carries_credentials_to_the_child(tmp_path):
    service = make_service(tmp_path, civitai_api_key="civitai-key-123")
    env = service._build_environment()

    assert env["RAINDROP_TOKEN"] == TOKEN
    assert env["RAINDROP_COLLECTION_ID"] == "12345678"
    assert env["LORA_RECIPE_DIR"] == str(tmp_path)
    assert env["CIVITAI_SYNC_EVENT_STREAM"] == "1"
    assert env["CIVITAI_SYNC_NON_INTERACTIVE"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["CIVITAI_API_KEY"] == "civitai-key-123"


def test_non_ascii_token_is_rejected_with_a_readable_message(tmp_path):
    """貼り付け事故を早期に落とす。

    非ASCIIのまま走らせると requests が Authorization ヘッダを latin-1 で
    エンコードできず「'latin-1' codec can't encode characters」しか出ない。
    実測でこれを踏んだので、位置と文字数を添えて先に止める。
    """
    service = make_service(tmp_path, raindrop_token="abcdef0123456789012345678公開用")
    with pytest.raises(RaindropSyncConfigError) as excinfo:
        service._build_environment()

    message = str(excinfo.value)
    assert "26 文字目以降" in message
    assert "全 28 文字" in message
    # 値そのものを画面へ出さない
    assert "公開用" not in message


def test_token_with_whitespace_is_rejected(tmp_path):
    service = make_service(tmp_path, raindrop_token="abcdef 0123456789")
    with pytest.raises(RaindropSyncConfigError) as excinfo:
        service._build_environment()
    assert "空白" in str(excinfo.value)
    assert "abcdef" not in str(excinfo.value)


def test_valid_ascii_token_still_passes(tmp_path):
    service = make_service(tmp_path)
    assert service._build_environment()["RAINDROP_TOKEN"] == TOKEN


def test_comfy_base_url_prefers_setting_then_caller_then_default(tmp_path):
    default_service = make_service(tmp_path)
    assert default_service._build_environment()["COMFY_BASE_URL"] == "http://127.0.0.1:8188"
    assert (
        default_service._build_environment("http://127.0.0.1:8189/")["COMFY_BASE_URL"]
        == "http://127.0.0.1:8189"
    )

    configured = make_service(
        tmp_path, raindrop_sync_comfy_base_url="http://192.168.0.5:9000/"
    )
    assert (
        configured._build_environment("http://127.0.0.1:8189")["COMFY_BASE_URL"]
        == "http://192.168.0.5:9000"
    )


def test_environment_drops_inherited_api_key_when_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_API_KEY", "inherited-should-not-leak")
    service = make_service(tmp_path)
    env = service._build_environment()
    assert "CIVITAI_API_KEY" not in env


# -- イベントの解釈 ---------------------------------------------------


def test_event_lines_drive_progress(tmp_path):
    service = make_service(tmp_path)
    service._state["status"] = "running"

    service._consume_line("[+] 人間向けのログ行")
    service._consume_line('@@RDSYNC@@ {"event": "started"}')
    service._consume_line(
        '@@RDSYNC@@ {"event": "planned", "bookmarks": 40, "total": 4, "already_synced": 36}'
    )
    service._consume_line('@@RDSYNC@@ {"event": "item_started", "image_id": "111"}')
    service._consume_line(
        '@@RDSYNC@@ {"event": "item_finished", "image_id": "111", "index": 1,'
        ' "total": 4, "ok": true, "success": 1, "failed": 0}'
    )

    progress = service.get_progress()
    assert progress["bookmarks"] == 40
    assert progress["already_synced"] == 36
    assert progress["total"] == 4
    assert progress["processed"] == 1
    assert progress["success"] == 1
    assert progress["progress_percent"] == 25.0
    assert progress["current_image_id"] is None
    assert "[+] 人間向けのログ行" in progress["log"]


def test_planned_event_carries_the_exclusion_breakdown(tmp_path):
    """ブックマーク数と対象数の差を、画面だけで説明できるようにする。

    実運用で「330件あるのに325件しか処理されない」と聞かれ、推測でしか
    答えられなかったので内訳を持ち回るようにした。
    """
    service = make_service(tmp_path)
    service._state["status"] = "running"
    service._consume_line(
        '@@RDSYNC@@ {"event": "planned", "bookmarks": 330, "total": 325,'
        ' "already_synced": 1, "excluded": 5, "excluded_not_civitai_image": 3,'
        ' "excluded_duplicate": 1, "excluded_no_link": 0,'
        ' "excluded_not_civitai_image_samples": ["https://civitai.com/posts/1"],'
        ' "duplicate_image_ids": ["12345"]}'
    )

    excluded = service.get_progress()["excluded"]
    assert excluded["total"] == 5
    assert excluded["not_civitai_image"] == 3
    assert excluded["duplicate"] == 1
    assert excluded["already_synced"] == 1
    assert excluded["no_link"] == 0
    assert excluded["not_civitai_image_samples"] == ["https://civitai.com/posts/1"]
    assert excluded["duplicate_image_ids"] == ["12345"]
    # 内訳の合計は差分と一致する
    assert service.get_progress()["bookmarks"] - service.get_progress()["total"] == 5


def test_exclusion_breakdown_defaults_to_zero(tmp_path):
    """内訳を出さない古いスクリプトと組んでも壊れない。"""
    service = make_service(tmp_path)
    service._consume_line(
        '@@RDSYNC@@ {"event": "planned", "bookmarks": 10, "total": 10}'
    )
    excluded = service.get_progress()["excluded"]
    assert excluded["total"] == 0
    assert excluded["not_civitai_image_samples"] == []


def test_malformed_event_line_is_kept_as_log(tmp_path):
    service = make_service(tmp_path)
    service._consume_line("@@RDSYNC@@ これはJSONではない")
    assert service.get_progress()["log"] == ["@@RDSYNC@@ これはJSONではない"]


def test_log_is_bounded(tmp_path):
    service = make_service(tmp_path)
    for index in range(400):
        service._consume_line(f"line {index}")
    log = service.get_progress()["log"]
    assert len(log) == 300
    assert log[-1] == "line 399"


def test_finalize_marks_failure_when_script_reports_error(tmp_path):
    service = make_service(tmp_path)
    service._state["status"] = "running"
    service._consume_line(
        '@@RDSYNC@@ {"event": "finished", "status": "error", "stage": "config",'
        ' "message": "設定が足りません", "total": 0, "success": 0, "failed": 0,'
        ' "failed_ids": []}'
    )
    service._finalize(0, [])

    progress = service.get_progress()
    assert progress["status"] == "failed"
    assert progress["message"] == "設定が足りません"


def test_finalize_marks_cancelled_when_requested(tmp_path):
    service = make_service(tmp_path)
    service._state["status"] = "running"
    service._cancel_requested = True
    service._finalize(-1, [])
    assert service.get_progress()["status"] == "cancelled"


def test_finalize_surfaces_stderr_tail_on_failure(tmp_path):
    service = make_service(tmp_path)
    service._state["status"] = "running"
    service._finalize(3, ["Traceback (most recent call last):", "RuntimeError: boom"])

    progress = service.get_progress()
    assert progress["status"] == "failed"
    assert "RuntimeError: boom" in progress["message"]
    assert any("[stderr] RuntimeError: boom" == line for line in progress["log"])


# -- 実際に子プロセスを起動する通し ------------------------------------


def _write_fake_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_sync.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return script


FAKE_SCRIPT_BODY = """
    import os
    import sys

    # 起動条件が正しく渡っていることを、値そのものを出さずに確認する。
    token_ok = os.environ.get("RAINDROP_TOKEN") == "test-raindrop-token-4f7a1c9e"
    collection = os.environ.get("RAINDROP_COLLECTION_ID", "")
    non_interactive = os.environ.get("CIVITAI_SYNC_NON_INTERACTIVE", "")
    args_ok = "--events" in sys.argv and "--non-interactive" in sys.argv

    print("[+] 開始しました")
    sys.stdout.write(
        '@@RDSYNC@@ {"event": "started", "token_ok": %s, "collection": "%s",'
        ' "non_interactive": "%s", "args_ok": %s}\\n'
        % (str(token_ok).lower(), collection, non_interactive, str(args_ok).lower())
    )
    sys.stdout.write('@@RDSYNC@@ {"event": "planned", "bookmarks": 3, "total": 2,'
                     ' "already_synced": 1}\\n')
    sys.stdout.write('@@RDSYNC@@ {"event": "item_finished", "image_id": "111",'
                     ' "index": 1, "total": 2, "ok": true, "success": 1, "failed": 0}\\n')
    sys.stdout.write('@@RDSYNC@@ {"event": "item_finished", "image_id": "222",'
                     ' "index": 2, "total": 2, "ok": false, "success": 1, "failed": 1}\\n')
    sys.stdout.write('@@RDSYNC@@ {"event": "finished", "status": "ok", "total": 2,'
                     ' "success": 1, "failed": 1, "failed_ids": ["222"]}\\n')
"""


async def _run_to_completion(service: RaindropSyncService):
    await service.start()
    assert service._task is not None
    await service._task
    return service.get_progress()


@pytest.mark.asyncio
async def test_child_process_round_trip(tmp_path):
    script = _write_fake_script(tmp_path, FAKE_SCRIPT_BODY)
    service = RaindropSyncService(
        settings_manager=make_settings(raindrop_sync_script_path=str(script)),
        recipes_dir_getter=lambda: str(tmp_path),
        python_executable=sys.executable,
    )

    progress = await _run_to_completion(service)

    assert progress["status"] == "completed"
    assert progress["total"] == 2
    assert progress["success"] == 1
    assert progress["failed"] == 1
    assert progress["failed_ids"] == ["222"]
    assert progress["already_synced"] == 1
    assert progress["progress_percent"] == 100.0
    assert "[+] 開始しました" in progress["log"]


@pytest.mark.asyncio
async def test_child_process_receives_credentials_and_flags(tmp_path):
    script = _write_fake_script(tmp_path, FAKE_SCRIPT_BODY)
    captured = {}

    service = RaindropSyncService(
        settings_manager=make_settings(raindrop_sync_script_path=str(script)),
        recipes_dir_getter=lambda: str(tmp_path),
        python_executable=sys.executable,
    )
    original_apply = service._apply_event

    def spy(event):
        if event.get("event") == "started":
            captured.update(event)
        return original_apply(event)

    service._apply_event = spy  # type: ignore[method-assign]

    await _run_to_completion(service)

    assert captured.get("token_ok") is True
    assert captured.get("collection") == "12345678"
    assert captured.get("non_interactive") == "1"
    assert captured.get("args_ok") is True


@pytest.mark.asyncio
async def test_progress_never_contains_the_token(tmp_path):
    # トークンを標準出力へ書かない普通のスクリプトでは、進捗のどこにも値が出ない。
    script = _write_fake_script(tmp_path, FAKE_SCRIPT_BODY)
    service = RaindropSyncService(
        settings_manager=make_settings(raindrop_sync_script_path=str(script)),
        recipes_dir_getter=lambda: str(tmp_path),
        python_executable=sys.executable,
    )

    progress = await _run_to_completion(service)
    assert TOKEN not in repr(progress)


@pytest.mark.asyncio
async def test_leak_check_is_falsifiable(tmp_path):
    """検査器の反証: トークンを出力するスクリプトなら、ちゃんと検出される。

    この検査が常に True を返すだけの飾りでないことを示す。
    """
    leaky = _write_fake_script(
        tmp_path,
        """
        import os
        import sys

        print("[!] token=" + os.environ.get("RAINDROP_TOKEN", ""))
        sys.stdout.write('@@RDSYNC@@ {"event": "finished", "status": "ok", "total": 0,'
                         ' "success": 0, "failed": 0, "failed_ids": []}\\n')
        """,
    )
    service = RaindropSyncService(
        settings_manager=make_settings(raindrop_sync_script_path=str(leaky)),
        recipes_dir_getter=lambda: str(tmp_path),
        python_executable=sys.executable,
    )

    progress = await _run_to_completion(service)
    assert TOKEN in repr(progress)


@pytest.mark.asyncio
async def test_second_start_is_rejected_while_running(tmp_path):
    from py.services.raindrop_sync_service import RaindropSyncBusyError

    script = _write_fake_script(
        tmp_path,
        """
        import time
        time.sleep(3)
        """,
    )
    service = RaindropSyncService(
        settings_manager=make_settings(raindrop_sync_script_path=str(script)),
        recipes_dir_getter=lambda: str(tmp_path),
        python_executable=sys.executable,
    )

    await service.start()
    try:
        with pytest.raises(RaindropSyncBusyError):
            await service.start()
    finally:
        await service.cancel()
        if service._task is not None:
            await service._task

    assert service.get_progress()["status"] == "cancelled"
