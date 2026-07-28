# このフォークが upstream から変えた点

このディレクトリは **[willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) v1.1.6 のフォーク**です。
upstream と同じ **GPL-3.0** で配布します。GPL-3.0 第5条が「変更したことを明示せよ」と求めているため、
何を変えたかをこのファイルに列挙します。

- 基点: upstream タグ **v1.1.6**
- 差分規模: **411ファイル**（既存ファイルの改変 36 ／ 新規追加 68 ／ 残りはテスト・アセット）
- upstream の **1.1.9 以降には追従していません**。upstream 側の新機能・修正は入っていません。

> 差分を自分で取り直す場合:
> ```
> git clone --depth 1 --branch v1.1.6 https://github.com/willmiao/ComfyUI-Lora-Manager.git upstream-1.1.6
> git diff --no-index --ignore-cr-at-eol --stat upstream-1.1.6 comfyui-lora-manager
> ```
> `--ignore-cr-at-eol` を付けないと改行コード差で全ファイルが変更扱いになります。

---

## 1. レシピの生成元ポリシーを「元画像優先」に変えた

upstream は画像ページ側の情報を優先していたが、このフォークは
**画像ファイルに埋め込まれた生成情報を先に使い、足りない分だけ外部から補う**方針に変えた。

- `py/services/recipes/analysis_service.py` — 判定を `generation_source` として記録する
  （`embedded_a1111` / `embedded_comfy` / `civitai_generation_data` / `reconstructed`）
- `py/utils/exif_utils.py` — JPEG / TIFF / WEBP の **EXIF UserComment** から A1111 パラメータを読む
  （upstream は PNG テキストチャンク中心）
- `py/utils/generation_metadata.py`（新規）— 生成情報の正規化
- この方針の識別子は `recipe_generation_source_policy = "embedded-first-v1"` として
  `/api/lm/health-check` が返す。付属の同期スクリプトはこの値で改造版かどうかを判定する。

## 2. ワークフロー復元・再現の追加

- `static/js/utils/recipeWorkflowBuilder.js`（新規・1,325行）— レシピから ComfyUI ワークフローを組み立てる
- `static/js/utils/recipeReplayCapability.js`（新規）— 再現可能かの判定
- `static/js/managers/RecipeWorkflowReplayManager.js`（新規）
- `py/services/recipes/replay_manifest_service.py`（新規）
- `web/comfyui/a1111_generation_patch.js`, `web/comfyui/a1111_lora_merge.js`（新規）—
  A1111 由来の生成情報を ComfyUI ノードへ流し込む
- 空 latent から始まる txt2img の再構築では denoise を 1.0 に保つ

## 3. レシピのパーサ強化

- `py/recipes/parsers/comfy.py` — ローカルパス形式（`Illustrious\anime\xxx.safetensors`）の
  LoRA / checkpoint を解決する。upstream は civitai URN 形式（`civitai:ID@VERSION`）しか見ず、
  ローカル名を黙って無視していた。`LoraLoaderModelOnly` と、本体独自の LoRA 運搬ノード5種
  （`Lora Loader` / `LoRA Text Loader` / `Lora Stacker` / `WanVideo Lora Select` ×2）に対応
- `py/recipes/parsers/automatic.py` — A1111 メタデータの解釈を拡張
- `py/recipes/parsers/civitai_image.py` — civitai 画像由来のメタデータ解釈を拡張
- `py/recipes/base.py`（新規）— パーサ共通処理

## 4. レシピ一覧・詳細の UI 追加

- **checkpoint 別の並び替え**（`sort_by=checkpoint:asc/desc`）— `py/services/recipe_scanner.py`,
  `templates/components/controls.html`
- **参考情報をコピー** — `static/js/utils/recipeReferenceInfo.js`（新規）。
  再現できないレシピからも、ハッシュ12桁・`※未所持` / `※配布終了` 付きで構成情報を回収する
- **実行リスト（レシピの連結実行）** — `static/js/utils/recipePlaylistStore.js`,
  `static/js/managers/RecipePlaylistManager.js`（新規）
- **不足リソースの一括ダウンロード** — `static/js/managers/BulkMissingLoraDownloadManager.js`
- 狭幅時のツールバー折返し修正 — `static/css/layout.css`

## 5. レシピの改訂履歴・下書き

- `py/services/recipes/revision_service.py`（新規・709行）
- `py/services/recipes/prompt_draft_service.py`（新規・1,021行）
- `static/js/managers/RecipeTrialManager.js`（新規）

## 6. ダウンロードの厳密化

- `py/services/download_manager.py` — CivitAI の**ファイルID と SHA-256 で実ファイルを特定**して落とす
  （同名別バージョンの取り違え防止）
- `py/services/model_hash_index.py` — ハッシュ索引の解決を強化

## 7. 【この配布版で追加】内部API（api/trpc）の撤去

**upstream v1.1.6 にも、これまでのフォークにも入っていた `api/trpc/image.getGenerationData` への
アクセスを撤去した。** Civitai の利用規約が自動アクセスを「明示的に提供するインターフェース
（公開API等）」に限っているため、配布物には残さない方針にした。

- `py/services/civitai_client.py` の `get_image_generation_data()` を、
  公開API **`GET /api/v1/images?imageId=..&withMeta=true`** ベースの実装に置き換えた。
  呼び出し側（`analysis_service.py`）から見た戻り値の形（`{"meta": .., "resources": [..]}`）は変えていない。
- `/api/v1/model-versions`, `/api/v1/model-versions/by-hash`, `/api/v1/models` は
  公開APIなのでそのまま使っている。

**回収率は落ちていない**（2026-07-26 実測・69枚のペア比較）:

| 母集団 | n | | プロンプト | 生成条件 | リソース有 |
|---|---|---|---|---|---|
| 既存レシピで `reconstructed` 判定だったもの | 19 | 公開API | 1 | 5 | 17 |
| | | 内部API | 1 | 5 | 17 |
| civitai.com Most Reactions / Day | 50 | 公開API | 32 | 32 | 35 |
| | | 内部API | 32 | 32 | 35 |

さらに 15枚で項目単位の照合を行い、**内部APIだけが持つキーは0件**、リソースID集合は 15/15 で一致した。
2枚では公開APIのほうが `resources` キーを余分に持っていた。

実測で分かった注意点:

- `imageId` フィルタは公式ドキュメントに記載が無いが**実際には機能する**
  （存在しないIDでは `items` が空になる＝無視されているのではない）
- **`withMeta=true` が無いと `meta` は空**で返る
- 成人向けを含める場合は `nsfw=X&browsingLevel=255` が要る
- `meta` は `{"id": .., "meta": {..}}` と**1段包まれて返ることがある**

## 8. 【この配布版で追加】Raindrop 同期をUIから起動できるようにした

レシピ画面のツールバーに「Raindrop 同期」ボタンを足し、
同梱スクリプト `civitai-recipe-sync/civitai_image_download.py` を**子プロセスとして起動**できるようにした。
コマンドラインを開かずに同期できる。

- `py/services/raindrop_sync_service.py`（新規）— スクリプトの起動・進捗集約・中断。
  **スクリプトを import しない**（ライセンス境界。ファイル冒頭に理由を書いてある）
- `py/routes/recipe_route_registrar.py` — `/api/lm/recipes/raindrop-sync/{start,progress,cancel}` を追加
- `py/routes/handlers/recipe_handlers.py` — `RaindropSyncHandler`（起動・進捗・中断の3本）
- `py/services/settings_manager.py` — `raindrop_token` / `raindrop_collection_id` /
  `raindrop_sync_script_path` / `raindrop_sync_comfy_base_url` を追加
- `py/routes/handlers/misc_handlers.py` — **`raindrop_token` はフロントへ返さない**。
  既存の `civitai_api_key` と同じ扱いで、設定済みかどうかの真偽値 `raindrop_token_set` だけを返す
- `templates/components/raindrop_sync_modal.html`（新規）,
  `static/js/managers/RaindropSyncManager.js`（新規）— 進捗・成功/失敗件数・失敗した画像IDを表示
- `templates/components/modals/settings_modal.html` — 設定画面に Raindrop 節を追加（トークンはマスク表示）
- `locales/en.json`, `locales/ja.json` — 上記の文言

子プロセスへは環境変数で設定を渡す（`RAINDROP_TOKEN` / `RAINDROP_COLLECTION_ID` /
`LORA_RECIPE_DIR` / `COMFY_BASE_URL` ほか）。スクリプトは
`CIVITAI_SYNC_EVENT_STREAM=1` のとき `@@RDSYNC@@ {...}` の1行1件で進捗を出し、
`CIVITAI_SYNC_NON_INTERACTIVE=1` のとき終了時の `input()` 待ちを飛ばす。
**引数なしで直接叩いたときの挙動は変えていない**（従来のダブルクリック起動がそのまま通る）。

`COMFY_BASE_URL` は「設定 → リクエストの出所 → `http://127.0.0.1:8188`」の順で決まるので、
ComfyUI を既定以外のポートで動かしていても当たる。

トークンは起動前に **ASCII のみ・空白なし** を検査する。非ASCIIのまま走らせると
`requests` が Authorization ヘッダを latin-1 でエンコードできず
「`'latin-1' codec can't encode characters`」しか出ず原因が読めないため
（実測で踏んだ。貼り付け事故だった）。エラー文には最初の非ASCII位置と全文字数だけを載せ、
**値そのものは画面にもログにも出さない**。

**実測（2026-07-28・330件のブックマークを持つ実コレクション）**:
レシピ0件の状態からボタンで起動し、**15件が保存されるところまで確認**（成功15/失敗0）。
中断ボタンで子プロセスが止まり、保存済みの15件はそのまま残る。

**ブックマーク数と対象数がずれる理由は画面に出す。** 上の実測では 330件中 325件が対象で、
残り5件の理由が分からなかった。対象外を
「Civitai画像URLでない／同じ画像IDが重複／同期済み／リンクが空」へ分類して数と実URLを出す
（`planned` イベントと進捗画面）。**`civitai.com` と `civitai.red` は同じ画像IDのミラー**なので、
両方ブックマークされていると重複として1件に畳まれる。
`civitai.com/posts/…`・`civitai.com/models/…`・`image.civitai.com/…`（CDN直リンク）は
画像ページURLではないので対象外になる。

**対象外5件の実内訳（2026-07-28 実測・レシピ0件と同条件）**:

| 分類 | 件数 |
|---|---|
| 同じ画像IDが重複 | **5** |
| Civitai画像URLでない | 0 |
| 同期済み | 0 |
| リンクが空 | 0 |

`330 = 325 + 5` が分類だけで閉じる（分類軸の追加は不要だった）。
**5件はすべて「まったく同じURLが2回ブックマークされていた」もの**で、
`.com` / `.red` のミラー対ではなかった。上の「両方ブックマークされていると畳まれる」は
コード上そうなるという説明であって、この実データの原因ではない。
根拠 — 重複した5つの画像IDについて Raindrop の `_id` は全10件が別物で、
各ペアの `created` は 0.5〜7秒差（例: `133999893` が `18:43:39.142Z` と `18:43:39.660Z`）。
ページングで同じ項目を2回拾った取得側の事故ではないことは、
330件の `_id` が全件一意であることで確認した。

## 9. 【この配布版で除外】同梱しないもの

配布物には作者のローカル環境由来のデータを入れていない。初回起動時に再生成される。

- `cache/` — モデル・レシピの SQLite キャッシュ（利用者のライブラリ内容そのもの）
- `backups/` — 作業中の zip バックアップ
- `stats/` — 利用統計
- `.tracking` — 作業用のファイル一覧
- `node_modules/`, `__pycache__/`, `.pytest_cache/`

---

## テストの状況

`pytest tests/`（`syrupy` / `hypothesis` を要する2モジュールを除く）:

- **1,478 passed / 16 failed**（2026-07-28 実測）。
- 残る16件は **上流 v1.1.6 でも同じ16件が落ちる**。
  - 内訳: `tests/config/test_config_save_paths.py` 10件 /
    `tests/nodes/test_lora_cycler.py` 3件 /
    `tests/services/test_download_manager_error.py` 3件（zip 展開時のパス区切り）。
  - 根拠は**件数の一致ではなくテスト名の集合比較**。素の v1.1.6 を同じ条件で走らせると
    19 failed / 1,357 passed で、その失敗集合はこのフォークの16件を**完全に含む**。
    「フォークだけで落ちるテスト」は**0件**。
  - 逆に上流だけで落ちる3件（`test_misc_routes` 2件・`test_persistent_recipe_cache` 1件）は
    このフォークでは通る。上流にはさらに Windows 固有の teardown エラー21件
    （`PermissionError: WinError 32`・一時sqliteの掴みっぱなし）があるが、こちらも出ない。

> **以前ここに書いていた「27 failed。改造前のフォークでも同じ27件」は誤りだった。**
> 件数だけを比べていたため、**i18n の9件がフォーク由来だったこと**を見落としていた。
> 上流は en 1,589キーに対し他ロケールの欠落が0で、i18n テストは通る。
> フォークが en へ59キー（Raindrop 同期 41 ＋ 実行リスト・一括DL等 18）を追加しながら
> **en と ja にしか入れていなかった**ため、残り8ロケールで各59キーが欠けていた。
> さらに static のコードが en に存在しない8キーを参照していた。2026-07-28 に全部埋めて解消。
> 併せて、フォークが変えた仕様に追随できていなかった2件も直した
> （`download_manager` のハッシュ照合が実バイト検証になった件、
> 永続キャッシュが壊れた safetensors を捨てるようになった件）。

**フロントエンド（vitest）**:

- `tests/frontend/**` — **54 files / 436 tests すべて通過**。
- `vue-widgets/tests/**` — **4 files / 68 tests すべて通過**。
- 実行手順: `npm ci` の後 `npx vitest run`、`cd vue-widgets && npm ci && npx vitest run`。
- 初回実行で1ファイルが収集できずに落ちた。`web/comfyui/a1111_generation_patch.js` が
  ブラウザへ配信される絶対URL `/loras_static/js/utils/genParamsMapper.js` を import しており、
  vitest はこのルートを知らないため。サーバ側の `add_static("/loras_static", static/)` と
  同じ対応付けを `vitest.config.js` の `resolve.alias` に足して解消した（実行時の欠陥ではない）。
