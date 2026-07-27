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

## 8. 【この配布版で除外】同梱しないもの

配布物には作者のローカル環境由来のデータを入れていない。初回起動時に再生成される。

- `cache/` — モデル・レシピの SQLite キャッシュ（利用者のライブラリ内容そのもの）
- `backups/` — 作業中の zip バックアップ
- `stats/` — 利用統計
- `.tracking` — 作業用のファイル一覧
- `node_modules/`, `__pycache__/`, `.pytest_cache/`

---

## テストの状況

`pytest tests/`（`syrupy` / `hypothesis` を要する2モジュールを除く）:

- **1,444 passed / 27 failed**
- この 27件は**改造前のフォークでも同じ27件が落ちる**（1,442 passed / 27 failed）。
  内訳は i18n のロケールキー欠落、`lora_cycler`、`download_manager` の zip 展開、`checkpoint_scanner`。
  上記7の変更による回帰ではない（同一条件の対照実行で確認済み）。
- フロントエンドの `vitest` は `node_modules/` を同梱していないため未実行。
  実行する場合は `npm install` の後に `npx vitest run`。
