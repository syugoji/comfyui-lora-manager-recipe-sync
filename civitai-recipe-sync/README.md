# civitai-recipe-sync

Raindrop に溜めた Civitai の画像ブックマークを、**LoRA Manager のレシピへ一括変換する**スクリプト。

ブックマークした画像1枚につき、
「元画像 → 生成情報の抽出 → 足りないモデル情報をハッシュで復元 → レシピとして保存」
までを自動で回す。

## これが解く問題

Civitai で「あとで再現したい」と思った画像をブックマークしても、
実際に再現しようとすると **モデルとLoRAを1つずつ手で探し直す**ことになる。
名前が一致しない、同名の別バージョンがある、消えている、といった理由で毎回詰まる。

このスクリプトは、**画像に埋め込まれた生成情報**と **SHA-256 ハッシュ**を索引にして、
その突き合わせを機械にやらせる。ハッシュで引くので、モデル名が変わっていても同定できる。

## 動かすのに要るもの（4つ揃わないと動かない）

| # | 必要なもの | 補足 |
|---|---|---|
| 1 | **Raindrop.io のアカウントとテストトークン** | 対象コレクションに Civitai の画像URLを入れておく |
| 2 | **本スクリプト** | このディレクトリ |
| 3 | **改造版 LoRA Manager** | 同梱の [`../comfyui-lora-manager/`](../comfyui-lora-manager/)。**本家では動かない**（後述） |
| 4 | **ComfyUI 本体が起動していること** | 既定 `http://127.0.0.1:8188`。LoRA Manager の API を叩くため |

加えて **Civitai のアカウント**（成人向けを含める場合は API キー）と、
**レシピの置き場所**（LoRA Manager が読むフォルダ）が要る。

> ### なぜ本家 LoRA Manager では動かないのか
> このスクリプトは起動時に `/api/lm/health-check` を叩き、
> `recipe_generation_source_policy == "embedded-first-v1"` を要求する。
> この識別子は**本家 1.1.9 には存在しない**（改造版が追加したもの）。
> 本家に向けると起動時チェックで止まる。

## セットアップ

```
cp config.example.json config.json
```

`config.json` を開いて埋める。必須は3つ。

| キー | 環境変数 | 中身 |
|---|---|---|
| `raindrop_token` | `RAINDROP_TOKEN` | Raindrop の設定 → Integrations → For Developers → Create new app → テストトークン |
| `collection_id` | `RAINDROP_COLLECTION_ID` | 対象コレクションを開いた時のURL末尾の数字 |
| `recipe_dir` | `LORA_RECIPE_DIR` | LoRA Manager のレシピ保存フォルダ（絶対パス） |

任意:

| キー | 環境変数 | 既定 |
|---|---|---|
| `civitai_api_key` | `CIVITAI_API_KEY` | 無し。成人向けや一部のフォールバックで必要 |
| `comfy_base_url` | `COMFY_BASE_URL` | `http://127.0.0.1:8188` |
| `lora_models_dir` | `LORA_MODELS_DIR` | `recipe_dir` の一つ上 |
| `checkpoint_models_dir` | `CHECKPOINT_MODELS_DIR` | 無し。**未設定だとチェックポイントの名前解決が効かない** |
| `prompt_lora_fallbacks` | (config.json のみ) | 空 |

**優先順位は 環境変数 > `config.json` > 既定値。**
`config.json` は `.gitignore` 済み。別の場所に置きたい場合は `CIVITAI_SYNC_CONFIG` にパスを渡す。

`recipe_dir` と `checkpoint_models_dir` は環境ごとに違う。
LoRA Manager 側の `settings.json` / `extra_model_paths.yaml` を見て合わせること。

## 実行

```
pip install requests
python civitai_image_download.py
```

必須項目が欠けていれば、何が足りないかを出して止まる。

初回だけ既存レシピの移行がまとめて走る。全件を一度に処理したい場合:

```
CIVITAI_REIMPORT_BATCH_SIZE=0 python civitai_image_download.py
```

## Civitai へのアクセスについて

**公開API（`/api/v1/…`）だけを使う。**

Civitai の利用規約 §11.4 は自動アクセスを「明示的に提供するインターフェース」に限っている。
そのため、以前のバージョンにあった

- 内部API `api/trpc/image.getGenerationData` / `api/trpc/image.get`
- 画像ページのHTMLを正規表現でスキャンして画像URLを抜く経路

は**どちらも撤去した**。現在使うのは次の公開エンドポイントだけ。

| エンドポイント | 用途 |
|---|---|
| `GET /api/v1/images?imageId=..&withMeta=true` | 画像本体URL・生成情報・使用モデルID |
| `GET /api/v1/model-versions/{id}` | モデルバージョンの詳細とDLリンク |
| `GET /api/v1/model-versions/by-hash/{sha256}` | ハッシュからモデルを同定 |

**撤去による回収率の低下は無い**（2026-07-26・69枚のペア実測）。
詳細は [`../comfyui-lora-manager/CHANGES-FORK.md`](../comfyui-lora-manager/CHANGES-FORK.md) の §7。

実測で分かった注意点:

- `imageId` フィルタは**公式ドキュメントに記載が無いが機能する**
- **`withMeta=true` を付けないと `meta` が空**で返る
- `meta` は `{"id": .., "meta": {..}}` と**1段包まれて返ることがある**

## `prompt_lora_fallbacks` について

プロンプトの `<lora:名前:強度>` にしか痕跡が無く、Civitai からも消えた LoRA を、
利用者が自分で用意したミラーから復元するための**枠だけ**を用意してある。**既定は空**。

`url` と `sha256` の両方が必須で、ダウンロード後に実ファイルのハッシュを照合し、
一致しなければ破棄する（すり替え防止）。

**何を登録するかは利用者の責任。** 配布物には何も入れていない。
実在人物の肖像を学習した LoRA など、権利・肖像権の問題があるものは登録しないこと。

## 制限

- **Windows 前提の箇所がある**（レシピの `file_path` をバックスラッシュで書く）
- 画像が**非公開・削除済み**なら取れない。生成情報を投稿者が伏せている場合も同じ
- 大量に回すと Civitai 側でレート制限に当たる
- レシピの再取り込み（reimport）は**レシピIDが変わる**

## ライセンス

**MIT。** 全文は [`LICENSE`](LICENSE) を参照。自分のワークフローへ自由に組み込んでよい。

同梱の LoRA Manager フォーク（[`../comfyui-lora-manager/`](../comfyui-lora-manager/)）は **GPL-3.0** で、
ライセンスが違う。このスクリプトは HTTP API 越しの別プロセスなので GPL の派生物条件には掛からず、
単体で MIT のまま扱える。**境界はディレクトリそのもの**で、
`civitai-recipe-sync/` 配下が MIT、`comfyui-lora-manager/` 配下が GPL-3.0。
