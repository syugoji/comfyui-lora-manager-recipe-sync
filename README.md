# lora-manager-release

Civitai で見つけた画像を **Raindrop に溜める → ハッシュで索引化する → LoRA Manager のレシピとして再現する**
までを繋ぐ2点セットの配布用ツリー。

| ディレクトリ | 中身 | ライセンス |
|---|---|---|
| [`comfyui-lora-manager/`](comfyui-lora-manager/) | ComfyUI-Lora-Manager **v1.1.6 の非公式フォーク** | GPL-3.0 |
| [`civitai-recipe-sync/`](civitai-recipe-sync/) | Raindrop → Civitai → レシピ の一括同期スクリプト | MIT |

**片方だけでは価値が出ない。** 溜める場所（Raindrop）と、照合できる索引（SHA-256ハッシュ）が
繋がっていることが本体で、個々のツールは部品にすぎない。
同期スクリプトは改造版のフォークを要求するので、動かすには両方が要る。

## まず読むもの

- 使い方・必要なもの4点 → [`civitai-recipe-sync/README.md`](civitai-recipe-sync/README.md)
- upstream から何を変えたか → [`comfyui-lora-manager/CHANGES-FORK.md`](comfyui-lora-manager/CHANGES-FORK.md)

## ライセンスについて

`comfyui-lora-manager/` は upstream 由来の **GPL-3.0** をそのまま継承する。
つまり **受け取った人は自由に再配布できる**。ここに「有料の壁」は作れない。

`civitai-recipe-sync/` は**独立したプロセス**として動き、フォークとは
**HTTP API と、標準出力を読むだけの子プロセス起動**でしか繋がらない。
同じプロセスへ取り込む（import・リンク）ことはしていないので GPL の派生物条件には掛からず、
別ライセンスにできる。こちらは **MIT**（[`civitai-recipe-sync/LICENSE`](civitai-recipe-sync/LICENSE)）。
読者が自分のワークフローへ組み込む摩擦を最小にするためにこうした。

繋がり方は2方向ある。どちらも別プロセスのままである点は変わらない。

| 向き | 何をするか | 実装 |
|---|---|---|
| スクリプト → フォーク | 解析・保存・再取込を HTTP で頼む | `civitai_image_download.py` が `/api/lm/recipes/*` を叩く |
| フォーク → スクリプト | レシピ画面のボタンから同期を起動し、進捗を受け取る | `py/services/raindrop_sync_service.py` が子プロセスとして起動し、標準出力の1行1件のJSONを読む |

**この配布ツリーは2ライセンス混在で、境界はディレクトリそのもの。**
`comfyui-lora-manager/` 配下が GPL-3.0、`civitai-recipe-sync/` 配下が MIT。
片方を改変して再配布するときは、そのディレクトリのライセンスに従うこと。
**同期ロジックをフォークの中へ移す（import・コピー）と、上の切り分けが崩れる。**

## Civitai へのアクセス方針

**公開API（`/api/v1/…`）のみを使う。**
内部API（`api/trpc/…`）と画像ページのHTML直接スキャンは、
Civitai 利用規約 §11.4（自動アクセスは明示的に提供されたインターフェースに限る）に合わせて
**両方とも撤去済み**。撤去しても回収率は落ちないことを69枚のペア実測で確認している
（[CHANGES-FORK.md §7](comfyui-lora-manager/CHANGES-FORK.md)）。

## 状態

**第2段階（UI統合）まで。**

- **公開先**: https://github.com/syugoji/comfyui-lora-manager-recipe-sync （2026-07-28 公開）
- **UI統合済み** — レシピ画面の「Raindrop 同期」ボタンから起動できる。
  トークンとコレクションIDは設定画面（ライブラリ → Raindrop 同期）で入力する。
  従来どおりスクリプトを直接叩く使い方も残してある。

## テストの状況（既知の失敗16件）

`comfyui-lora-manager/` を `pytest tests/` で走らせると
**1,478 passed / 16 failed** になる（`syrupy` / `hypothesis` を要する2モジュールは除外）。
**この16件はフォークの欠陥ではなく、上流 ComfyUI-Lora-Manager v1.1.6 でも同じく落ちる。**

| ファイル | 件数 | 内容 |
|---|---|---|
| `tests/config/test_config_save_paths.py` | 10 | 設定パスの正規化 |
| `tests/nodes/test_lora_cycler.py` | 3 | LoRA サイクラーの強度既定 |
| `tests/services/test_download_manager_error.py` | 3 | zip 展開時のパス区切り（`C:/` と `C:\`） |

根拠は件数の一致ではない。**素の v1.1.6 を同じ条件で実行してテスト名の集合を突き合わせた**。
上流は 19 failed / 1,357 passed で、その失敗集合はこの16件を完全に含み、
**このフォークだけで落ちるテストは0件**（逆に上流だけで落ちる3件はフォークでは通る）。
再現手順と数字は [`CHANGES-FORK.md`](comfyui-lora-manager/CHANGES-FORK.md) の「テストの状況」にある。

フロントエンドは `vitest` で **436 + 68 テストすべて通過**する。
