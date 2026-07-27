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

`civitai-recipe-sync/` は HTTP API 越しの別プロセスなので GPL の派生物条件には掛からず、
別ライセンスにできる。こちらは **MIT**（[`civitai-recipe-sync/LICENSE`](civitai-recipe-sync/LICENSE)）。
読者が自分のワークフローへ組み込む摩擦を最小にするためにこうした。

**この配布ツリーは2ライセンス混在で、境界はディレクトリそのもの。**
`comfyui-lora-manager/` 配下が GPL-3.0、`civitai-recipe-sync/` 配下が MIT。
片方を改変して再配布するときは、そのディレクトリのライセンスに従うこと。

## Civitai へのアクセス方針

**公開API（`/api/v1/…`）のみを使う。**
内部API（`api/trpc/…`）と画像ページのHTML直接スキャンは、
Civitai 利用規約 §11.4（自動アクセスは明示的に提供されたインターフェースに限る）に合わせて
**両方とも撤去済み**。撤去しても回収率は落ちないことを69枚のペア実測で確認している
（[CHANGES-FORK.md §7](comfyui-lora-manager/CHANGES-FORK.md)）。

## 状態

**第1段階（配布できる状態にする）まで。** 未了なのは次の2点。

- **GitHub へは未公開**（公開は人間が判断する）
- **UI統合は未着手** — 同期機能はスクリプトを直接叩く。フォークのボタンからは呼べない
