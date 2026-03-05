# CosmosReason2-SFT（日本語ガイド）

このリポジトリは、CosmosReason2 の Supervised Fine-Tuning（SFT）を行うためのスクリプトと設定を提供します。

## 1. SFT 手順

学習に必要なマシン要件は、公式の Cosmos-Reason2 の [Tested Platforms](https://github.com/nvidia-cosmos/cosmos-reason2/tree/main?tab=readme-ov-file#tested-platforms) を参照してください。

SFT の詳細な環境構築・学習実行手順は、`SFT/README_ja.md` を参照してください。

## 2. デプロイ手順

> TODO: ここにデプロイの詳細手順を記載予定です。

- （ダミー）推論サーバー環境準備
- （ダミー）モデル配置
- （ダミー）API 起動
- （ダミー）疎通確認

## 3. デモアプリ起動手順

`cosmos-reason2-app` の内容が `demo_app/` 配下にある前提で、FastAPI デモアプリを起動します。

### 3.1 前提条件

- Python 3.10 以上
- `uv` がインストール済みであること

`uv` が未インストールの場合（Linux）:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3.2 セットアップ

```bash
cd demo_app
uv venv
uv sync
```

### 3.3 起動

```bash
cd demo_app
uv run python run_fastapi_uvicorn.py
```

起動後、ブラウザで `http://localhost:8001` を開いてください。

### 3.4 任意の環境変数（API設定）

必要に応じて、以下を設定できます。

- `API_BASE_URL`
- `API_KEY`
- `MODEL`
- `MAX_TOKENS`
- `TEMPERATURE`
- `JPEG_QUALITY`
- `REQUEST_INTERVAL_SEC`
- `MIN_STEP_DURATION_SEC`
