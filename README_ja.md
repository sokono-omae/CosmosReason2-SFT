# CosmosReason2-SFT（日本語ガイド）

このリポジトリは、CosmosReason2 の Supervised Fine-Tuning（SFT）を行うためのスクリプトと設定を提供します。

## 1. SFT 手順

学習に必要なマシン要件は、公式の Cosmos-Reason2 の [Tested Platforms](https://github.com/nvidia-cosmos/cosmos-reason2/tree/main?tab=readme-ov-file#tested-platforms) を参照してください。

SFT の詳細な環境構築・学習実行手順は、`SFT/README_ja.md` を参照してください。

## 2. デプロイ手順

ここでは、学習済みモデルを NVIDIA NIM (VLM) でサービングする手順を示します。

参照:
- モデル配布先: [mito-w/CosmosReason2](https://huggingface.co/mito-w/CosmosReason2/tree/main)
- NIM 公式手順（Cosmos Reason2 / Usage）: [Fine-Tune a Model - Usage](https://docs.nvidia.com/nim/vision-language-models/latest/fine-tune-model.html#usage)

### 2.1 前提条件

- NVIDIA GPU が利用可能であること
- Docker および NVIDIA Container Toolkit が利用可能であること
- NIM の Cosmos Reason2 対応イメージを取得済みであること（`$NIM_IMAGE`）
- Hugging Face からモデルを取得できること（`huggingface-cli` など）

### 2.2 モデルをダウンロード

`mito-w/CosmosReason2` をローカルに取得します。

```bash
huggingface-cli download mito-w/CosmosReason2 --local-dir /path/to/CosmosReason2
```

このリポジトリの `server` ディレクトリを、NIM 起動時のモデルパスとして使用します。

```bash
export CUSTOM_WEIGHTS=/path/to/CosmosReason2/server
```

### 2.3 NIM コンテナを起動（Cosmos Reason2）

以下は NIM 公式ドキュメント（Cosmos Reason2）に沿った起動例です。

```bash
docker run -it --rm --name=cosmos-reason2-2b \
    --gpus all \
    --shm-size=32GB \
    -e NIM_MODEL_NAME=$CUSTOM_WEIGHTS \
    -e NIM_SERVED_MODEL_NAME="cosmos-reason2-2b" \
    -v $CUSTOM_WEIGHTS:$CUSTOM_WEIGHTS \
    -u $(id -u) \
    -p 8000:8000 \
    $NIM_IMAGE
```

### 2.4 疎通確認（任意）

サーバー起動後、`http://localhost:8000` に対して推論クライアントや API から接続して動作確認してください。

## 3. デモアプリ起動手順

デモアプリの詳細なセットアップ・起動方法は `demo_app/README.md` を参照してください。

最低限の実行コマンド:

```bash
cd demo_app
uv venv
uv sync
uv run python run_fastapi_uvicorn.py
```

## 4. 使用データ・使用モデル配布先

- 使用データセット: [mito-w/CosmosReason2-dataset](https://huggingface.co/datasets/mito-w/CosmosReason2-dataset/tree/main)
- 使用モデル: [mito-w/CosmosReason2](https://huggingface.co/mito-w/CosmosReason2/tree/main)
