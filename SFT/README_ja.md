# CosmosReason2 SFT Environment Setup (Docker)

このREADMEは、 Cosmos-Reason2 のSFT実行環境を構築する手順です。

> 学習に必要なマシン要件は、公式の Cosmos-Reason2 の [Tested Platforms](https://github.com/nvidia-cosmos/cosmos-reason2/tree/main?tab=readme-ov-file#tested-platforms) を参照してください。

## 1. Docker イメージの取得と起動

```bash
docker pull nvcr.io/nvidia/pytorch:25.03-py3

docker run --gpus all -it --rm \
  --ipc=host \
  -v /your/local/workspace:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.03-py3
```

- コンテナ内の作業ディレクトリは `/workspace` を前提にしています。
- `-v /your/local/workspace:/workspace` の `/your/local/workspace` は、ホスト側の実際の作業ディレクトリに置き換えてください。
- 本スクリプトは `apt-get` を実行するため、rootユーザーでの実行が必要です（上記コンテナ起動なら通常root）。

## 2. リポジトリをクローン

```bash
cd /workspace
git clone https://github.com/sokono-omae/CosmosReason2-SFT.git
cd /workspace/CosmosReason2-SFT/SFT
```

## 3. `.env` を作成

`/workspace/CosmosReason2-SFT/SFT/.env` を作成し、以下を設定してください。

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

- `HF_TOKEN`: Hugging Face へのログインに使用
- `WANDB_API_KEY`: Weights & Biases へのログインに使用

## 4. セットアップスクリプトを実行

```bash
cd /workspace/CosmosReason2-SFT/SFT
chmod +x setup_cosmos-reason2_post-training.sh
source setup_cosmos-reason2_post-training.sh
```

このスクリプトは以下を実施します。

1. システム依存パッケージのインストール（`ffmpeg`, `git-lfs`, `redis-server` など）
2. `uv` のインストール
3. `nvidia-cosmos/cosmos-reason2` を **`/workspace/cosmos-reason2`** にクローン
4. `cosmos-reason2` 本体と `examples/cosmos_rl` の依存解決
5. Hugging Face / W&B ログイン
6. `grouped_gemm` の追加インストール
7. `SFT/llava_sft.toml` を  
   `/workspace/cosmos-reason2/examples/cosmos_rl/configs/llava_sft.toml` にコピー（置換）
8. `SFT/llava_sft.py` を  
   `/workspace/cosmos-reason2/examples/cosmos_rl/scripts/llava_sft.py` にコピー（置換）

※ 公式（オリジナル）の `llava_sft.py` は、学習データを動画のみで構成した場合に不具合が発生するため、本リポジトリでは修正版に差し替えています。

## 5. Hugging Face データセットを `/workspace` 直下に取得

対象データセット:
- `mito-w/CosmosReason2-dataset`

```bash
cd /workspace
git lfs install
git clone https://huggingface.co/datasets/mito-w/CosmosReason2-dataset
```

> 代替手段（`git clone` が難しい環境）  
> `huggingface-cli download mito-w/CosmosReason2-dataset --repo-type dataset --local-dir /workspace/CosmosReason2-dataset`

## 6. 学習開始例

```bash
cd /workspace/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate
uv run cosmos-rl --config configs/llava_sft.toml --log-dir outputs/llava_sft scripts/llava_sft.py
```
学習結果は`/workspace/results/llava_sft_output`に保存されます。

## 参考

- [sokono-omae/CosmosReason2-SFT](https://github.com/sokono-omae/CosmosReason2-SFT.git)
- [nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2.git)
- [mito-w/CosmosReason2-dataset](https://huggingface.co/datasets/mito-w/CosmosReason2-dataset/tree/main)
