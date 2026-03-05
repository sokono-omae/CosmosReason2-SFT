# CosmosReason2 SFT Environment Setup (Docker)

This README describes how to set up an environment for Cosmos-Reason2 SFT.

> For machine requirements needed for training, please refer to the official Cosmos-Reason2 [Tested Platforms](https://github.com/nvidia-cosmos/cosmos-reason2/tree/main?tab=readme-ov-file#tested-platforms).

## 1. Pull and run the Docker image

```bash
docker pull nvcr.io/nvidia/pytorch:25.03-py3

docker run --gpus all -it --rm \
  --ipc=host \
  -v /your/local/workspace:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.03-py3
```

- The working directory inside the container is assumed to be `/workspace`.
- Replace `/your/local/workspace` in `-v /your/local/workspace:/workspace` with your actual host-side working directory.
- This script runs `apt-get`, so it must be executed as root (typically root by default in the container command above).

## 2. Clone the repository

```bash
cd /workspace
git clone https://github.com/sokono-omae/CosmosReason2-SFT.git
cd /workspace/CosmosReason2-SFT/SFT
```

## 3. Create `.env`

Create `/workspace/CosmosReason2-SFT/SFT/.env` and set:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

- `HF_TOKEN`: Used for Hugging Face login
- `WANDB_API_KEY`: Used for Weights & Biases login

## 4. Run the setup script

```bash
cd /workspace/CosmosReason2-SFT/SFT
chmod +x setup_cosmos-reason2_post-training.sh
source setup_cosmos-reason2_post-training.sh
```

This script performs:

1. Installs system dependencies (`ffmpeg`, `git-lfs`, `redis-server`, etc.)
2. Installs `uv`
3. Clones `nvidia-cosmos/cosmos-reason2` to **`/workspace/cosmos-reason2`**
4. Resolves dependencies for `cosmos-reason2` and `examples/cosmos_rl`
5. Logs in to Hugging Face / W&B
6. Installs `grouped_gemm`
7. Copies (replaces) `SFT/llava_sft.toml` to  
   `/workspace/cosmos-reason2/examples/cosmos_rl/configs/llava_sft.toml`

## 5. Download the Hugging Face dataset under `/workspace`

Target dataset:
- `mito-w/CosmosReason2-dataset`

```bash
cd /workspace
git lfs install
git clone https://huggingface.co/datasets/mito-w/CosmosReason2-dataset
```

> Alternative (for environments where `git clone` is difficult):  
> `huggingface-cli download mito-w/CosmosReason2-dataset --repo-type dataset --local-dir /workspace/CosmosReason2-dataset`

## 6. Training command example

```bash
cd /workspace/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate
uv run cosmos-rl --config configs/llava_sft.toml --log-dir outputs/llava_sft scripts/llava_sft.py
```
Training outputs are saved to `/workspace/results/llava_sft_output`.

## References

- [sokono-omae/CosmosReason2-SFT](https://github.com/sokono-omae/CosmosReason2-SFT.git)
- [nvidia-cosmos/cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2.git)
- [mito-w/CosmosReason2-dataset](https://huggingface.co/datasets/mito-w/CosmosReason2-dataset/tree/main)
