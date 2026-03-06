#!/usr/bin/env bash
set -euo pipefail

echo "=== cosmos-reason2 setup start ==="

# ----------------------------
# path setup
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

COSMOS_REASON2_DIR="${WORKSPACE_ROOT}/cosmos-reason2"
SFT_LLAVA_TOML="${SCRIPT_DIR}/llava_sft.toml"
COSMOS_LLAVA_TOML="${COSMOS_REASON2_DIR}/examples/cosmos_rl/configs/llava_sft.toml"
SFT_LLAVA_SCRIPT="${SCRIPT_DIR}/llava_sft.py"
COSMOS_LLAVA_SCRIPT="${COSMOS_REASON2_DIR}/examples/cosmos_rl/scripts/llava_sft.py"

echo "[path] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[path] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[path] WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "[path] COSMOS_REASON2_DIR=${COSMOS_REASON2_DIR}"

# ----------------------------
# root check
# ----------------------------
if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: Please run as root (use sudo)"
  exit 1
fi

# ----------------------------
# load env (tokens)
# ----------------------------
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
  echo "[env] loading ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "ERROR: .env file not found at ${ENV_FILE}"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set in ${ENV_FILE}"
  exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is not set in ${ENV_FILE}"
  exit 1
fi

# ----------------------------
# apt & repositories
# ----------------------------
echo "[1/8] enable universe & update apt"
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y universe
apt-get update

# ----------------------------
# system packages
# ----------------------------
echo "[2/8] install system packages"
apt-get install -y \
  curl \
  ffmpeg \
  git \
  unzip \
  gnupg \
  redis-server

# ----------------------------
# git-lfs
# ----------------------------
echo "[3/8] install git-lfs"
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get update
apt-get install -y git-lfs
git lfs install

# ----------------------------
# uv install
# ----------------------------
echo "[4/8] install uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ----------------------------
# clone cosmos-reason2
# ----------------------------
echo "[5/8] clone cosmos-reason2"
if [ ! -d "${COSMOS_REASON2_DIR}/.git" ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-reason2.git "${COSMOS_REASON2_DIR}"
fi
cd "${COSMOS_REASON2_DIR}"

# ----------------------------
# python env & deps (CUDA 12.8)
# ----------------------------
echo "[6/8] setup python venv & dependencies"
uv sync --extra cu128
source .venv/bin/activate

# ----------------------------
# HuggingFace auth (non-interactive)
# ----------------------------
echo "[7/8] huggingface login"
uvx hf auth login --token "$HF_TOKEN"

# ----------------------------
# cosmos_rl setup
# ----------------------------
echo "[8/8] setup cosmos_rl & wandb"
cd examples/cosmos_rl
uv sync
uv tool install -U wandb
wandb login "$WANDB_API_KEY"

source .venv/bin/activate
uv pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4 --no-deps --no-build-isolation

# ----------------------------
# copy SFT llava config
# ----------------------------
echo "[config] copy custom llava_sft.toml"
if [ ! -f "${SFT_LLAVA_TOML}" ]; then
  echo "ERROR: source config not found: ${SFT_LLAVA_TOML}"
  exit 1
fi
if [ ! -d "$(dirname "${COSMOS_LLAVA_TOML}")" ]; then
  echo "ERROR: target config directory not found: $(dirname "${COSMOS_LLAVA_TOML}")"
  exit 1
fi
cp -f "${SFT_LLAVA_TOML}" "${COSMOS_LLAVA_TOML}"
echo "[config] replaced ${COSMOS_LLAVA_TOML}"

# ----------------------------
# copy SFT llava script
# ----------------------------
echo "[config] copy custom llava_sft.py"
if [ ! -f "${SFT_LLAVA_SCRIPT}" ]; then
  echo "ERROR: source script not found: ${SFT_LLAVA_SCRIPT}"
  exit 1
fi
if [ ! -d "$(dirname "${COSMOS_LLAVA_SCRIPT}")" ]; then
  echo "ERROR: target script directory not found: $(dirname "${COSMOS_LLAVA_SCRIPT}")"
  exit 1
fi
cp -f "${SFT_LLAVA_SCRIPT}" "${COSMOS_LLAVA_SCRIPT}"
echo "[config] replaced ${COSMOS_LLAVA_SCRIPT}"

echo "=== cosmos-reason2 setup completed ==="
