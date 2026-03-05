# CosmosReason2-SFT

This repository provides scripts and configurations for Supervised Fine-Tuning (SFT) of CosmosReason2.

## 1. SFT Procedure

For the machine requirements needed for training, refer to the official Cosmos-Reason2 [Tested Platforms](https://github.com/nvidia-cosmos/cosmos-reason2/tree/main?tab=readme-ov-file#tested-platforms).

For detailed environment setup and training execution steps, refer to `SFT/README.md`.

## 2. Deployment Procedure

This section describes how to serve a trained model with NVIDIA NIM (VLM).

References:
- Model distribution: [mito-w/CosmosReason2](https://huggingface.co/mito-w/CosmosReason2/tree/main)
- Official NIM guide (Cosmos Reason2 / Usage): [Fine-Tune a Model - Usage](https://docs.nvidia.com/nim/vision-language-models/latest/fine-tune-model.html#usage)

### 2.1 Prerequisites

- NVIDIA GPU is available
- Docker and NVIDIA Container Toolkit are available
- NIM image for Cosmos Reason2 is available (`$NIM_IMAGE`)
- You can download models from Hugging Face (for example, using `huggingface-cli`)

### 2.2 Download the model

Download `mito-w/CosmosReason2` locally.

```bash
huggingface-cli download mito-w/CosmosReason2 --local-dir /path/to/CosmosReason2
```

Use the `server` directory in this repository as the model path for NIM startup.

```bash
export CUSTOM_WEIGHTS=/path/to/CosmosReason2/server
```

### 2.3 Start NIM container (Cosmos Reason2)

The following example follows the official NIM documentation for Cosmos Reason2.

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

### 2.4 Connectivity check (optional)

After startup, verify connectivity by accessing `http://localhost:8000` from your inference client or API caller.

## 3. Demo App Startup

For detailed setup and startup instructions for the demo app, refer to `demo_app/README.md`.

## 4. Dataset and Model Distribution

- Dataset: [mito-w/CosmosReason2-dataset](https://huggingface.co/datasets/mito-w/CosmosReason2-dataset/tree/main)
- Model: [mito-w/CosmosReason2](https://huggingface.co/mito-w/CosmosReason2/tree/main)

## 5. Demo Video

<video src="resources/demo_movie.mp4" controls width="960">
  Your environment does not support embedded video playback.
</video>

If embedded playback is not available, open [demo video](resources/demo_movie.mp4).
