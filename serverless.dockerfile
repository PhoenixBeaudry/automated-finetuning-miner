FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

USER root

# System dependencies and performance tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# Install ML packages with specific versions for compatibility
RUN pip install -U --no-cache-dir \
    numpy \
    transformers \
    accelerate \
    datasets\
    sentencepiece\
    huggingface_hub \
    wandb \
    peft \
    bitsandbytes \
    safetensors \
    tokenizers

# Install training frameworks
RUN pip install -U --no-cache-dir \
    trl \
    liger-kernel \
    optuna \
    mlflow \
    protobuf

# Install Flash Attention 2 (much faster than flash-attn v1)
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Install Triton for kernel compilation
RUN pip install --no-cache-dir triton

# Install DeepSpeed
RUN pip install --no-cache-dir -U deepspeed

# Install additional optimizations
RUN pip install --no-cache-dir \
    einops \
    scipy \
    numba \
    packaging \
    toml \
    hf_xet \
    psutil

# RunPod specific
RUN pip install -U runpod
RUN pip install textstat


WORKDIR /workspace
RUN mkdir -p /workspace/configs /workspace/outputs /workspace/data /workspace/input_data /workspace/training

# Environment variables for optimal performance
ENV TOKENIZERS_PARALLELISM=false

# Ensure high-speed P2P/NCCL comms and fault tolerance
ENV NCCL_DEBUG=WARN
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
ENV DEEPSPEED_TIMEOUT=3600
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV NCCL_IB_DISABLE=1
ENV NCCL_SHARP_DISABLE=1

# AWS credentials (keep existing)
ENV CONFIG_DIR="/workspace/configs"
ENV OUTPUT_DIR="/workspace/outputs"
ENV AWS_ENDPOINT_URL="https://5a301a635a9d0ac3cb7fcc3bf373c3c3.r2.cloudflarestorage.com"
ENV AWS_ACCESS_KEY_ID=REMOVED_EXPOSED_CREDENTIAL
ENV AWS_DEFAULT_REGION="us-east-1"
ENV AWS_SECRET_ACCESS_KEY=REMOVED_EXPOSED_CREDENTIAL

RUN mkdir -p /root/.aws && \
    echo "[default]\naws_access_key_id=dummy_access_key\naws_secret_access_key=dummy_secret_key" > /root/.aws/credentials && \
    echo "[default]\nregion=us-east-1" > /root/.aws/config

# Copy configuration files
COPY runpod_handler.py /workspace
COPY configs/ /workspace/configs
COPY training_helpers/ /workspace/training/training_helpers
COPY training/ /workspace/training


CMD echo 'Preparing logging...' && \
    echo "Attempting to log in to Hugging Face" && \
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential && \
    echo "Attempting to log in to W&B" && \
    wandb login "$WANDB_TOKEN" && \
    python -u /workspace/runpod_handler.py