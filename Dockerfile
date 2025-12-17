# DeepSeek-OCR RunPod Serverless Container
#
# Model: deepseek-ai/DeepSeek-OCR (3B params, BF16)
# Inference: vLLM 0.8.5 with Gundam mode
# Requirements: CUDA 11.8, Python 3.12.9, PyTorch 2.6.0
# GPU: A100-40G recommended (~40GB VRAM)
#
# Based on official requirements from:
# https://github.com/deepseek-ai/DeepSeek-OCR

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install system dependencies including Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    wget \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.6.0 with CUDA 11.8 (official DeepSeek-OCR requirement)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install vLLM 0.8.5 (official DeepSeek-OCR requirement)
# Download the specific wheel from vLLM releases
RUN wget -q https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl \
    && pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl \
    && rm vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install flash-attention 2.7.3 (official DeepSeek-OCR requirement)
RUN pip install ninja packaging \
    && pip install flash-attn==2.7.3 --no-build-isolation

# Install additional dependencies
RUN pip install \
    runpod \
    pillow \
    huggingface_hub \
    transformers==4.46.3 \
    einops \
    addict \
    easydict

# Download model at build time to bake into image
# This avoids download delays on cold starts
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR')"

# Copy handler
COPY handler.py /app/handler.py

# Run handler
CMD ["python", "-u", "handler.py"]
