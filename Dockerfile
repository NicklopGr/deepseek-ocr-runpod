# DeepSeek-OCR RunPod Serverless Container
#
# Model: deepseek-ai/DeepSeek-OCR (3B params, BF16)
# Inference: vLLM with Gundam mode
# Environment: CUDA 11.8 + PyTorch 2.6.0 (OFFICIAL)
#
# GPU: A100/A10/L40S recommended (needs ~8GB VRAM)

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV HF_HOME=/root/.cache/huggingface
# Extend initialization timeout for model loading (default is too short)
ENV RUNPOD_INIT_TIMEOUT=800

WORKDIR /app

# Install system dependencies including Python 3.11
# Note: Python 3.12 has issues with some vLLM builds, 3.11 is safer
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.6.0 with CUDA 11.8 support (OFFICIAL for DeepSeek-OCR)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install vLLM nightly (required for DeepSeek-OCR support)
# DeepSeek-OCR is supported in vLLM 0.8.5+ with CUDA 11.8
RUN pip install --no-cache-dir --pre vllm --extra-index-url https://wheels.vllm.ai/nightly

# Install flash-attention 2.7.3 (OFFICIAL version for DeepSeek-OCR)
RUN pip install --no-cache-dir ninja packaging
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Install additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    pillow \
    huggingface_hub \
    transformers

# Download model at build time to bake into image
# This avoids download delays on cold starts (~2.5GB model)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR')"

# Copy handler
COPY handler.py /app/handler.py

# Run handler
CMD ["python", "-u", "handler.py"]
