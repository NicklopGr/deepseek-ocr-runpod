# DeepSeek-OCR RunPod Serverless Container
#
# Model: deepseek-ai/DeepSeek-OCR (3B params, BF16)
# Inference: vLLM with Gundam mode
# GPU: Requires CUDA 12.x compatible GPU (A100/A10/L40S recommended)

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install system dependencies including Python 3.11
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

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install vLLM nightly (required for DeepSeek-OCR support)
# DeepSeek-OCR is supported in vLLM 0.11.1+ / nightly builds
RUN pip install --no-cache-dir --pre vllm --extra-index-url https://wheels.vllm.ai/nightly

# Install flash-attention (build from source for CUDA 12.8)
RUN pip install --no-cache-dir ninja packaging
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    pillow \
    huggingface_hub

# Download model at build time to bake into image
# This avoids download delays on cold starts
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR')"

# Copy handler
COPY handler.py /app/handler.py

# Run handler
CMD ["python", "-u", "handler.py"]
