# DeepSeek-OCR RunPod Serverless Container - Gundam Mode (Transformers)
#
# Model: deepseek-ai/DeepSeek-OCR (3B params, BF16)
# Inference: Transformers with Gundam mode (NOT vLLM)
# Environment: CUDA 11.8 + PyTorch 2.6.0 + Flash Attention 2.7.3
#
# Official DeepSeek-OCR tested stack:
#   - PyTorch 2.6.0, Transformers 4.46.3, Flash Attention 2.7.3, CUDA 11.8
#
# Gundam mode parameters (matches HF demo exactly):
#   base_size=1024, image_size=640, crop_mode=True
#   eval_mode=True, test_compress=True, save_results=True
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

# Install PyTorch 2.6.0 with CUDA 11.8 support (OFFICIAL DeepSeek-OCR stack)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install flash-attention 2.7.3 (OFFICIAL DeepSeek-OCR version)
# Required for _attn_implementation="flash_attention_2"
# Using MAX_JOBS=4 to limit parallel compilation and avoid OOM
RUN pip install --no-cache-dir ninja packaging
ENV MAX_JOBS=4
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Install Transformers and dependencies (NOT vLLM - using native Transformers)
# Pin transformers to 4.46.3 (official DeepSeek-OCR tested version)
# addict is required by DeepSeek-OCR's custom model code
RUN pip install --no-cache-dir \
    runpod \
    pillow \
    huggingface_hub \
    transformers==4.46.3 \
    accelerate \
    safetensors \
    addict

# Download model at build time to bake into image
# This avoids download delays on cold starts (~2.5GB model)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR')"

# Copy Gundam handler (Transformers-based with model.infer())
COPY handler_gundam.py /app/handler.py

# Run handler
CMD ["python", "-u", "handler.py"]
