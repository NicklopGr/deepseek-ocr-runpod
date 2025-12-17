# DeepSeek-OCR RunPod Serverless Container
#
# Model: deepseek-ai/DeepSeek-OCR (3B params, BF16)
# Inference: vLLM with Gundam mode
# GPU: Requires CUDA 12.x compatible GPU (A100/A10/L40S recommended)

FROM runpod/pytorch:2.6.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM nightly (required for DeepSeek-OCR support)
# DeepSeek-OCR is supported in vLLM 0.11.1+ / nightly builds
RUN pip install --no-cache-dir --pre vllm --extra-index-url https://wheels.vllm.ai/nightly

# Install additional dependencies
RUN pip install --no-cache-dir \
    flash-attn==2.7.3 \
    runpod \
    pillow \
    huggingface_hub

# Download model at build time to bake into image
# This avoids download delays on cold starts
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-OCR')"

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Run handler
CMD ["python", "-u", "handler.py"]
