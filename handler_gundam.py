"""
DeepSeek-OCR RunPod Serverless Handler - Gundam Mode (Transformers)

Uses native DeepSeek-OCR transformers implementation with Gundam mode
for highest quality document OCR. This matches the Hugging Face demo.

Model: deepseek-ai/DeepSeek-OCR (3B params, BF16)
Mode: Gundam (base_size=1024, image_size=640, crop_mode=True)
Quality: 97% OCR precision at <10x compression ratio

Input: {"image_base64": "...", "prompt": "optional custom prompt"}
Output: {"markdown": "..."}

Key difference from vLLM handler:
- Uses model.infer() with Gundam mode parameters
- Supports base_size, image_size, crop_mode, save_results, test_compress
- Better quality but slightly slower than vLLM

NOTE: eval_mode is NOT a valid parameter for model.infer() - removed Dec 2025
"""

import runpod
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import base64
import io
import tempfile
import os
import time

print("[DeepSeek-OCR Gundam] Loading model and tokenizer...")
start_load = time.time()

model_name = "deepseek-ai/DeepSeek-OCR"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model with flash attention 2 (matches HF demo)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
)

# Move to GPU and set to eval mode
model = model.cuda().eval()

print(f"[DeepSeek-OCR Gundam] Model loaded in {time.time() - start_load:.2f}s")
print(f"[DeepSeek-OCR Gundam] Device: {next(model.parameters()).device}")
print(f"[DeepSeek-OCR Gundam] Dtype: {next(model.parameters()).dtype}")

# Default prompt for markdown extraction with grounding (matches HF demo exactly)
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def handler(job):
    """
    Process a single image with DeepSeek-OCR Gundam mode (Transformers).

    Args:
        job: RunPod job with input containing:
            - image_base64: Base64 encoded image (PNG/JPG)
            - prompt: Optional custom prompt (defaults to markdown conversion)

    Returns:
        dict with:
            - markdown: Extracted markdown text
            - processing_time: Time taken in seconds
            - pipeline: "deepseek-ocr-gundam"
    """
    start_time = time.time()

    try:
        job_input = job.get("input", {})
        image_base64 = job_input.get("image_base64")

        # Support custom prompt override
        custom_prompt = job_input.get("prompt")
        prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT

        if not image_base64:
            return {"error": "Missing image_base64 in input"}

        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        print(f"[DeepSeek-OCR Gundam] Processing image: {image.size[0]}x{image.size[1]}")
        if custom_prompt:
            print(f"[DeepSeek-OCR Gundam] Using custom prompt: {prompt[:100]}...")

        # Save image to temp file (required by model.infer)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_image_path = os.path.join(temp_dir, "input.png")
            image.save(temp_image_path)

            # Gundam mode parameters - production settings
            # save_results=False to get direct return (True returns None + writes to disk)
            # test_compress=False for standard inference (True is for diagnostics)
            markdown = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=temp_dir,
                base_size=1024,      # Gundam: base resolution
                image_size=640,      # Gundam: target resolution
                crop_mode=True,      # Gundam: crop for detail
                save_results=False,  # Return markdown directly (not to disk)
                test_compress=False, # Standard inference mode
            )

        processing_time = time.time() - start_time

        print(f"[DeepSeek-OCR Gundam] Completed in {processing_time:.2f}s, output length: {len(markdown)} chars")

        return {
            "status": "success",
            "markdown": markdown,
            "processing_time": processing_time,
            "pipeline": "deepseek-ocr-gundam"
        }

    except Exception as e:
        import traceback
        print(f"[DeepSeek-OCR Gundam] Error: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "processing_time": time.time() - start_time
        }


# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
