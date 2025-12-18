"""
DeepSeek-OCR RunPod Serverless Handler

Uses DeepSeek-OCR (3B model) with vLLM in Gundam mode for high-quality
document OCR with markdown output.

Model: deepseek-ai/DeepSeek-OCR
- 3B parameters, BF16 precision
- Gundam mode: base_size=1024, image_size=640, crop_mode=True
- 97% OCR precision at <10x compression ratio
- ~2500 tokens/s on A100-40G

Input: {"image_base64": "...", "prompt": "optional custom prompt"}
Output: {"markdown": "..."}
"""

import runpod
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import base64
import io
import time

print("[DeepSeek-OCR] Loading model with vLLM (BF16)...")
start_load = time.time()

# Load model once at startup (BF16 native precision)
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    dtype="bfloat16",  # BF16 native precision
    trust_remote_code=True,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

print(f"[DeepSeek-OCR] Model loaded in {time.time() - start_load:.2f}s")


# Default prompt for markdown extraction with grounding
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def handler(job):
    """
    Process a single image with DeepSeek-OCR Gundam mode.

    Args:
        job: RunPod job with input containing:
            - image_base64: Base64 encoded image (PNG/JPG)
            - prompt: Optional custom prompt (defaults to markdown conversion)

    Returns:
        dict with:
            - markdown: Extracted markdown text
            - processing_time: Time taken in seconds
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

        print(f"[DeepSeek-OCR] Processing image: {image.size[0]}x{image.size[1]}")
        if custom_prompt:
            print(f"[DeepSeek-OCR] Using custom prompt: {prompt[:100]}...")

        # Sampling parameters optimized for OCR
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic output
            max_tokens=8192,  # Max tokens per page
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # <td>, </td> tokens
            ),
            skip_special_tokens=False,
        )

        # Run inference
        model_input = [{"prompt": prompt, "multi_modal_data": {"image": image}}]
        outputs = llm.generate(model_input, sampling_params)

        markdown = outputs[0].outputs[0].text
        processing_time = time.time() - start_time

        print(f"[DeepSeek-OCR] Completed in {processing_time:.2f}s, output length: {len(markdown)} chars")

        return {
            "status": "success",
            "markdown": markdown,
            "processing_time": processing_time
        }

    except Exception as e:
        print(f"[DeepSeek-OCR] Error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "processing_time": time.time() - start_time
        }


# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
