"""
OvisOCR2 — Multimodal Document & Receipt OCR Model Inference.
Compact 0.9B parameter state-of-the-art vision-language document parser.
"""

import os
import sys
import gc
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.image_optimizer import optimize_image_for_ocr

MODEL_ID = "ATH-MaaS/OvisOCR2"


def get_device():
    """Apple Silicon MPS, CUDA veya CPU seçimi."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_ovis_ocr2(model_id=MODEL_ID):
    """
    OvisOCR2 model ve işlemcisini Apple Silicon MPS / float16 ile optimize yükler.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    device = get_device()
    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32

    print(f"[*] OvisOCR2 optimize yükleniyor: '{model_id}' (Cihaz: {device}, Veri Tipi: {dtype})...")

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(device).eval()
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device).eval()

    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return processor, model, device


def run_ovis_ocr2(image_path, prompt="Read all text in this image.", processor=None, model=None, device=None, max_new_tokens=1024):
    """
    Görsel üzerinden OvisOCR2 çıkarımı yapar.
    """
    if processor is None or model is None:
        processor, model, device = load_ovis_ocr2()

    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # Fiş ve belge akıllı ölçekleme
    image = optimize_image_for_ocr(image, model_key="OvisOCR2")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)

    eos_ids = [248044, 248054]
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "eos_token_id") and processor.tokenizer.eos_token_id is not None:
        eos_ids.append(processor.tokenizer.eos_token_id)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
            do_sample=False
        )
        input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

    # Temizlik: think etiketleri ve boşluklar
    cleaned = decoded.replace("<think>\n\n</think>", "").replace("<think></think>", "").strip()

    del inputs
    del output
    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return cleaned


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_img or not os.path.exists(test_img):
        test_img = os.path.join(PROJECT_ROOT, "ocr", "GLM-OCR", "sample_test.png")

    print(f"[*] Görsel test ediliyor: {test_img}")
    result = run_ovis_ocr2(test_img)
    print("\n" + "=" * 50)
    print("🎯 OvisOCR2 Çıktısı:")
    print("=" * 50)
    print(result)
    print("=" * 50)
