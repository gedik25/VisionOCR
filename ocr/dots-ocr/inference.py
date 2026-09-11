"""
dots.ocr — Multilingual Document Layout & OCR Model Inference.
High-accuracy layout parsing, text extraction, and Markdown table output.
"""

import os
import sys
import gc
import json
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.image_optimizer import optimize_image_for_ocr

MODEL_ID = "rednote-hilab/dots.ocr"

OFFICIAL_LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def get_device():
    """Apple Silicon MPS, CUDA veya CPU seçimi."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dots_ocr(model_id=MODEL_ID):
    """
    dots.ocr model ve işlemcisini düşük bellek kullanımıyla yerel önbellekten yükler.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM

    device = get_device()
    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32

    print(f"[*] dots.ocr optimize yükleniyor: '{model_id}' (Cihaz: {device}, Veri Tipi: {dtype})...")

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(device).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(device).eval()

    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return processor, model, device


def layout_json_to_markdown(raw_output: str) -> str:
    """
    dots.ocr JSON çıktısını okunabilir Markdown ve düz metne dönüştürür.
    """
    cleaned = raw_output.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            lines = []
            for item in data:
                cat = item.get("category", "Text")
                text = item.get("text", "").strip()
                bbox = item.get("bbox", [])
                if text:
                    if cat == "Title":
                        lines.append(f"# {text}")
                    elif cat == "Section-header":
                        lines.append(f"## {text}")
                    elif cat == "List-item":
                        lines.append(f"- {text}")
                    else:
                        lines.append(text)
                elif cat == "Picture" and bbox:
                    lines.append(f"*[Görsel/Grafik Bölgesi: {bbox}]*")
            if lines:
                return "\n\n".join(lines)
    except Exception:
        pass
    return cleaned


def run_dots_ocr(image_path, prompt=None, processor=None, model=None, device=None, max_new_tokens=512):
    """
    Görsel üzerinden dots.ocr çıkarımı yapar.
    """
    if processor is None or model is None:
        processor, model, device = load_dots_ocr()

    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # Fiş / Belge akıllı ölçekleme
    image = optimize_image_for_ocr(image, model_key="dots.ocr")

    prompt_content = prompt or OFFICIAL_LAYOUT_PROMPT
    prompt_text = f"<|user|><|img|><|imgpad|><|endofimg|>{prompt_content}<|endofuser|><|assistant|>"

    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt"
    )

    device = model.device if hasattr(model, "device") else get_device()
    model_dtype = getattr(model, "dtype", torch.float16 if str(device) in ["mps", "cuda"] else torch.float32)
    inputs = {k: v.to(device=device, dtype=model_dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    inputs.pop("mm_token_type_ids", None)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=[151673, 151643],
            do_sample=False
        )
        input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

    del inputs
    del output
    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    formatted_md = layout_json_to_markdown(decoded)
    return formatted_md if formatted_md else decoded.strip()


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_img or not os.path.exists(test_img):
        test_img = os.path.join(PROJECT_ROOT, "ocr", "GLM-OCR", "sample_test.png")

    print(f"[*] Görsel test ediliyor: {test_img}")
    result = run_dots_ocr(test_img)
    print("\n" + "=" * 50)
    print("🎯 dots.ocr Çıktısı:")
    print("=" * 50)
    print(result)
    print("=" * 50)
