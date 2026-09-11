import os
import sys
import gc
import torch
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────
# Cihaz ve dtype seçimi  (MPS → float32, CPU → float32)
# ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
    dtype  = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype  = torch.float32          # bfloat16 MPS'te desteklenmiyor
else:
    device = "cpu"
    dtype  = torch.float32

MODEL_ID = "lightonai/LightOnOCR-2-1B"
MAX_IMAGE_EDGE = 1536

def resize_image_if_needed(image: Image.Image, max_edge: int = MAX_IMAGE_EDGE) -> Image.Image:
    from ocr.image_optimizer import optimize_image_for_ocr
    return optimize_image_for_ocr(image, model_key="LightOnOCR-2-1B", max_edge=max_edge)

# ──────────────────────────────────────────────────────────
# Test görseli oluştur
# ──────────────────────────────────────────────────────────
def create_test_image(path: str) -> str:
    img  = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 40),  "LightOnOCR-2-1B Testi",     fill=(0, 0, 0),   font=font)
    draw.text((30, 90),  "Belge Okuma: Başarılı mı?",  fill=(0, 50, 200), font=font)
    draw.text((30, 140), "Tarih: 2026-08-30 | Skor: 100/100", fill=(0, 150, 0), font=font)
    img.save(path)
    print(f"[✓] Test görseli oluşturuldu: {path}")
    return path


def run_ocr(image_path: str) -> str:
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    print(f"[*] LightOnOCR yükleniyor: '{MODEL_ID}' (Cihaz: {device}, dtype: {dtype})...")
    try:
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
        ).to(device)
        processor = LightOnOcrProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)

    image = Image.open(image_path).convert("RGB")
    image = resize_image_if_needed(image)

    conversation = [
        {"role": "user", "content": [{"type": "image"}]}
    ]

    prompt_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    print("[*] OCR çıkarımı yapılıyor...")
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    result = processor.decode(outputs[0], skip_special_tokens=True)

    del inputs
    del outputs
    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return result


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(script_dir, "sample_test.png")

    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        create_test_image(test_image)

    print(f"[*] Görsel test ediliyor: {test_image}")
    result = run_ocr(test_image)

    print("\n" + "=" * 50)
    print("🎯 LightOnOCR-2-1B Çıktısı:")
    print("=" * 50)
    print(result)
    print("=" * 50)
