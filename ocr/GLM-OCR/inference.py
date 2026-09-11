import os
import sys
import gc
import torch
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "zai-org/GLM-OCR"
MAX_IMAGE_EDGE = 1536  # Maksimum kenar boyutu (Görsel token patlamasını ve RAM şişmesini önler)

def get_device():
    """Apple Silicon MPS veya CPU/CUDA seçimi."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resize_image_if_needed(image: Image.Image, max_edge: int = MAX_IMAGE_EDGE) -> Image.Image:
    """
    Yüksek çözünürlüklü ve dikey market fişlerini akıllı ölçekler.
    """
    from ocr.image_optimizer import optimize_image_for_ocr
    return optimize_image_for_ocr(image, model_key="GLM-OCR", max_edge=max_edge)

def load_glm_ocr(model_id=MODEL_ID):
    """GLM-OCR model ve işlemcisini düşük bellek kullanımıyla yerel önbellekten yükler."""
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM

    device = get_device()
    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32

    print(f"[*] GLM-OCR optimize yükleniyor: '{model_id}' (Cihaz: {device}, Veri Tipi: {dtype})...")
    
    # Yerel önbellekten anında yükleme dene
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    try:
        from transformers import GlmOcrForConditionalGeneration
        model = GlmOcrForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(device).eval()
    except Exception:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
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

    # Yükleme sonrası çöp toplayıcı
    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return processor, model, device

def run_glm_ocr(image_path, prompt="Text Recognition:", processor=None, model=None, device=None):
    """Bellek optimizasyonlu GLM-OCR çıkarımı."""
    if processor is None or model is None:
        processor, model, device = load_glm_ocr()

    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")

    # 1. Görseli akıllı ölçekle (RAM patlamasını engeller)
    image = resize_image_if_needed(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    # 2. torch.inference_mode() ile en hafif çıkarım
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
        input_len = inputs['input_ids'].shape[1]
        decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)

    # 3. Çıkarım sonrası belleği serbest bırak
    del inputs
    del output
    gc.collect()
    if device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return decoded.strip()

def create_sample_test_image(output_path=None):
    """GLM-OCR için örnek test görseli oluşturur."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sample_test.png")

    img = Image.new('RGB', (500, 180), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    title_font = ImageFont.truetype(font_path, 28) if os.path.exists(font_path) else ImageFont.load_default()
    body_font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()

    d.text((30, 25), "GLM-OCR Bitirme Projesi", fill=(0, 0, 0), font=title_font)
    d.text((30, 80), "Yapay Zeka Destekli Belge Ayiklama", fill=(30, 30, 30), font=body_font)
    d.text((30, 120), "Tarih: 2026-08-29 | Durum: Aktif", fill=(50, 50, 50), font=body_font)

    img.save(output_path)
    print(f"[✓] GLM-OCR test görseli oluşturuldu: {output_path}")
    return output_path

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None

    if test_img is None or not os.path.exists(test_img):
        test_img = os.path.join(os.path.dirname(__file__), "sample_test.png")
        create_sample_test_image(test_img)

    print(f"[*] Görsel test ediliyor: {test_img}")
    result = run_glm_ocr(test_img)

    print("\n" + "=" * 50)
    print("🎯 GLM-OCR Çıktısı:")
    print("=" * 50)
    print(result)
    print("=" * 50)
