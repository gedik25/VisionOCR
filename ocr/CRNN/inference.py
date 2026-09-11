import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

def load_crnn_model(device=None):
    """
    CRNN modelini PyTorch Hub (baudm/parseq Scene Text Recognition Benchmark)
    üzerinden önceden eğitilmiş ağırlıklarıyla yükler.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    print(f"[*] CRNN modeli yükleniyor (Cihaz: {device})...")
    # baudm/parseq reposu CRNN mimarisini ve pretrained ağırlıklarını içerir
    model = torch.hub.load('baudm/parseq', 'crnn', pretrained=True, trust_repo=True).eval().to(device)
    
    # Modelin resmi preprocess dönüşümü
    img_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    return model, img_transform, device

def predict(image_path, model=None, img_transform=None, device=None):
    """Verilen görseldeki metni tanır ve olasılık değerleriyle döndürür."""
    if model is None:
        model, img_transform, device = load_crnn_model(device)

    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):
        image = image_path.convert('RGB')
    else:
        raise ValueError("image_path bir dosya yolu veya PIL.Image nesnesi olmalıdır.")

    image_tensor = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = logits.softmax(-1)
        label, _ = model.tokenizer.decode(probs)

    recognized_text = label[0] if isinstance(label, list) else str(label)
    return recognized_text

def create_sample_test_image(text="PROJECT", output_path=None):
    """Test amaçlı sentetik bir kelime görseli üretir."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sample_test.png")
    
    # macOS ve Linux için standart font kontrolü
    font = None
    for fp in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 32)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    img = Image.new('RGB', (280, 50), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((15, 8), text, fill=(0, 0, 0), font=font)
    img.save(output_path)
    print(f"[✓] Sentetik test görseli oluşturuldu: {output_path} (İçerik: '{text}')")
    return output_path

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None

    if test_img is None or not os.path.exists(test_img):
        test_img = os.path.join(os.path.dirname(__file__), "sample_test.png")
        create_sample_test_image("graduation", test_img)

    print(f"[*] Görsel test ediliyor: {test_img}")
    model, img_transform, device = load_crnn_model()
    result = predict(test_img, model, img_transform, device)
    
    print("\n" + "=" * 45)
    print(f"🎯 CRNN OCR Sonucu: '{result}'")
    print("=" * 45 + "\n")
