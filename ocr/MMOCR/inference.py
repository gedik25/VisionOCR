import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont

def get_device():
    """Cihazı belirler (CUDA varsa GPU, macOS/ARM için kararlı CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def run_mmocr(image_path, det='DBNet', rec='CRNN', save_vis=True, out_dir=None):
    """
    MMOCR ile görsel üzerindeki metinleri tespit eder ve tanır.
    det: 'DBNet', 'DBNetpp', 'TextSnake', vb.
    rec: 'CRNN', 'SAR', 'ABINet', 'MASTER', vb.
    """
    from mmocr.apis import MMOCRInferencer

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    print(f"[*] MMOCR Inferencer başlatılıyor (Det: {det}, Rec: {rec}, Cihaz: {device})...")

    inferencer = MMOCRInferencer(det=det, rec=rec, device=device)

    print(f"[*] Görsel işleniyor: {image_path}")
    result = inferencer(image_path, save_vis=save_vis, save_pred=True, out_dir=out_dir)

    # MMOCR sonucundan metinleri ayıkla
    predictions = result.get('predictions', [])
    recognized_texts = []
    if predictions:
        pred_item = predictions[0]
        rec_texts = pred_item.get('rec_texts', [])
        rec_scores = pred_item.get('rec_scores', [])
        for text, score in zip(rec_texts, rec_scores):
            recognized_texts.append(f"{text} (Güven: %{score*100:.1f})")

    return {
        "texts": rec_texts if predictions else [],
        "scores": rec_scores if predictions else [],
        "formatted": recognized_texts,
        "raw_result": result,
        "out_dir": out_dir
    }

def create_sample_page_image(output_path=None):
    """MMOCR'ın çoklu satır/kelime tespiti için örnek bir doküman görseli oluşturur."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sample_doc.png")

    img = Image.new('RGB', (450, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 24) if os.path.exists(font_path) else ImageFont.load_default()
    small_font = ImageFont.truetype(font_path, 18) if os.path.exists(font_path) else ImageFont.load_default()

    d.text((25, 25), "BITIRME PROJESI", fill=(0, 0, 0), font=font)
    d.text((25, 75), "OpenMMLab MMOCR Sistemi", fill=(30, 30, 30), font=small_font)
    d.text((25, 120), "Karakter ve Metin Tanima 2026", fill=(50, 50, 50), font=small_font)

    img.save(output_path)
    print(f"[✓] MMOCR test doküman görseli oluşturuldu: {output_path}")
    return output_path

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None

    if test_img is None or not os.path.exists(test_img):
        test_img = os.path.join(os.path.dirname(__file__), "sample_doc.png")
        create_sample_page_image(test_img)

    result_data = run_mmocr(test_img)

    print("\n" + "=" * 45)
    print("🎯 MMOCR Tespit ve Tanıma Sonuçları:")
    print("=" * 45)
    if result_data["formatted"]:
        for idx, item in enumerate(result_data["formatted"], 1):
            print(f"  {idx}. {item}")
    else:
        print("  Metin tespit edilemedi.")
    print("=" * 45)
    print(f"[*] Çıktı görseli kaydedildi: {result_data['out_dir']}\n")
