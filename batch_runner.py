#!/usr/bin/env python3
"""
VisionOCR Batch Processing CLI Tool.
Processes directories of receipt/document images using warm model cache,
outputs individual .json/.txt files and summary.json/summary.csv tables.

Usage:
    python batch_runner.py --input_dir /path/to/images --output_dir /path/to/output --model GLM-OCR --format all
"""

import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "scratch", "matplotlib"))

from ocr.batch_pipeline import BatchPipeline, process_batch, find_image_files


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


MODEL_ALIASES = {
    "glm-ocr": "GLM-OCR",
    "glm_ocr": "GLM-OCR",
    "lightonocr-2-1b": "LightOnOCR-2-1B",
    "lightonocr": "LightOnOCR-2-1B",
    "mmocr": "MMOCR",
    "crnn": "CRNN",
    "ovisocr2": "OvisOCR2",
    "ovis-ocr2": "OvisOCR2",
    "ovis": "OvisOCR2",
    "dots.ocr": "dots.ocr",
    "dots_ocr": "dots.ocr",
    "dots": "dots.ocr",
}


def parse_model(val: str) -> str:
    key = val.strip()
    norm = MODEL_ALIASES.get(key.lower(), key)
    if norm not in ("GLM-OCR", "LightOnOCR-2-1B", "MMOCR", "CRNN", "OvisOCR2", "dots.ocr"):
        raise argparse.ArgumentTypeError(f"Geçersiz model: '{val}'. Desteklenen modeller: GLM-OCR, LightOnOCR-2-1B, MMOCR, CRNN, OvisOCR2, dots.ocr")
    return norm


def parse_format(val: str) -> str:
    f = val.strip().lower()
    if f not in ("all", "json", "txt"):
        raise argparse.ArgumentTypeError(f"Geçersiz format: '{val}'. Seçenekler: all, json, txt")
    return f


def main():
    parser = argparse.ArgumentParser(
        description="VisionOCR High-Performance Batch Receipt & Document Processing Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Path to folder containing receipt/document images"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Destination directory for individual .json/.txt files and summary reports"
    )
    parser.add_argument(
        "--model",
        type=parse_model,
        default="GLM-OCR",
        help="OCR Model Key (GLM-OCR, LightOnOCR-2-1B, MMOCR, CRNN)"
    )
    parser.add_argument(
        "--format",
        type=parse_format,
        default="all",
        help="Output format: 'all' (both json and txt), 'json', or 'txt'"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for images"
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("\n" + "=" * 65)
    print("⚡ VisionOCR Toplu Fiş & Görsel İşleme Motoru (Batch Runner)")
    print("=" * 65)
    print(f"📁 Girdi Klasörü : {input_dir}")
    print(f"📂 Çıktı Klasörü : {output_dir}")
    print(f"🧠 Model         : {args.model}")
    print(f"📄 Format        : {args.format}")
    print(f"🔍 Özyinelemeli  : {'Evet' if args.recursive else 'Hayır'}")
    print("=" * 65 + "\n")

    if not os.path.isdir(input_dir):
        print(f"❌ Hata: Girdi klasörü bulunamadı: {input_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        image_files = find_image_files(input_dir, recursive=args.recursive)
    except Exception as e:
        print(f"❌ Dosyalar taranırken hata: {e}", file=sys.stderr)
        sys.exit(1)

    total_images = len(image_files)
    if total_images == 0:
        print(f"⚠️ Uyarı: '{input_dir}' içinde desteklenen görsel formatı (PNG, JPG, WEBP, BMP, TIFF) bulunamadı.")
        # Generate empty summary and empty zip
        os.makedirs(output_dir, exist_ok=True)
        pipeline = BatchPipeline(input_dir=input_dir, output_dir=output_dir, model_key=args.model, output_format=args.format)
        pipeline.run(image_paths=[])
        zip_path = BatchPipeline.create_zip_archive(output_dir)
        print(f"✓ Boş özet dosyaları oluşturuldu: {output_dir}")
        print(f"📦 Boş Arşiv ZIP: {zip_path}")
        sys.exit(0)

    print(f"🔍 {total_images} adet görsel tespit edildi. Sıcak model önbelleği yükleniyor...")

    pipeline = BatchPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        model_key=args.model,
        output_format=args.format,
        recursive=args.recursive
    )

    def on_progress(data: dict):
        idx = data["index"]
        total = data["total"]
        pct = data["percent"]
        fname = data["filename"]
        status = data["status"]
        elapsed = data["elapsed_sec"]

        if status == "success":
            status_symbol = "✓"
            status_text = "BAŞARILI"
            print(f"[{idx:3d}/{total:3d}] ({pct:3d}%) {status_symbol} {fname:<30} - {elapsed:.3f}s [{status_text}]")
        else:
            status_symbol = "✗"
            status_text = "HATA"
            err = data.get("error", "Bilinmeyen hata")
            print(f"[{idx:3d}/{total:3d}] ({pct:3d}%) {status_symbol} {fname:<30} - {elapsed:.3f}s [{status_text}: {err}]")

    start_wall_time = time.time()
    summary = pipeline.run(image_paths=image_files, progress_callback=on_progress)
    total_duration = time.time() - start_wall_time

    # Generate ZIP archive as well
    zip_path = BatchPipeline.create_zip_archive(output_dir)

    print("\n" + "=" * 65)
    print("🎉 TOPLU İŞLEM TAMAMLANDI")
    print("=" * 65)
    print(f"📊 Toplam Görsel      : {summary['total_images']}")
    print(f"✅ Başarılı İşlenen   : {summary['successful']}")
    print(f"❌ Hatalı / Atlanan   : {summary['failed']}")
    print(f"⏱️ Toplam Süre        : {format_duration(total_duration)} (Ortalama: {format_duration(summary['average_latency_sec'])} / fiş)")
    print(f"📄 Bireysel Çıktılar  : {output_dir}")
    print(f"📋 Özet JSON          : {os.path.join(output_dir, 'summary.json')}")
    print(f"📊 Özet CSV (Excel)   : {os.path.join(output_dir, 'summary.csv')}")
    print(f"📦 Arşiv ZIP          : {zip_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
