"""
Universal OCR CLI Runner.
Outputs results as JSON to stdout.
"""

import sys
import os
import json
import argparse
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "scratch", "matplotlib"))

from ocr.model_manager import get_model_manager


def run_inference(model_key: str, image_path: str) -> dict:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    mgr = get_model_manager()
    return mgr.predict(model_key, image_path)


def main():
    parser = argparse.ArgumentParser(description="Isolated OCR Runner")
    parser.add_argument("--model", required=True, help="Model Key (CRNN, MMOCR, GLM-OCR, LightOnOCR-2-1B, OvisOCR2, dots.ocr)")
    parser.add_argument("--image", required=True, help="Input Image File Path")
    args = parser.parse_args()

    try:
        result = run_inference(args.model, args.image)
        # Wrap JSON output with special markers for reliable extraction
        print("__OCR_RESULT_START__")
        print(json.dumps(result, ensure_ascii=False))
        print("__OCR_RESULT_END__")
    except Exception as e:
        err_res = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("__OCR_RESULT_START__")
        print(json.dumps(err_res, ensure_ascii=False))
        print("__OCR_RESULT_END__")
        sys.exit(1)


if __name__ == "__main__":
    main()
