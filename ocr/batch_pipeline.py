"""
VisionOCR High-Performance Batch Processing Pipeline.
Processes folders of receipts and documents using the warm model cache,
exports structured per-file outputs (.json, .txt), generates summary reports
(summary.json, summary.csv), and creates downloadable ZIP packages.
"""

import os
import sys
import csv
import json
import time
import shutil
import zipfile
import datetime
from typing import List, Dict, Any, Optional, Callable, Generator
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "scratch", "matplotlib"))

from ocr.model_manager import get_model_manager

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

MODEL_KEY_MAP = {
    "glm-ocr": "GLM-OCR",
    "glm_ocr": "GLM-OCR",
    "glm": "GLM-OCR",
    "lightonocr-2-1b": "LightOnOCR-2-1B",
    "lightonocr": "LightOnOCR-2-1B",
    "lighton": "LightOnOCR-2-1B",
    "mmocr": "MMOCR",
    "crnn": "CRNN",
}


def normalize_model_key(key: str) -> str:
    cleaned = (key or "").strip()
    return MODEL_KEY_MAP.get(cleaned.lower(), cleaned)


def find_image_files(input_dir: str, recursive: bool = False) -> List[str]:
    """
    Scans a directory for supported image files.
    Returns sorted list of absolute file paths.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Girdi klasörü bulunamadı: {input_dir}")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Girdi bir klasör değil: {input_dir}")

    image_paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            full_path = os.path.join(input_dir, f)
            if os.path.isfile(full_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_paths.append(full_path)

    image_paths.sort()
    return image_paths


def _make_serializable(obj: Any) -> Any:
    """Recursively converts numpy types, tensors, sets, and paths to native JSON-serializable types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return _make_serializable(obj.tolist())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, set):
        return [_make_serializable(x) for x in sorted(obj, key=str)]
    return str(obj)


class BatchPipeline:
    """
    Batch OCR Pipeline Engine.
    Executes warm-cache inference over multiple images, handles faulty files gracefully,
    and automatically outputs structured per-item and summary files.
    """

    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        model_key: str = "GLM-OCR",
        output_format: str = "all",  # "all", "json", "txt"
        recursive: bool = False
    ):
        self.input_dir = os.path.abspath(input_dir) if input_dir else None
        self.output_dir = os.path.abspath(output_dir) if output_dir else None
        self.model_key = normalize_model_key(model_key)
        
        norm_fmt = (output_format or "all").lower().strip()
        valid_formats = {"all", "json", "txt"}
        if norm_fmt not in valid_formats:
            raise ValueError(f"Geçersiz çıktı formatı: '{output_format}'. Desteklenen formatlar: {', '.join(sorted(valid_formats))}")
        self.output_format = norm_fmt
        self.recursive = recursive
        self.mgr = get_model_manager()

    def process_single_image(
        self,
        image_path: str,
        output_dir: str,
        model_key: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a single image file, saves output files, and returns result item dictionary.
        Does not raise exceptions on corrupt/invalid images to ensure pipeline continuity.
        Preserves subdirectory structures when input_dir is defined.
        """
        active_model = normalize_model_key(model_key or self.model_key)
        active_format = (output_format or self.output_format).lower().strip()
        os.makedirs(output_dir, exist_ok=True)

        abs_img_path = os.path.abspath(image_path)
        filename = os.path.basename(abs_img_path)
        stem, _ = os.path.splitext(filename)
        start_time = time.perf_counter()

        # Preserve relative subdirectory if input_dir is set
        if self.input_dir:
            try:
                rel_path = os.path.relpath(abs_img_path, self.input_dir)
                if rel_path.startswith("..") or os.path.isabs(rel_path):
                    rel_path = filename
            except ValueError:
                rel_path = filename
        else:
            rel_path = filename

        rel_dir = os.path.dirname(rel_path) if rel_path != filename else ""
        target_item_dir = os.path.join(output_dir, rel_dir) if rel_dir else output_dir
        os.makedirs(target_item_dir, exist_ok=True)

        json_filename = f"{stem}.json"
        txt_filename = f"{stem}.txt"
        json_path = os.path.join(target_item_dir, json_filename)
        txt_path = os.path.join(target_item_dir, txt_filename)

        rel_json_path = os.path.join(rel_dir, json_filename) if rel_dir else json_filename
        rel_txt_path = os.path.join(rel_dir, txt_filename) if rel_dir else txt_filename

        item_result: Dict[str, Any] = {
            "filename": filename,
            "relative_path": rel_path,
            "stem": stem,
            "status": "error",
            "model_key": active_model,
            "elapsed_sec": 0.0,
            "char_count": 0,
            "word_count": 0,
            "preview_text": "",
            "json_file": rel_json_path if active_format in ("all", "json") else None,
            "txt_file": rel_txt_path if active_format in ("all", "txt") else None,
            "error": None
        }

        try:
            # 1. Validation checks
            if not os.path.isfile(abs_img_path):
                raise FileNotFoundError(f"Görsel dosyası bulunamadı: {abs_img_path}")

            file_size = os.path.getsize(abs_img_path)
            if file_size == 0:
                raise ValueError("Görsel dosyası 0 bayt (boş dosya).")

            # Verify image can be parsed by PIL
            try:
                with Image.open(abs_img_path) as test_img:
                    test_img.verify()
            except Exception as pil_err:
                raise ValueError(f"Bozuk veya geçersiz görsel formatı: {pil_err}")

            # 2. Run warm prediction
            pred = self.mgr.predict(active_model, abs_img_path)
            elapsed = time.perf_counter() - start_time

            extracted_text = pred.get("text", "")
            words = extracted_text.strip().split() if extracted_text else []
            preview = " ".join(extracted_text.split()[:20]) if extracted_text else ""

            item_result.update({
                "status": "success",
                "elapsed_sec": round(elapsed, 4),
                "char_count": len(extracted_text),
                "word_count": len(words),
                "preview_text": preview,
                "text": extracted_text,
                "markdown": pred.get("markdown", ""),
                "boxes": _make_serializable(pred.get("boxes", [])),
                "device": pred.get("device", ""),
                "is_warm": pred.get("is_warm", True),
                "error": None
            })

            # 3. Write individual output files
            if active_format in ("all", "json"):
                full_json_payload = {
                    "filename": filename,
                    "relative_path": rel_path,
                    "stem": stem,
                    "status": "success",
                    "model_key": active_model,
                    "text": extracted_text,
                    "markdown": pred.get("markdown", ""),
                    "boxes": _make_serializable(pred.get("boxes", [])),
                    "raw": _make_serializable(pred.get("raw", {})),
                    "device": pred.get("device", ""),
                    "elapsed_sec": round(elapsed, 4),
                    "char_count": len(extracted_text),
                    "word_count": len(words),
                    "processed_at": datetime.datetime.now().isoformat()
                }
                try:
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(_make_serializable(full_json_payload), jf, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pass

            if active_format in ("all", "txt"):
                try:
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(extracted_text)
                except Exception:
                    pass

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            err_msg = str(e)
            item_result.update({
                "status": "error",
                "elapsed_sec": round(elapsed, 4),
                "error": err_msg,
                "text": "",
                "markdown": "",
                "boxes": []
            })

            # Even on error, write output files if requested
            if active_format in ("all", "json"):
                err_json_payload = {
                    "filename": filename,
                    "relative_path": rel_path,
                    "stem": stem,
                    "status": "error",
                    "model_key": active_model,
                    "error": err_msg,
                    "elapsed_sec": round(elapsed, 4),
                    "processed_at": datetime.datetime.now().isoformat()
                }
                try:
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(_make_serializable(err_json_payload), jf, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pass

            if active_format in ("all", "txt"):
                try:
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(f"HATA: {err_msg}")
                except Exception:
                    pass

        return item_result

    def run(
        self,
        image_paths: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes the entire batch job synchronously.
        Preloads the warm model once and runs all images sequentially.
        """
        if not self.output_dir:
            raise ValueError("Çıktı klasörü (output_dir) belirtilmelidir.")

        os.makedirs(self.output_dir, exist_ok=True)

        if image_paths is None:
            if not self.input_dir:
                raise ValueError("Girdi klasörü (input_dir) veya görsel listesi (image_paths) belirtilmelidir.")
            image_paths = find_image_files(self.input_dir, recursive=self.recursive)

        total_images = len(image_paths)
        batch_start_time = time.perf_counter()

        # Warm up the model once before the batch loop
        if total_images > 0:
            self.mgr.load_model(self.model_key)

        results = []
        successful_count = 0
        failed_count = 0

        for idx, img_path in enumerate(image_paths, start=1):
            item_res = self.process_single_image(
                image_path=img_path,
                output_dir=self.output_dir,
                model_key=self.model_key,
                output_format=self.output_format
            )

            if item_res["status"] == "success":
                successful_count += 1
            else:
                failed_count += 1

            results.append(item_res)

            if progress_callback:
                percent = int((idx / total_images) * 100) if total_images > 0 else 100
                progress_callback({
                    "index": idx,
                    "total": total_images,
                    "percent": percent,
                    "filename": item_res["filename"],
                    "status": item_res["status"],
                    "elapsed_sec": item_res["elapsed_sec"],
                    "preview": item_res.get("preview_text", ""),
                    "error": item_res.get("error")
                })

        total_elapsed = round(time.perf_counter() - batch_start_time, 4)
        avg_latency = round(total_elapsed / total_images, 4) if total_images > 0 else 0.0

        summary = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "model": self.model_key,
            "format": self.output_format,
            "total_images": total_images,
            "successful": successful_count,
            "failed": failed_count,
            "total_elapsed_sec": total_elapsed,
            "average_latency_sec": avg_latency,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results
        }

        # Write summary.json and summary.csv
        self.export_summary_json(summary, os.path.join(self.output_dir, "summary.json"))
        self.export_summary_csv(summary, os.path.join(self.output_dir, "summary.csv"))

        return summary

    def stream_run(
        self,
        image_paths: Optional[List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generator for Server-Sent Events (SSE) streaming during batch processing.
        Yields start, progress per item, and done events.
        """
        if not self.output_dir:
            raise ValueError("Çıktı klasörü (output_dir) belirtilmelidir.")

        os.makedirs(self.output_dir, exist_ok=True)

        if image_paths is None:
            if not self.input_dir:
                raise ValueError("Girdi klasörü veya görsel listesi belirtilmelidir.")
            image_paths = find_image_files(self.input_dir, recursive=self.recursive)

        total_images = len(image_paths)
        yield {
            "event": "start",
            "total": total_images,
            "model": self.model_key
        }

        batch_start_time = time.perf_counter()
        if total_images > 0:
            self.mgr.load_model(self.model_key)

        results = []
        successful_count = 0
        failed_count = 0

        for idx, img_path in enumerate(image_paths, start=1):
            item_res = self.process_single_image(
                image_path=img_path,
                output_dir=self.output_dir,
                model_key=self.model_key,
                output_format=self.output_format
            )

            if item_res["status"] == "success":
                successful_count += 1
            else:
                failed_count += 1

            results.append(item_res)
            percent = int((idx / total_images) * 100) if total_images > 0 else 100

            yield {
                "event": "progress",
                "index": idx,
                "total": total_images,
                "percent": percent,
                "filename": item_res["filename"],
                "status": item_res["status"],
                "elapsed_sec": item_res["elapsed_sec"],
                "char_count": item_res["char_count"],
                "word_count": item_res["word_count"],
                "preview": item_res.get("preview_text", ""),
                "error": item_res.get("error")
            }

        total_elapsed = round(time.perf_counter() - batch_start_time, 4)
        avg_latency = round(total_elapsed / total_images, 4) if total_images > 0 else 0.0

        summary = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "model": self.model_key,
            "format": self.output_format,
            "total_images": total_images,
            "successful": successful_count,
            "failed": failed_count,
            "total_elapsed_sec": total_elapsed,
            "average_latency_sec": avg_latency,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results
        }

        self.export_summary_json(summary, os.path.join(self.output_dir, "summary.json"))
        self.export_summary_csv(summary, os.path.join(self.output_dir, "summary.csv"))

        yield {
            "event": "done",
            "summary": summary
        }

    @staticmethod
    def export_summary_json(summary_data: Dict[str, Any], output_path: str) -> str:
        """Exports summary data to formatted JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_make_serializable(summary_data), f, ensure_ascii=False, indent=2, default=str)
        return output_path

    @staticmethod
    def export_summary_csv(summary_data: Dict[str, Any], output_path: str) -> str:
        """
        Exports summary data to an Excel-compatible CSV file (UTF-8 with BOM).
        Ensures quotes, formula injection prevention, and multiline previews are properly handled.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        results = summary_data.get("results", [])
        fieldnames = [
            "filename",
            "status",
            "elapsed_sec",
            "char_count",
            "word_count",
            "preview_text",
            "error"
        ]

        with open(output_path, "w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for r in results:
                raw_preview = str(r.get("preview_text") or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                raw_error = str(r.get("error") or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

                # Excel formula injection protection
                if raw_preview and raw_preview[0] in ("=", "+", "-", "@"):
                    raw_preview = "'" + raw_preview
                if raw_error and raw_error[0] in ("=", "+", "-", "@"):
                    raw_error = "'" + raw_error

                disp_filename = r.get("relative_path") or r.get("filename", "")
                if disp_filename and disp_filename[0] in ("=", "+", "-", "@"):
                    disp_filename = "'" + disp_filename

                elapsed_val = r.get("elapsed_sec")
                char_val = r.get("char_count")
                word_val = r.get("word_count")

                writer.writerow({
                    "filename": disp_filename,
                    "status": r.get("status", ""),
                    "elapsed_sec": elapsed_val if elapsed_val is not None else 0.0,
                    "char_count": char_val if char_val is not None else 0,
                    "word_count": word_val if word_val is not None else 0,
                    "preview_text": raw_preview,
                    "error": raw_error
                })
        return output_path

    @staticmethod
    def create_zip_archive(output_dir: str, zip_path: Optional[str] = None) -> str:
        """
        Creates a zip archive containing all files in the output directory.
        """
        if zip_path is None:
            clean_dir = output_dir.rstrip("/\\")
            base_name = os.path.basename(clean_dir) or "batch_output"
            zip_path = os.path.join(os.path.dirname(clean_dir) or ".", f"{base_name}.zip")

        abs_zip_path = os.path.abspath(zip_path)
        os.makedirs(os.path.dirname(abs_zip_path), exist_ok=True)

        with zipfile.ZipFile(abs_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for f in files:
                    if f.lower().endswith(".zip"):
                        continue
                    full_file_path = os.path.join(root, f)
                    if os.path.abspath(full_file_path) == abs_zip_path:
                        continue
                    rel_path = os.path.relpath(full_file_path, output_dir)
                    zf.write(full_file_path, arcname=rel_path)

        return zip_path


def process_batch(
    input_dir: str,
    output_dir: str,
    model: str = "GLM-OCR",
    output_format: str = "all",
    recursive: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Convenience function for running batch processing.
    """
    pipeline = BatchPipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        model_key=model,
        output_format=output_format,
        recursive=recursive
    )
    return pipeline.run(progress_callback=progress_callback)
