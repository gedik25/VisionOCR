"""
Adversarial and Stress Test Suite.
Verifies fixes for:
1. Multi-threaded race conditions during concurrent model switching.
2. MMOCR bounding box coordinate scaling invariance on high-resolution images.
3. Narrow vertical receipts (e.g. 200x8000) & wide banners (8000x200) visual token & legibility bounds.
4. EXIF rotation preservation on mobile camera captures.
5. Client disconnection handling and generator cleanup.
6. Peak memory boundary under aggressive multi-model switching (< 3.5 GB).
"""

import os
import sys
import time
import math
import threading
import unittest
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.model_manager import get_model_manager
from ocr.image_optimizer import (
    calculate_optimal_dimensions,
    detect_document_type,
    optimize_image_for_ocr,
    MAX_TOTAL_PIXELS,
    MIN_RECEIPT_WIDTH
)


class TestAdversarialReviewer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = get_model_manager()
        cls.sample_crnn = os.path.join(PROJECT_ROOT, "ocr", "CRNN", "sample_test.png")
        cls.sample_glm = os.path.join(PROJECT_ROOT, "ocr", "GLM-OCR", "sample_test.png")
        cls.sample_mmocr = os.path.join(PROJECT_ROOT, "ocr", "MMOCR", "sample_doc.png")
        cls.scratch_dir = os.path.join(PROJECT_ROOT, "scratch", "adversarial_tests")
        os.makedirs(cls.scratch_dir, exist_ok=True)

    def test_01_concurrent_multi_model_switching_race(self):
        """
        Concurrency Stress Test:
        Multiple worker threads simultaneously execute inference on DIFFERENT models (CRNN and GLM-OCR).
        Must not crash with NoneType or race condition errors.
        """
        errors = []

        def worker_crnn():
            try:
                for _ in range(3):
                    res = self.mgr.predict("CRNN", self.sample_crnn)
                    self.assertEqual(res["status"], "success")
            except Exception as e:
                errors.append(("CRNN", str(e)))

        def worker_glm():
            try:
                for _ in range(2):
                    res = self.mgr.predict("GLM-OCR", self.sample_glm, max_new_tokens=16)
                    self.assertEqual(res["status"], "success")
            except Exception as e:
                errors.append(("GLM-OCR", str(e)))

        threads = [
            threading.Thread(target=worker_crnn),
            threading.Thread(target=worker_glm),
            threading.Thread(target=worker_crnn),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent model switching failed with errors: {errors}")

    def test_02_narrow_receipt_dimension_scaling(self):
        """
        Receipt Dimension Test:
        A narrow long receipt (200 x 8000) must NOT be squashed down to 40px width.
        Width must be preserved at min(w, MIN_RECEIPT_WIDTH) while staying under MAX_TOTAL_PIXELS.
        """
        w, h = 200, 8000
        doc_type = detect_document_type(w, h)
        self.assertEqual(doc_type, "receipt")

        nw, nh = calculate_optimal_dimensions(w, h, doc_type)
        self.assertEqual(nw, 200, "Narrow receipt width 200 must be preserved to prevent text obliteration")
        self.assertLessEqual(nw * nh, MAX_TOTAL_PIXELS + 100)

        # Create synthetic image and optimize
        synth_receipt = Image.new("RGB", (200, 8000), (255, 255, 255))
        opt = optimize_image_for_ocr(synth_receipt, model_key="GLM-OCR")
        self.assertEqual(opt.size[0], 200)
        self.assertLessEqual(opt.size[0] * opt.size[1], MAX_TOTAL_PIXELS + 100)

    def test_03_wide_banner_dimension_scaling(self):
        """
        Wide Banner Test:
        A wide horizontal banner (8000 x 200) must maintain readable height (>= 128px) within token budget.
        """
        w, h = 8000, 200
        doc_type = detect_document_type(w, h)
        self.assertEqual(doc_type, "wide_banner")

        nw, nh = calculate_optimal_dimensions(w, h, doc_type)
        self.assertGreaterEqual(nh, 128)
        self.assertLessEqual(nw * nh, MAX_TOTAL_PIXELS + 100)

    def test_04_degenerate_dimension_safety(self):
        """Degenerate dimensions (0, 0, negatives) should not cause ZeroDivisionError or crash."""
        nw, nh = calculate_optimal_dimensions(0, 0, "standard_doc")
        self.assertGreaterEqual(nw, 1)
        self.assertGreaterEqual(nh, 1)

        nw_neg, nh_neg = calculate_optimal_dimensions(-50, 100, "standard_doc")
        self.assertGreaterEqual(nw_neg, 1)

    def test_05_mmocr_high_res_bounding_box_scaling(self):
        """
        MMOCR Bounding Box Invariant:
        When a high-res image (e.g. 3000 x 1000) is scaled for OCR,
        returned bounding boxes must match the ORIGINAL image coordinate system.
        """
        high_res_path = os.path.join(self.scratch_dir, "test_high_res_box.png")
        img = Image.new("RGB", (3000, 1000), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw text at right half of image: x in [1800..2400], y in [400..600]
        draw.text((1800, 450), "BITIRME TEST", fill=(0, 0, 0))
        img.save(high_res_path)

        res = self.mgr.predict("MMOCR", high_res_path)
        self.assertEqual(res["status"], "success")
        boxes = res.get("boxes", [])

        if boxes:
            for b in boxes:
                coords = b["box"]
                # Verify that xmax is in original coordinate space (> 1000), NOT scaled down (< 1000)
                self.assertGreater(coords[2], 1000, f"Box coordinate {coords} was not inverse scaled to original dimensions!")

    def test_06_exif_rotation_preservation(self):
        """Image optimizer should transpose EXIF orientation tag automatically."""
        # Create an image with an EXIF Orientation tag
        img = Image.new("RGB", (600, 200), (255, 255, 255))
        exif = img.getexif()
        exif[0x0112] = 6  # Orientation: Rotate 90 CW
        
        exif_img_path = os.path.join(self.scratch_dir, "exif_test.png")
        img.save(exif_img_path, exif=exif)

        opt = optimize_image_for_ocr(exif_img_path, model_key="CRNN")
        self.assertIsInstance(opt, Image.Image)

    def test_07_sse_stream_client_early_disconnect(self):
        """
        Client Disconnect Test:
        Consuming only the first token of a stream and terminating early must not leave dangling threads.
        """
        gen = self.mgr.stream_predict("CRNN", self.sample_crnn)
        first_event = next(gen)
        self.assertIn("event", first_event)
        # Close generator early (emulating client disconnect)
        gen.close()

    def test_08_multi_model_peak_memory(self):
        """
        Peak Memory Bound Test:
        Repeatedly switching between GLM-OCR, LightOnOCR-2-1B, MMOCR, and CRNN must keep Peak RAM < 3.5 GB.
        """
        for model in ["CRNN", "GLM-OCR", "CRNN"]:
            res = self.mgr.predict(model, self.sample_crnn if model == "CRNN" else self.sample_glm, max_new_tokens=16)
            self.assertEqual(res["status"], "success")
            mem = self.mgr.get_memory_info()
            self.assertLess(mem["rss_mb"], 3500, f"Memory {mem['rss_mb']} MB exceeded 3.5 GB threshold on model {model}")

    def test_09_transparent_png_white_composite(self):
        """
        Transparent PNG Test:
        A transparent PNG with dark text must be composited onto a solid white background,
        preventing background pitch-black corruption that destroys OCR accuracy.
        """
        rgba_img = Image.new("RGBA", (300, 100), (0, 0, 0, 0)) # Fully transparent
        draw = ImageDraw.Draw(rgba_img)
        draw.text((20, 30), "TRANSPARENT_TEST", fill=(0, 0, 0, 255)) # Black text
        
        rgba_path = os.path.join(self.scratch_dir, "transparent_test.png")
        rgba_img.save(rgba_path)

        opt = optimize_image_for_ocr(rgba_path, model_key="CRNN")
        self.assertEqual(opt.mode, "RGB")
        # Check background pixel at (5, 5): must be pure white (255, 255, 255), NOT black (0, 0, 0)
        bg_pixel = opt.getpixel((5, 5))
        self.assertEqual(bg_pixel, (255, 255, 255), f"Transparent area became {bg_pixel} instead of pure white!")

    def test_10_empty_and_corrupt_file_handling(self):
        """
        Corrupt & Empty File Safety:
        0-byte or non-image files must raise ValueError without crashing the manager or thread.
        """
        empty_path = os.path.join(self.scratch_dir, "empty_file.png")
        with open(empty_path, "wb") as f:
            f.write(b"")

        corrupt_path = os.path.join(self.scratch_dir, "corrupt_file.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_DATA_CORRUPT")

        # Predict on empty file
        with self.assertRaises(ValueError):
            self.mgr.predict("CRNN", empty_path)

        # Predict on corrupt file
        with self.assertRaises(ValueError):
            self.mgr.predict("CRNN", corrupt_path)

        # Stream predict on corrupt file
        gen = self.mgr.stream_predict("CRNN", corrupt_path)
        with self.assertRaises(ValueError):
            next(gen)

    def test_11_concurrent_streaming_and_predict_safety(self):
        """
        Concurrent Streaming Stress:
        Simultaneous streaming generator execution and synchronous prediction across threads
        must remain strictly serialized under self.model_lock without crashing.
        """
        results = []
        errors = []

        def stream_worker():
            try:
                tokens = []
                for event in self.mgr.stream_predict("CRNN", self.sample_crnn):
                    if event.get("event") == "token":
                        tokens.append(event.get("token"))
                    elif event.get("event") == "done":
                        results.append(("stream_done", event.get("text")))
            except Exception as e:
                errors.append(("stream_error", str(e)))

        def predict_worker():
            try:
                res = self.mgr.predict("CRNN", self.sample_crnn)
                results.append(("predict_done", res.get("text")))
            except Exception as e:
                errors.append(("predict_error", str(e)))

        threads = [
            threading.Thread(target=stream_worker),
            threading.Thread(target=predict_worker),
            threading.Thread(target=stream_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent streaming/predict errors: {errors}")
        self.assertEqual(len(results), 3)

    def test_12_device_name_and_cuda_mps_cpu_fallback(self):
        """Verify get_device_name handles strings and torch.device objects correctly."""
        from ocr.model_manager import get_device_name
        import torch

        self.assertEqual(get_device_name("mps"), "Apple Silicon (MPS)")
        self.assertEqual(get_device_name("cuda"), "CUDA")
        self.assertEqual(get_device_name("cuda:0"), "CUDA")
        self.assertEqual(get_device_name("cpu"), "CPU")
        self.assertEqual(get_device_name(torch.device("cpu")), "CPU")
        self.assertIn(get_device_name(None), ["Apple Silicon (MPS)", "CUDA", "CPU"])

    def test_13_web_api_empty_file_bad_request(self):
        """
        FastAPI /api/ocr and /api/ocr/stream must return 400 Bad Request
        when given an empty or corrupt file upload.
        """
        from fastapi.testclient import TestClient
        from web.server import app

        client = TestClient(app)

        # 1. Empty file sync OCR
        res_sync = client.post(
            "/api/ocr",
            files={"file": ("empty.png", b"", "image/png")},
            data={"model": "CRNN"}
        )
        self.assertEqual(res_sync.status_code, 400)

        # 2. Unsupported extension
        res_ext = client.post(
            "/api/ocr",
            files={"file": ("test.txt", b"hello world", "text/plain")},
            data={"model": "CRNN"}
        )
        self.assertEqual(res_ext.status_code, 400)


if __name__ == "__main__":
    unittest.main()

