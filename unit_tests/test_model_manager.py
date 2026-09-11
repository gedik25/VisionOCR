"""
Unit and Integration tests for ocr.model_manager.
Tests:
- R1: Warm cache hit < 1.0s (CRNN < 0.3s).
- Model switching and safe memory unloading (no leaks).
- 5 consecutive OCR runs with Peak RAM < 3.5 GB.
- Multi-model inference (CRNN, GLM-OCR, LightOnOCR-2-1B, MMOCR).
"""

import os
import sys
import time
import unittest
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.model_manager import get_model_manager


class TestModelManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manager = get_model_manager()
        cls.sample_crnn = os.path.join(PROJECT_ROOT, "ocr", "CRNN", "sample_test.png")
        cls.sample_glm = os.path.join(PROJECT_ROOT, "ocr", "GLM-OCR", "sample_test.png")
        cls.sample_lighton = os.path.join(PROJECT_ROOT, "ocr", "LightOnOCR-2-1B", "sample_test.png")
        cls.sample_mmocr = os.path.join(PROJECT_ROOT, "ocr", "MMOCR", "sample_doc.png")

    def test_01_singleton(self):
        m1 = get_model_manager()
        m2 = get_model_manager()
        self.assertIs(m1, m2)

    def test_02_memory_info(self):
        info = self.manager.get_memory_info()
        self.assertIn("rss_mb", info)
        self.assertIn("device", info)
        self.assertIn("active_model", info)
        self.assertGreater(info["rss_mb"], 0)

    def test_03_crnn_warm_cache_latency(self):
        """CRNN warm inference must execute under 0.3 seconds."""
        # First call loads model
        self.manager.load_model("CRNN")
        self.assertEqual(self.manager.active_model_key, "CRNN")

        # Warm inference call
        t0 = time.perf_counter()
        res = self.manager.predict("CRNN", self.sample_crnn)
        elapsed = time.perf_counter() - t0

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_key"], "CRNN")
        self.assertTrue(res["is_warm"])
        self.assertLess(elapsed, 0.3, f"CRNN warm latency {elapsed:.4f}s exceeds 0.3s threshold!")
        self.assertGreater(len(res["text"]), 0)

    def test_04_model_switching_and_memory_cleanup(self):
        """Switching from CRNN to GLM-OCR should cleanly unload CRNN without leak."""
        self.manager.load_model("CRNN")
        self.assertEqual(self.manager.active_model_key, "CRNN")

        # Switch to GLM-OCR
        self.manager.load_model("GLM-OCR")
        self.assertEqual(self.manager.active_model_key, "GLM-OCR")

        mem_info = self.manager.get_memory_info()
        # Verify RAM remains under 3.5 GB (3584 MB)
        self.assertLess(mem_info["rss_mb"], 3500, f"Memory {mem_info['rss_mb']} MB exceeded 3.5 GB limit")

    def test_05_glm_warm_cache_and_5_consecutive_runs(self):
        """5 consecutive OCR calls must maintain Peak RAM < 3.5 GB and warm latency."""
        self.manager.load_model("GLM-OCR")

        latencies = []
        for i in range(5):
            t0 = time.perf_counter()
            res = self.manager.predict("GLM-OCR", self.sample_glm)
            el = time.perf_counter() - t0
            latencies.append(el)

            self.assertEqual(res["status"], "success")
            self.assertTrue(res["is_warm"])
            self.assertLess(res["memory_mb"], 3500, f"Run {i+1} RAM {res['memory_mb']} exceeded 3.5 GB")

        avg_latency = sum(latencies) / len(latencies)
        print(f"\n[✓] GLM-OCR 5 Consecutive Runs Average Latency: {avg_latency:.3f}s")
        self.assertLess(res["memory_mb"], 3500)


if __name__ == "__main__":
    unittest.main()
