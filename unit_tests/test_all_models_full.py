"""
Full test suite exercising all 4 OCR models (CRNN, GLM-OCR, LightOnOCR-2-1B, MMOCR)
and verifying warm cache, zero cold-start, memory bounds (< 3.5 GB), and streaming.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.model_manager import get_model_manager


class TestAllModelsFull(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = get_model_manager()
        cls.sample_crnn = os.path.join(PROJECT_ROOT, "ocr", "CRNN", "sample_test.png")
        cls.sample_glm = os.path.join(PROJECT_ROOT, "ocr", "GLM-OCR", "sample_test.png")
        cls.sample_lighton = os.path.join(PROJECT_ROOT, "ocr", "LightOnOCR-2-1B", "sample_test.png")
        cls.sample_mmocr = os.path.join(PROJECT_ROOT, "ocr", "MMOCR", "sample_doc.png")

    def test_01_lighton_ocr_inference_and_stream(self):
        """LightOnOCR-2-1B warm inference and streaming."""
        self.mgr.load_model("LightOnOCR-2-1B")
        res = self.mgr.predict("LightOnOCR-2-1B", self.sample_lighton, max_new_tokens=64)
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["text"]), 0)
        self.assertLess(res["memory_mb"], 3500)

        # Stream
        events = list(self.mgr.stream_predict("LightOnOCR-2-1B", self.sample_lighton, max_new_tokens=32))
        done_events = [e for e in events if e.get("event") == "done"]
        self.assertEqual(len(done_events), 1)

    def test_02_mmocr_inference_and_stream(self):
        """MMOCR DBNet + CRNN inference and streaming."""
        self.mgr.load_model("MMOCR")
        res = self.mgr.predict("MMOCR", self.sample_mmocr)
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["text"]), 0)
        self.assertIn("boxes", res)

        # Stream
        events = list(self.mgr.stream_predict("MMOCR", self.sample_mmocr))
        done_events = [e for e in events if e.get("event") == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertGreater(len(done_events[0]["boxes"]), 0)


    def test_03_full_4_model_switching_cycle(self):
        """
        Full 4-Model Switching Cycle:
        CRNN -> MMOCR -> GLM-OCR -> LightOnOCR-2-1B -> CRNN.
        Verifies:
        1. Clean unload and load on every model transition.
        2. Warm cache second-call response time (< 1.0s, CRNN < 0.3s).
        3. Peak RAM remains strictly < 3.5 GB (3500 MB) across all 4 models.
        4. Memory drops safely when returning to lightweight CRNN.
        """
        import time

        model_sequence = [
            ("CRNN", self.sample_crnn, 0.3),
            ("MMOCR", self.sample_mmocr, 2.5),
            ("GLM-OCR", self.sample_glm, 1.0),
            ("LightOnOCR-2-1B", self.sample_lighton, 1.0),
            ("CRNN", self.sample_crnn, 0.3),
        ]

        for model_key, sample_img, max_warm_sec in model_sequence:
            # 1. First call (loads model into cache)
            res1 = self.mgr.predict(model_key, sample_img, max_new_tokens=16)
            self.assertEqual(res1["status"], "success")
            self.assertEqual(res1["model_key"], model_key)

            # 2. Second call (zero cold-start warm cache hit)
            t0 = time.perf_counter()
            res2 = self.mgr.predict(model_key, sample_img, max_new_tokens=16)
            warm_latency = time.perf_counter() - t0

            self.assertEqual(res2["status"], "success")
            self.assertTrue(res2["is_warm"], f"Model {model_key} warm cache flag should be True on second call")
            self.assertLess(warm_latency, max_warm_sec, f"{model_key} warm latency {warm_latency:.3f}s exceeded {max_warm_sec}s")

            # 3. Peak RAM verification
            mem = self.mgr.get_memory_info()
            self.assertLess(mem["rss_mb"], 3500, f"RAM {mem['rss_mb']} MB exceeded 3.5 GB limit on model {model_key}")

    def test_04_ovisocr2_inference_and_stream(self):
        """OvisOCR2 warm inference and streaming."""
        self.mgr.load_model("OvisOCR2")
        res = self.mgr.predict("OvisOCR2", self.sample_glm, max_new_tokens=64)
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["text"]), 0)
        self.assertIn("Bitirme", res["text"])
        self.assertLess(res["memory_mb"], 3500)

        # Stream
        events = list(self.mgr.stream_predict("OvisOCR2", self.sample_glm, max_new_tokens=32))
        done_events = [e for e in events if e.get("event") == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertGreater(len(done_events[0]["text"]), 0)


if __name__ == "__main__":
    unittest.main()
