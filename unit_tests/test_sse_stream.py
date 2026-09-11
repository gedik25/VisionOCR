"""
Unit and Integration tests for SSE streaming (R2).
Tests stream_predict generator on GLM-OCR, LightOnOCR, and CRNN.
Verifies token deltas, accumulated text, and final 'done' event payload.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ocr.model_manager import get_model_manager


class TestSSEStreaming(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manager = get_model_manager()
        cls.sample_crnn = os.path.join(PROJECT_ROOT, "ocr", "CRNN", "sample_test.png")
        cls.sample_glm = os.path.join(PROJECT_ROOT, "ocr", "GLM-OCR", "sample_test.png")

    def test_crnn_stream(self):
        """CRNN stream should emit token chunks and a final done event."""
        events = list(self.manager.stream_predict("CRNN", self.sample_crnn))
        self.assertGreater(len(events), 1)

        token_events = [e for e in events if e.get("event") == "token"]
        done_events = [e for e in events if e.get("event") == "done"]

        self.assertGreater(len(token_events), 0)
        self.assertEqual(len(done_events), 1)

        done = done_events[0]
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["model_key"], "CRNN")
        self.assertIn("text", done)
        self.assertIn("elapsed_sec", done)
        self.assertLess(done["elapsed_sec"], 0.3)

    def test_glm_stream(self):
        """GLM-OCR stream should emit live decoded tokens and a final done event."""
        events = list(self.manager.stream_predict("GLM-OCR", self.sample_glm, max_new_tokens=64))
        self.assertGreater(len(events), 1)

        token_events = [e for e in events if e.get("event") == "token"]
        done_events = [e for e in events if e.get("event") == "done"]

        self.assertGreater(len(token_events), 0)
        self.assertEqual(len(done_events), 1)

        done = done_events[0]
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["model_key"], "GLM-OCR")
        self.assertGreater(len(done["text"]), 0)
        self.assertIn("markdown", done)
        self.assertIn("device", done)


if __name__ == "__main__":
    unittest.main()
