"""
Integration tests for FastAPI Server (web.server).
Tests /api/models, /api/ocr, /api/ocr/stream (SSE), /api/models/preload, /api/models/unload,
and static frontend delivery.
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.server import app

client = TestClient(app)


class TestWebAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_crnn = os.path.join(PROJECT_ROOT, "ocr", "CRNN", "sample_test.png")

    def test_01_get_models(self):
        response = client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("device", data)
        self.assertIn("memory", data)
        self.assertGreaterEqual(len(data["models"]), 4)

    def test_02_static_index(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("VisionOCR", response.text)

    def test_03_ocr_sync_endpoint(self):
        with open(self.sample_crnn, "rb") as f:
            response = client.post(
                "/api/ocr",
                files={"file": ("sample.png", f, "image/png")},
                data={"model": "CRNN"}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_key"], "CRNN")
        self.assertIn("text", data)
        self.assertIn("elapsed_sec", data)

    def test_04_ocr_stream_endpoint(self):
        with open(self.sample_crnn, "rb") as f:
            response = client.post(
                "/api/ocr/stream",
                files={"file": ("sample.png", f, "image/png")},
                data={"model": "CRNN"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))
        
        # Verify event stream content
        body = response.text
        self.assertIn("data:", body)
        self.assertIn('"event": "done"', body)
        self.assertIn('"status": "success"', body)

    def test_05_preload_and_unload(self):
        # Preload
        res_pre = client.post("/api/models/preload", data={"model": "CRNN"})
        self.assertEqual(res_pre.status_code, 200)
        self.assertEqual(res_pre.json()["status"], "success")

        # Unload
        res_un = client.post("/api/models/unload")
        self.assertEqual(res_un.status_code, 200)
        self.assertEqual(res_un.json()["status"], "success")


    def test_06_preload_invalid_model_bad_request(self):
        res = client.post("/api/models/preload", data={"model": "NonExistentModel"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("detail", res.json())

    def test_07_upload_retrieval_and_not_found(self):
        res_not_found = client.get("/uploads/non_existent_image_12345.png")
        self.assertEqual(res_not_found.status_code, 404)


if __name__ == "__main__":
    unittest.main()
