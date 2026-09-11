"""
Integration Tests for VisionOCR Web Batch Endpoints (web.server).
Verifies multi-file upload batch processing, Server-Sent Events (SSE) batch streaming,
ZIP download delivery, summary.json & summary.csv downloads, and error resilience.
"""

import os
import sys
import io
import json
import zipfile
import unittest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "scratch", "matplotlib"))

from web.server import app

client = TestClient(app)


def make_test_image_bytes(text: str = "Test Receipt 15.00 TL") -> bytes:
    img = Image.new("RGB", (180, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestBatchWebAPI(unittest.TestCase):

    def test_01_batch_sync_endpoint(self):
        """Tests POST /api/batch synchronous batch processing endpoint."""
        img1 = make_test_image_bytes("Market 01 25.50 TL")
        img2 = make_test_image_bytes("Market 02 89.90 TL")
        corrupt = b"NOT_A_VALID_IMAGE_DATA"

        files = [
            ("files", ("receipt_01.png", img1, "image/png")),
            ("files", ("receipt_02.png", img2, "image/png")),
            ("files", ("corrupt.png", corrupt, "image/png")),
        ]

        response = client.post("/api/batch", files=files, data={"model": "CRNN", "format": "all"})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertIn("batch_id", data)
        self.assertIn("zip_url", data)
        self.assertIn("csv_url", data)
        self.assertIn("json_url", data)

        summary = data["summary"]
        self.assertEqual(summary["total_images"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["failed"], 1)

        batch_id = data["batch_id"]

        # Test GET /api/batch/download/{batch_id}
        zip_res = client.get(f"/api/batch/download/{batch_id}")
        self.assertEqual(zip_res.status_code, 200)
        self.assertEqual(zip_res.headers.get("content-type"), "application/zip")

        # Verify ZIP contains expected files
        zip_buf = io.BytesIO(zip_res.content)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("summary.json", namelist)
            self.assertIn("summary.csv", namelist)
            self.assertIn("receipt_01.json", namelist)
            self.assertIn("receipt_01.txt", namelist)
            self.assertIn("corrupt.json", namelist)

        # Test GET /api/batch/summary/{batch_id}
        summary_res = client.get(f"/api/batch/summary/{batch_id}")
        self.assertEqual(summary_res.status_code, 200)
        self.assertEqual(summary_res.json()["successful"], 2)

        # Test GET /api/batch/csv/{batch_id}
        csv_res = client.get(f"/api/batch/csv/{batch_id}")
        self.assertEqual(csv_res.status_code, 200)
        self.assertIn("text/csv", csv_res.headers.get("content-type", ""))
        self.assertIn("receipt_01.png", csv_res.text)

    def test_02_batch_stream_endpoint(self):
        """Tests POST /api/batch/stream live SSE streaming endpoint."""
        img1 = make_test_image_bytes("Receipt A 10.00 TL")
        img2 = make_test_image_bytes("Receipt B 20.00 TL")

        files = [
            ("files", ("receipt_a.png", img1, "image/png")),
            ("files", ("receipt_b.png", img2, "image/png")),
        ]

        response = client.post("/api/batch/stream", files=files, data={"model": "CRNN", "format": "all"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

        body = response.text
        self.assertIn("data:", body)
        self.assertIn('"event": "start"', body)
        self.assertIn('"event": "progress"', body)
        self.assertIn('"event": "done"', body)
        self.assertIn('"zip_url":', body)

    def test_03_batch_not_found_endpoints(self):
        """Tests 404 error handling for non-existent batch jobs."""
        res_zip = client.get("/api/batch/download/non_existent_job_12345")
        self.assertEqual(res_zip.status_code, 404)

        res_sum = client.get("/api/batch/summary/non_existent_job_12345")
        self.assertEqual(res_sum.status_code, 404)

        res_csv = client.get("/api/batch/csv/non_existent_job_12345")
        self.assertEqual(res_csv.status_code, 404)

    def test_04_batch_empty_upload(self):
        """Tests handling when no files are submitted."""
        res = client.post("/api/batch", files=[], data={"model": "CRNN"})
        self.assertIn(res.status_code, [400, 422])

    def test_05_batch_upload_with_subdirectories_and_non_image_filtering(self):
        """Tests folder upload with subpaths (e.g. folderA/rec.png) and verifies non-images (.txt, .DS_Store) are filtered out."""
        img1 = make_test_image_bytes("Sub A Receipt 50 TL")
        img2 = make_test_image_bytes("Sub B Receipt 75 TL")

        files = [
            ("files", ("subA/receipt.png", img1, "image/png")),
            ("files", ("subB/receipt.png", img2, "image/png")),
            ("files", (".DS_Store", b"DS_STORE_BINARY_DATA", "application/octet-stream")),
            ("files", ("notes.txt", b"just a text note", "text/plain")),
        ]

        response = client.post("/api/batch", files=files, data={"model": "CRNN", "format": "all"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        summary = data["summary"]

        # Only the 2 valid image files should have been processed
        self.assertEqual(summary["total_images"], 2)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["failed"], 0)

        # Download ZIP and verify subfolder contents
        batch_id = data["batch_id"]
        zip_res = client.get(f"/api/batch/download/{batch_id}")
        self.assertEqual(zip_res.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(zip_res.content), "r") as zf:
            namelist = zf.namelist()
            self.assertIn("subA/receipt.json", namelist)
            self.assertIn("subB/receipt.json", namelist)

    def test_06_batch_stream_format_json_only(self):
        """Tests SSE batch stream with format='json'."""
        img = make_test_image_bytes("JSON only receipt")
        files = [("files", ("test_json.png", img, "image/png"))]
        res = client.post("/api/batch/stream", files=files, data={"model": "CRNN", "format": "json"})
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers.get("content-type", ""))
        self.assertIn('"event": "done"', res.text)

    def test_07_batch_id_security_boundary_checks(self):
        """Tests that invalid, traversal, or root batch IDs are rejected with 400."""
        from web.server import get_safe_batch_job_dir, HTTPException

        # Direct validator testing
        for invalid_id in [".", "..", "job/123", "job\\123", "../secret", "bad..id", "", "   "]:
            with self.assertRaises(HTTPException) as ctx:
                get_safe_batch_job_dir(invalid_id)
            self.assertEqual(ctx.exception.status_code, 400)

        # HTTP Endpoint testing with invalid single-segment ID
        for invalid_param in ["bad..id", "traversal..batch"]:
            res_zip = client.get(f"/api/batch/download/{invalid_param}")
            self.assertEqual(res_zip.status_code, 400)

            res_sum = client.get(f"/api/batch/summary/{invalid_param}")
            self.assertEqual(res_sum.status_code, 400)

            res_csv = client.get(f"/api/batch/csv/{invalid_param}")
            self.assertEqual(res_csv.status_code, 400)


if __name__ == "__main__":
    unittest.main()
