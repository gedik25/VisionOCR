"""
Unit & Integration Tests for Batch Runner & Pipeline (batch_runner.py, ocr/batch_pipeline.py).
Verifies warm model caching, per-file .json/.txt output generation, summary.json & Excel-compatible summary.csv,
fault resilience with 0-byte and corrupted images, and CLI execution.
"""

import os
import sys
import csv
import json
import shutil
import zipfile
import tempfile
import unittest
import subprocess
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "scratch", "matplotlib"))

from ocr.batch_pipeline import BatchPipeline, process_batch, find_image_files
from ocr.model_manager import get_model_manager


class TestBatchRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="visionocr_batch_test_")
        cls.input_dir = os.path.join(cls.temp_dir, "input")
        cls.output_dir = os.path.join(cls.temp_dir, "output")
        os.makedirs(cls.input_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        # Create 8 valid images with varying formats (PNG, JPG, WEBP, BMP, TIFF)
        cls.valid_files = []
        formats = [
            ("receipt_01.png", "PNG"),
            ("receipt_02.jpg", "JPEG"),
            ("receipt_03.jpeg", "JPEG"),
            ("receipt_04.webp", "WEBP"),
            ("receipt_05.bmp", "BMP"),
            ("receipt_06.tiff", "TIFF"),
            ("receipt_07.png", "PNG"),
            ("receipt_08.png", "PNG"),
        ]

        for fname, fmt in formats:
            path = os.path.join(cls.input_dir, fname)
            img = Image.new("RGB", (220, 60), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 20), f"Fis No: {fname} 45.90 TL", fill=(0, 0, 0))
            img.save(path, format=fmt)
            cls.valid_files.append(fname)

        # 1 corrupt image
        cls.corrupt_file = "corrupt_receipt.png"
        corrupt_path = os.path.join(cls.input_dir, cls.corrupt_file)
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_DATA_CORRUPT_BYTES")

        # 1 zero-byte empty image
        cls.empty_file = "empty_zero_byte.png"
        empty_path = os.path.join(cls.input_dir, cls.empty_file)
        with open(empty_path, "wb") as f:
            pass

        # Non-image files to verify scanner filtering
        with open(os.path.join(cls.input_dir, "ignore_me.txt"), "w") as f:
            f.write("text file")
        with open(os.path.join(cls.input_dir, "notes.pdf"), "w") as f:
            f.write("pdf dummy")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_find_image_files_filtering(self):
        """Verifies that find_image_files only returns supported image extensions."""
        files = find_image_files(self.input_dir, recursive=False)
        self.assertEqual(len(files), 10)  # 8 valid + 1 corrupt + 1 empty (all have image ext)
        base_names = [os.path.basename(f) for f in files]
        self.assertNotIn("ignore_me.txt", base_names)
        self.assertNotIn("notes.pdf", base_names)
        self.assertIn("receipt_01.png", base_names)
        self.assertIn("receipt_04.webp", base_names)
        self.assertIn("receipt_06.tiff", base_names)

    def test_02_batch_pipeline_run_crnn(self):
        """Runs 10 images batch with CRNN and verifies warm cache execution and fault resilience."""
        out_dir = os.path.join(self.temp_dir, "crnn_out")
        pipeline = BatchPipeline(
            input_dir=self.input_dir,
            output_dir=out_dir,
            model_key="CRNN",
            output_format="all"
        )

        progress_events = []
        def on_prog(e):
            progress_events.append(e)

        summary = pipeline.run(progress_callback=on_prog)

        self.assertEqual(summary["total_images"], 10)
        self.assertEqual(summary["successful"], 8)
        self.assertEqual(summary["failed"], 2)
        self.assertGreater(summary["total_elapsed_sec"], 0)
        self.assertGreater(summary["average_latency_sec"], 0)
        self.assertEqual(len(progress_events), 10)

        # Verify output files
        # 1. summary.json
        summary_json_path = os.path.join(out_dir, "summary.json")
        self.assertTrue(os.path.isfile(summary_json_path))
        with open(summary_json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
            self.assertEqual(loaded_json["successful"], 8)
            self.assertEqual(loaded_json["failed"], 2)
            self.assertEqual(len(loaded_json["results"]), 10)

        # 2. summary.csv
        summary_csv_path = os.path.join(out_dir, "summary.csv")
        self.assertTrue(os.path.isfile(summary_csv_path))
        with open(summary_csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 10)
            # Check columns
            self.assertIn("filename", rows[0])
            self.assertIn("status", rows[0])
            self.assertIn("elapsed_sec", rows[0])
            self.assertIn("preview_text", rows[0])
            self.assertIn("error", rows[0])

            # Check status counts
            success_rows = [r for r in rows if r["status"] == "success"]
            error_rows = [r for r in rows if r["status"] == "error"]
            self.assertEqual(len(success_rows), 8)
            self.assertEqual(len(error_rows), 2)

        # 3. Individual files
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "receipt_01.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "receipt_01.txt")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "corrupt_receipt.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "empty_zero_byte.json")))

    def test_03_batch_format_filtering(self):
        """Verifies that format='json' creates only json, format='txt' creates only txt."""
        # JSON only
        out_json = os.path.join(self.temp_dir, "json_only_out")
        process_batch(self.input_dir, out_json, model="CRNN", output_format="json")
        self.assertTrue(os.path.isfile(os.path.join(out_json, "receipt_01.json")))
        self.assertFalse(os.path.isfile(os.path.join(out_json, "receipt_01.txt")))

        # TXT only
        out_txt = os.path.join(self.temp_dir, "txt_only_out")
        process_batch(self.input_dir, out_txt, model="CRNN", output_format="txt")
        self.assertTrue(os.path.isfile(os.path.join(out_txt, "receipt_01.txt")))
        self.assertFalse(os.path.isfile(os.path.join(out_txt, "receipt_01.json")))

    def test_04_batch_stream_run(self):
        """Verifies stream_run generator produces valid SSE event dicts."""
        out_dir = os.path.join(self.temp_dir, "stream_out")
        pipeline = BatchPipeline(input_dir=self.input_dir, output_dir=out_dir, model_key="CRNN")

        events = list(pipeline.stream_run())
        self.assertGreaterEqual(len(events), 12)  # start + 10 progress + done
        self.assertEqual(events[0]["event"], "start")
        self.assertEqual(events[0]["total"], 10)
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["summary"]["total_images"], 10)

    def test_05_zip_archive_creation(self):
        """Verifies create_zip_archive bundles all output files properly."""
        out_dir = os.path.join(self.temp_dir, "zip_test_out")
        process_batch(self.input_dir, out_dir, model="CRNN", output_format="all")
        
        zip_path = BatchPipeline.create_zip_archive(out_dir)
        self.assertTrue(os.path.isfile(zip_path))
        self.assertTrue(zip_path.endswith(".zip"))

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("summary.json", namelist)
            self.assertIn("summary.csv", namelist)
            self.assertIn("receipt_01.json", namelist)
            self.assertIn("receipt_01.txt", namelist)

    def test_06_empty_directory_handling(self):
        """Verifies that an empty input directory produces a valid empty summary without errors."""
        empty_dir = os.path.join(self.temp_dir, "empty_dir")
        out_dir = os.path.join(self.temp_dir, "empty_out")
        os.makedirs(empty_dir, exist_ok=True)

        summary = process_batch(empty_dir, out_dir, model="CRNN")
        self.assertEqual(summary["total_images"], 0)
        self.assertEqual(summary["successful"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "summary.json")))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "summary.csv")))

    def test_07_batch_runner_cli_invocation(self):
        """Tests running batch_runner.py via command line subprocess."""
        cli_out = os.path.join(self.temp_dir, "cli_out")
        python_bin = sys.executable

        cmd = [
            python_bin,
            os.path.join(PROJECT_ROOT, "batch_runner.py"),
            "--input_dir", self.input_dir,
            "--output_dir", cli_out,
            "--model", "CRNN",
            "--format", "all"
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI stderr: {proc.stderr}")
        self.assertIn("TOPLU İŞLEM TAMAMLANDI", proc.stdout)
        self.assertTrue(os.path.isfile(os.path.join(cli_out, "summary.json")))
        self.assertTrue(os.path.isfile(os.path.join(cli_out, "summary.csv")))
        self.assertTrue(os.path.isfile(os.path.join(cli_out, "receipt_01.json")))

    def test_08_recursive_batch_with_subdirectories_and_stem_collision(self):
        """Tests that recursive scanning preserves subdirectories and does not overwrite colliding stems."""
        rec_in = os.path.join(self.temp_dir, "rec_in")
        sub1 = os.path.join(rec_in, "sub1")
        sub2 = os.path.join(rec_in, "sub2")
        os.makedirs(sub1, exist_ok=True)
        os.makedirs(sub2, exist_ok=True)

        # Same filename in both subdirectories
        img1 = Image.new("RGB", (150, 40), color=(255, 255, 255))
        ImageDraw.Draw(img1).text((5, 10), "Store 1 Receipt 100 TL", fill=(0, 0, 0))
        img1.save(os.path.join(sub1, "receipt.png"), "PNG")

        img2 = Image.new("RGB", (150, 40), color=(255, 255, 255))
        ImageDraw.Draw(img2).text((5, 10), "Store 2 Receipt 200 TL", fill=(0, 0, 0))
        img2.save(os.path.join(sub2, "receipt.png"), "PNG")

        rec_out = os.path.join(self.temp_dir, "rec_out")
        summary = process_batch(rec_in, rec_out, model="CRNN", output_format="all", recursive=True)
        self.assertEqual(summary["total_images"], 2)
        self.assertEqual(summary["successful"], 2)

        # Both output files should exist in their respective subdirectories
        self.assertTrue(os.path.isfile(os.path.join(rec_out, "sub1", "receipt.json")))
        self.assertTrue(os.path.isfile(os.path.join(rec_out, "sub1", "receipt.txt")))
        self.assertTrue(os.path.isfile(os.path.join(rec_out, "sub2", "receipt.json")))
        self.assertTrue(os.path.isfile(os.path.join(rec_out, "sub2", "receipt.txt")))

        # Verify ZIP contains both subpaths
        zip_path = BatchPipeline.create_zip_archive(rec_out)
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("sub1/receipt.json", namelist)
            self.assertIn("sub2/receipt.json", namelist)

    def test_09_all_errors_input_dir(self):
        """Tests that a folder containing only corrupt and 0-byte files completes safely with 0 successes."""
        bad_in = os.path.join(self.temp_dir, "bad_in")
        bad_out = os.path.join(self.temp_dir, "bad_out")
        os.makedirs(bad_in, exist_ok=True)

        with open(os.path.join(bad_in, "bad1.png"), "wb") as f:
            f.write(b"CORRUPT_HEADER_XYZ")
        with open(os.path.join(bad_in, "bad2.jpg"), "wb") as f:
            pass  # 0 bytes

        summary = process_batch(bad_in, bad_out, model="CRNN", output_format="all")
        self.assertEqual(summary["total_images"], 2)
        self.assertEqual(summary["successful"], 0)
        self.assertEqual(summary["failed"], 2)
        self.assertTrue(os.path.isfile(os.path.join(bad_out, "summary.json")))
        self.assertTrue(os.path.isfile(os.path.join(bad_out, "summary.csv")))

    def test_10_csv_excel_formula_and_newline_escaping(self):
        """Tests that summary data with formula injection triggers and newlines is safely sanitized in CSV."""
        out_csv = os.path.join(self.temp_dir, "safe_test.csv")
        summary_data = {
            "results": [
                {
                    "filename": "formula.png",
                    "status": "success",
                    "elapsed_sec": 0.12,
                    "char_count": 10,
                    "word_count": 2,
                    "preview_text": "=SUM(1,2)\nSecond Line\r\nThird Line",
                    "error": None
                },
                {
                    "filename": "error_formula.png",
                    "status": "error",
                    "elapsed_sec": 0.05,
                    "char_count": 0,
                    "word_count": 0,
                    "preview_text": "",
                    "error": "+CMD|' /C calc'!A0\nError detail"
                }
            ]
        }
        BatchPipeline.export_summary_csv(summary_data, out_csv)
        self.assertTrue(os.path.isfile(out_csv))

        with open(out_csv, "r", encoding="utf-8-sig") as f:
            content = f.read()
            # Newlines inside preview should be flattened to spaces
            self.assertNotIn("\nSecond Line", content)
            # Formula characters should be prepended with single quote
            self.assertIn("'=SUM(1,2)", content)
            self.assertIn("'+CMD", content)

    def test_11_invalid_format_error(self):
        """Verifies that passing an invalid format raises ValueError."""
        with self.assertRaises(ValueError):
            BatchPipeline(input_dir=self.input_dir, output_dir=self.output_dir, output_format="invalid_format")

    def test_12_cli_case_insensitive_args(self):
        """Tests that CLI handles lowercase model and uppercase format arguments cleanly."""
        cli_out = os.path.join(self.temp_dir, "cli_case_out")
        python_bin = sys.executable
        cmd = [
            python_bin,
            os.path.join(PROJECT_ROOT, "batch_runner.py"),
            "--input_dir", self.input_dir,
            "--output_dir", cli_out,
            "--model", "crnn",
            "--format", "JSON"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI stderr: {proc.stderr}")
        self.assertTrue(os.path.isfile(os.path.join(cli_out, "summary.json")))
        self.assertTrue(os.path.isfile(os.path.join(cli_out, "receipt_01.json")))
        self.assertFalse(os.path.isfile(os.path.join(cli_out, "receipt_01.txt")))

    def test_13_make_serializable_and_nested_conversions(self):
        """Tests that _make_serializable handles tensors, numpy arrays, sets, and custom types recursively."""
        import numpy as np
        import torch
        from ocr.batch_pipeline import _make_serializable

        class CustomWithToList:
            def tolist(self):
                return [np.float32(1.5), np.int64(42), {"sub": np.array([1, 2, 3])}]

        payload = {
            "tensor": torch.tensor([1.0, 2.0, 3.0]),
            "scalar_np": np.float32(3.1415),
            "array_np": np.array([[1, 2], [3, 4]]),
            "set_val": {"banana", "apple", "cherry"},
            "custom": CustomWithToList(),
            "nested": {
                "inner_list": [torch.tensor(99), np.bool_(True)]
            }
        }

        serialized = _make_serializable(payload)
        # Verify it can be dumped to JSON without errors
        dumped = json.dumps(serialized)
        self.assertIsInstance(dumped, str)
        self.assertIn("apple", dumped)
        self.assertIn("inner_list", dumped)
        self.assertEqual(serialized["set_val"], ["apple", "banana", "cherry"])
        self.assertEqual(serialized["tensor"], [1.0, 2.0, 3.0])
        self.assertEqual(serialized["custom"][0], 1.5)
        self.assertEqual(serialized["custom"][2]["sub"], [1, 2, 3])

    def test_14_path_prefix_containment_edge_case(self):
        """Tests that an external image with shared directory prefix doesn't falsely claim to be a child."""
        base_dir = os.path.join(self.temp_dir, "prefix_test")
        input_dir = os.path.join(base_dir, "dataset")
        outside_dir = os.path.join(base_dir, "dataset_other")
        out_dir = os.path.join(base_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(outside_dir, exist_ok=True)

        outside_img = os.path.join(outside_dir, "external.png")
        img = Image.new("RGB", (100, 30), color=(255, 255, 255))
        ImageDraw.Draw(img).text((5, 5), "Outside", fill=(0, 0, 0))
        img.save(outside_img)

        pipeline = BatchPipeline(input_dir=input_dir, output_dir=out_dir, model_key="CRNN")
        res = pipeline.process_single_image(outside_img, out_dir)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["relative_path"], "external.png")
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "external.json")))

    def test_15_csv_null_field_fallbacks(self):
        """Tests that summary data containing None numeric fields correctly writes default numeric zeros."""
        out_csv = os.path.join(self.temp_dir, "null_fields_test.csv")
        summary_data = {
            "results": [
                {
                    "filename": "null_item.png",
                    "status": "success",
                    "elapsed_sec": None,
                    "char_count": None,
                    "word_count": None,
                    "preview_text": None,
                    "error": None
                }
            ]
        }
        BatchPipeline.export_summary_csv(summary_data, out_csv)
        self.assertTrue(os.path.isfile(out_csv))
        with open(out_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            self.assertEqual(row["elapsed_sec"], "0.0")
            self.assertEqual(row["char_count"], "0")
            self.assertEqual(row["word_count"], "0")

    def test_16_csv_formula_injection_on_filename(self):
        """Tests that filenames starting with formula trigger chars (=, +, -, @) are sanitized in CSV."""
        out_csv = os.path.join(self.temp_dir, "filename_formula_test.csv")
        summary_data = {
            "results": [
                {
                    "filename": "=SUM(1,2).png",
                    "relative_path": "+cmd_calc.png",
                    "status": "success",
                    "elapsed_sec": 0.1,
                    "char_count": 5,
                    "word_count": 1,
                    "preview_text": "Sample",
                    "error": None
                },
                {
                    "filename": "-minus_file.jpg",
                    "status": "error",
                    "elapsed_sec": 0.05,
                    "char_count": 0,
                    "word_count": 0,
                    "preview_text": "",
                    "error": "@formula_err"
                }
            ]
        }
        BatchPipeline.export_summary_csv(summary_data, out_csv)
        self.assertTrue(os.path.isfile(out_csv))
        with open(out_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["filename"], "'+cmd_calc.png")
            self.assertEqual(rows[1]["filename"], "'-minus_file.jpg")
            self.assertEqual(rows[1]["error"], "'@formula_err")

    def test_17_create_zip_archive_with_uppercase_zip_extension_and_destination_collision(self):
        """Tests that existing .ZIP archives and in-place target zip paths are excluded from packaging."""
        out_dir = os.path.join(self.temp_dir, "zip_edge_dir")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "test.txt"), "w") as f:
            f.write("test content")
        with open(os.path.join(out_dir, "existing_archive.ZIP"), "wb") as f:
            f.write(b"PK_DUMMY_ZIP_DATA")

        # Destination zip placed directly inside output_dir
        dest_zip = os.path.join(out_dir, "my_bundled_output.zip")
        created_zip = BatchPipeline.create_zip_archive(out_dir, zip_path=dest_zip)
        self.assertTrue(os.path.isfile(created_zip))

        with zipfile.ZipFile(created_zip, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("test.txt", namelist)
            self.assertNotIn("existing_archive.ZIP", namelist)
            self.assertNotIn("my_bundled_output.zip", namelist)


if __name__ == "__main__":
    unittest.main()
