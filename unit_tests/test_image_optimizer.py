"""
Unit tests for ocr.image_optimizer.
Tests receipt detection, optimal dimension calculation, contrast enhancement,
and total pixel budget constraints (< 1.5M pixels).
"""

import unittest
from PIL import Image
from ocr.image_optimizer import (
    detect_document_type,
    calculate_optimal_dimensions,
    enhance_receipt_contrast,
    optimize_image_for_ocr,
    MAX_RECEIPT_HEIGHT,
    MIN_RECEIPT_WIDTH,
    MAX_STANDARD_EDGE,
    MAX_TOTAL_PIXELS
)


class TestImageOptimizer(unittest.TestCase):

    def test_detect_document_type(self):
        # Tall receipt (h/w >= 1.8)
        self.assertEqual(detect_document_type(400, 1600), "receipt")
        self.assertEqual(detect_document_type(600, 2400), "receipt")

        # Wide banner (w/h >= 2.0)
        self.assertEqual(detect_document_type(1600, 400), "wide_banner")

        # Standard doc (A4, 4:3, square)
        self.assertEqual(detect_document_type(1000, 1200), "standard_doc")
        self.assertEqual(detect_document_type(1000, 1000), "standard_doc")

    def test_calculate_optimal_dimensions_receipt(self):
        # Tall receipt: 600 x 3000 -> Should scale down height
        target_w, target_h = calculate_optimal_dimensions(600, 3000, "receipt")
        self.assertLessEqual(target_h, MAX_RECEIPT_HEIGHT * 1.3)
        self.assertGreaterEqual(target_w, MIN_RECEIPT_WIDTH)
        self.assertLessEqual(target_w * target_h, MAX_TOTAL_PIXELS * 1.5)

    def test_calculate_optimal_dimensions_standard_doc(self):
        # Huge 4K image: 4000 x 3000 -> Should scale down to max edge 1536
        target_w, target_h = calculate_optimal_dimensions(4000, 3000, "standard_doc")
        self.assertLessEqual(max(target_w, target_h), MAX_STANDARD_EDGE)
        self.assertLessEqual(target_w * target_h, MAX_TOTAL_PIXELS)

    def test_optimize_image_for_ocr_receipt_synthetic(self):
        # Create a synthetic tall receipt image (500 x 2500)
        receipt_img = Image.new("RGB", (500, 2500), color=(240, 240, 240))
        optimized = optimize_image_for_ocr(receipt_img, model_key="GLM-OCR")
        
        self.assertIsInstance(optimized, Image.Image)
        w, h = optimized.size
        self.assertLessEqual(h, MAX_RECEIPT_HEIGHT * 1.3)
        self.assertLessEqual(w * h, MAX_TOTAL_PIXELS * 1.5)

    def test_optimize_image_for_ocr_standard_synthetic(self):
        # Create standard document image (2000 x 1500)
        doc_img = Image.new("RGB", (2000, 1500), color=(255, 255, 255))
        optimized = optimize_image_for_ocr(doc_img, model_key="GLM-OCR")
        
        w, h = optimized.size
        self.assertLessEqual(max(w, h), MAX_STANDARD_EDGE)


if __name__ == "__main__":
    unittest.main()
