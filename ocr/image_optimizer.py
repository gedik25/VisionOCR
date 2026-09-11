"""
VisionOCR Image Optimizer.
Smart scaling and preprocessing for receipts (market fişleri) and vertical/large documents.
Prevents visual token explosion in Vision-Language Models (GLM-OCR, LightOnOCR)
and keeps peak memory tightly bounded under 3.5 GB without losing OCR legibility.
"""

import os
import math
from typing import Union, Tuple
from PIL import Image, ImageOps, ImageEnhance


# Maximum dimension constraints
MAX_STANDARD_EDGE = 1536       # Max edge for standard documents (4:3, 16:9, square)
MAX_RECEIPT_HEIGHT = 1600      # Max height for tall vertical receipts (aspect ratio > 2.0)
MIN_RECEIPT_WIDTH = 384        # Minimum width for receipts to preserve character stroke details
MAX_TOTAL_PIXELS = 1500000     # ~1.5 Megapixels maximum visual budget (prevents patch explosion)


def detect_document_type(w: int, h: int) -> str:
    """
    Classifies image layout based on aspect ratio:
    - 'receipt': Tall vertical receipt (aspect ratio h/w >= 1.8)
    - 'wide_banner': Wide horizontal document (w/h >= 2.0)
    - 'standard_doc': Standard A4, letter, or document image
    """
    if h <= 0 or w <= 0:
        return "standard_doc"
    
    aspect_vertical = h / float(w)
    aspect_horizontal = w / float(h)

    if aspect_vertical >= 1.8:
        return "receipt"
    elif aspect_horizontal >= 2.0:
        return "wide_banner"
    return "standard_doc"


def calculate_optimal_dimensions(
    w: int,
    h: int,
    doc_type: str,
    max_edge: int = MAX_STANDARD_EDGE,
    max_pixels: int = MAX_TOTAL_PIXELS
) -> Tuple[int, int]:
    """
    Calculates target dimensions preserving aspect ratio and stroke readability
    while strictly capping total visual token budget.
    """
    if w <= 0 or h <= 0:
        return (max(1, w), max(1, h))

    # 1. Receipts (market fişi) - tall and narrow
    if doc_type == "receipt":
        scale = 1.0
        if h > MAX_RECEIPT_HEIGHT:
            scale = min(scale, MAX_RECEIPT_HEIGHT / float(h))
        
        # Check if total pixels exceed budget
        if (w * scale) * (h * scale) > max_pixels:
            scale = min(scale, math.sqrt(max_pixels / float(w * h)))

        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        # Minimum legible receipt width protection
        target_min_w = min(w, MIN_RECEIPT_WIDTH)
        if new_w < target_min_w:
            new_w = target_min_w
            # Maintain aspect ratio as much as total pixels allow
            aspect = h / float(w)
            new_h = max(int(new_w * aspect), 1)
            if new_w * new_h > max_pixels:
                new_h = max(int(max_pixels / float(new_w)), 1)

        return (new_w, new_h)

    # 2. Wide banners - short and wide
    if doc_type == "wide_banner":
        scale = 1.0
        if w > max_edge:
            scale = min(scale, max_edge / float(w))
        if (w * scale) * (h * scale) > max_pixels:
            scale = min(scale, math.sqrt(max_pixels / float(w * h)))

        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        target_min_h = min(h, 128)
        if new_h < target_min_h:
            new_h = target_min_h
            aspect = w / float(h)
            new_w = max(int(new_h * aspect), 1)
            if new_w * new_h > max_pixels:
                new_w = max(int(max_pixels / float(new_h)), 1)

        return (new_w, new_h)

    # 3. Standard document
    longest = max(w, h)
    scale = 1.0
    if longest > max_edge:
        scale = min(scale, max_edge / float(longest))

    if (w * scale) * (h * scale) > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / float(w * h)))

    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    return (new_w, new_h)


def enhance_receipt_contrast(image: Image.Image) -> Image.Image:
    """
    Applies adaptive contrast normalization for thermal receipt paper
    and low-contrast mobile camera captures.
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Autocontrast with gentle 0.5% cutoff to eliminate background gray noise
    try:
        image = ImageOps.autocontrast(image, cutoff=0.5)
    except Exception:
        pass

    # Slight sharpness enhancement (1.1x) to define character boundaries
    try:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
    except Exception:
        pass

    return image


def optimize_image_for_ocr(
    image: Union[str, Image.Image],
    model_key: str = "GLM-OCR",
    max_edge: int = MAX_STANDARD_EDGE,
    enhance_receipts: bool = True
) -> Image.Image:
    """
    Main entry point for intelligent OCR image optimization.
    
    1. Loads image (if file path) with error handling.
    2. Transposes EXIF orientation for mobile camera photos.
    3. Handles alpha transparency by compositing onto a solid white background.
    4. Detects document geometry (receipt vs standard vs wide banner).
    5. Scales intelligently with Lanczos resampling.
    6. Optionally enhances contrast on thermal receipts.
    """
    if isinstance(image, str):
        if not os.path.isfile(image) or os.path.getsize(image) == 0:
            raise ValueError(f"Görsel dosyası bulunamadı veya boş: {image}")
        try:
            img = Image.open(image)
        except Exception as e:
            raise ValueError(f"Görsel açılamadı veya bozuk: {str(e)}")
    elif isinstance(image, Image.Image):
        img = image.copy()
    else:
        raise TypeError(f"Desteklenmeyen görsel girdisi: {type(image)}")

    # Automatically correct mobile photo rotation from EXIF metadata
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Safely composite transparent images (RGBA, LA, transparent Palette) onto white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in getattr(img, "info", {})):
        rgba_img = img.convert("RGBA")
        white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, rgba_img).convert("RGB")
    else:
        img = img.convert("RGB")

    w, h = img.size
    doc_type = detect_document_type(w, h)
    target_w, target_h = calculate_optimal_dimensions(w, h, doc_type, max_edge=max_edge)

    # Only resize if dimensions changed
    if (target_w, target_h) != (w, h):
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((target_w, target_h), resample=resample)

    # Thermal receipt preprocessing
    if doc_type == "receipt" and enhance_receipts and model_key in ["GLM-OCR", "LightOnOCR-2-1B"]:
        img = enhance_receipt_contrast(img)

    return img

