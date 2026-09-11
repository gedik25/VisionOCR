"""
VisionOCR Dynamic Warm Model Cache & Manager.
Zero Cold-Start (< 1.0s for warm cache, < 0.3s for CRNN), live token streaming (SSE),
controlled memory management, and automatic garbage collection on model switching.
"""

import os
import sys
import gc
import time
import psutil
import threading
from typing import Optional, Dict, Any, Generator, Union
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure matplotlib scratch directory to avoid warnings and delays
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "scratch", "matplotlib"))

import torch
from ocr.image_optimizer import optimize_image_for_ocr


def is_mps_available() -> bool:
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def get_default_device(model_key: str) -> str:
    """Returns optimal device for each model."""
    if model_key == "MMOCR":
        # MMOCR is most stable on CPU/CUDA on macOS ARM
        return "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def get_device_name(device_str: Optional[Union[str, torch.device]] = None) -> str:
    """Returns standardized human-readable device name."""
    if device_str is not None:
        d = str(device_str).lower()
        if "mps" in d:
            return "Apple Silicon (MPS)"
        if "cuda" in d:
            return "CUDA"
        if "cpu" in d:
            return "CPU"
    if torch.cuda.is_available():
        return "CUDA"
    if is_mps_available():
        return "Apple Silicon (MPS)"
    return "CPU"



class ModelManager:
    """
    Thread-safe Singleton Model Cache Manager.
    Keeps the active OCR model warm in memory for zero cold-start inference.
    Safely purges prior models on switch to avoid memory leaks.
    """
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.active_model_key: Optional[str] = None
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.img_transform: Any = None
        self.device: Optional[str] = None
        self.model_lock = threading.RLock()
        self.process = psutil.Process(os.getpid())
        self._initialized = True

    def get_memory_info(self) -> Dict[str, Any]:
        """Returns current process RAM usage and device info."""
        rss_bytes = self.process.memory_info().rss
        rss_mb = round(rss_bytes / (1024 * 1024), 2)
        rss_gb = round(rss_bytes / (1024 * 1024 * 1024), 3)
        return {
            "rss_mb": rss_mb,
            "rss_gb": rss_gb,
            "device": get_device_name(self.device),
            "is_mps": is_mps_available(),
            "active_model": self.active_model_key
        }

    def unload_model(self) -> None:
        """Safely cleans current model from memory and purges GPU/MPS cache."""
        with self.model_lock:
            if self.active_model_key is not None or self.model is not None:
                self.model = None
                self.processor = None
                self.tokenizer = None
                self.img_transform = None
                self.active_model_key = None
                self.device = None
                gc.collect()
                if is_mps_available() and hasattr(torch.mps, "empty_cache"):
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                if torch.cuda.is_available() and hasattr(torch.cuda, "empty_cache"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    def load_model(self, model_key: str) -> None:
        """
        Loads requested model into warm cache.
        If requested model is already active, returns immediately (0 cold-start).
        If switching models, cleans old model before loading new one.
        """
        with self.model_lock:
            if self.active_model_key == model_key and self.model is not None:
                return

            # Clean previous model
            self.unload_model()

            device = get_default_device(model_key)
            self.device = device

            # 1. CRNN
            if model_key == "CRNN":
                import torchvision.transforms as transforms
                # baudm/parseq Scene Text Recognition Benchmark
                model = torch.hub.load(
                    'baudm/parseq', 'crnn', pretrained=True, trust_repo=True
                ).eval().to(device)
                
                img_transform = transforms.Compose([
                    transforms.Resize((32, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)
                ])
                # Warmup Metal/PyTorch graph
                try:
                    dummy = torch.zeros((1, 3, 32, 128), device=device)
                    with torch.no_grad():
                        logits = model(dummy)
                        probs = logits.softmax(-1)
                        model.tokenizer.decode(probs)
                except Exception:
                    pass

                self.model = model
                self.img_transform = img_transform
                self.active_model_key = "CRNN"

            # 2. GLM-OCR
            elif model_key == "GLM-OCR":
                import importlib
                glm_mod = importlib.import_module("ocr.GLM-OCR.inference")
                processor, model, dev = glm_mod.load_glm_ocr()
                self.processor = processor
                self.tokenizer = getattr(processor, "tokenizer", processor)
                self.model = model
                self.device = dev
                self.active_model_key = "GLM-OCR"

            # 3. LightOnOCR-2-1B
            elif model_key == "LightOnOCR-2-1B":
                import importlib
                lighton_mod = importlib.import_module("ocr.LightOnOCR-2-1B.inference")
                from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

                dtype = torch.float32  # MPS and CPU float32
                if device == "cuda":
                    dtype = torch.bfloat16

                try:
                    model = LightOnOcrForConditionalGeneration.from_pretrained(
                        lighton_mod.MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
                    ).to(device).eval()
                except Exception:
                    model = LightOnOcrForConditionalGeneration.from_pretrained(
                        lighton_mod.MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
                    ).to(device).eval()

                try:
                    processor = LightOnOcrProcessor.from_pretrained(
                        lighton_mod.MODEL_ID, local_files_only=True, fix_mistral_regex=True
                    )
                except TypeError:
                    processor = LightOnOcrProcessor.from_pretrained(lighton_mod.MODEL_ID, local_files_only=True)
                except Exception:
                    try:
                        processor = LightOnOcrProcessor.from_pretrained(lighton_mod.MODEL_ID, fix_mistral_regex=True)
                    except Exception:
                        processor = LightOnOcrProcessor.from_pretrained(lighton_mod.MODEL_ID)


                self.model = model
                self.processor = processor
                self.tokenizer = getattr(processor, "tokenizer", processor)
                self.active_model_key = "LightOnOCR-2-1B"

            # 4. MMOCR
            elif model_key == "MMOCR":
                from mmocr.apis import MMOCRInferencer
                inferencer = MMOCRInferencer(det='DBNet', rec='CRNN', device=device)
                self.model = inferencer
                self.active_model_key = "MMOCR"

            # 5. OvisOCR2
            elif model_key == "OvisOCR2":
                from transformers import AutoProcessor, AutoModelForImageTextToText
                dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
                try:
                    model = AutoModelForImageTextToText.from_pretrained(
                        "ATH-MaaS/OvisOCR2", torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=True
                    ).to(device).eval()
                except Exception:
                    model = AutoModelForImageTextToText.from_pretrained(
                        "ATH-MaaS/OvisOCR2", torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
                    ).to(device).eval()

                try:
                    processor = AutoProcessor.from_pretrained("ATH-MaaS/OvisOCR2", trust_remote_code=True, local_files_only=True)
                except Exception:
                    processor = AutoProcessor.from_pretrained("ATH-MaaS/OvisOCR2", trust_remote_code=True)

                self.model = model
                self.processor = processor
                self.tokenizer = getattr(processor, "tokenizer", processor)
                self.active_model_key = "OvisOCR2"

            # 6. dots.ocr
            elif model_key == "dots.ocr":
                from transformers import AutoProcessor, AutoModelForCausalLM
                dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        "rednote-hilab/dots.ocr", torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True, local_files_only=True
                    ).to(device).eval()
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(
                        "rednote-hilab/dots.ocr", torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
                    ).to(device).eval()

                try:
                    processor = AutoProcessor.from_pretrained("rednote-hilab/dots.ocr", trust_remote_code=True, local_files_only=True)
                except Exception:
                    processor = AutoProcessor.from_pretrained("rednote-hilab/dots.ocr", trust_remote_code=True)

                self.model = model
                self.processor = processor
                self.tokenizer = getattr(processor, "tokenizer", processor)
                self.active_model_key = "dots.ocr"

            else:
                raise ValueError(f"Desteklenmeyen model anahtarı: {model_key}")

    def predict(
        self,
        model_key: str,
        image_input: Union[str, Image.Image],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes fast synchronous OCR inference on warm model.
        Returns standardized response dict.
        """
        with self.model_lock:
            is_warm = (self.active_model_key == model_key and self.model is not None)
            self.load_model(model_key)
            start_time = time.perf_counter()

            # Obtain original image dimensions for bounding box inverse scaling
            if isinstance(image_input, str):
                if not os.path.isfile(image_input) or os.path.getsize(image_input) == 0:
                    raise ValueError(f"Görsel dosyası bulunamadı veya boş: {image_input}")
                try:
                    with Image.open(image_input) as _img:
                        orig_w, orig_h = _img.size
                except Exception as e:
                    raise ValueError(f"Görsel açılamadı veya bozuk: {str(e)}") from e
            elif isinstance(image_input, Image.Image):
                orig_w, orig_h = image_input.size
            else:
                raise TypeError(f"Desteklenmeyen görsel girdisi: {type(image_input)}")


            # Smart image optimization for receipts & docs
            image = optimize_image_for_ocr(image_input, model_key=model_key)
            target_w, target_h = image.size
            scale_x = orig_w / float(target_w) if target_w > 0 else 1.0
            scale_y = orig_h / float(target_h) if target_h > 0 else 1.0

            # 1. CRNN
            if model_key == "CRNN":
                image_tensor = self.img_transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(image_tensor)
                    probs = logits.softmax(-1)
                    label, _ = self.model.tokenizer.decode(probs)
                
                raw_text = label[0] if (isinstance(label, list) and len(label) > 0) else str(label)
                # Remove CTC trailing dots if present
                cleaned_text = raw_text.rstrip(".")
                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                return {
                    "status": "success",
                    "model_key": "CRNN",
                    "text": cleaned_text,
                    "markdown": f"### CRNN Tanıma Çıktısı\n\n```\n{cleaned_text}\n```",
                    "boxes": [],
                    "raw": {"recognized_text": cleaned_text, "model": "baudm/parseq-crnn"},
                    "device": get_device_name(self.device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 2. GLM-OCR
            elif model_key == "GLM-OCR":
                prompt = kwargs.get("prompt", "Text Recognition:")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)

                max_tokens = kwargs.get("max_new_tokens", 1024)
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False
                    )
                    input_len = inputs['input_ids'].shape[1]
                    decoded = self.processor.decode(output[0][input_len:], skip_special_tokens=True)

                del inputs
                del output
                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                return {
                    "status": "success",
                    "model_key": "GLM-OCR",
                    "text": decoded.strip(),
                    "markdown": f"### GLM-OCR Çıktısı\n\n{decoded.strip()}",
                    "boxes": [],
                    "raw": {"output": decoded.strip(), "model": "zai-org/GLM-OCR"},
                    "device": get_device_name(self.device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 3. LightOnOCR-2-1B
            elif model_key == "LightOnOCR-2-1B":
                conversation = [
                    {"role": "user", "content": [{"type": "image"}]}
                ]
                prompt_text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=prompt_text,
                    images=image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    param_dtype = next(self.model.parameters()).dtype if hasattr(self.model, "parameters") else torch.float32
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=param_dtype)

                max_tokens = kwargs.get("max_new_tokens", 1024)
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

                raw_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                cleaned_text = raw_text
                if "assistant\n" in raw_text:
                    cleaned_text = raw_text.split("assistant\n", 1)[1]

                del inputs
                del outputs
                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                return {
                    "status": "success",
                    "model_key": "LightOnOCR-2-1B",
                    "text": cleaned_text.strip(),
                    "markdown": f"### LightOnOCR-2-1B Çıktısı\n\n{cleaned_text.strip()}",
                    "boxes": [],
                    "raw": {"output": raw_text, "model": "lightonai/LightOnOCR-2-1B"},
                    "device": get_device_name(self.device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 4. MMOCR
            elif model_key == "MMOCR":
                import numpy as np
                img_np = np.array(image)
                mmocr_res = self.model(img_np, save_vis=False, save_pred=False)
                
                predictions = mmocr_res.get('predictions', [])
                rec_texts = []
                rec_scores = []
                boxes = []
                
                if predictions:
                    pred_item = predictions[0]
                    rec_texts = pred_item.get('rec_texts', [])
                    rec_scores = pred_item.get('rec_scores', [])
                    polys = pred_item.get('det_polygons', [])
                    
                    for i, poly in enumerate(polys):
                        t = rec_texts[i] if i < len(rec_texts) else ""
                        if len(poly) >= 4:
                            xs = [x * scale_x for x in poly[0::2]]
                            ys = [y * scale_y for y in poly[1::2]]
                            box = [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]
                            boxes.append({"box": box, "text": t})

                formatted = [f"{t} (Güven: %{s*100:.1f})" for t, s in zip(rec_texts, rec_scores)]
                full_text = "\n".join(rec_texts)
                md_lines = ["### MMOCR (DBNet + CRNN) Çıktısı\n"]
                for f in formatted:
                    md_lines.append(f"- {f}")
                md_text = "\n".join(md_lines)

                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                return {
                    "status": "success",
                    "model_key": "MMOCR",
                    "text": full_text,
                    "markdown": md_text,
                    "boxes": boxes,
                    "raw": mmocr_res,
                    "device": get_device_name(self.device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 5. OvisOCR2
            elif model_key == "OvisOCR2":
                prompt = kwargs.get("prompt", "Read all text in this image.")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(self.device)

                max_tokens = kwargs.get("max_new_tokens", 1024)
                eos_ids = [248044, 248054]
                if hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "eos_token_id", None):
                    eos_ids.append(self.processor.tokenizer.eos_token_id)

                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        eos_token_id=eos_ids,
                        do_sample=False
                    )
                    input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                    decoded = self.processor.decode(output[0][input_len:], skip_special_tokens=True)

                cleaned = decoded.replace("<think>\n\n</think>", "").replace("<think></think>", "").strip()
                del inputs
                del output
                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                return {
                    "status": "success",
                    "model_key": "OvisOCR2",
                    "text": cleaned,
                    "markdown": f"### OvisOCR2 Çıktısı\n\n{cleaned}",
                    "boxes": [],
                    "raw": {"output": cleaned, "model": "ATH-MaaS/OvisOCR2"},
                    "device": get_device_name(self.device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 6. dots.ocr
            elif model_key == "dots.ocr":
                prompt_content = kwargs.get("prompt") or (
                    "Please output the layout information from the PDF image, including each layout element's bbox, "
                    "its category, and the corresponding text content within the bbox.\n"
                    "1. Bbox format: [x1, y1, x2, y2]\n"
                    "2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n"
                    "3. Text Extraction: Picture text omitted, Formula LaTeX, Table HTML, Others Markdown.\n"
                    "4. Constraints: Original text, sorted by reading order.\n"
                    "5. Final Output: Single JSON object."
                )
                prompt_text = f"<|user|><|img|><|imgpad|><|endofimg|>{prompt_content}<|endofuser|><|assistant|>"
                inputs = self.processor(text=prompt_text, images=image, return_tensors="pt")
                model_dtype = getattr(self.model, "dtype", torch.float16 if str(self.device) in ["mps", "cuda"] else torch.float32)
                inputs = {k: v.to(device=self.device, dtype=model_dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                inputs.pop("mm_token_type_ids", None)

                max_tokens = kwargs.get("max_new_tokens", 512)
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        eos_token_id=[151673, 151643],
                        do_sample=False
                    )
                    input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
                    decoded = self.processor.decode(output[0][input_len:], skip_special_tokens=True)

                del inputs
                del output
                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                # Parse JSON layout if available
                formatted_text = decoded.strip()
                try:
                    import json
                    parsed = json.loads(decoded.strip())
                    if isinstance(parsed, list):
                        lines = []
                        for item in parsed:
                            cat = item.get("category", "Text")
                            text = item.get("text", "").strip()
                            bbox = item.get("bbox", [])
                            if text:
                                if cat == "Title":
                                    lines.append(f"# {text}")
                                elif cat == "Section-header":
                                    lines.append(f"## {text}")
                                elif cat == "List-item":
                                    lines.append(f"- {text}")
                                else:
                                    lines.append(text)
                            elif cat == "Picture" and bbox:
                                lines.append(f"*[Görsel/Grafik Bölgesi: {bbox}]*")
                        if lines:
                            formatted_text = "\n\n".join(lines)
                except Exception:
                    pass

                return {
                    "status": "success",
                    "model_key": "dots.ocr",
                    "text": formatted_text,
                    "markdown": f"### dots.ocr Çıktısı\n\n{formatted_text}",
                    "boxes": [],
                    "raw": {"output": decoded.strip(), "model": "rednote-hilab/dots.ocr"},
                    "device": get_device_name(self.device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            else:
                raise ValueError(f"Bilinmeyen model anahtarı: {model_key}")

    def stream_predict(
        self,
        model_key: str,
        image_input: Union[str, Image.Image],
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Server-Sent Events (SSE) streaming OCR generator.
        Safely locked under self.model_lock for the entire generation lifecycle.
        Yields token-level delta dictionaries in real time:
        - `{"event": "token", "token": delta, "accumulated": text}`
        - `{"event": "done", "status": "success", ...}`
        """
        from transformers import TextIteratorStreamer

        with self.model_lock:
            is_warm = (self.active_model_key == model_key and self.model is not None)
            self.load_model(model_key)
            start_time = time.perf_counter()

            # Obtain original image dimensions for bounding box inverse scaling
            if isinstance(image_input, str):
                if not os.path.isfile(image_input) or os.path.getsize(image_input) == 0:
                    raise ValueError(f"Görsel dosyası bulunamadı veya boş: {image_input}")
                try:
                    with Image.open(image_input) as _img:
                        orig_w, orig_h = _img.size
                except Exception as e:
                    raise ValueError(f"Görsel açılamadı veya bozuk: {str(e)}") from e
            elif isinstance(image_input, Image.Image):
                orig_w, orig_h = image_input.size
            else:
                raise TypeError(f"Desteklenmeyen görsel girdisi: {type(image_input)}")


            image = optimize_image_for_ocr(image_input, model_key=model_key)
            target_w, target_h = image.size
            scale_x = orig_w / float(target_w) if target_w > 0 else 1.0
            scale_y = orig_h / float(target_h) if target_h > 0 else 1.0

            model = self.model
            processor = self.processor
            tokenizer = self.tokenizer
            device = self.device
            img_transform = self.img_transform

            # 1. GLM-OCR Streaming
            if model_key == "GLM-OCR":
                prompt = kwargs.get("prompt", "Text Recognition:")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(device)

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
                max_tokens = kwargs.get("max_new_tokens", 1024)
                gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False, streamer=streamer)

                gen_error = []
                def run_glm_gen():
                    try:
                        with torch.inference_mode():
                            model.generate(**gen_kwargs)
                    except Exception as ex:
                        gen_error.append(ex)

                thread = threading.Thread(target=run_glm_gen, daemon=True)
                thread.start()

                accumulated_tokens = []
                try:
                    for token_delta in streamer:
                        accumulated_tokens.append(token_delta)
                        current_text = "".join(accumulated_tokens)
                        yield {
                            "event": "token",
                            "token": token_delta,
                            "accumulated": current_text
                        }
                finally:
                    thread.join(timeout=5.0)
                    del inputs

                if gen_error:
                    raise gen_error[0]

                full_text = "".join(accumulated_tokens).strip()
                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                yield {
                    "event": "done",
                    "status": "success",
                    "model_key": "GLM-OCR",
                    "text": full_text,
                    "markdown": f"### GLM-OCR Çıktısı\n\n{full_text}",
                    "boxes": [],
                    "raw": {"output": full_text, "model": "zai-org/GLM-OCR"},
                    "device": get_device_name(device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 2. LightOnOCR-2-1B Streaming
            elif model_key == "LightOnOCR-2-1B":
                conversation = [
                    {"role": "user", "content": [{"type": "image"}]}
                ]
                prompt_text = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=prompt_text,
                    images=image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    param_dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else torch.float32
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=param_dtype)

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
                max_tokens = kwargs.get("max_new_tokens", 1024)
                gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=False, streamer=streamer)

                gen_error = []
                def run_lighton_gen():
                    try:
                        with torch.inference_mode():
                            model.generate(**gen_kwargs)
                    except Exception as ex:
                        gen_error.append(ex)

                thread = threading.Thread(target=run_lighton_gen, daemon=True)
                thread.start()

                accumulated_tokens = []
                try:
                    for token_delta in streamer:
                        accumulated_tokens.append(token_delta)
                        current_text = "".join(accumulated_tokens)
                        yield {
                            "event": "token",
                            "token": token_delta,
                            "accumulated": current_text
                        }
                finally:
                    thread.join(timeout=5.0)
                    del inputs

                if gen_error:
                    raise gen_error[0]

                full_text = "".join(accumulated_tokens).strip()
                # Clean possible assistant prefix if escaped through streamer
                if "assistant\n" in full_text:
                    full_text = full_text.split("assistant\n", 1)[1].strip()

                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                yield {
                    "event": "done",
                    "status": "success",
                    "model_key": "LightOnOCR-2-1B",
                    "text": full_text,
                    "markdown": f"### LightOnOCR-2-1B Çıktısı\n\n{full_text}",
                    "boxes": [],
                    "raw": {"output": full_text, "model": "lightonai/LightOnOCR-2-1B"},
                    "device": get_device_name(device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 3. CRNN (Fast Single Line Streaming)
            elif model_key == "CRNN":
                image_tensor = img_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(image_tensor)
                    probs = logits.softmax(-1)
                    label, _ = model.tokenizer.decode(probs)
                
                raw_text = label[0] if (isinstance(label, list) and len(label) > 0) else str(label)
                cleaned_text = raw_text.rstrip(".")

                # Stream token chunks for UI typewriter consistency
                chunk_size = max(1, len(cleaned_text) // 3)
                for idx in range(0, len(cleaned_text), chunk_size):
                    chunk = cleaned_text[idx:idx + chunk_size]
                    yield {
                        "event": "token",
                        "token": chunk,
                        "accumulated": cleaned_text[:idx + len(chunk)]
                    }

                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                yield {
                    "event": "done",
                    "status": "success",
                    "model_key": "CRNN",
                    "text": cleaned_text,
                    "markdown": f"### CRNN Tanıma Çıktısı\n\n```\n{cleaned_text}\n```",
                    "boxes": [],
                    "raw": {"recognized_text": cleaned_text, "model": "baudm/parseq-crnn"},
                    "device": get_device_name(device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 4. MMOCR (Line-by-Line Streaming)
            elif model_key == "MMOCR":
                import numpy as np
                img_np = np.array(image)
                mmocr_res = model(img_np, save_vis=False, save_pred=False)
                
                predictions = mmocr_res.get('predictions', [])
                rec_texts = []
                rec_scores = []
                boxes = []
                
                if predictions:
                    pred_item = predictions[0]
                    rec_texts = pred_item.get('rec_texts', [])
                    rec_scores = pred_item.get('rec_scores', [])
                    polys = pred_item.get('det_polygons', [])
                    
                    for i, poly in enumerate(polys):
                        t = rec_texts[i] if i < len(rec_texts) else ""
                        if len(poly) >= 4:
                            xs = [x * scale_x for x in poly[0::2]]
                            ys = [y * scale_y for y in poly[1::2]]
                            box = [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]
                            boxes.append({"box": box, "text": t})

                # Stream line by line
                acc_lines = []
                for t in rec_texts:
                    acc_lines.append(t)
                    yield {
                        "event": "token",
                        "token": t + "\n",
                        "accumulated": "\n".join(acc_lines)
                    }

                formatted = [f"{t} (Güven: %{s*100:.1f})" for t, s in zip(rec_texts, rec_scores)]
                full_text = "\n".join(rec_texts)
                md_lines = ["### MMOCR (DBNet + CRNN) Çıktısı\n"]
                for f in formatted:
                    md_lines.append(f"- {f}")
                md_text = "\n".join(md_lines)

                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                yield {
                    "event": "done",
                    "status": "success",
                    "model_key": "MMOCR",
                    "text": full_text,
                    "markdown": md_text,
                    "boxes": boxes,
                    "raw": mmocr_res,
                    "device": get_device_name(device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 5. OvisOCR2 Streaming
            elif model_key == "OvisOCR2":
                prompt = kwargs.get("prompt", "Read all text in this image.")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)

                max_tokens = kwargs.get("max_new_tokens", 1024)
                eos_ids = [248044, 248054]
                if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "eos_token_id", None):
                    eos_ids.append(processor.tokenizer.eos_token_id)

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
                gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, eos_token_id=eos_ids, do_sample=False, streamer=streamer)

                gen_error = []
                def run_ovis_gen():
                    try:
                        with torch.inference_mode():
                            model.generate(**gen_kwargs)
                    except Exception as ex:
                        gen_error.append(ex)

                thread = threading.Thread(target=run_ovis_gen, daemon=True)
                thread.start()

                accumulated_tokens = []
                try:
                    for token_delta in streamer:
                        if "<think>" in token_delta or "</think>" in token_delta:
                            continue
                        accumulated_tokens.append(token_delta)
                        current_text = "".join(accumulated_tokens).strip()
                        yield {
                            "event": "token",
                            "token": token_delta,
                            "accumulated": current_text
                        }
                finally:
                    thread.join(timeout=5.0)
                    del inputs

                if gen_error:
                    raise gen_error[0]

                full_text = "".join(accumulated_tokens).strip()
                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                yield {
                    "event": "done",
                    "status": "success",
                    "model_key": "OvisOCR2",
                    "text": full_text,
                    "markdown": f"### OvisOCR2 Çıktısı\n\n{full_text}",
                    "boxes": [],
                    "raw": {"output": full_text, "model": "ATH-MaaS/OvisOCR2"},
                    "device": get_device_name(device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            # 6. dots.ocr Streaming
            elif model_key == "dots.ocr":
                prompt_content = kwargs.get("prompt") or (
                    "Please output the layout information from the PDF image, including each layout element's bbox, "
                    "its category, and the corresponding text content within the bbox.\n"
                    "1. Bbox format: [x1, y1, x2, y2]\n"
                    "2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n"
                    "3. Text Extraction: Picture text omitted, Formula LaTeX, Table HTML, Others Markdown.\n"
                    "4. Constraints: Original text, sorted by reading order.\n"
                    "5. Final Output: Single JSON object."
                )
                prompt_text = f"<|user|><|img|><|imgpad|><|endofimg|>{prompt_content}<|endofuser|><|assistant|>"
                inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                model_dtype = getattr(model, "dtype", torch.float16 if str(device) in ["mps", "cuda"] else torch.float32)
                inputs = {k: v.to(device=device, dtype=model_dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                inputs.pop("mm_token_type_ids", None)

                max_tokens = kwargs.get("max_new_tokens", 512)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)
                gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, eos_token_id=[151673, 151643], do_sample=False, streamer=streamer)

                gen_error = []
                def run_dots_gen():
                    try:
                        with torch.inference_mode():
                            model.generate(**gen_kwargs)
                    except Exception as ex:
                        gen_error.append(ex)

                thread = threading.Thread(target=run_dots_gen, daemon=True)
                thread.start()

                accumulated_tokens = []
                try:
                    for token_delta in streamer:
                        accumulated_tokens.append(token_delta)
                        current_text = "".join(accumulated_tokens)
                        yield {
                            "event": "token",
                            "token": token_delta,
                            "accumulated": current_text
                        }
                finally:
                    thread.join(timeout=5.0)
                    del inputs

                if gen_error:
                    raise gen_error[0]

                raw_text = "".join(accumulated_tokens).strip()
                formatted_text = raw_text
                try:
                    import json
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, list):
                        lines = []
                        for item in parsed:
                            cat = item.get("category", "Text")
                            text = item.get("text", "").strip()
                            bbox = item.get("bbox", [])
                            if text:
                                if cat == "Title":
                                    lines.append(f"# {text}")
                                elif cat == "Section-header":
                                    lines.append(f"## {text}")
                                elif cat == "List-item":
                                    lines.append(f"- {text}")
                                else:
                                    lines.append(text)
                            elif cat == "Picture" and bbox:
                                lines.append(f"*[Görsel/Grafik Bölgesi: {bbox}]*")
                        if lines:
                            formatted_text = "\n\n".join(lines)
                except Exception:
                    pass

                elapsed = time.perf_counter() - start_time
                mem_info = self.get_memory_info()

                yield {
                    "event": "done",
                    "status": "success",
                    "model_key": "dots.ocr",
                    "text": formatted_text,
                    "markdown": f"### dots.ocr Çıktısı\n\n{formatted_text}",
                    "boxes": [],
                    "raw": {"output": raw_text, "model": "rednote-hilab/dots.ocr"},
                    "device": get_device_name(device),
                    "elapsed_sec": elapsed,
                    "is_warm": is_warm,
                    "memory_mb": mem_info["rss_mb"]
                }

            else:
                raise ValueError(f"Desteklenmeyen akış modeli: {model_key}")



# Singleton accessor
_manager_instance: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance

