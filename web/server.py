"""
FastAPI Server for VisionOCR Web.
Provides high-performance REST and Server-Sent Events (SSE) streaming endpoints
with Dynamic Warm Model Caching (Zero Cold-Start < 1.0s, CRNN < 0.3s),
real-time token streaming, and controlled peak memory (< 3.5 GB).
"""

import os
import sys
import json
import time
import uuid
import shutil
import asyncio
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, "scratch", "matplotlib"))

from ocr.model_manager import get_model_manager, is_mps_available, get_device_name

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "scratch", "web_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="VisionOCR Web API",
    description="High-Performance OCR Server with Warm Caching and Live SSE Token Streaming",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
async def get_available_models():
    """Returns supported OCR models, hardware backend status, and active warm cache info."""
    mgr = get_model_manager()
    mem_info = mgr.get_memory_info()
    is_mps = is_mps_available()
    device = get_device_name(mgr.device)

    models = [
        {
            "id": "GLM-OCR",
            "name": "GLM-OCR",
            "tag": "0.9B ViT",
            "description": "Yapay Zeka Destekli Belge & Metin Ayrıştırma (SSE Akışı)",
            "default": True,
            "supports_streaming": True
        },
        {
            "id": "OvisOCR2",
            "name": "OvisOCR2",
            "tag": "0.9B Markdown VLM",
            "description": "ATH-MaaS Yüksek Doğruluklu Markdown Belge Ayrıştırma (SSE Akışı)",
            "default": False,
            "supports_streaming": True
        },
        {
            "id": "LightOnOCR-2-1B",
            "name": "LightOnOCR-2-1B",
            "tag": "2.1B VLM",
            "description": "Belge, Rapor & Tablo Ayrıştırma (SSE Akışı)",
            "default": False,
            "supports_streaming": True
        },
        {
            "id": "dots.ocr",
            "name": "dots.ocr",
            "tag": "Çok Dilli Belge",
            "description": "Rednote HiLab Çok Dilli Belge & Düzen Analizi",
            "default": False,
            "supports_streaming": True
        },
        {
            "id": "MMOCR",
            "name": "MMOCR",
            "tag": "DBNet + CRNN",
            "description": "OpenMMLab Çoklu Satır Tespit & Tanıma",
            "default": False,
            "supports_streaming": True
        },
        {
            "id": "CRNN",
            "name": "CRNN",
            "tag": "Hub Pretrained",
            "description": "Ultra Hızlı Tek Satır & Kelime Tanıma (< 0.3s)",
            "default": False,
            "supports_streaming": True
        }
    ]
    return {
        "models": models,
        "device": device,
        "is_mps": is_mps,
        "active_model": mgr.active_model_key,
        "memory": mem_info
    }


def save_upload_file(file: UploadFile) -> str:
    """Validates and saves uploaded image file to scratch directory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Geçersiz dosya.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"]:
        raise HTTPException(status_code=400, detail="Desteklenmeyen görsel formatı.")

    safe_filename = f"upload_{int(time.time() * 1000)}_{os.path.basename(file.filename)}"
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not os.path.isfile(saved_path) or os.path.getsize(saved_path) == 0:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=400, detail="Boş veya geçersiz görsel dosyası yüklendi.")

    return saved_path


@app.post("/api/ocr")
async def process_ocr_sync(
    file: UploadFile = File(...),
    model: str = Form("GLM-OCR")
):
    """
    Direct synchronous OCR inference with warm model cache (Zero Cold-Start < 1.0s).
    """
    saved_path = save_upload_file(file)
    mgr = get_model_manager()

    try:
        # Run inference in worker thread pool so we do not block async event loop
        result = await asyncio.to_thread(mgr.predict, model, saved_path)
        result["image_filename"] = os.path.basename(saved_path)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Hatası: {str(e)}")



@app.post("/api/ocr/stream")
async def process_ocr_stream(
    file: UploadFile = File(...),
    model: str = Form("GLM-OCR")
):
    """
    Live Server-Sent Events (SSE) token streaming endpoint.
    Streams incremental tokens as they are decoded in real-time.
    """
    saved_path = save_upload_file(file)
    filename = os.path.basename(saved_path)
    mgr = get_model_manager()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def stream_worker():
            try:
                for event_dict in mgr.stream_predict(model, saved_path):
                    if event_dict.get("event") == "done":
                        event_dict["image_filename"] = filename
                    asyncio.run_coroutine_threadsafe(queue.put(event_dict), loop)
                # Signal completion
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as exc:
                err_dict = {
                    "event": "error",
                    "status": "error",
                    "error": str(exc)
                }
                asyncio.run_coroutine_threadsafe(queue.put(err_dict), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # Start inference in background thread
        import threading
        t = threading.Thread(target=stream_worker, daemon=True)
        t.start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                json_data = json.dumps(item, ensure_ascii=False, default=str)
                yield f"data: {json_data}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/models/preload")
async def preload_model(model: str = Form(...)):
    """Preloads a model into warm memory ahead of time."""
    mgr = get_model_manager()
    try:
        await asyncio.to_thread(mgr.load_model, model)
        return {
            "status": "success",
            "message": f"Model '{model}' sıcak önbelleğe yüklendi.",
            "memory": mgr.get_memory_info()
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/unload")
async def unload_model():
    """Frees the currently loaded model and clears GPU/MPS cache."""
    mgr = get_model_manager()
    mgr.unload_model()
    return {
        "status": "success",
        "message": "Model bellekten temizlendi.",
        "memory": mgr.get_memory_info()
    }


# Serve uploaded images for web canvas
@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Görsel bulunamadı.")


# ─────────────────────────────────────────────────────────────
# ⚡ Batch Receipt & Document Processing API
# ─────────────────────────────────────────────────────────────
BATCH_JOBS_DIR = os.path.join(PROJECT_ROOT, "scratch", "batch_jobs")
os.makedirs(BATCH_JOBS_DIR, exist_ok=True)

from ocr.batch_pipeline import BatchPipeline, SUPPORTED_EXTENSIONS


def get_safe_batch_job_dir(batch_id: str) -> str:
    """Safely validates batch_id and returns the absolute path to the job directory within BATCH_JOBS_DIR."""
    if not batch_id or not isinstance(batch_id, str):
        raise HTTPException(status_code=400, detail="Geçersiz batch ID.")
    clean_id = os.path.basename(batch_id.strip())
    if clean_id != batch_id or ".." in batch_id or "/" in batch_id or "\\" in batch_id or clean_id in (".", "..", ""):
        raise HTTPException(status_code=400, detail="Geçersiz veya güvensiz batch ID.")
    job_dir = os.path.abspath(os.path.join(BATCH_JOBS_DIR, clean_id))
    abs_base = os.path.abspath(BATCH_JOBS_DIR)
    if not job_dir.startswith(abs_base) or job_dir == abs_base:
        raise HTTPException(status_code=400, detail="Geçersiz erişim yolu.")
    return job_dir


def save_batch_upload_files(files: list[UploadFile], job_id: str) -> list[str]:
    job_input_dir = os.path.join(BATCH_JOBS_DIR, job_id, "input")
    os.makedirs(job_input_dir, exist_ok=True)
    saved_paths = []

    for file in files:
        if not file.filename:
            continue

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        raw_rel = file.filename.replace("\\", "/")
        safe_rel = os.path.normpath(raw_rel).lstrip("/.")
        norm_parts = safe_rel.replace("\\", "/").split("/")
        if ".." in norm_parts or "." in norm_parts or not safe_rel or os.path.isabs(safe_rel):
            safe_rel = os.path.basename(file.filename)

        dest_path = os.path.join(job_input_dir, safe_rel)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Handle duplicate filenames in same batch to prevent data overwrite
        if os.path.exists(dest_path):
            base_dir = os.path.dirname(dest_path)
            fstem, fext = os.path.splitext(os.path.basename(dest_path))
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(base_dir, f"{fstem}_{counter}{fext}")
                counter += 1

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(dest_path)

    return saved_paths


@app.post("/api/batch/process")
@app.post("/api/batch")
async def process_batch_sync(
    files: list[UploadFile] = File(...),
    model: str = Form("GLM-OCR"),
    format: str = Form("all")
):
    """
    Synchronous batch OCR inference over multiple uploaded receipt/document images.
    Returns complete batch summary, item results, and ZIP download URL.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Hiçbir görsel dosyası yüklenmedi.")

    job_id = f"job_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    job_dir = os.path.join(BATCH_JOBS_DIR, job_id)
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = save_batch_upload_files(files, job_id)
    if not saved_paths:
        raise HTTPException(status_code=400, detail="Yüklenen dosyalarda geçerli görsel bulunamadı.")

    pipeline = BatchPipeline(
        input_dir=os.path.join(job_dir, "input"),
        output_dir=output_dir,
        model_key=model,
        output_format=format
    )

    try:
        summary = await asyncio.to_thread(pipeline.run, saved_paths)
        zip_path = os.path.join(job_dir, f"visionocr_batch_{job_id}.zip")
        await asyncio.to_thread(BatchPipeline.create_zip_archive, output_dir, zip_path)

        return {
            "status": "success",
            "batch_id": job_id,
            "zip_url": f"/api/batch/download/{job_id}",
            "csv_url": f"/api/batch/csv/{job_id}",
            "json_url": f"/api/batch/summary/{job_id}",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Toplu OCR işlemi hatası: {str(e)}")


@app.post("/api/batch/stream")
async def process_batch_stream(
    files: list[UploadFile] = File(...),
    model: str = Form("GLM-OCR"),
    format: str = Form("all")
):
    """
    Live Server-Sent Events (SSE) streaming endpoint for batch processing.
    Emits real-time progress events (%65 - 32/50 fiş, current item latency, preview)
    and ends with summary and ZIP download link.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Hiçbir görsel dosyası yüklenmedi.")

    job_id = f"job_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    job_dir = os.path.join(BATCH_JOBS_DIR, job_id)
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = save_batch_upload_files(files, job_id)
    if not saved_paths:
        raise HTTPException(status_code=400, detail="Yüklenen dosyalarda geçerli görsel bulunamadı.")

    pipeline = BatchPipeline(
        input_dir=os.path.join(job_dir, "input"),
        output_dir=output_dir,
        model_key=model,
        output_format=format
    )

    async def batch_event_generator() -> AsyncGenerator[str, None]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def stream_worker():
            try:
                for event_dict in pipeline.stream_run(image_paths=saved_paths):
                    event_dict["batch_id"] = job_id
                    if event_dict.get("event") == "done":
                        zip_path = os.path.join(job_dir, f"visionocr_batch_{job_id}.zip")
                        BatchPipeline.create_zip_archive(output_dir, zip_path)
                        event_dict["zip_url"] = f"/api/batch/download/{job_id}"
                        event_dict["csv_url"] = f"/api/batch/csv/{job_id}"
                        event_dict["json_url"] = f"/api/batch/summary/{job_id}"

                    asyncio.run_coroutine_threadsafe(queue.put(event_dict), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception as exc:
                err_dict = {
                    "event": "error",
                    "batch_id": job_id,
                    "status": "error",
                    "error": str(exc)
                }
                asyncio.run_coroutine_threadsafe(queue.put(err_dict), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        import threading
        t = threading.Thread(target=stream_worker, daemon=True)
        t.start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                json_data = json.dumps(item, ensure_ascii=False, default=str)
                yield f"data: {json_data}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        batch_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/batch/download/{batch_id}")
async def download_batch_zip(batch_id: str):
    """Downloads all output files of a batch job as a ZIP archive."""
    job_dir = get_safe_batch_job_dir(batch_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Toplu işlem kaydı bulunamadı.")

    zip_path = os.path.join(job_dir, f"visionocr_batch_{batch_id}.zip")
    output_dir = os.path.join(job_dir, "output")

    if not os.path.isfile(zip_path):
        if os.path.isdir(output_dir):
            await asyncio.to_thread(BatchPipeline.create_zip_archive, output_dir, zip_path)
        else:
            raise HTTPException(status_code=404, detail="Çıktı arşivi bulunamadı.")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"visionocr_batch_{batch_id}.zip"
    )


@app.get("/api/batch/summary/{batch_id}")
async def get_batch_summary_json(batch_id: str):
    """Returns summary.json for a completed batch job."""
    job_dir = get_safe_batch_job_dir(batch_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Toplu işlem kaydı bulunamadı.")
    summary_path = os.path.join(job_dir, "output", "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Özet JSON dosyası bulunamadı.")


@app.get("/api/batch/csv/{batch_id}")
async def get_batch_summary_csv(batch_id: str):
    """Returns summary.csv for a completed batch job."""
    job_dir = get_safe_batch_job_dir(batch_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Toplu işlem kaydı bulunamadı.")
    csv_path = os.path.join(job_dir, "output", "summary.csv")
    if os.path.isfile(csv_path):
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename=f"summary_{batch_id}.csv"
        )
    raise HTTPException(status_code=404, detail="Özet CSV dosyası bulunamadı.")


# Serve static web frontend
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "VisionOCR Web Backend Active", "docs": "/docs"})

