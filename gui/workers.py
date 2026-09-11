"""
Asynchronous OCR Worker (QThread).
Executes model inference in an isolated subprocess to prevent PyTorch/Metal
crashes with Qt's Cocoa/Metal render thread on macOS Apple Silicon.
"""

import sys
import os
import json
import subprocess
from PySide6.QtCore import QThread, Signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
RUNNER_SCRIPT = os.path.join(PROJECT_ROOT, "ocr_runner.py")


class OcrWorker(QThread):
    """
    Subprocess-backed OCR worker thread.
    Safely isolates deep learning inference from the GUI main thread.
    """
    started_signal  = Signal()
    finished_signal = Signal(dict)   # Emits parsed result dictionary
    error_signal    = Signal(str)    # Emits error message

    def __init__(self, model_key: str, image_path: str, parent=None):
        super().__init__(parent)
        self.model_key  = model_key
        self.image_path = image_path

    def run(self):
        self.started_signal.emit()

        python_exec = VENV_PYTHON if os.path.isfile(VENV_PYTHON) else sys.executable
        cmd = [
            python_exec,
            RUNNER_SCRIPT,
            "--model", self.model_key,
            "--image", self.image_path
        ]

        # Pass HF token if available in current env
        env = os.environ.copy()
        if "HF_TOKEN" in os.environ:
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=PROJECT_ROOT
            )
            stdout, stderr = process.communicate()

            if "__OCR_RESULT_START__" in stdout:
                json_part = stdout.split("__OCR_RESULT_START__")[1].split("__OCR_RESULT_END__")[0].strip()
                result = json.loads(json_part)

                if result.get("status") == "error":
                    err_msg = result.get("error", "Bilinmeyen model hatası")
                    tb = result.get("traceback", "")
                    self.error_signal.emit(f"{err_msg}\n\n{tb}")
                else:
                    self.finished_signal.emit(result)
            else:
                err_text = stderr.strip() if stderr else stdout.strip()
                self.error_signal.emit(f"Model çalıştırılamadı (Çıkış Kodu: {process.returncode}):\n\n{err_text}")

        except Exception as e:
            self.error_signal.emit(f"İşlem başlatma hatası: {str(e)}")
