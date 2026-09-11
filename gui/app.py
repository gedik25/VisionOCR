"""
Main Application Window (MainWindow).
Integrates TopBar, Interactive Canvas, Results Panel, and Worker Threads into a cohesive UI.
"""

import os
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeySequence, QShortcut, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QFrame
)

from gui.theme import DARK_STYLESHEET, BORDER_SUBTLE
from gui.canvas import InteractiveImageCanvas
from gui.widgets import TopBar, ResultsPanel
from gui.workers import OcrWorker


class MainWindow(QMainWindow):
    """
    Main VisionOCR Studio application window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionOCR Studio — Deep Learning OCR Workbench")
        self.resize(1200, 780)
        self.setMinimumSize(950, 600)

        self.current_image_path = None
        self.current_worker = None

        # Apply global dark stylesheet
        self.setStyleSheet(DARK_STYLESHEET)

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Control Bar
        self.top_bar = TopBar(self)
        self.top_bar.image_selected.connect(self._open_image_dialog)
        self.top_bar.run_ocr_requested.connect(self._start_ocr_inference)
        main_layout.addWidget(self.top_bar)

        # 2. Main Content Splitter (Side-by-side Inspection)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setContentsMargins(12, 12, 12, 12)
        self.splitter.setChildrenCollapsible(False)

        # Left: Interactive Canvas
        self.canvas = InteractiveImageCanvas(self)
        self.canvas.image_dropped.connect(self._on_image_loaded)
        self.splitter.addWidget(self.canvas)

        # Right: Results Panel
        self.results_panel = ResultsPanel(self)
        self.splitter.addWidget(self.results_panel)

        # 55% Left (Image) / 45% Right (Text) Initial Split Ratio
        self.splitter.setSizes([650, 500])

        main_layout.addWidget(self.splitter)

    def _setup_shortcuts(self):
        """Standard keyboard shortcuts for fast workflows."""
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_image_dialog)
        QShortcut(QKeySequence("Ctrl+Return"), self, self._trigger_current_run)
        QShortcut(QKeySequence("Ctrl+0"), self, self.canvas.fit_to_window)
        QShortcut(QKeySequence("Ctrl+1"), self, self.canvas.set_actual_size)

    def _trigger_current_run(self):
        selected_model = self.top_bar.model_combo.currentData()
        self._start_ocr_inference(selected_model)

    def _open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Görsel Seç",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp *.bmp *.tiff);;All Files (*)"
        )
        if file_path:
            self._load_image_file(file_path)

    def _load_image_file(self, file_path: str):
        if self.canvas.load_image(file_path):
            self._on_image_loaded(file_path)

    def _on_image_loaded(self, file_path: str):
        self.current_image_path = file_path
        file_name = os.path.basename(file_path)
        self.top_bar.status_text.setText(f"Görsel: {file_name}")
        self.results_panel.clear()
        self.canvas.clear_bounding_boxes()

    def _start_ocr_inference(self, model_key: str):
        if not self.current_image_path or not os.path.isfile(self.current_image_path):
            QMessageBox.warning(
                self,
                "Görsel Eksik",
                "Lütfen önce sol panele bir görsel yükleyin veya 'Görsel Seç' butonuna tıklayın."
            )
            return

        # Start Async Worker
        self.top_bar.set_loading_state(True, model_name=model_key)
        self.canvas.clear_bounding_boxes()

        self.current_worker = OcrWorker(model_key, self.current_image_path)
        self.current_worker.finished_signal.connect(self._on_ocr_finished)
        self.current_worker.error_signal.connect(self._on_ocr_error)
        self.current_worker.start()

    def _on_ocr_finished(self, result: dict):
        self.top_bar.set_loading_state(False)
        self.top_bar.set_success_state(
            elapsed_sec=result.get("elapsed_sec", 0.0),
            device=result.get("device", "MPS")
        )

        # Update Results Tabs
        self.results_panel.display_results(result)

        # Draw Bounding Boxes on Canvas if any
        boxes = result.get("boxes", [])
        if boxes:
            self.canvas.set_bounding_boxes(boxes)

    def _on_ocr_error(self, err_msg: str):
        self.top_bar.set_loading_state(False)
        QMessageBox.critical(
            self,
            "OCR Hatası",
            f"Model çalıştırılırken bir hata oluştu:\n\n{err_msg}"
        )
