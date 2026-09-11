"""
Custom Widgets for PySide6 OCR Studio.
Contains TopBar, ResultsPanel, and StatusBadge components.
"""

import json
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QComboBox, QLabel, QFrame, QTabWidget, QTextBrowser,
    QPlainTextEdit, QFileDialog, QMessageBox, QProgressBar
)
from gui.theme import (
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, FONT_MONO
)


class TopBar(QFrame):
    """
    Top navigation and action bar with model selector and execution trigger.
    """
    image_selected = Signal()
    run_ocr_requested = Signal(str)  # Emits selected model key

    MODELS = [
        ("GLM-OCR", "GLM-OCR (Yapay Zeka Destekli Belge OCR)"),
        ("LightOnOCR-2-1B", "LightOnOCR-2-1B (Belge & Tablo OCR)"),
        ("MMOCR", "MMOCR (DBNet Tespit + CRNN Tanıma)"),
        ("CRNN", "CRNN (Hızlı Karakter Tanıma)")
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBarFrame")
        self.setFixedHeight(56)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # 1. Branding / Title
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_box.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel("⚡ VisionOCR Studio")
        self.title_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            letter-spacing: -0.2px;
        """)
        
        self.subtitle_label = QLabel("Yapay Zeka Destekli Doküman & Metin Ayrıştırma")
        self.subtitle_label.setStyleSheet(f"""
            font-size: 10.5px;
            color: {TEXT_MUTED};
        """)
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.subtitle_label)
        layout.addLayout(title_box)

        layout.addSpacing(12)

        # 2. Model Selector
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(34)
        for key, display_name in self.MODELS:
            self.model_combo.addItem(display_name, key)
        self.model_combo.setToolTip("Kullanılacak OCR Modelini Seçin")
        layout.addWidget(self.model_combo)

        # 3. Open Image Button
        self.btn_open = QPushButton("📁 Görsel Seç")
        self.btn_open.setFixedHeight(34)
        self.btn_open.setToolTip("Bilgisayardan bir görsel seçin (⌘+O)")
        self.btn_open.clicked.connect(self.image_selected.emit)
        layout.addWidget(self.btn_open)

        # 4. Run OCR Button (Primary)
        self.btn_run = QPushButton("⚡ OCR Çalıştır")
        self.btn_run.setObjectName("PrimaryBtn")
        self.btn_run.setFixedHeight(34)
        self.btn_run.setToolTip("Seçilen modeli çalıştır (⌘+Return)")
        self.btn_run.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.btn_run)

        layout.addStretch()

        # 5. Status / Latency Badge
        self.badge_frame = QFrame()
        self.badge_frame.setObjectName("StatusCard")
        self.badge_frame.setFixedHeight(30)
        badge_layout = QHBoxLayout(self.badge_frame)
        badge_layout.setContentsMargins(10, 2, 10, 2)
        badge_layout.setSpacing(6)
        badge_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 10px;")
        
        self.status_text = QLabel("Hazır")
        self.status_text.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 500;")

        badge_layout.addWidget(self.status_indicator)
        badge_layout.addWidget(self.status_text)
        layout.addWidget(self.badge_frame)

    def _on_run_clicked(self):
        selected_model_key = self.model_combo.currentData()
        self.run_ocr_requested.emit(selected_model_key)

    def set_loading_state(self, is_loading: bool, model_name: str = ""):
        """Updates buttons and status badge during processing."""
        if is_loading:
            self.btn_run.setEnabled(False)
            self.btn_run.setText("⏳ İşleniyor...")
            self.btn_open.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.status_indicator.setStyleSheet(f"color: {ACCENT_AMBER}; font-size: 10px;")
            self.status_text.setText(f"{model_name} Çalışıyor...")
        else:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("⚡ OCR Çalıştır")
            self.btn_open.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.status_indicator.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 10px;")
            self.status_text.setText("Hazır")

    def set_success_state(self, elapsed_sec: float, device: str):
        """Displays completion metrics in badge."""
        self.status_indicator.setStyleSheet(f"color: {ACCENT_GREEN}; font-size: 10px;")
        self.status_text.setText(f"{device} | {elapsed_sec:.2f} sn")


class ResultsPanel(QWidget):
    """
    Multi-tab inspection and export panel for recognized OCR content.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_result = {}
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. Tabs
        self.tabs = QTabWidget()

        # Tab 1: Formatted / Markdown View
        self.txt_markdown = QTextBrowser()
        self.txt_markdown.setOpenExternalLinks(True)
        self.txt_markdown.setPlaceholderText("OCR çalıştırıldığında biçimlendirilmiş metin burada görünecektir...")
        self.tabs.addTab(self.txt_markdown, "📝 Markdown / Tablo")

        # Tab 2: Plain Text View
        self.txt_plain = QPlainTextEdit()
        self.txt_plain.setPlaceholderText("Düz metin çıktısı...")
        self.tabs.addTab(self.txt_plain, "📄 Düz Metin")

        # Tab 3: JSON Output
        self.txt_json = QPlainTextEdit()
        self.txt_json.setStyleSheet(f"font-family: {FONT_MONO}; font-size: 12px;")
        self.txt_json.setPlaceholderText("Yapılandırılmış JSON verisi...")
        self.tabs.addTab(self.txt_json, "⚙️ Ham JSON")

        # Tab 4: Performance & Metadata
        self.txt_stats = QTextBrowser()
        self.txt_stats.setPlaceholderText("Model ve çıkarım istatistikleri...")
        self.tabs.addTab(self.txt_stats, "📊 İstatistik")

        main_layout.addWidget(self.tabs)

        # 2. Bottom Action Bar
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.btn_copy = QPushButton("📋 Panoya Kopyala")
        self.btn_copy.setToolTip("Tanınan metni panoya kopyala")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)

        self.btn_export_txt = QPushButton("💾 .txt")
        self.btn_export_txt.setToolTip("Düz metin dosyası olarak kaydet")
        self.btn_export_txt.clicked.connect(self._export_txt)

        self.btn_export_json = QPushButton("💾 .json")
        self.btn_export_json.setToolTip("JSON formatında kaydet")
        self.btn_export_json.clicked.connect(self._export_json)

        action_layout.addWidget(self.btn_copy)
        action_layout.addStretch()
        action_layout.addWidget(self.btn_export_txt)
        action_layout.addWidget(self.btn_export_json)

        main_layout.addLayout(action_layout)

    def display_results(self, result: dict):
        """Populates all tabs with inference results."""
        self.current_result = result
        raw_text = result.get("text", "")
        markdown_text = result.get("markdown", raw_text)
        raw_dict = result.get("raw", {})
        elapsed_sec = result.get("elapsed_sec", 0.0)
        device = result.get("device", "Bilinmiyor")
        model_key = result.get("model_key", "OCR Model")
        boxes = result.get("boxes", [])

        # 1. Markdown
        self.txt_markdown.setMarkdown(markdown_text)

        # 2. Plain Text
        self.txt_plain.setPlainText(raw_text)

        # 3. JSON
        clean_json = {
            "model": model_key,
            "device": device,
            "latency_seconds": round(elapsed_sec, 4),
            "text": raw_text,
            "detected_boxes_count": len(boxes),
            "raw_output": raw_dict
        }
        self.txt_json.setPlainText(json.dumps(clean_json, indent=2, ensure_ascii=False))

        # 4. Stats
        word_count = len(raw_text.split())
        char_count = len(raw_text)
        stats_html = f"""
        <div style='font-family: -apple-system, sans-serif; line-height: 1.6; color: #f1f5f9;'>
            <h3 style='color: #0a84ff; margin-bottom: 12px;'>Çıkarım Raporu</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr><td style='color: #94a3b8; padding: 6px 0;'>Model:</td><td style='font-weight: 600;'>{model_key}</td></tr>
                <tr><td style='color: #94a3b8; padding: 6px 0;'>Hesaplama Cihazı:</td><td style='color: #10b981; font-weight: 600;'>{device}</td></tr>
                <tr><td style='color: #94a3b8; padding: 6px 0;'>İşlem Süresi:</td><td style='font-weight: 600;'>{elapsed_sec:.3f} saniye</td></tr>
                <tr><td style='color: #94a3b8; padding: 6px 0;'>Toplam Kelime:</td><td>{word_count}</td></tr>
                <tr><td style='color: #94a3b8; padding: 6px 0;'>Toplam Karakter:</td><td>{char_count}</td></tr>
                <tr><td style='color: #94a3b8; padding: 6px 0;'>Tespit Edilen Kutu:</td><td>{len(boxes)} adet</td></tr>
            </table>
        </div>
        """
        self.txt_stats.setHtml(stats_html)

    def clear(self):
        """Clears all result fields."""
        self.current_result = {}
        self.txt_markdown.clear()
        self.txt_plain.clear()
        self.txt_json.clear()
        self.txt_stats.clear()

    def _copy_to_clipboard(self):
        text = self.txt_plain.toPlainText()
        if text:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(text)
            QMessageBox.information(self, "Kopyalandı", "Metin panoya kopyalandı! ✅")
        else:
            QMessageBox.warning(self, "Uyarı", "Kopyalanacak metin bulunamadı.")

    def _export_txt(self):
        text = self.txt_plain.toPlainText()
        if not text:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek metin bulunamadı.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Metin Olarak Kaydet", "ocr_cikti.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(self, "Kaydedildi", f"Dosya başarıyla kaydedildi:\n{path}")

    def _export_json(self):
        json_content = self.txt_json.toPlainText()
        if not json_content:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek JSON verisi bulunamadı.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "JSON Olarak Kaydet", "ocr_cikti.json", "JSON Files (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_content)
            QMessageBox.information(self, "Kaydedildi", f"Dosya başarıyla kaydedildi:\n{path}")
