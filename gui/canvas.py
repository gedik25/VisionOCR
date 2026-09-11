"""
Interactive Image Canvas (QGraphicsView).
Provides smooth mouse-wheel zooming, panning, drag & drop,
rotation, fitting, and bounding-box overlay support.
"""

import os
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QFrame
)


class InteractiveImageCanvas(QGraphicsView):
    """
    High-performance zoomable, pannable, interactive document canvas.
    """
    image_dropped = Signal(str)      # Emits image file path
    zoom_changed  = Signal(float)    # Emits current zoom level (1.0 = 100%)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Rendering & Performance hints
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAcceptDrops(True)

        # State
        self.pixmap_item = None
        self.image_path  = None
        self.current_zoom = 1.0
        self.rotation_angle = 0
        self.box_items = []

        # Empty state label
        self._setup_empty_state()

        # Floating Toolbar Overlay
        self._setup_floating_toolbar()

    def _setup_empty_state(self):
        """Displays friendly drop-zone text when no image is loaded."""
        self.empty_label = QLabel(
            "🖼️ Görseli buraya sürükleyip bırakın\nveya yukarıdan 'Görsel Seç'e tıklayın",
            self
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 14px;
                font-weight: 500;
                line-height: 1.6;
                background: transparent;
            }
        """)
        self.empty_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _setup_floating_toolbar(self):
        """Creates floating translucent action buttons in the top-right corner."""
        self.toolbar_container = QWidget(self)
        self.toolbar_container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self.toolbar_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.zoom_label = QLabel("100%", self.toolbar_container)
        self.zoom_label.setStyleSheet("""
            color: #94a3b8;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 8px;
            background-color: rgba(22, 24, 29, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
        """)

        self.btn_fit = QPushButton("⤢ Sığdır", self.toolbar_container)
        self.btn_fit.setObjectName("FloatingBtn")
        self.btn_fit.setToolTip("Görseli ekrana sığdır (⌘+0)")
        self.btn_fit.clicked.connect(self.fit_to_window)

        self.btn_actual = QPushButton("1:1", self.toolbar_container)
        self.btn_actual.setObjectName("FloatingBtn")
        self.btn_actual.setToolTip("Gerçek piksel boyutu (⌘+1)")
        self.btn_actual.clicked.connect(self.set_actual_size)

        self.btn_rotate = QPushButton("↷ 90°", self.toolbar_container)
        self.btn_rotate.setObjectName("FloatingBtn")
        self.btn_rotate.setToolTip("Saat yönünde 90° döndür")
        self.btn_rotate.clicked.connect(self.rotate_clockwise)

        self.btn_reset = QPushButton("↺ Sıfırla", self.toolbar_container)
        self.btn_reset.setObjectName("FloatingBtn")
        self.btn_reset.setToolTip("Görünümü ve dönüşümü sıfırla")
        self.btn_reset.clicked.connect(self.reset_view)

        layout.addWidget(self.zoom_label)
        layout.addWidget(self.btn_fit)
        layout.addWidget(self.btn_actual)
        layout.addWidget(self.btn_rotate)
        layout.addWidget(self.btn_reset)

        self.toolbar_container.hide()  # Show only when image is loaded

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position empty state
        self.empty_label.setGeometry(self.rect())
        # Position floating toolbar at top right with 12px margin
        tb_size = self.toolbar_container.sizeHint()
        self.toolbar_container.setGeometry(
            self.width() - tb_size.width() - 14,
            14,
            tb_size.width(),
            tb_size.height()
        )

    # ─────────────────────────────────────────────────────────
    # Image Loading & Management
    # ─────────────────────────────────────────────────────────
    def load_image(self, file_path: str) -> bool:
        """Loads an image into the scene and fits it to view."""
        if not os.path.isfile(file_path):
            return False

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            return False

        self.image_path = file_path
        self.scene.clear()
        self.box_items.clear()
        self.rotation_angle = 0
        self.resetTransform()

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        self.empty_label.hide()
        self.toolbar_container.show()
        self.toolbar_container.raise_()

        self.fit_to_window()
        return True

    # ─────────────────────────────────────────────────────────
    # Zoom, Pan & Transformation Actions
    # ─────────────────────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent):
        """Smooth zooming anchored under the cursor."""
        if not self.pixmap_item:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Zoom delta
        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        new_zoom = self.current_zoom * factor
        # Clamp zoom level between 5% and 3000%
        if 0.05 <= new_zoom <= 30.0:
            self.scale(factor, factor)
            self.current_zoom = new_zoom
            self._update_zoom_label()

    def fit_to_window(self):
        """Fits the entire image inside the current viewport."""
        if not self.pixmap_item:
            return
        self.resetTransform()
        if self.rotation_angle != 0:
            self.rotate(self.rotation_angle)

        scene_rect = self.scene.sceneRect()
        if scene_rect.isEmpty():
            return

        self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
        # Calculate current effective zoom factor
        transform = self.transform()
        self.current_zoom = transform.m11()
        self._update_zoom_label()

    def set_actual_size(self):
        """Resets zoom to exactly 100% (1:1 pixel mapping)."""
        if not self.pixmap_item:
            return
        self.resetTransform()
        if self.rotation_angle != 0:
            self.rotate(self.rotation_angle)
        self.current_zoom = 1.0
        self._update_zoom_label()
        self.centerOn(self.pixmap_item)

    def rotate_clockwise(self):
        """Rotates image 90 degrees clockwise."""
        if not self.pixmap_item:
            return
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.rotate(90)
        self.fit_to_window()

    def reset_view(self):
        """Resets zoom, rotation, and center."""
        if not self.pixmap_item:
            return
        self.rotation_angle = 0
        self.resetTransform()
        self.fit_to_window()

    def _update_zoom_label(self):
        pct = int(self.current_zoom * 100)
        self.zoom_label.setText(f"{pct}%")
        self.zoom_changed.emit(self.current_zoom)

    # ─────────────────────────────────────────────────────────
    # Bounding Boxes Overlay (for Detection Models / MMOCR)
    # ─────────────────────────────────────────────────────────
    def set_bounding_boxes(self, boxes: list):
        """
        Draws semi-transparent bounding boxes over detected text regions.
        boxes: list of dicts [{'box': [x1, y1, x2, y2], 'text': '...'}]
        """
        self.clear_bounding_boxes()
        if not self.pixmap_item or not boxes:
            return

        pen = QPen(QColor(10, 132, 255, 220), 2)
        brush = QBrush(QColor(10, 132, 255, 45))

        for item in boxes:
            box = item.get("box", [])
            text = item.get("text", "")
            if len(box) >= 4:
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                rect_item = QGraphicsRectItem(x, y, w, h)
                rect_item.setPen(pen)
                rect_item.setBrush(brush)
                if text:
                    rect_item.setToolTip(f"Metin: {text}")
                self.scene.addItem(rect_item)
                self.box_items.append(rect_item)

    def clear_bounding_boxes(self):
        """Removes all bounding box items from the scene."""
        for item in self.box_items:
            self.scene.removeItem(item)
        self.box_items.clear()

    # ─────────────────────────────────────────────────────────
    # Drag and Drop Events
    # ─────────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                    if self.load_image(file_path):
                        self.image_dropped.emit(file_path)
                        event.acceptProposedAction()
                        return
        event.ignore()
