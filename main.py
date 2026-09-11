"""
VisionOCR Studio — Main Entry Point.
Initializes High-DPI application, macOS dark title bar, and launches the UI.
"""

import sys
import os
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt

# Ensure root directory is on Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.app import MainWindow


def main():
    # 1. Enable High DPI Scaling (Must be set BEFORE QApplication)
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    if hasattr(Qt.HighDpiScaleFactorRoundingPolicy, "PassThrough"):
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("VisionOCR Studio")
    app.setOrganizationName("BitirmeProjesi")

    # 2. Instantiate and show Main Window
    window = MainWindow()
    window.show()

    # 3. Application Event Loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
