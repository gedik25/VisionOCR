"""
Modern Dark Theme & Design Tokens for PySide6 Desktop OCR App.
Inspired by Apple HIG, Linear-clean aesthetics, and Impeccable standards.
"""

# ─────────────────────────────────────────────────────────────
# Color Tokens
# ─────────────────────────────────────────────────────────────
BG_BASE      = "#0f1013"   # Deep neutral background
BG_SURFACE   = "#16181d"   # Panels, toolbars, sidebar
BG_CARD      = "#1c1f26"   # Inputs, inner containers, canvas bg
BG_HOVER     = "#262a34"   # Hover state for controls
BG_ACTIVE    = "#323744"   # Active / Pressed state

BORDER_SUBTLE = "#282c37"  # 1px border for containers
BORDER_FOCUS  = "#3b82f6"  # Highlight / Focus border

TEXT_PRIMARY   = "#f1f5f9" # Bright, high readability
TEXT_SECONDARY = "#94a3b8" # Subtext, labels
TEXT_MUTED     = "#64748b" # Placeholders, hints

ACCENT_BLUE    = "#0a84ff" # Apple System Blue
ACCENT_BLUE_HOVER = "#0071e3"
ACCENT_GREEN   = "#10b981" # Success badge
ACCENT_AMBER   = "#f59e0b" # Warning / Processing badge

FONT_FAMILY = "-apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_MONO   = "Menlo, Monaco, 'Courier New', monospace"

# ─────────────────────────────────────────────────────────────
# Complete Qt Style Sheet (QSS)
# ─────────────────────────────────────────────────────────────
DARK_STYLESHEET = f"""
/* ── Global Reset ── */
QWidget {{
    background-color: {BG_BASE};
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: 13px;
    selection-background-color: {ACCENT_BLUE};
    selection-color: #ffffff;
}}

/* ── Main Window ── */
QMainWindow {{
    background-color: {BG_BASE};
}}

/* ── Top Bar Container ── */
#TopBarFrame {{
    background-color: {BG_SURFACE};
    border-bottom: 1px solid {BORDER_SUBTLE};
    max-height: 54px;
}}

/* ── Status & Info Cards ── */
#StatusCard {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 3px 10px;
}}

/* ── Standard Buttons ── */
QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: #3b4252;
}}

QPushButton:pressed {{
    background-color: {BG_ACTIVE};
}}

QPushButton:disabled {{
    background-color: #14161b;
    color: {TEXT_MUTED};
    border-color: #1e2129;
}}

/* ── Primary Action Button (Run OCR) ── */
QPushButton#PrimaryBtn {{
    background-color: {ACCENT_BLUE};
    color: #ffffff;
    border: 1px solid #1a75ff;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 7px;
}}

QPushButton#PrimaryBtn:hover {{
    background-color: {ACCENT_BLUE_HOVER};
    border-color: #0064cc;
}}

QPushButton#PrimaryBtn:pressed {{
    background-color: #0056b3;
}}

QPushButton#PrimaryBtn:disabled {{
    background-color: #1a2a44;
    color: #5a7094;
    border-color: #1c2e4a;
}}

/* ── Floating Canvas Buttons ── */
QPushButton#FloatingBtn {{
    background-color: rgba(28, 31, 38, 0.85);
    color: {TEXT_PRIMARY};
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
}}

QPushButton#FloatingBtn:hover {{
    background-color: rgba(45, 50, 62, 0.95);
    border-color: rgba(255, 255, 255, 0.25);
}}

/* ── Combo Box (Model Selector) ── */
QComboBox {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 7px;
    padding: 6px 12px;
    min-width: 180px;
    font-weight: 500;
}}

QComboBox:hover {{
    background-color: {BG_HOVER};
    border-color: #3b4252;
}}

QComboBox:focus {{
    border-color: {ACCENT_BLUE};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: none;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    selection-background-color: {ACCENT_BLUE};
    selection-color: #ffffff;
    padding: 4px;
    outline: none;
}}

/* ── Tabs (Results Panel) ── */
QTabWidget::pane {{
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    background-color: {BG_SURFACE};
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 500;
    border-bottom: 2px solid transparent;
}}

QTabBar::tab:hover {{
    color: {TEXT_PRIMARY};
    background-color: {BG_HOVER};
}}

QTabBar::tab:selected {{
    color: {TEXT_PRIMARY};
    background-color: {BG_SURFACE};
    border-bottom: 2px solid {ACCENT_BLUE};
    font-weight: 600;
}}

/* ── Text Areas & Outputs ── */
QTextBrowser, QPlainTextEdit, QTextEdit {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 12px;
    font-family: {FONT_FAMILY};
    font-size: 13.5px;
    line-height: 1.5;
}}

QTextBrowser:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT_BLUE};
}}

/* ── Graphics View (Canvas) ── */
QGraphicsView {{
    background-color: #0b0c0e;
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
}}

/* ── Splitter ── */
QSplitter::handle {{
    background-color: {BORDER_SUBTLE};
}}

QSplitter::handle:horizontal {{
    width: 2px;
    margin: 4px 0;
}}

QSplitter::handle:hover {{
    background-color: {ACCENT_BLUE};
}}

/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px 4px 0;
}}

QScrollBar::handle:vertical {{
    background: #2d323f;
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: #3f4657;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
    border: none;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0 4px 2px 4px;
}}

QScrollBar::handle:horizontal {{
    background: #2d323f;
    min-width: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: #3f4657;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
    border: none;
}}

/* ── Tooltips ── */
QToolTip {{
    background-color: #1e222b;
    color: {TEXT_PRIMARY};
    border: 1px solid #3b4252;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
}}
"""
