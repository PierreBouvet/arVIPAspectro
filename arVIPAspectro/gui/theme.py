"""
theme.py: Dynamic System Day/Night (Light/Dark) Theme Manager for arVIPAspectro GUI.
"""

from dataclasses import dataclass
from typing import Dict, Any
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class MplTheme:
    fig_bg: str
    ax_bg: str
    text_title: str
    text_label: str
    text_tick: str
    spine: str
    grid: str
    grid_alpha: float
    legend_bg: str
    legend_edge: str
    cursor_text: str


MPL_DARK = MplTheme(
    fig_bg="#1e1e1e",
    ax_bg="#121212",
    text_title="#ffffff",
    text_label="#cccccc",
    text_tick="#aaaaaa",
    spine="#444444",
    grid="#555555",
    grid_alpha=0.35,
    legend_bg="#262626",
    legend_edge="#555555",
    cursor_text="#00e5ff",
)

MPL_LIGHT = MplTheme(
    fig_bg="#ffffff",
    ax_bg="#fafafa",
    text_title="#111827",
    text_label="#374151",
    text_tick="#4b5563",
    spine="#d1d5db",
    grid="#9ca3af",
    grid_alpha=0.30,
    legend_bg="#ffffff",
    legend_edge="#d1d5db",
    cursor_text="#0288d1",
)


def get_system_theme() -> str:
    """Detects whether macOS / system is currently in Dark or Light mode."""
    hints = QtGui.QGuiApplication.styleHints()
    if hasattr(hints, "colorScheme"):
        scheme = hints.colorScheme()
        if scheme == QtCore.Qt.ColorScheme.Dark:
            return "dark"
        elif scheme == QtCore.Qt.ColorScheme.Light:
            return "light"

    # Fallback to palette luminance check
    palette = QtGui.QGuiApplication.palette()
    bg_lum = palette.color(QtGui.QPalette.Window).value()
    return "dark" if bg_lum < 128 else "light"


def get_mpl_theme(theme_name: str) -> MplTheme:
    """Returns the Matplotlib theme settings for 'dark' or 'light'."""
    return MPL_DARK if theme_name == "dark" else MPL_LIGHT


def apply_axis_theme(ax, theme: MplTheme, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Applies theme colors to a Matplotlib axis."""
    ax.set_facecolor(theme.ax_bg)
    if title:
        ax.set_title(title, color=theme.text_title, fontsize=11, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=theme.text_label, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme.text_label, fontsize=10)
    ax.tick_params(colors=theme.text_tick)
    for spine in ax.spines.values():
        spine.set_color(theme.spine)


# ==============================================================================
# Dark Theme Stylesheet
# ==============================================================================

DARK_QSS = """
QMainWindow, QWidget#centralWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
}
QSplitter::handle {
    background-color: #2d2d2d;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}
QTabWidget::pane {
    border: 1px solid #333333;
    background-color: #1e1e1e;
}
QTabBar::tab {
    background-color: #282828;
    color: #aaaaaa;
    padding: 7px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #0288d1;
    color: #ffffff;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background-color: #383838;
    color: #ffffff;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #2b2b2b;
    color: #ffffff;
    border: 1px solid #444444;
    border-radius: 3px;
    padding: 3px 6px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #00bcd4;
}
QGroupBox {
    border: 1px solid #383838;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: bold;
    color: #dcdcdc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
CollapsibleCard[isMajor="true"] {
    background-color: #262626;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    margin: 2px 0px;
}
CollapsibleCard[isMajor="false"] {
    background-color: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 4px;
    margin: 2px 0px;
}
QTreeWidget {
    background-color: #242424;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
}
QTreeWidget::item:selected {
    background-color: #0288d1;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #1a1a1a;
    color: #aaaaaa;
    border: none;
    padding: 4px;
}
QScrollBar:vertical {
    background: #1e1e1e;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #444444;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #666666;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QProgressBar {
    border: 1px solid #444444;
    border-radius: 3px;
    text-align: center;
    color: white;
    background-color: #252525;
}
QProgressBar::chunk {
    background-color: #00bcd4;
}
QPlainTextEdit {
    background-color: #121212;
    color: #a5d6a7;
    font-family: Menlo, Monaco, "Courier New", monospace;
    font-size: 11px;
    border: 1px solid #333333;
}
QStatusBar {
    background-color: #1a1a1a;
    color: #9e9e9e;
    border-top: 1px solid #2d2d2d;
}
"""


# ==============================================================================
# Light Theme Stylesheet (Day Mode)
# ==============================================================================

LIGHT_QSS = """
QMainWindow, QWidget#centralWidget {
    background-color: #f3f4f6;
    color: #111827;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
}
QSplitter::handle {
    background-color: #e5e7eb;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}
QTabWidget::pane {
    border: 1px solid #d1d5db;
    background-color: #ffffff;
}
QTabBar::tab {
    background-color: #e5e7eb;
    color: #4b5563;
    padding: 7px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #0288d1;
    color: #ffffff;
    font-weight: bold;
}
QTabBar::tab:hover:!selected {
    background-color: #d1d5db;
    color: #111827;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 3px 6px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0288d1;
}
QPushButton {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 4px 10px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #f1f5f9;
    border: 1px solid #94a3b8;
}
QPushButton:pressed {
    background-color: #e2e8f0;
}
QGroupBox {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-weight: bold;
    color: #1e293b;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
CollapsibleCard[isMajor="true"] {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    margin: 2px 0px;
}
CollapsibleCard[isMajor="false"] {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    margin: 2px 0px;
}
QTreeWidget {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
}
QTreeWidget::item:selected {
    background-color: #0288d1;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #f1f5f9;
    color: #475569;
    border: none;
    padding: 4px;
    font-weight: 600;
}
QScrollBar:vertical {
    background: #f1f5f9;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    text-align: center;
    color: #1e293b;
    background-color: #e2e8f0;
}
QProgressBar::chunk {
    background-color: #0288d1;
}
QPlainTextEdit {
    background-color: #ffffff;
    color: #1e3a1e;
    font-family: Menlo, Monaco, "Courier New", monospace;
    font-size: 11px;
    border: 1px solid #cbd5e1;
}
QStatusBar {
    background-color: #f1f5f9;
    color: #475569;
    border-top: 1px solid #e2e8f0;
}
"""
