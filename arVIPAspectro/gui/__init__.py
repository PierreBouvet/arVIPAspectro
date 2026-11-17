"""
GUI package for arVIPAspectro hardware acquisition and on-the-fly analysis.
"""

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtWidgets
from .app import MainWindow


def launch_gui() -> int:
    """Launches the PySide6 acquisition application."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("arVIPAspectro Acquisition")

    window = MainWindow()
    window.show()
    return app.exec()


__all__ = ["MainWindow", "launch_gui"]
