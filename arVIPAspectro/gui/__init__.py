"""
arVIPAspectro GUI Package
"""

import sys
from typing import Optional
from PySide6 import QtWidgets, QtCore
from .app import MainWindow


def launch_gui(default_h5: Optional[str] = None):
    """
    Initializes the PySide6 application and starts the main GUI loop.
    """
    # High DPI scaling attributes
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    app.setApplicationName("arVIPAspectro")
    app.setOrganizationName("arVIPAspectro")

    window = MainWindow(default_h5=default_h5)
    window.show()

    return app.exec()


__all__ = ["launch_gui", "MainWindow"]
