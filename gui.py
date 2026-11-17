#!/usr/bin/env python3
"""
gui.py: Interactive Graphical User Interface launcher for arVIPAspectro Hardware Acquisition.

Usage:
    python gui.py
"""

import os
import sys

# Force Qt bindings (such as pylablib/pyqtgraph) to use PySide6
os.environ.setdefault("QT_API", "pyside6")

# Ensure repository root is in sys.path
_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from arVIPAspectro.gui import launch_gui


def main():
    sys.exit(launch_gui())


if __name__ == "__main__":
    main()
