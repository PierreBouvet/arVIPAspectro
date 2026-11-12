#!/usr/bin/env python3
"""
gui.py: Interactive Graphical User Interface launcher for arVIPAspectro.

Usage:
    python gui.py
    python gui.py path/to/Measures.h5
"""

import os
import sys

# Ensure repository root is in sys.path
_repo_root = os.path.abspath(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from arVIPAspectro.gui import launch_gui


def main():
    default_h5 = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        candidate = sys.argv[1]
        if os.path.exists(candidate):
            default_h5 = candidate

    # Fallback to config path if present
    if default_h5 is None:
        try:
            from arVIPAspectro import config
            if hasattr(config, "DATA_DIR") and hasattr(config, "TARGET_FILE"):
                cfg_path = os.path.join(config.DATA_DIR, config.TARGET_FILE)
                if os.path.exists(cfg_path):
                    default_h5 = cfg_path
        except Exception:
            pass

    sys.exit(launch_gui(default_h5=default_h5))


if __name__ == "__main__":
    main()
