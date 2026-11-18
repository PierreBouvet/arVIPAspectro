"""
Hardware drivers package for arVIPAspectro:
  - `MS2000Stage`: Applied Scientific Instrumentation (ASI) MS-2000 motorized stage.
  - `OrcaFlashCamera`: Hamamatsu Orca Flash 4.0 v3 sCMOS camera (via DCAM-API / `pylablib`).
  - `TeledyneBrightfieldCamera`: Teledyne FLIR CM3-U3-31S4M-CS brightfield camera (via Spinnaker / `pylablib`).
"""

from .stage_ms2000 import MS2000Stage, MS2000Error, list_serial_ports
from .camera_orca import OrcaFlashCamera, OrcaCameraError, count_dcam_cameras
from .camera_teledyne import (
    TeledyneBrightfieldCamera,
    TeledyneCameraError,
    count_teledyne_cameras,
    list_teledyne_cameras,
    find_chameleon3_camera,
    diagnose_teledyne_backend,
)

__all__ = [
    "MS2000Stage",
    "MS2000Error",
    "list_serial_ports",
    "OrcaFlashCamera",
    "OrcaCameraError",
    "count_dcam_cameras",
    "TeledyneBrightfieldCamera",
    "TeledyneCameraError",
    "count_teledyne_cameras",
    "list_teledyne_cameras",
    "find_chameleon3_camera",
    "diagnose_teledyne_backend",
]
