"""
Control & State Machine Layer for arVIPAspectro.

Sits between the pure Hardware Abstraction Layer (`arVIPAspectro.hardware`) and
the PySide6 GUI (`arVIPAspectro.gui`), centralizing rig state transitions,
canonical move-and-settle / acquisition sequences, safety halt, and dry-run proxies.
"""

from .state import RigState
from .rig_controller import (
    DryRunStage,
    RigController,
    StageActionWorker,
    PlateAutoCalibrateWorker,
    PlateJogCaptureWorker,
)

__all__ = [
    "RigState",
    "DryRunStage",
    "RigController",
    "StageActionWorker",
    "PlateAutoCalibrateWorker",
    "PlateJogCaptureWorker",
]
