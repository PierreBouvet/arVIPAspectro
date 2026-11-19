"""
Hardware Abstraction Layer (HAL) protocols and shared acquisition helpers.

Defines runtime-checkable `Protocol` interfaces for motorized stages and cameras
so that control-layer state machines, dry-run proxies, and unit tests can interact
with real hardware drivers and simulators uniformly.
"""

from typing import Callable, Dict, Optional, Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class StageProtocol(Protocol):
    """Abstract interface for a 2- or 3-axis motorized stage (µm coordinates)."""

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_simulated(self) -> bool: ...

    @property
    def has_z_axis(self) -> bool: ...

    def connect(
        self,
        port: str = "auto",
        baudrate: int = 115200,
        simulate: bool = False,
    ) -> Dict[str, object]: ...

    def disconnect(self) -> None: ...

    def is_busy(self) -> bool: ...

    def get_position(self) -> Dict[str, Optional[float]]: ...

    def move_to(
        self,
        x_um: Optional[float] = None,
        y_um: Optional[float] = None,
        z_um: Optional[float] = None,
        wait: bool = True,
        timeout_s: float = 30.0,
    ) -> Dict[str, Optional[float]]: ...

    def move_relative(
        self,
        dx_um: float = 0.0,
        dy_um: float = 0.0,
        dz_um: float = 0.0,
        wait: bool = True,
        timeout_s: float = 30.0,
    ) -> Dict[str, Optional[float]]: ...

    def wait_for_stop(self, timeout_s: float = 30.0, poll_interval_s: float = 0.02) -> None: ...

    def zero_position(self, axes: Tuple[str, ...] = ("X", "Y")) -> Dict[str, Optional[float]]: ...

    def start_continuous_move(self, vx_um_s: float = 0.0, vy_um_s: float = 0.0) -> None: ...

    def halt(self) -> None: ...

    def get_status(self) -> Dict[str, object]: ...


@runtime_checkable
class CameraProtocol(Protocol):
    """Abstract interface for a scientific camera supporting snapped/averaged frames."""

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_simulated(self) -> bool: ...

    @property
    def model(self) -> str: ...

    @property
    def serial_number(self) -> str: ...

    @property
    def detector_size(self) -> Tuple[int, int]: ...

    def disconnect(self) -> None: ...

    def get_exposure(self) -> float: ...

    def set_exposure(self, exposure_s: float) -> float: ...

    def get_frame_averaging(self) -> int: ...

    def set_frame_averaging(self, n_frames: int) -> int: ...

    def get_roi(self) -> Tuple[int, ...]: ...

    def snap_frame(
        self,
        n_frames: Optional[int] = None,
        timeout_s: Optional[float] = None,
        stage_pos_um: Optional[Dict[str, Optional[float]]] = None,
    ) -> np.ndarray: ...

    def get_status(self) -> Dict[str, object]: ...


@runtime_checkable
class LiveCameraProtocol(CameraProtocol, Protocol):
    """Extended camera interface supporting continuous live streaming."""

    def start_live(self, buffsize: int = 16) -> None: ...

    def read_live_frame(self, timeout_s: float = 0.5) -> Optional[np.ndarray]: ...

    def stop_live(self) -> None: ...


def average_frames(acquire_one: Callable[[], np.ndarray], n_frames: int) -> np.ndarray:
    """
    Canonical multi-frame averaging helper shared across HAL drivers and simulators.
    Acquires `n_frames` frames via `acquire_one` and returns their rounded `uint16` mean.
    """
    n = max(1, int(n_frames))
    first = acquire_one()
    if n == 1:
        return np.asarray(first, dtype=np.uint16)
    acc = np.asarray(first, dtype=np.float64)
    for _ in range(n - 1):
        acc += np.asarray(acquire_one(), dtype=np.float64)
    return np.clip(np.rint(acc / n), 0, 65535).astype(np.uint16)
