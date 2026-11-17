"""
Driver for the Hamamatsu Orca Flash 4.0 v3 sCMOS camera (C13440-20CU).

Uses `pylablib.devices.DCAM` (wrapping Windows `dcamapi4.dll`) for hardware control
and includes a built-in Simulation / Mock mode (`simulate=True`) for offline development
and UI testing.
"""

import time
import threading
from typing import Dict, Optional, Tuple, Callable
import numpy as np


class OrcaCameraError(Exception):
    """Exception raised for Hamamatsu Orca Flash 4.0 v3 communication or DCAM errors."""


def count_dcam_cameras() -> int:
    """
    Returns the number of Hamamatsu DCAM cameras detected on the system.
    Returns 0 if DCAM-API (`dcamapi4.dll`) is not installed or no cameras are connected.
    """
    try:
        from pylablib.devices import DCAM
        return int(DCAM.get_cameras_number())
    except Exception:
        return 0


class OrcaFlashCamera:
    """
    Hardware driver for the Hamamatsu Orca Flash 4.0 v3 sCMOS camera.
    """

    DEFAULT_DETECTOR_SIZE = (2048, 2048)  # (width, height) in pixels

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        self._log_cb = logger
        self._lock = threading.RLock()
        self._connected: bool = False
        self._simulate: bool = False
        self._camera_idx: int = 0
        self._cam = None

        # Cached device metadata
        self._vendor: str = ""
        self._model: str = ""
        self._serial_number: str = ""
        self._camera_version: str = ""
        self._detector_size: Tuple[int, int] = self.DEFAULT_DETECTOR_SIZE

        # Simulated state
        self._sim_exposure_s: float = 0.050  # 50 ms default
        self._sim_roi: Tuple[int, int, int, int, int, int] = (0, 2048, 0, 2048, 1, 1)
        self._sim_readout_speed: str = "fast"

    def _log(self, message: str):
        if self._log_cb is not None:
            self._log_cb(f"[Orca Flash 4.0 v3] {message}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulated(self) -> bool:
        return self._simulate

    @property
    def model(self) -> str:
        return self._model

    @property
    def serial_number(self) -> str:
        return self._serial_number

    @property
    def detector_size(self) -> Tuple[int, int]:
        return self._detector_size

    # ==========================================================================
    # Connection & Disconnection
    # ==========================================================================

    def connect(self, camera_idx: int = 0, simulate: bool = False) -> Dict[str, object]:
        """
        Opens a connection to the Hamamatsu Orca Flash 4.0 v3 camera.

        Parameters
        ----------
        camera_idx : int
            Zero-based DCAM camera index (default: 0).
        simulate : bool
            If True, connects to an internal sCMOS camera simulator (`2048x2048`, 16-bit).

        Returns
        -------
        status : dict
            Complete dictionary of camera identity, sensor geometry, and current parameters.
        """
        with self._lock:
            if self._connected:
                self.disconnect()

            if simulate:
                self._connected = True
                self._simulate = True
                self._camera_idx = camera_idx
                self._vendor = "Hamamatsu (Simulated)"
                self._model = "ORCA-Flash4.0 V3 (C13440-20CU Sim)"
                self._serial_number = "SIM-ORCA-001"
                self._camera_version = "3.00.Sim"
                self._detector_size = self.DEFAULT_DETECTOR_SIZE
                self._log(
                    f"Connected in SIMULATION mode: {self._model} "
                    f"[{self._detector_size[0]}x{self._detector_size[1]} px]."
                )
                return self.get_status()

            try:
                from pylablib.devices import DCAM
            except ImportError as exc:
                raise OrcaCameraError(
                    "The `pylablib` package is required for DCAM camera control. "
                    f"Import error: {exc}"
                ) from exc

            try:
                n_cams = DCAM.get_cameras_number()
            except Exception as exc:
                raise OrcaCameraError(
                    "Failed to initialize Hamamatsu DCAM-API (`dcamapi4.dll`). "
                    "Ensure the Hamamatsu DCAM-API driver is installed on this Windows 11 PC "
                    "and the Orca Flash 4.0 v3 is powered on, or enable Simulation Mode.\n"
                    f"Details: {exc}"
                ) from exc

            if n_cams <= 0:
                raise OrcaCameraError(
                    "Hamamatsu DCAM-API initialized, but 0 cameras were detected. "
                    "Verify that the Orca Flash 4.0 v3 is powered on and connected via USB 3.0 / CameraLink."
                )

            if camera_idx >= n_cams:
                raise OrcaCameraError(
                    f"Requested camera index {camera_idx}, but only {n_cams} DCAM camera(s) found."
                )

            try:
                self._log(f"Opening DCAM camera index {camera_idx} (out of {n_cams} detected)...")
                cam = DCAM.DCAMCamera(idx=camera_idx)
                info = cam.get_device_info()
                det_size = cam.get_detector_size()

                self._cam = cam
                self._connected = True
                self._simulate = False
                self._camera_idx = camera_idx
                self._vendor = str(getattr(info, "vendor", "Hamamatsu"))
                self._model = str(getattr(info, "model", "ORCA-Flash4.0 V3"))
                self._serial_number = str(getattr(info, "serial_number", "Unknown"))
                self._camera_version = str(getattr(info, "camera_version", "Unknown"))
                self._detector_size = (int(det_size[0]), int(det_size[1]))

                self._log(
                    f"Connected to {self._vendor} {self._model} "
                    f"(S/N: {self._serial_number}, Sensor: {self._detector_size[0]}x{self._detector_size[1]} px)."
                )
                return self.get_status()
            except Exception as exc:
                if self._cam is not None:
                    try:
                        self._cam.close()
                    except Exception:
                        pass
                    self._cam = None
                raise OrcaCameraError(f"Error opening Hamamatsu camera (idx={camera_idx}): {exc}") from exc

    def disconnect(self):
        """Closes the connection to the Hamamatsu camera."""
        with self._lock:
            model_info = self._model
            if self._cam is not None:
                try:
                    self._cam.close()
                except Exception:
                    pass
                self._cam = None
            was_connected = self._connected
            self._connected = False
            self._simulate = False
            if was_connected:
                self._log(f"Disconnected from {model_info}.")

    # ==========================================================================
    # Status & Basic Queries
    # ==========================================================================

    def get_exposure(self) -> float:
        """Returns the current exposure time in seconds."""
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            if self._simulate:
                return float(self._sim_exposure_s)
            return float(self._cam.get_exposure())

    def set_exposure(self, exposure_s: float) -> float:
        """Sets the exposure time in seconds and returns the actual applied exposure time."""
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            exposure_s = max(1e-5, float(exposure_s))
            if self._simulate:
                self._sim_exposure_s = exposure_s
                self._log(f"Set exposure (sim) to {exposure_s * 1e3:.2f} ms.")
                return self._sim_exposure_s
            actual = float(self._cam.set_exposure(exposure_s))
            self._log(f"Set exposure to {actual * 1e3:.2f} ms.")
            return actual

    def get_roi(self) -> Tuple[int, int, int, int, int, int]:
        """Returns `(hstart, hend, vstart, vend, hbin, vbin)`."""
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            if self._simulate:
                return self._sim_roi
            roi = self._cam.get_roi()
            return tuple(int(v) for v in roi)

    def get_sensor_temperature(self) -> Optional[float]:
        """Returns the sensor temperature in °C if supported by the camera."""
        with self._lock:
            if not self._connected:
                return None
            if self._simulate:
                return -10.0
            try:
                temp = self._cam.get_attribute_value("SENSOR TEMPERATURE", default=None)
                return float(temp) if temp is not None else None
            except Exception:
                return None

    def get_status(self) -> Dict[str, object]:
        """Returns a complete snapshot of the camera connection state and parameters."""
        with self._lock:
            if not self._connected:
                return {
                    "connected": False,
                    "simulate": False,
                    "camera_idx": self._camera_idx,
                    "vendor": "",
                    "model": "",
                    "serial_number": "",
                    "camera_version": "",
                    "detector_size": self.DEFAULT_DETECTOR_SIZE,
                    "roi": None,
                    "exposure_s": None,
                    "temperature_c": None,
                }
            return {
                "connected": True,
                "simulate": self._simulate,
                "camera_idx": self._camera_idx,
                "vendor": self._vendor,
                "model": self._model,
                "serial_number": self._serial_number,
                "camera_version": self._camera_version,
                "detector_size": self._detector_size,
                "roi": self.get_roi(),
                "exposure_s": self.get_exposure(),
                "temperature_c": self.get_sensor_temperature(),
            }
