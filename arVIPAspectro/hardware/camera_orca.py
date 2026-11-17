"""
Driver for the Hamamatsu Orca Flash 4.0 v3 sCMOS camera (C13440-20CU).

Uses `pylablib.devices.DCAM` (wrapping Windows `dcamapi4.dll`) for hardware control
and includes a built-in Simulation / Mock mode (`simulate=True`) for offline development
and UI testing.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
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
    Supports drivable acquisition parameters:
      - Exposure time (s / ms)
      - Hardware binning (1x1, 2x2, 4x4)
      - Sensor ROI (hstart, hend, vstart, vend in multiples of 4 px)
      - Readout speed ('fast' standard scan vs 'slow' low-noise scan)
      - Defect pixel correction (True / False)
      - Multi-frame averaging (N frames averaged per snap)
    """

    DEFAULT_DETECTOR_SIZE = (2048, 2048)  # (width, height) in pixels
    VALID_BINNINGS = (1, 2, 4)

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

        # Software-managed multi-frame averaging count
        self._frame_averaging: int = 1

        # Simulated state
        self._sim_exposure_s: float = 0.050  # 50 ms default
        self._sim_roi: Tuple[int, int, int, int, int, int] = (0, 2048, 0, 2048, 1, 1)
        self._sim_readout_speed: str = "fast"
        self._sim_defect_correct: bool = True
        self._sim_frame_counter: int = 0
        self._sim_grid_cache: Optional[Tuple[int, int, np.ndarray, np.ndarray]] = None

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
                self._sim_roi = (0, self._detector_size[0], 0, self._detector_size[1], 1, 1)
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
    # Drivable Acquisition Parameters (Step 3)
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
                self._log(f"Exposure set (sim) to {exposure_s * 1e3:.3f} ms.")
                return self._sim_exposure_s
            actual = float(self._cam.set_exposure(exposure_s))
            self._log(f"Exposure set to {actual * 1e3:.3f} ms.")
            return actual

    def get_roi(self) -> Tuple[int, int, int, int, int, int]:
        """
        Returns the active sensor ROI and hardware binning as
        `(hstart, hend, vstart, vend, hbin, vbin)` in unbinned detector pixels.
        """
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            if self._simulate:
                return self._sim_roi
            roi = self._cam.get_roi()
            return tuple(int(v) for v in roi)

    def set_roi(
        self,
        hstart: int = 0,
        hend: Optional[int] = None,
        vstart: int = 0,
        vend: Optional[int] = None,
        binning: int = 1,
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Sets the sensor sub-array ROI and hardware binning (`1`, `2`, or `4`).
        Coordinates `hstart, hend, vstart, vend` are automatically aligned to 4-pixel boundaries
        required by the Orca Flash 4.0 v3 hardware.
        """
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")

            det_w, det_h = self._detector_size
            if hend is None:
                hend = det_w
            if vend is None:
                vend = det_h

            # Quantize to valid binning (1, 2, 4)
            binning = int(binning)
            if binning not in self.VALID_BINNINGS:
                binning = min(self.VALID_BINNINGS, key=lambda b: abs(b - binning))

            # Align ROI limits to multiples of 4 pixels (minimum 4x4 subarray size)
            hstart = max(0, min((int(hstart) // 4) * 4, det_w - 4))
            hend = max(hstart + 4, min(((int(hend) + 3) // 4) * 4, det_w))
            vstart = max(0, min((int(vstart) // 4) * 4, det_h - 4))
            vend = max(vstart + 4, min(((int(vend) + 3) // 4) * 4, det_h))

            if self._simulate:
                self._sim_roi = (hstart, hend, vstart, vend, binning, binning)
                self._sim_grid_cache = None
                out_w = (hend - hstart) // binning
                out_h = (vend - vstart) // binning
                self._log(
                    f"ROI & Binning set (sim): X=[{hstart}:{hend}], Y=[{vstart}:{vend}], "
                    f"Bin={binning}x{binning} -> Frame {out_w}x{out_h} px."
                )
                return self._sim_roi

            applied = self._cam.set_roi(
                hstart=hstart,
                hend=hend,
                vstart=vstart,
                vend=vend,
                hbin=binning,
                vbin=binning,
            )
            roi_tuple = tuple(int(v) for v in applied)
            out_w = (roi_tuple[1] - roi_tuple[0]) // max(1, roi_tuple[4])
            out_h = (roi_tuple[3] - roi_tuple[2]) // max(1, roi_tuple[5])
            self._log(
                f"ROI & Binning set: X=[{roi_tuple[0]}:{roi_tuple[1]}], Y=[{roi_tuple[2]}:{roi_tuple[3]}], "
                f"Bin={roi_tuple[4]}x{roi_tuple[5]} -> Frame {out_w}x{out_h} px."
            )
            return roi_tuple

    def get_binning(self) -> int:
        """Returns the current hardware binning factor (`1`, `2`, or `4`)."""
        roi = self.get_roi()
        return int(roi[4])

    def set_binning(self, binning: int) -> Tuple[int, int, int, int, int, int]:
        """Updates the hardware binning factor (`1`, `2`, or `4`) while preserving the current ROI bounds."""
        hstart, hend, vstart, vend, _, _ = self.get_roi()
        return self.set_roi(hstart=hstart, hend=hend, vstart=vstart, vend=vend, binning=binning)

    def reset_full_roi(self, binning: Optional[int] = None) -> Tuple[int, int, int, int, int, int]:
        """Resets the sensor ROI to the full detector area (`2048x2048`)."""
        if binning is None:
            binning = self.get_binning()
        det_w, det_h = self._detector_size
        return self.set_roi(hstart=0, hend=det_w, vstart=0, vend=det_h, binning=binning)

    def get_available_readout_speeds(self) -> List[str]:
        """Returns the list of supported readout speeds (e.g. `['slow', 'fast']`)."""
        with self._lock:
            if not self._connected or self._simulate:
                return ["fast", "slow"]
            try:
                speeds = list(self._cam.get_all_readout_speeds())
                return speeds if speeds else ["fast", "slow"]
            except Exception:
                return ["fast", "slow"]

    def get_readout_speed(self) -> str:
        """Returns the current readout speed (`'fast'` or `'slow'`)."""
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            if self._simulate:
                return self._sim_readout_speed
            try:
                return str(self._cam.get_readout_speed())
            except Exception:
                return "fast"

    def set_readout_speed(self, speed: str) -> str:
        """Sets the camera readout speed (`'fast'` standard scan or `'slow'` low-noise scan)."""
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            speed = str(speed).lower().strip()
            if self._simulate:
                self._sim_readout_speed = speed if speed in ("fast", "slow", "normal") else "fast"
                self._log(f"Readout speed set (sim) to '{self._sim_readout_speed}'.")
                return self._sim_readout_speed
            actual = str(self._cam.set_readout_speed(speed))
            self._log(f"Readout speed set to '{actual}'.")
            return actual

    def get_defect_correct_mode(self) -> bool:
        """Returns True if on-camera defect pixel correction is enabled."""
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            if self._simulate:
                return self._sim_defect_correct
            try:
                return bool(self._cam.get_defect_correct_mode())
            except Exception:
                return True

    def set_defect_correct_mode(self, enabled: bool) -> bool:
        """Enables or disables on-camera defect pixel correction."""
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")
            enabled = bool(enabled)
            if self._simulate:
                self._sim_defect_correct = enabled
                self._log(f"Defect pixel correction set (sim) to {enabled}.")
                return self._sim_defect_correct
            actual = bool(self._cam.set_defect_correct_mode(enabled))
            self._log(f"Defect pixel correction set to {actual}.")
            return actual

    def get_frame_averaging(self) -> int:
        """Returns the number of frames averaged per acquisition (`>= 1`)."""
        with self._lock:
            return int(self._frame_averaging)

    def set_frame_averaging(self, n_frames: int) -> int:
        """Sets the number of frames (`1..100`) to average per acquisition."""
        with self._lock:
            self._frame_averaging = max(1, min(int(n_frames), 500))
            self._log(f"Frame averaging set to {self._frame_averaging} frame(s) per acquisition.")
            return self._frame_averaging

    def get_frame_readout_time(self) -> Optional[float]:
        """Returns the sensor frame readout time in seconds."""
        with self._lock:
            if not self._connected:
                return None
            if self._simulate:
                return 0.010 if self._sim_readout_speed == "fast" else 0.030
            try:
                return float(self._cam.get_frame_readout_time())
            except Exception:
                return None

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

    # ==========================================================================
    # Image Acquisition
    # ==========================================================================

    def snap_frame(
        self,
        n_frames: Optional[int] = None,
        timeout_s: Optional[float] = None,
        stage_pos_um: Optional[Dict[str, Optional[float]]] = None,
    ) -> np.ndarray:
        """
        Acquires a 2D image frame (`uint16`) from the camera.
        If `n_frames` (or `self.get_frame_averaging()`) > 1, acquires `n_frames`
        consecutive frames and returns their pixel-wise average as `uint16`.

        Parameters
        ----------
        n_frames : int, optional
            Number of frames to average. Defaults to `self.get_frame_averaging()`.
        timeout_s : float, optional
            Per-frame timeout in seconds. Defaults to `max(5.0, exposure_s * 1.5 + 5.0)`.
        stage_pos_um : dict, optional
            Optional stage coordinates `{'X': x, 'Y': y, 'Z': z}` (used in Simulation Mode).

        Returns
        -------
        frame : numpy.ndarray
            2D image array (`uint16`, shape `(H, W)`).
        """
        with self._lock:
            if not self._connected:
                raise OrcaCameraError("Camera is not connected.")

            if n_frames is None:
                n_frames = self._frame_averaging
            n_frames = max(1, int(n_frames))

            exp_s = self.get_exposure()
            if timeout_s is None:
                timeout_s = max(5.0, exp_s * 1.5 + 5.0)

            if self._simulate:
                t0 = time.time()
                time.sleep(min(exp_s * n_frames, 0.25))
                if n_frames == 1:
                    frame = self._generate_simulated_frame(stage_pos_um=stage_pos_um)
                else:
                    acc = np.zeros_like(
                        self._generate_simulated_frame(stage_pos_um=stage_pos_um),
                        dtype=np.float64,
                    )
                    for _ in range(n_frames):
                        acc += self._generate_simulated_frame(stage_pos_um=stage_pos_um)
                    frame = np.clip(np.rint(acc / n_frames), 0, 65535).astype(np.uint16)

                elapsed_ms = (time.time() - t0) * 1e3
                self._log(
                    f"Acquired simulated frame ({frame.shape[1]}x{frame.shape[0]} px, "
                    f"avg={n_frames}, {elapsed_ms:.1f} ms) [min={frame.min()}, max={frame.max()}]."
                )
                return frame

            try:
                t0 = time.time()
                if n_frames == 1:
                    raw_frame = self._cam.snap(timeout=timeout_s)
                    frame = np.asarray(raw_frame, dtype=np.uint16)
                else:
                    frames = self._cam.grab(nframes=n_frames, frame_timeout=timeout_s)
                    if not frames:
                        raise OrcaCameraError("Camera grab returned 0 frames.")
                    acc = np.mean(np.asarray(frames, dtype=np.float64), axis=0)
                    frame = np.clip(np.rint(acc), 0, 65535).astype(np.uint16)

                elapsed_ms = (time.time() - t0) * 1e3
                self._log(
                    f"Acquired frame ({frame.shape[1]}x{frame.shape[0]} px, uint16, avg={n_frames}) "
                    f"in {elapsed_ms:.1f} ms [min={frame.min()}, max={frame.max()}]."
                )
                return frame
            except Exception as exc:
                raise OrcaCameraError(f"Failed to acquire frame from camera: {exc}") from exc

    def _generate_simulated_frame(
        self, stage_pos_um: Optional[Dict[str, Optional[float]]] = None
    ) -> np.ndarray:
        """
        Generates a realistic synthetic 2D ar-VIPA Brillouin interferogram (`uint16`)
        whose dimensions match the active ROI & hardware binning, and whose fringes
        vary smoothly with `(X, Y)` stage coordinates and exposure time.
        """
        self._sim_frame_counter += 1
        det_w, det_h = self._detector_size
        hstart, hend, vstart, vend, hbin, vbin = self._sim_roi
        w = max(1, (hend - hstart) // max(1, hbin))
        h = max(1, (vend - vstart) // max(1, vbin))

        if self._sim_grid_cache is None or self._sim_grid_cache[0] != h or self._sim_grid_cache[1] != w:
            # Map binned ROI pixel indices back to full-sensor pixel coordinates
            y_sensor = vstart + (np.arange(h, dtype=np.float32) + 0.5) * vbin
            x_sensor = hstart + (np.arange(w, dtype=np.float32) + 0.5) * hbin
            yy, xx = np.meshgrid(y_sensor, x_sensor, indexing="ij")
            self._sim_grid_cache = (h, w, xx, yy)
        else:
            _, _, xx, yy = self._sim_grid_cache

        x_um = float((stage_pos_um or {}).get("X") or 0.0)
        y_um = float((stage_pos_um or {}).get("Y") or 0.0)

        # Normalize detector coordinates to [-1, 1] relative to full sensor center
        xn = (xx - 0.5 * det_w) / (0.5 * det_w)
        yn = (yy - 0.5 * det_h) / (0.5 * det_h)

        # Curved VIPA orders (parabolic curvature along y + non-linear dispersion along x)
        order_phase = 6.0 * (xn + 1.05) - 0.45 * (xn + 1.05) ** 2 + 0.18 * (yn ** 2)
        frac = order_phase - np.floor(order_phase)

        # Modulate Brillouin shift ratio slightly with stage (X, Y)
        shift_frac = 0.18 + 0.03 * np.sin(x_um / 50.0) * np.cos(y_um / 50.0)
        shift_y = shift_frac * np.sqrt(np.clip(1.0 - 0.25 * yn ** 2, 0.2, 1.0))

        # Elastic Rayleigh peak + Stokes & Anti-Stokes doublets
        elastic = np.exp(-0.5 * (np.minimum(frac, 1.0 - frac) / 0.018) ** 2)
        stokes = 0.42 * np.exp(-0.5 * ((frac - shift_y) / 0.026) ** 2)
        antistokes = 0.42 * np.exp(-0.5 * ((frac - (1.0 - shift_y)) / 0.026) ** 2)

        env_x0 = 0.08 * np.sin(x_um / 100.0)
        env_y0 = 0.08 * np.cos(y_um / 100.0)
        envelope = np.exp(-0.6 * ((xn - env_x0) ** 2 + (yn - env_y0) ** 2))

        exp_scale = np.clip(self._sim_exposure_s / 0.050, 0.02, 20.0) * (hbin * vbin)
        signal = (120.0 + 18000.0 * exp_scale * envelope * (elastic + stokes + antistokes)).astype(np.float32)

        noise_std = 12.0 if self._sim_readout_speed == "slow" else 25.0
        rng = np.random.default_rng(seed=1000 + self._sim_frame_counter)
        noise = rng.normal(loc=0.0, scale=noise_std, size=(h, w)).astype(np.float32)
        frame = np.clip(signal + noise, 0, 65535).astype(np.uint16)
        return frame

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
                    "binning": 1,
                    "output_shape": None,
                    "exposure_s": None,
                    "readout_speed": None,
                    "available_readout_speeds": ["fast", "slow"],
                    "defect_correct": None,
                    "frame_averaging": self._frame_averaging,
                    "readout_time_s": None,
                    "temperature_c": None,
                }
            roi = self.get_roi()
            out_w = (roi[1] - roi[0]) // max(1, roi[4])
            out_h = (roi[3] - roi[2]) // max(1, roi[5])
            return {
                "connected": True,
                "simulate": self._simulate,
                "camera_idx": self._camera_idx,
                "vendor": self._vendor,
                "model": self._model,
                "serial_number": self._serial_number,
                "camera_version": self._camera_version,
                "detector_size": self._detector_size,
                "roi": roi,
                "binning": int(roi[4]),
                "output_shape": (out_h, out_w),
                "exposure_s": self.get_exposure(),
                "readout_speed": self.get_readout_speed(),
                "available_readout_speeds": self.get_available_readout_speeds(),
                "defect_correct": self.get_defect_correct_mode(),
                "frame_averaging": self._frame_averaging,
                "readout_time_s": self.get_frame_readout_time(),
                "temperature_c": self.get_sensor_temperature(),
            }
