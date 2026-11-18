"""
Driver for the Teledyne FLIR Chameleon3 CM3-U3-31S4M-CS monochrome USB 3.0 camera
(Sony IMX265 Pregius global shutter CMOS, 2048 x 1536 px, 3.45 um pixel pitch).

Uses `pylablib.devices.Teledyne` (wrapping the Teledyne Spinnaker C SDK) for
hardware control and includes a built-in Simulation / Mock mode (`simulate=True`)
for offline development and UI testing.
"""

import glob
import os
import sys
import time
import threading
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


_SPINNAKER_DLL_CONFIGURED = False


def _ensure_spinnaker_dll_configured() -> Optional[str]:
    """
    On Windows, locates ``SpinnakerC*.dll`` in standard Teledyne / FLIR Spinnaker
    installation directories (or PATH) and registers it with ``pylablib`` so
    3-digit Spinnaker 4.x DLL names (e.g. ``SpinnakerC_v440.dll``) and non-PATH
    installs load cleanly.
    """
    global _SPINNAKER_DLL_CONFIGURED
    if _SPINNAKER_DLL_CONFIGURED or sys.platform != "win32":
        return None

    try:
        import pylablib as pll
    except ImportError:
        return None

    if ("devices/dlls", "teledyne_spinnaker") in pll.par:
        _SPINNAKER_DLL_CONFIGURED = True
        return str(pll.par["devices/dlls/teledyne_spinnaker"])

    search_dirs: List[str] = []
    for base_env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        base = os.environ.get(base_env)
        if not base:
            continue
        for vendor in ("Teledyne", "FLIR Systems", "Point Grey Research"):
            root = os.path.join(base, vendor, "Spinnaker")
            if os.path.isdir(root):
                search_dirs.extend(sorted(glob.glob(os.path.join(root, "bin64", "vs*")), reverse=True))
                search_dirs.append(os.path.join(root, "bin64"))
                search_dirs.append(os.path.join(root, "bin"))

    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p and os.path.isdir(p):
            search_dirs.append(p)

    seen = set()
    for folder in search_dirs:
        norm = os.path.normcase(os.path.abspath(folder))
        if norm in seen or not os.path.isdir(folder):
            continue
        seen.add(norm)
        candidates = sorted(glob.glob(os.path.join(folder, "SpinnakerC_v*.dll")), reverse=True)
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(folder, "SpinnakerC*.dll")), reverse=True)
        # Prefer release DLL over debug (*d.dll)
        release = [c for c in candidates if not os.path.basename(c).lower().endswith("d.dll")]
        chosen = release[0] if release else (candidates[0] if candidates else None)
        if chosen:
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(folder)
                except OSError:
                    pass
            cur_path = os.environ.get("PATH", "")
            if folder not in cur_path.split(os.pathsep):
                os.environ["PATH"] = folder + os.pathsep + cur_path
            pll.par["devices/dlls/teledyne_spinnaker"] = chosen
            _SPINNAKER_DLL_CONFIGURED = True
            return chosen

    _SPINNAKER_DLL_CONFIGURED = True
    return None


class TeledyneCameraError(Exception):
    """Exception raised for Teledyne Spinnaker camera communication or SDK errors."""


def list_teledyne_cameras() -> List[Dict[str, str]]:
    """
    Returns a list of detected Teledyne Spinnaker cameras as dictionaries with
    keys ``index``, ``name``, ``model``, ``serial``, and ``vendor``.
    Returns ``[]`` if the Spinnaker SDK is not installed or no cameras are found.
    """
    try:
        _ensure_spinnaker_dll_configured()
        from pylablib.devices import Teledyne
        infos = Teledyne.list_cameras(desc=True)
    except Exception:
        return []

    cameras: List[Dict[str, str]] = []
    for idx, info in enumerate(infos):
        cameras.append({
            "index": idx,
            "name": str(getattr(info, "name", "")),
            "model": str(getattr(info, "model", "")),
            "serial": str(getattr(info, "serial", "")),
            "vendor": str(getattr(info, "vendor", "")),
        })
    return cameras


def count_teledyne_cameras() -> int:
    """
    Returns the number of Teledyne Spinnaker cameras detected on the system.
    Returns 0 if the Spinnaker SDK is not installed or no cameras are connected.
    """
    try:
        _ensure_spinnaker_dll_configured()
        from pylablib.devices import Teledyne
        return int(Teledyne.get_cameras_number())
    except Exception:
        return 0


def find_chameleon3_camera(model_substr: str = "CM3-U3-31S4M") -> Optional[int]:
    """
    Searches connected Spinnaker cameras for a Chameleon3 CM3-U3-31S4M-CS model.
    Returns the matching zero-based camera index, falls back to ``0`` if another
    Teledyne camera is connected, or returns ``None`` if no cameras are detected.
    """
    cams = list_teledyne_cameras()
    if not cams:
        return None
    target = model_substr.lower()
    for cam in cams:
        if target in cam["model"].lower() or target in cam["name"].lower():
            return int(cam["index"])
    return 0


class TeledyneBrightfieldCamera:
    """
    Hardware driver for the Teledyne FLIR CM3-U3-31S4M-CS brightfield camera.
    Supports drivable acquisition parameters:
      - Exposure time (seconds)
      - Analog gain (dB)
      - Sensor ROI (hstart, hend, vstart, vend)
      - Multi-frame averaging (N frames averaged per snap)
      - Continuous live acquisition (`start_live`, `read_live_frame`, `stop_live`)
    """

    DEFAULT_DETECTOR_SIZE: Tuple[int, int] = (2048, 1536)  # (width, height) in pixels
    PIXEL_SIZE_UM: float = 3.45

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        self._log_cb = logger
        self._lock = threading.RLock()
        self._connected: bool = False
        self._simulate: bool = False
        self._live_active: bool = False
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
        self._sim_exposure_s: float = 0.020  # 20 ms default for brightfield
        self._sim_gain_db: float = 0.0
        self._sim_roi: Tuple[int, int, int, int] = (0, 2048, 0, 1536)
        self._sim_frame_counter: int = 0
        self._sim_grid_cache: Optional[Tuple[int, int, np.ndarray, np.ndarray]] = None

    def _log(self, message: str):
        if self._log_cb is not None:
            self._log_cb(f"[Teledyne CM3-U3-31S4M] {message}")

    def __enter__(self) -> "TeledyneBrightfieldCamera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulated(self) -> bool:
        return self._simulate

    @property
    def is_live(self) -> bool:
        return self._live_active

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

    def connect(
        self,
        camera_idx: Optional[int] = None,
        serial: Optional[str] = None,
        simulate: bool = False,
    ) -> Dict[str, object]:
        """
        Opens a connection to the Teledyne CM3-U3-31S4M-CS camera.

        Parameters
        ----------
        camera_idx : int, optional
            Zero-based Spinnaker camera index. If None (and `serial` is None),
            automatically searches for a connected CM3-U3-31S4M camera.
        serial : str, optional
            Specific camera serial number to open.
        simulate : bool
            If True, connects to an internal brightfield camera simulator (2048x1536).

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
                self._live_active = False
                self._camera_idx = 0 if camera_idx is None else int(camera_idx)
                self._vendor = "Teledyne FLIR (Simulated)"
                self._model = "Chameleon3 CM3-U3-31S4M-CS (Sim)"
                self._serial_number = serial or "SIM-CM3-001"
                self._camera_version = "Sim-1.0"
                self._detector_size = self.DEFAULT_DETECTOR_SIZE
                self._sim_roi = (0, self._detector_size[0], 0, self._detector_size[1])
                self._log(
                    f"Connected in SIMULATION mode: {self._model} "
                    f"[{self._detector_size[0]}x{self._detector_size[1]} px]."
                )
                return self.get_status()

            try:
                _ensure_spinnaker_dll_configured()
                from pylablib.devices import Teledyne
            except ImportError as exc:
                raise TeledyneCameraError(
                    "The `pylablib` package is required for Teledyne Spinnaker camera control. "
                    f"Import error: {exc}"
                ) from exc

            try:
                n_cams = Teledyne.get_cameras_number()
            except Exception as exc:
                raise TeledyneCameraError(
                    "Failed to initialize Teledyne Spinnaker SDK. "
                    "Ensure the Teledyne Spinnaker SDK is installed and the camera is connected via USB 3.0.\n"
                    f"Details: {exc}"
                ) from exc

            if n_cams <= 0:
                raise TeledyneCameraError(
                    "Teledyne Spinnaker SDK initialized, but 0 cameras were detected. "
                    "Verify that the CM3-U3-31S4M-CS is connected via USB 3.0."
                )

            if serial is None and camera_idx is None:
                camera_idx = find_chameleon3_camera()
                if camera_idx is None:
                    camera_idx = 0

            if serial is None and camera_idx is not None and camera_idx >= n_cams:
                raise TeledyneCameraError(
                    f"Requested camera index {camera_idx}, but only {n_cams} Spinnaker camera(s) found."
                )

            try:
                self._log(
                    f"Opening Teledyne Spinnaker camera "
                    f"({'serial=' + serial if serial else f'idx={camera_idx}'})..."
                )
                cam = (
                    Teledyne.TeledyneSpinnakerCamera(serial=serial)
                    if serial is not None
                    else Teledyne.TeledyneSpinnakerCamera(idx=int(camera_idx))
                )
                info = cam.get_device_info()
                det_size = cam.get_detector_size()

                self._cam = cam
                self._connected = True
                self._simulate = False
                self._live_active = False
                self._camera_idx = 0 if camera_idx is None else int(camera_idx)
                self._vendor = str(getattr(info, "vendor", "Teledyne FLIR"))
                self._model = str(getattr(info, "model", "CM3-U3-31S4M-CS"))
                self._serial_number = str(getattr(info, "serial", "Unknown"))
                self._camera_version = str(getattr(info, "devversion", "Unknown"))
                self._detector_size = (int(det_size[0]), int(det_size[1]))

                self._configure_defaults()

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
                raise TeledyneCameraError(f"Error opening Teledyne camera: {exc}") from exc

    def _configure_defaults(self):
        """Sets manual exposure/gain and Mono16 (fallback Mono8) pixel format when supported."""
        if self._cam is None:
            return
        for attr, val in (("ExposureAuto", "Off"), ("GainAuto", "Off")):
            try:
                self._cam.set_attribute_value(attr, val, error_on_missing=False)
            except Exception:
                pass
        for fmt in ("Mono16", "Mono8"):
            try:
                self._cam.set_attribute_value("PixelFormat", fmt, error_on_missing=False)
                break
            except Exception:
                continue

    def disconnect(self):
        """Stops any active acquisition and closes the connection to the camera."""
        with self._lock:
            model_info = self._model
            if self._live_active:
                try:
                    self.stop_live()
                except Exception:
                    self._live_active = False
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
    # Drivable Acquisition Parameters
    # ==========================================================================

    def get_exposure(self) -> float:
        """Returns the current exposure time in seconds."""
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")
            if self._simulate:
                return float(self._sim_exposure_s)
            return float(self._cam.get_exposure())

    def set_exposure(self, exposure_s: float) -> float:
        """Sets the exposure time in seconds and returns the actual applied exposure time."""
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")
            exposure_s = max(1e-5, float(exposure_s))
            if self._simulate:
                self._sim_exposure_s = exposure_s
                self._log(f"Exposure set (sim) to {exposure_s * 1e3:.3f} ms.")
                return self._sim_exposure_s
            try:
                self._cam.set_attribute_value("ExposureAuto", "Off", error_on_missing=False)
            except Exception:
                pass
            actual = float(self._cam.set_exposure(exposure_s))
            self._log(f"Exposure set to {actual * 1e3:.3f} ms.")
            return actual

    def get_gain(self) -> Optional[float]:
        """Returns the camera analog gain in dB if supported."""
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")
            if self._simulate:
                return float(self._sim_gain_db)
            try:
                val = self._cam.get_attribute_value("Gain", default=None)
                return float(val) if val is not None else None
            except Exception:
                return None

    def set_gain(self, gain_db: float) -> Optional[float]:
        """Sets the camera analog gain in dB and returns the actual applied gain."""
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")
            gain_db = max(0.0, float(gain_db))
            if self._simulate:
                self._sim_gain_db = min(gain_db, 24.0)
                self._log(f"Gain set (sim) to {self._sim_gain_db:.2f} dB.")
                return self._sim_gain_db
            try:
                self._cam.set_attribute_value("GainAuto", "Off", error_on_missing=False)
                self._cam.set_attribute_value("Gain", gain_db, error_on_missing=False)
                actual = self.get_gain()
                self._log(f"Gain set to {actual} dB.")
                return actual
            except Exception as exc:
                self._log(f"Failed to set gain: {exc}")
                return self.get_gain()

    def get_roi(self) -> Tuple[int, int, int, int]:
        """Returns the active sensor ROI as ``(hstart, hend, vstart, vend)`` in pixels."""
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")
            if self._simulate:
                return self._sim_roi
            roi = self._cam.get_roi()
            return tuple(int(v) for v in roi[:4])

    def set_roi(
        self,
        hstart: int = 0,
        hend: Optional[int] = None,
        vstart: int = 0,
        vend: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """Sets the sensor sub-array ROI ``(hstart, hend, vstart, vend)``."""
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")

            det_w, det_h = self._detector_size
            if hend is None:
                hend = det_w
            if vend is None:
                vend = det_h

            hstart = max(0, min((int(hstart) // 4) * 4, det_w - 4))
            hend = max(hstart + 4, min(((int(hend) + 3) // 4) * 4, det_w))
            vstart = max(0, min((int(vstart) // 4) * 4, det_h - 4))
            vend = max(vstart + 4, min(((int(vend) + 3) // 4) * 4, det_h))

            if self._simulate:
                self._sim_roi = (hstart, hend, vstart, vend)
                self._sim_grid_cache = None
                self._log(f"ROI set (sim): X=[{hstart}:{hend}], Y=[{vstart}:{vend}].")
                return self._sim_roi

            applied = self._cam.set_roi(hstart=hstart, hend=hend, vstart=vstart, vend=vend)
            roi_tuple = tuple(int(v) for v in applied[:4])
            self._log(
                f"ROI set: X=[{roi_tuple[0]}:{roi_tuple[1]}], Y=[{roi_tuple[2]}:{roi_tuple[3]}]."
            )
            return roi_tuple

    def reset_full_roi(self) -> Tuple[int, int, int, int]:
        """Resets the sensor ROI to the full detector area."""
        det_w, det_h = self._detector_size
        return self.set_roi(hstart=0, hend=det_w, vstart=0, vend=det_h)

    def get_frame_averaging(self) -> int:
        """Returns the number of frames averaged per snap acquisition (`>= 1`)."""
        with self._lock:
            return int(self._frame_averaging)

    def set_frame_averaging(self, n_frames: int) -> int:
        """Sets the number of frames (`1..500`) to average per snap acquisition."""
        with self._lock:
            self._frame_averaging = max(1, min(int(n_frames), 500))
            return self._frame_averaging

    def get_sensor_temperature(self) -> Optional[float]:
        """Returns the sensor/device temperature in °C if supported by the camera."""
        with self._lock:
            if not self._connected:
                return None
            if self._simulate:
                return 32.5
            try:
                temp = self._cam.get_attribute_value("DeviceTemperature", default=None)
                return float(temp) if temp is not None else None
            except Exception:
                return None

    # ==========================================================================
    # Image Acquisition (Snap & Continuous Live Stream)
    # ==========================================================================

    def snap_frame(
        self,
        n_frames: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> np.ndarray:
        """
        Acquires a single 2D image frame (or averaged stack of `n_frames`) as `uint16`/`uint8`.
        """
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")

            if n_frames is None:
                n_frames = self._frame_averaging
            n_frames = max(1, int(n_frames))

            exp_s = self.get_exposure()
            if timeout_s is None:
                timeout_s = max(5.0, exp_s * 1.5 + 5.0)

            if self._simulate:
                time.sleep(min(exp_s * n_frames, 0.1))
                if n_frames == 1:
                    return self._generate_simulated_frame()
                acc = self._generate_simulated_frame().astype(np.float64)
                for _ in range(n_frames - 1):
                    acc += self._generate_simulated_frame()
                return np.clip(np.rint(acc / n_frames), 0, 65535).astype(np.uint16)

            try:
                if n_frames == 1:
                    return np.asarray(self._cam.snap(timeout=timeout_s))
                frames = self._cam.grab(nframes=n_frames, frame_timeout=timeout_s)
                if not frames:
                    raise TeledyneCameraError("Camera grab returned 0 frames.")
                acc = np.mean(np.asarray(frames, dtype=np.float64), axis=0)
                return np.clip(np.rint(acc), 0, 65535).astype(np.uint16)
            except Exception as exc:
                raise TeledyneCameraError(f"Failed to acquire frame: {exc}") from exc

    def start_live(self, buffsize: int = 16):
        """Starts continuous acquisition for live viewing."""
        with self._lock:
            if not self._connected:
                raise TeledyneCameraError("Camera is not connected.")
            if self._live_active:
                return
            if self._simulate:
                self._live_active = True
                self._log("Live acquisition started (sim).")
                return
            try:
                self._cam.start_acquisition(buffsize=int(buffsize))
                self._live_active = True
                self._log("Live acquisition started.")
            except Exception as exc:
                raise TeledyneCameraError(f"Failed to start live acquisition: {exc}") from exc

    def read_live_frame(self, timeout_s: float = 0.5) -> Optional[np.ndarray]:
        """
        Waits up to `timeout_s` for the newest frame during live acquisition.
        Returns None if no new frame arrived before timeout.
        """
        with self._lock:
            if not self._connected or not self._live_active:
                return None
            if self._simulate:
                return self._generate_simulated_frame()
            try:
                self._cam.wait_for_frame(since="now", timeout=float(timeout_s))
                frame = self._cam.read_newest_image()
                return np.asarray(frame) if frame is not None else None
            except self._cam.TimeoutError:
                return None
            except Exception as exc:
                raise TeledyneCameraError(f"Error reading live frame: {exc}") from exc

    def stop_live(self):
        """Stops continuous acquisition and clears camera buffers."""
        with self._lock:
            if not self._live_active:
                return
            if not self._simulate and self._cam is not None:
                try:
                    self._cam.stop_acquisition()
                finally:
                    try:
                        self._cam.clear_acquisition()
                    except Exception:
                        pass
            self._live_active = False
            self._log("Live acquisition stopped.")

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generates a synthetic brightfield microscopy image (`uint16`) matching the active ROI."""
        self._sim_frame_counter += 1
        det_w, det_h = self._detector_size
        hstart, hend, vstart, vend = self._sim_roi
        w = max(1, hend - hstart)
        h = max(1, vend - vstart)

        if self._sim_grid_cache is None or self._sim_grid_cache[0] != h or self._sim_grid_cache[1] != w:
            yy, xx = np.mgrid[vstart:vend, hstart:hend].astype(np.float32)
            self._sim_grid_cache = (h, w, xx, yy)
        else:
            _, _, xx, yy = self._sim_grid_cache

        xn = (xx - 0.5 * det_w) / (0.5 * det_w)
        yn = (yy - 0.5 * det_h) / (0.5 * det_h)

        # Vignetted brightfield illumination + drifting synthetic sample features + crosshair
        illum = np.exp(-0.35 * (xn ** 2 + yn ** 2))
        drift_x = 0.04 * np.sin(self._sim_frame_counter * 0.08)
        drift_y = 0.04 * np.cos(self._sim_frame_counter * 0.06)

        obj1 = 0.45 * np.exp(-(((xn - 0.15 - drift_x) / 0.18) ** 2 + ((yn + 0.10 - drift_y) / 0.12) ** 2))
        obj2 = 0.35 * np.exp(-(((xn + 0.25 - drift_x) / 0.10) ** 2 + ((yn - 0.20 - drift_y) / 0.10) ** 2))
        rings = 0.12 * (0.5 + 0.5 * np.cos(28.0 * np.hypot(xn - drift_x, yn - drift_y)))

        exp_scale = np.clip(self._sim_exposure_s / 0.020, 0.05, 10.0)
        gain_scale = 10.0 ** (self._sim_gain_db / 20.0)
        signal = 42000.0 * exp_scale * gain_scale * illum * (1.0 - obj1 - obj2 - rings * 0.2)

        rng = np.random.default_rng(seed=2000 + self._sim_frame_counter)
        noise = rng.normal(loc=0.0, scale=220.0 * gain_scale, size=(h, w)).astype(np.float32)
        return np.clip(signal + noise, 0, 65535).astype(np.uint16)

    def get_status(self) -> Dict[str, object]:
        """Returns a snapshot of the camera connection state and parameters."""
        with self._lock:
            if not self._connected:
                return {
                    "connected": False,
                    "simulate": False,
                    "live": False,
                    "camera_idx": self._camera_idx,
                    "vendor": "",
                    "model": "",
                    "serial_number": "",
                    "camera_version": "",
                    "detector_size": self.DEFAULT_DETECTOR_SIZE,
                    "pixel_size_um": self.PIXEL_SIZE_UM,
                    "roi": None,
                    "output_shape": None,
                    "exposure_s": None,
                    "gain_db": None,
                    "frame_averaging": self._frame_averaging,
                    "temperature_c": None,
                }
            roi = self.get_roi()
            out_w = roi[1] - roi[0]
            out_h = roi[3] - roi[2]
            return {
                "connected": True,
                "simulate": self._simulate,
                "live": self._live_active,
                "camera_idx": self._camera_idx,
                "vendor": self._vendor,
                "model": self._model,
                "serial_number": self._serial_number,
                "camera_version": self._camera_version,
                "detector_size": self._detector_size,
                "pixel_size_um": self.PIXEL_SIZE_UM,
                "roi": roi,
                "output_shape": (out_h, out_w),
                "exposure_s": self.get_exposure(),
                "gain_db": self.get_gain(),
                "frame_averaging": self._frame_averaging,
                "temperature_c": self.get_sensor_temperature(),
            }
