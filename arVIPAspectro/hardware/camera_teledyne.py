"""
Driver for the Teledyne FLIR (Point Grey) Chameleon3 CM3-U3-31S4M-CS monochrome
USB 3.0 camera (Sony IMX265 Pregius global shutter CMOS, 2048 x 1536 px, 3.45 um).

Supports both:
  1. **PyCapture2** (FlyCapture2 Python wrapper — primary when FlyCapture2 is installed)
  2. **pylablib.devices.Teledyne** (Teledyne Spinnaker C SDK — fallback/alternative)

Includes a built-in Simulation / Mock mode (`simulate=True`) for offline
development and UI testing.
"""

import glob
import os
import time
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class TeledyneCameraError(Exception):
    """Exception raised for Teledyne / Point Grey camera communication or SDK errors."""


def _decode_str(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    return str(val).strip()


def _find_spinnaker_c_dll_dir() -> Optional[str]:
    """
    Searches common Teledyne / FLIR Spinnaker SDK installation paths on Windows
    for ``SpinnakerC_v*.dll`` and returns the containing directory if found.
    """
    if os.name != "nt":
        return None

    candidate_dirs: List[Path] = []
    try:
        import pylablib as pll
        configured = pll.par.get("devices/dlls/teledyne_spinnaker", None)
        if configured:
            candidate_dirs.append(Path(configured))
    except Exception:
        pass

    for env_var in ("SPINNAKER_GENTL64_CTI", "SPINNAKER_GENTL32_CTI"):
        val = os.environ.get(env_var)
        if val:
            p = Path(val).parent
            candidate_dirs.extend([
                p,
                p / "vs2015",
                p / "vs2017",
                p / "vs2019",
                p / "vs2022",
                p.parent / "bin64" / "vs2015",
                p.parent / "bin64",
            ])

    prog_roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]
    for root in prog_roots:
        if not root:
            continue
        for vendor_dir in ("Teledyne", "FLIR Systems", "Point Grey Research"):
            base = Path(root) / vendor_dir / "Spinnaker"
            for sub in (
                "bin64/vs2015",
                "bin64/vs2017",
                "bin64/vs2019",
                "bin64/vs2022",
                "bin64",
                "bin/vs2015",
                "bin",
            ):
                candidate_dirs.append(base / sub)

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry.strip():
            candidate_dirs.append(Path(entry.strip()))

    seen = set()
    for d in candidate_dirs:
        try:
            resolved = str(d.resolve())
        except Exception:
            resolved = str(d)
        if resolved in seen or not d.is_dir():
            continue
        seen.add(resolved)
        if glob.glob(str(d / "SpinnakerC_v*.dll")):
            return resolved
    return None


def _configure_spinnaker_dll_path() -> Optional[str]:
    """Configures `pylablib` and Windows DLL search paths for `SpinnakerC_v*.dll`."""
    dll_dir = _find_spinnaker_c_dll_dir()
    if dll_dir:
        try:
            import pylablib as pll
            pll.par["devices/dlls/teledyne_spinnaker"] = dll_dir
        except Exception:
            pass
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
            parent_dir = str(Path(dll_dir).parent)
            if Path(parent_dir).is_dir():
                try:
                    os.add_dll_directory(parent_dir)
                except Exception:
                    pass
    return dll_dir


def _list_pycapture2_cameras() -> Tuple[List[Dict[str, object]], Optional[str]]:
    """Lists connected FlyCapture2 cameras via `PyCapture2`."""
    try:
        import PyCapture2
    except ImportError as exc:
        return [], f"PyCapture2 not installed: {exc}"

    try:
        bus = PyCapture2.BusManager()
        n_cams = int(bus.getNumOfCameras())
        cams: List[Dict[str, object]] = []
        for idx in range(n_cams):
            uid = bus.getCameraFromIndex(idx)
            entry: Dict[str, object] = {
                "index": idx,
                "name": f"FlyCapture2 Camera #{idx}",
                "model": "",
                "serial": "",
                "vendor": "Point Grey / FLIR",
                "backend": "pycapture2",
            }
            tmp = PyCapture2.Camera()
            try:
                tmp.connect(uid)
                info = tmp.getCameraInfo()
                entry["model"] = _decode_str(getattr(info, "modelName", ""))
                entry["serial"] = str(getattr(info, "serialNumber", ""))
                entry["vendor"] = _decode_str(getattr(info, "vendorName", "Point Grey / FLIR"))
                entry["name"] = f"{entry['vendor']} {entry['model']}".strip()
            except Exception:
                try:
                    entry["serial"] = str(bus.getCameraSerialNumberFromIndex(idx))
                except Exception:
                    pass
            finally:
                try:
                    tmp.disconnect()
                except Exception:
                    pass
            cams.append(entry)
        return cams, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _list_spinnaker_cameras() -> Tuple[List[Dict[str, object]], Optional[str]]:
    """Lists connected Spinnaker cameras via `pylablib.devices.Teledyne`."""
    _configure_spinnaker_dll_path()
    try:
        from pylablib.devices import Teledyne
        infos = Teledyne.list_cameras(desc=True)
        cams = [
            {
                "index": idx,
                "name": str(getattr(info, "name", "")),
                "model": str(getattr(info, "model", "")),
                "serial": str(getattr(info, "serial", "")),
                "vendor": str(getattr(info, "vendor", "")),
                "backend": "spinnaker",
            }
            for idx, info in enumerate(infos)
        ]
        return cams, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def diagnose_teledyne_backend() -> Dict[str, object]:
    """
    Collects diagnostic information for both PyCapture2 (FlyCapture2) and
    pylablib Spinnaker backends.
    """
    fc2_cams, fc2_err = _list_pycapture2_cameras()
    dll_dir = _find_spinnaker_c_dll_dir()
    spin_cams, spin_err = _list_spinnaker_cameras()
    return {
        "os": os.name,
        "pycapture2_available": fc2_err is None or not str(fc2_err).startswith("PyCapture2 not installed"),
        "pycapture2_cameras": fc2_cams,
        "pycapture2_error": fc2_err,
        "spinnaker_c_dll_dir": dll_dir,
        "spinnaker_cameras": spin_cams,
        "spinnaker_error": spin_err,
    }


def list_teledyne_cameras() -> List[Dict[str, object]]:
    """
    Returns a list of detected Teledyne / Point Grey cameras.
    Checks PyCapture2 (FlyCapture2) first, then Teledyne Spinnaker (`pylablib`).
    """
    fc2_cams, _ = _list_pycapture2_cameras()
    if fc2_cams:
        return fc2_cams
    spin_cams, _ = _list_spinnaker_cameras()
    return spin_cams


def count_teledyne_cameras() -> int:
    """Returns the number of detected cameras across PyCapture2 and Spinnaker."""
    return len(list_teledyne_cameras())


def find_chameleon3_camera(model_substr: str = "CM3-U3-31S4M") -> Optional[int]:
    """
    Searches connected cameras for a Chameleon3 CM3-U3-31S4M-CS model.
    Returns the matching zero-based camera index, falls back to ``0`` if another
    Teledyne / Point Grey camera is connected, or returns ``None`` if none found.
    """
    cams = list_teledyne_cameras()
    if not cams:
        return None
    target = model_substr.lower()
    for cam in cams:
        if target in str(cam.get("model", "")).lower() or target in str(cam.get("name", "")).lower():
            return int(cam["index"])
    return 0


def _pycapture2_image_to_numpy(image) -> np.ndarray:
    """Converts a `PyCapture2.Image` buffer into a 2D NumPy array."""
    rows = int(image.getRows())
    cols = int(image.getCols())
    raw = np.asarray(image.getData(), dtype=np.uint8)
    expected_8 = rows * cols
    if raw.size == expected_8:
        return raw.reshape((rows, cols)).copy()
    if raw.size == expected_8 * 2:
        return raw.view(np.uint16).reshape((rows, cols)).copy()
    if raw.size == expected_8 * 3:
        return raw.reshape((rows, cols, 3)).copy()
    # Fallback using stride if padded
    data_size = int(image.getDataSize())
    stride = data_size // max(1, rows)
    if stride >= cols:
        return raw[: rows * stride].reshape((rows, stride))[:, :cols].copy()
    raise TeledyneCameraError(
        f"Unexpected PyCapture2 image buffer size {raw.size} for shape ({rows}, {cols})."
    )


class TeledyneBrightfieldCamera:
    """
    Hardware driver for the Teledyne FLIR (Point Grey) CM3-U3-31S4M-CS brightfield camera.
    Supports PyCapture2 (FlyCapture2) and Teledyne Spinnaker (`pylablib`) backends,
    plus a built-in Simulation Mode (`simulate=True`).
    """

    DEFAULT_DETECTOR_SIZE: Tuple[int, int] = (2048, 1536)  # (width, height) in pixels
    PIXEL_SIZE_UM: float = 3.45

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        self._log_cb = logger
        self._lock = threading.RLock()
        self._connected: bool = False
        self._simulate: bool = False
        self._live_active: bool = False
        self._backend: str = "none"  # "pycapture2" | "spinnaker" | "simulate"
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
        self._sim_exposure_s: float = 0.020
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
    def backend(self) -> str:
        return self._backend

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
        backend: str = "auto",
    ) -> Dict[str, object]:
        """
        Opens a connection to the Teledyne CM3-U3-31S4M-CS camera.

        Parameters
        ----------
        camera_idx : int, optional
            Zero-based camera index. If None (and `serial` is None), automatically
            searches for a connected CM3-U3-31S4M camera.
        serial : str, optional
            Specific camera serial number to open.
        simulate : bool
            If True, connects to an internal brightfield camera simulator (2048x1536).
        backend : str
            ``"auto"`` (prefers PyCapture2/FlyCapture2, falls back to Spinnaker),
            ``"pycapture2"``, or ``"spinnaker"``.
        """
        with self._lock:
            if self._connected:
                self.disconnect()

            if simulate:
                self._connected = True
                self._simulate = True
                self._live_active = False
                self._backend = "simulate"
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

            chosen = backend.lower().strip()
            if chosen == "auto":
                fc2_cams, _ = _list_pycapture2_cameras()
                if fc2_cams:
                    chosen = "pycapture2"
                else:
                    chosen = "spinnaker"

            if chosen == "pycapture2":
                return self._connect_pycapture2(camera_idx=camera_idx, serial=serial)
            if chosen == "spinnaker":
                return self._connect_spinnaker(camera_idx=camera_idx, serial=serial)
            raise TeledyneCameraError(f"Unknown backend '{backend}'. Use 'auto', 'pycapture2', or 'spinnaker'.")

    def _connect_pycapture2(
        self,
        camera_idx: Optional[int] = None,
        serial: Optional[str] = None,
    ) -> Dict[str, object]:
        try:
            import PyCapture2
        except ImportError as exc:
            raise TeledyneCameraError(f"PyCapture2 is not installed: {exc}") from exc

        try:
            bus = PyCapture2.BusManager()
            n_cams = int(bus.getNumOfCameras())
            if n_cams <= 0:
                raise TeledyneCameraError(
                    "PyCapture2 initialized, but 0 FlyCapture2 cameras were detected."
                )

            if serial is not None:
                uid = bus.getCameraFromSerialNumber(int(serial))
                idx_used = 0
            else:
                if camera_idx is None:
                    fc2_cams, _ = _list_pycapture2_cameras()
                    idx_used = 0
                    for c in fc2_cams:
                        if "cm3-u3-31s4m" in str(c.get("model", "")).lower():
                            idx_used = int(c["index"])
                            break
                else:
                    idx_used = int(camera_idx)
                if idx_used >= n_cams:
                    raise TeledyneCameraError(
                        f"Requested camera index {idx_used}, but only {n_cams} FlyCapture2 camera(s) found."
                    )
                uid = bus.getCameraFromIndex(idx_used)

            cam = PyCapture2.Camera()
            cam.connect(uid)

            info = cam.getCameraInfo()
            try:
                fmt7_info, _ = cam.getFormat7Info(PyCapture2.MODE.MODE_0)
                det_size = (int(fmt7_info.maxWidth), int(fmt7_info.maxHeight))
            except Exception:
                det_size = self.DEFAULT_DETECTOR_SIZE

            # Configure low-latency DROP_FRAMES mode for live viewing
            try:
                cfg = cam.getConfiguration()
                cfg.grabMode = PyCapture2.GRAB_MODE.DROP_FRAMES
                cfg.grabTimeout = 500
                cam.setConfiguration(cfg)
            except Exception:
                pass

            # Disable auto-exposure / auto-shutter / auto-gain for deterministic manual control
            for ptype in (
                PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE,
                PyCapture2.PROPERTY_TYPE.SHUTTER,
                PyCapture2.PROPERTY_TYPE.GAIN,
            ):
                try:
                    prop = cam.getProperty(ptype)
                    prop.autoManualMode = False
                    prop.onOff = True
                    if ptype in (PyCapture2.PROPERTY_TYPE.SHUTTER, PyCapture2.PROPERTY_TYPE.GAIN):
                        prop.absControl = True
                    cam.setProperty(prop)
                except Exception:
                    pass

            self._cam = cam
            self._connected = True
            self._simulate = False
            self._live_active = False
            self._backend = "pycapture2"
            self._camera_idx = idx_used
            self._vendor = _decode_str(getattr(info, "vendorName", "Point Grey / FLIR"))
            self._model = _decode_str(getattr(info, "modelName", "CM3-U3-31S4M-CS"))
            self._serial_number = str(getattr(info, "serialNumber", "Unknown"))
            self._camera_version = _decode_str(getattr(info, "firmwareVersion", "Unknown"))
            self._detector_size = det_size

            self._log(
                f"Connected via PyCapture2 to {self._vendor} {self._model} "
                f"(S/N: {self._serial_number}, Sensor: {det_size[0]}x{det_size[1]} px)."
            )
            return self.get_status()
        except TeledyneCameraError:
            raise
        except Exception as exc:
            raise TeledyneCameraError(f"Error opening camera via PyCapture2: {exc}") from exc

    def _connect_spinnaker(
        self,
        camera_idx: Optional[int] = None,
        serial: Optional[str] = None,
    ) -> Dict[str, object]:
        dll_dir = _configure_spinnaker_dll_path()
        if dll_dir:
            self._log(f"Using Spinnaker C DLL directory: {dll_dir}")

        try:
            from pylablib.devices import Teledyne
        except ImportError as exc:
            raise TeledyneCameraError(
                f"The `pylablib` package is required for Spinnaker camera control: {exc}"
            ) from exc

        try:
            n_cams = Teledyne.get_cameras_number()
        except Exception as exc:
            raise TeledyneCameraError(
                "Failed to initialize Teledyne Spinnaker SDK (and no PyCapture2 camera found).\n"
                f"Details: {exc}"
            ) from exc

        if n_cams <= 0:
            raise TeledyneCameraError(
                "Teledyne Spinnaker SDK initialized, but 0 cameras were detected."
            )

        if serial is None and camera_idx is None:
            spin_cams, _ = _list_spinnaker_cameras()
            camera_idx = 0
            for c in spin_cams:
                if "cm3-u3-31s4m" in str(c.get("model", "")).lower():
                    camera_idx = int(c["index"])
                    break

        if serial is None and camera_idx is not None and camera_idx >= n_cams:
            raise TeledyneCameraError(
                f"Requested camera index {camera_idx}, but only {n_cams} Spinnaker camera(s) found."
            )

        try:
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
            self._backend = "spinnaker"
            self._camera_idx = 0 if camera_idx is None else int(camera_idx)
            self._vendor = str(getattr(info, "vendor", "Teledyne FLIR"))
            self._model = str(getattr(info, "model", "CM3-U3-31S4M-CS"))
            self._serial_number = str(getattr(info, "serial", "Unknown"))
            self._camera_version = str(getattr(info, "devversion", "Unknown"))
            self._detector_size = (int(det_size[0]), int(det_size[1]))

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

            self._log(
                f"Connected via Spinnaker to {self._vendor} {self._model} "
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
            raise TeledyneCameraError(f"Error opening Teledyne camera via Spinnaker: {exc}") from exc

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
                    if self._backend == "pycapture2":
                        self._cam.disconnect()
                    elif self._backend == "spinnaker":
                        self._cam.close()
                except Exception:
                    pass
                self._cam = None
            was_connected = self._connected
            self._connected = False
            self._simulate = False
            self._backend = "none"
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
            if self._backend == "pycapture2":
                import PyCapture2
                prop = self._cam.getProperty(PyCapture2.PROPERTY_TYPE.SHUTTER)
                return float(prop.absValue) * 1e-3
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
            if self._backend == "pycapture2":
                import PyCapture2
                prop = self._cam.getProperty(PyCapture2.PROPERTY_TYPE.SHUTTER)
                prop.onOff = True
                prop.autoManualMode = False
                prop.absControl = True
                prop.absValue = float(exposure_s * 1e3)
                self._cam.setProperty(prop)
                actual = float(self._cam.getProperty(PyCapture2.PROPERTY_TYPE.SHUTTER).absValue) * 1e-3
                self._log(f"Exposure set to {actual * 1e3:.3f} ms.")
                return actual
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
            if self._backend == "pycapture2":
                try:
                    import PyCapture2
                    prop = self._cam.getProperty(PyCapture2.PROPERTY_TYPE.GAIN)
                    return float(prop.absValue)
                except Exception:
                    return None
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
            if self._backend == "pycapture2":
                try:
                    import PyCapture2
                    prop = self._cam.getProperty(PyCapture2.PROPERTY_TYPE.GAIN)
                    prop.onOff = True
                    prop.autoManualMode = False
                    prop.absControl = True
                    prop.absValue = float(gain_db)
                    self._cam.setProperty(prop)
                    actual = self.get_gain()
                    self._log(f"Gain set to {actual} dB.")
                    return actual
                except Exception as exc:
                    self._log(f"Failed to set gain: {exc}")
                    return self.get_gain()
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
            if self._backend == "pycapture2":
                try:
                    fmt7_cfg, _, _ = self._cam.getFormat7Configuration()
                    ox = int(fmt7_cfg.offsetX)
                    oy = int(fmt7_cfg.offsetY)
                    w = int(fmt7_cfg.width)
                    h = int(fmt7_cfg.height)
                    return (ox, ox + w, oy, oy + h)
                except Exception:
                    det_w, det_h = self._detector_size
                    return (0, det_w, 0, det_h)
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

            if self._backend == "pycapture2":
                was_live = self._live_active
                if was_live:
                    self.stop_live()
                try:
                    fmt7_cfg, pkt_size, _ = self._cam.getFormat7Configuration()
                    fmt7_cfg.offsetX = hstart
                    fmt7_cfg.offsetY = vstart
                    fmt7_cfg.width = hend - hstart
                    fmt7_cfg.height = vend - vstart
                    pkt_info, is_valid = self._cam.validateFormat7Settings(fmt7_cfg)
                    if is_valid:
                        self._cam.setFormat7Configuration(fmt7_cfg, pkt_info.recommendedBytesPerPacket)
                    else:
                        self._cam.setFormat7Configuration(fmt7_cfg, pkt_size)
                finally:
                    if was_live:
                        self.start_live()
                return self.get_roi()

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
            if self._backend == "pycapture2":
                try:
                    import PyCapture2
                    prop = self._cam.getProperty(PyCapture2.PROPERTY_TYPE.TEMPERATURE)
                    # FlyCapture2 reports temperature in tenths of Kelvin
                    return float(prop.absValue) - 273.15 if prop.absControl else (float(prop.valueA) / 10.0 - 273.15)
                except Exception:
                    return None
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
        """Acquires a single 2D image frame (or averaged stack of `n_frames`)."""
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

            if self._backend == "pycapture2":
                was_live = self._live_active
                try:
                    if not was_live:
                        self._cam.startCapture()
                    if n_frames == 1:
                        return _pycapture2_image_to_numpy(self._cam.retrieveBuffer())
                    first = _pycapture2_image_to_numpy(self._cam.retrieveBuffer())
                    acc = first.astype(np.float64)
                    for _ in range(n_frames - 1):
                        acc += _pycapture2_image_to_numpy(self._cam.retrieveBuffer())
                    out_dtype = np.uint16 if first.dtype == np.uint16 else np.uint8
                    max_val = 65535 if out_dtype == np.uint16 else 255
                    return np.clip(np.rint(acc / n_frames), 0, max_val).astype(out_dtype)
                except Exception as exc:
                    raise TeledyneCameraError(f"Failed to acquire PyCapture2 frame: {exc}") from exc
                finally:
                    if not was_live:
                        try:
                            self._cam.stopCapture()
                        except Exception:
                            pass

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
            if self._backend == "pycapture2":
                try:
                    import PyCapture2
                    cfg = self._cam.getConfiguration()
                    cfg.grabMode = PyCapture2.GRAB_MODE.DROP_FRAMES
                    cfg.grabTimeout = 200
                    self._cam.setConfiguration(cfg)
                    self._cam.startCapture()
                    self._live_active = True
                    self._log("Live acquisition started (PyCapture2).")
                    return
                except Exception as exc:
                    raise TeledyneCameraError(f"Failed to start PyCapture2 live capture: {exc}") from exc
            try:
                self._cam.start_acquisition(buffsize=int(buffsize))
                self._live_active = True
                self._log("Live acquisition started (Spinnaker).")
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
            if self._backend == "pycapture2":
                try:
                    image = self._cam.retrieveBuffer()
                    return _pycapture2_image_to_numpy(image)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "timeout" in msg or "timed out" in msg:
                        return None
                    raise TeledyneCameraError(f"Error reading PyCapture2 frame: {exc}") from exc
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
                if self._backend == "pycapture2":
                    try:
                        self._cam.stopCapture()
                    except Exception:
                        pass
                else:
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
                    "backend": "none",
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
                "backend": self._backend,
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
