"""
Driver for the Applied Scientific Instrumentation (ASI) MS-2000 motorized stage.

Communicates with the MS-2000 controller over USB / RS-232 virtual COM port
using the official ASI ASCII serial protocol (`pyserial`), and includes a
built-in Simulation / Mock mode (`simulate=True`) for offline development and UI testing.

All public position coordinates and movements are expressed in micrometers (µm).
"""

import time
import threading
from typing import Dict, List, Optional, Callable


class MS2000Error(Exception):
    """Exception raised for ASI MS-2000 communication or hardware errors."""


def list_serial_ports() -> List[Dict[str, str]]:
    """
    Enumerates available serial COM ports on the system.
    Prioritizes USB-to-Serial adapters commonly used by ASI MS-2000 (CP210x, FTDI, USB Serial).
    """
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
    except Exception:
        return []

    def _port_priority(p) -> int:
        desc = f"{p.description or ''} {p.manufacturer or ''} {p.hwid or ''}".lower()
        if any(k in desc for k in ("cp210", "silicon labs", "ftdi", "asi", "usb serial", "usb-serial")):
            return 0
        if any(k in desc for k in ("bluetooth", "debug", "wlan")):
            return 2
        return 1

    ports.sort(key=_port_priority)
    result = []
    for p in ports:
        result.append({
            "port": p.device,
            "description": p.description or "Serial Port",
            "hwid": p.hwid or "",
        })
    return result


class MS2000Stage:
    """
    Hardware driver for the ASI MS-2000 XY(Z) motorized stage controller.

    All positions are reported and commanded in micrometers (µm).
    Note on hardware units:
      - Native ASI MS-2000 serial protocol uses tenths of a micrometer (1 ASI unit = 0.1 µm).
    """

    ASI_UNITS_PER_UM = 10.0  # 1 µm = 10 ASI serial units (0.1 µm)

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        self._log_cb = logger
        self._lock = threading.RLock()
        self._connected: bool = False
        self._simulate: bool = False
        self._backend: str = "none"  # 'pyserial' or 'simulation'
        self._port: Optional[str] = None
        self._baudrate: int = 115200
        self._firmware_info: str = ""
        self._has_z: bool = True

        self._serial = None

        # Simulated state (in µm)
        self._sim_pos: Dict[str, float] = {"X": 0.0, "Y": 0.0, "Z": 0.0}

    def _log(self, message: str):
        if self._log_cb is not None:
            self._log_cb(f"[ASI MS-2000] {message}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulated(self) -> bool:
        return self._simulate

    @property
    def port(self) -> Optional[str]:
        return self._port

    @property
    def baudrate(self) -> int:
        return self._baudrate

    @property
    def backend_name(self) -> str:
        return self._backend

    @property
    def firmware_info(self) -> str:
        return self._firmware_info

    @property
    def has_z_axis(self) -> bool:
        return self._has_z

    # ==========================================================================
    # Connection & Discovery
    # ==========================================================================

    def connect(
        self,
        port: str = "auto",
        baudrate: int = 115200,
        simulate: bool = False,
    ) -> Dict[str, object]:
        """
        Connects to the ASI MS-2000 controller.

        Parameters
        ----------
        port : str
            Serial port name (e.g. 'COM3' on Windows 11) or 'auto' to scan available ports.
        baudrate : int
            Serial baudrate (typically 115200 for USB or 9600 for RS-232).
        simulate : bool
            If True, initializes an in-memory simulated stage without hardware.

        Returns
        -------
        status : dict
            Dictionary with connection details, firmware version, and initial position (µm).
        """
        with self._lock:
            if self._connected:
                self.disconnect()

            if simulate:
                self._connected = True
                self._simulate = True
                self._backend = "simulation"
                self._port = "SIMULATED_COM"
                self._baudrate = baudrate
                self._firmware_info = "ASI MS-2000 Simulator v1.0 (XY+Z)"
                self._has_z = True
                self._log(f"Connected in SIMULATION mode ({self._firmware_info}).")
                return self.get_status()

            # Build candidate port list
            if port and port.lower() != "auto":
                candidate_ports = [port]
                baudrates_to_try = [baudrate]
            else:
                detected = list_serial_ports()
                candidate_ports = [p["port"] for p in detected]
                if not candidate_ports:
                    raise MS2000Error(
                        "No serial COM ports detected on this system. "
                        "Check that the ASI MS-2000 USB/serial cable is plugged in and drivers are installed, "
                        "or enable Simulation Mode."
                    )
                baudrates_to_try = [baudrate]
                for b in (115200, 9600, 28800):
                    if b not in baudrates_to_try:
                        baudrates_to_try.append(b)

            last_error = None
            for cand_port in candidate_ports:
                for cand_baud in baudrates_to_try:
                    self._log(f"Probing port {cand_port} at {cand_baud} baud...")
                    try:
                        self._connect_pyserial(cand_port, cand_baud)
                        self._log(
                            f"Connected on {cand_port} @ {cand_baud} baud "
                            f"({self._firmware_info}, Z-axis={'yes' if self._has_z else 'no'})."
                        )
                        return self.get_status()
                    except Exception as exc:
                        last_error = exc
                        self._cleanup_handles()

            raise MS2000Error(
                f"Could not connect to ASI MS-2000 on ports {candidate_ports} "
                f"(last error: {last_error})."
            )

    def _connect_pyserial(self, port: str, baudrate: int):
        import serial

        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.8,
            write_timeout=0.8,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        self._serial = ser

        # Verify stage communication with 'W X Y' (WHERE X Y)
        resp_where = self._serial_command("W X Y")
        if not resp_where.startswith(":A"):
            ser.close()
            self._serial = None
            raise MS2000Error(f"Unexpected response to 'W X Y': {resp_where!r}")

        # Query firmware version with 'V'
        fw_info = "ASI MS-2000"
        try:
            resp_ver = self._serial_command("V")
            if resp_ver.startswith(":A"):
                ver_str = resp_ver[2:].strip()
                if ver_str:
                    fw_info = f"ASI MS-2000 ({ver_str})"
        except Exception:
            pass

        # Check Z axis availability
        has_z = False
        try:
            resp_z = self._serial_command("W Z")
            if resp_z.startswith(":A"):
                has_z = True
        except Exception:
            has_z = False

        self._connected = True
        self._simulate = False
        self._backend = "pyserial"
        self._port = port
        self._baudrate = baudrate
        self._firmware_info = fw_info
        self._has_z = has_z

    def _serial_command(self, cmd: str) -> str:
        """Sends an ASCII command terminated with CR (\\r) and reads the response line."""
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise MS2000Error("Serial port is not open.")
            full_cmd = (cmd.strip() + "\r").encode("ascii")
            self._serial.reset_input_buffer()
            self._serial.write(full_cmd)
            self._serial.flush()
            raw = self._serial.read_until(b"\n")
            if not raw:
                raise MS2000Error(f"Timeout waiting for response to command {cmd!r}.")
            return raw.decode("ascii", errors="replace").strip()

    def _cleanup_handles(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def disconnect(self):
        """Closes the connection to the ASI MS-2000 controller."""
        with self._lock:
            port_info = self._port
            self._cleanup_handles()
            self._connected = False
            self._simulate = False
            self._backend = "none"
            self._port = None
            if port_info:
                self._log(f"Disconnected from {port_info}.")

    # ==========================================================================
    # Position & Status Queries (in µm)
    # ==========================================================================

    def is_busy(self) -> bool:
        """Returns True if any stage axis is currently moving (`/` STATUS query)."""
        with self._lock:
            if not self._connected:
                raise MS2000Error("Stage is not connected.")
            if self._simulate:
                return False
            resp = self._serial_command("/")
            return resp.upper().startswith("B")

    def get_position(self) -> Dict[str, Optional[float]]:
        """
        Queries the current stage coordinates in micrometers (µm).

        Returns
        -------
        pos : dict
            `{'X': float, 'Y': float, 'Z': float or None}` in µm.
        """
        with self._lock:
            if not self._connected:
                raise MS2000Error("Stage is not connected.")

            if self._simulate:
                return {
                    "X": float(self._sim_pos["X"]),
                    "Y": float(self._sim_pos["Y"]),
                    "Z": float(self._sim_pos["Z"]) if self._has_z else None,
                }

            axes_query = "W X Y Z" if self._has_z else "W X Y"
            resp = self._serial_command(axes_query)
            if not resp.startswith(":A"):
                raise MS2000Error(f"Failed to read stage position: {resp!r}")
            parts = resp[2:].strip().split()
            vals = [float(v) / self.ASI_UNITS_PER_UM for v in parts]
            x_um = vals[0] if len(vals) > 0 else 0.0
            y_um = vals[1] if len(vals) > 1 else 0.0
            z_um = vals[2] if (self._has_z and len(vals) > 2) else None
            return {"X": x_um, "Y": y_um, "Z": z_um}

    def move_to(
        self,
        x_um: Optional[float] = None,
        y_um: Optional[float] = None,
        z_um: Optional[float] = None,
        wait: bool = True,
        timeout_s: float = 30.0,
    ) -> Dict[str, Optional[float]]:
        """
        Moves the stage to the specified absolute coordinates in micrometers (µm).
        """
        with self._lock:
            if not self._connected:
                raise MS2000Error("Stage is not connected.")

            if self._simulate:
                if x_um is not None:
                    self._sim_pos["X"] = float(x_um)
                if y_um is not None:
                    self._sim_pos["Y"] = float(y_um)
                if z_um is not None and self._has_z:
                    self._sim_pos["Z"] = float(z_um)
                time.sleep(0.02)
                pos = self.get_position()
                self._log(f"Moved (sim) to X={pos['X']:.2f} µm, Y={pos['Y']:.2f} µm.")
                return pos

            tokens = ["M"]
            if x_um is not None:
                tokens.append(f"X={x_um * self.ASI_UNITS_PER_UM:.1f}")
            if y_um is not None:
                tokens.append(f"Y={y_um * self.ASI_UNITS_PER_UM:.1f}")
            if z_um is not None and self._has_z:
                tokens.append(f"Z={z_um * self.ASI_UNITS_PER_UM:.1f}")

            if len(tokens) > 1:
                resp = self._serial_command(" ".join(tokens))
                if not resp.startswith(":A"):
                    raise MS2000Error(f"Move command rejected: {resp!r}")
                if wait:
                    self.wait_for_stop(timeout_s=timeout_s)
            return self.get_position()

    def move_relative(
        self,
        dx_um: float = 0.0,
        dy_um: float = 0.0,
        dz_um: float = 0.0,
        wait: bool = True,
        timeout_s: float = 30.0,
    ) -> Dict[str, Optional[float]]:
        """Moves the stage by a relative displacement in micrometers (µm)."""
        with self._lock:
            if not self._connected:
                raise MS2000Error("Stage is not connected.")

            if self._simulate:
                self._sim_pos["X"] += float(dx_um)
                self._sim_pos["Y"] += float(dy_um)
                if self._has_z:
                    self._sim_pos["Z"] += float(dz_um)
                time.sleep(0.02)
                pos = self.get_position()
                self._log(f"Relative move (sim) -> X={pos['X']:.2f} µm, Y={pos['Y']:.2f} µm.")
                return pos

            tokens = ["R"]
            if dx_um != 0.0:
                tokens.append(f"X={dx_um * self.ASI_UNITS_PER_UM:.1f}")
            if dy_um != 0.0:
                tokens.append(f"Y={dy_um * self.ASI_UNITS_PER_UM:.1f}")
            if dz_um != 0.0 and self._has_z:
                tokens.append(f"Z={dz_um * self.ASI_UNITS_PER_UM:.1f}")

            if len(tokens) > 1:
                resp = self._serial_command(" ".join(tokens))
                if not resp.startswith(":A"):
                    raise MS2000Error(f"Relative move rejected: {resp!r}")
                if wait:
                    self.wait_for_stop(timeout_s=timeout_s)
            return self.get_position()

    def wait_for_stop(self, timeout_s: float = 30.0, poll_interval_s: float = 0.02):
        """Blocks until all stage axes have finished moving."""
        if not self._connected or self._simulate:
            return
        t0 = time.time()
        while self.is_busy():
            if time.time() - t0 > timeout_s:
                raise MS2000Error("Timeout waiting for ASI MS-2000 stage movement to complete.")
            time.sleep(poll_interval_s)

    def halt(self):
        """Emergency stop of all stage axes (`\\` command)."""
        with self._lock:
            if not self._connected or self._simulate:
                return
            try:
                if self._serial is not None and self._serial.is_open:
                    self._serial.write(b"\\\r")
                    self._serial.flush()
                self._log("Emergency HALT sent to stage.")
            except Exception as exc:
                self._log(f"Error sending halt: {exc}")

    def get_status(self) -> Dict[str, object]:
        """Returns a complete snapshot of the stage connection state and coordinates."""
        with self._lock:
            if not self._connected:
                return {
                    "connected": False,
                    "simulate": False,
                    "backend": "none",
                    "port": None,
                    "baudrate": self._baudrate,
                    "firmware": "",
                    "has_z": False,
                    "busy": False,
                    "position_um": {"X": None, "Y": None, "Z": None},
                }
            pos = self.get_position()
            return {
                "connected": True,
                "simulate": self._simulate,
                "backend": self._backend,
                "port": self._port,
                "baudrate": self._baudrate,
                "firmware": self._firmware_info,
                "has_z": self._has_z,
                "busy": self.is_busy(),
                "position_um": pos,
            }
