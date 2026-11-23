"""
Centralized RigController, canonical hardware workers, and dry-run proxies.

Replaces duplicated `HardwareActionWorker` / `_StageMoveWorker` and
`AutoCalibratePlateWorker` / `_PlateJogCaptureWorker` implementations across the GUI.
"""

import time
from typing import Any, Callable, Dict, Optional
from PySide6 import QtCore

from ..hardware import (
    StageProtocol,
    CameraProtocol,
    LiveCameraProtocol,
    MS2000Error,
    OrcaCameraError,
    TeledyneCameraError,
)
from ..brightfield_align import build_plate_calibration_from_jogs, compute_jog_correlations
from .state import RigState


class DryRunStage:
    """
    Dry-run proxy implementing `StageProtocol` that logs motion commands
    without energizing physical motors.
    """

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        self._log_cb = logger
        self._connected = True
        self._pos: Dict[str, Optional[float]] = {"X": 0.0, "Y": 0.0, "Z": 0.0}

    def _log(self, msg: str) -> None:
        if self._log_cb is not None:
            self._log_cb(f"[DRY-RUN Stage] {msg}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulated(self) -> bool:
        return True

    @property
    def has_z_axis(self) -> bool:
        return True

    def connect(self, port: str = "auto", baudrate: int = 115200, simulate: bool = True) -> Dict[str, object]:
        self._connected = True
        self._log(f"connect(port={port}, baudrate={baudrate})")
        return self.get_status()

    def disconnect(self) -> None:
        self._connected = False
        self._log("disconnect()")

    def is_busy(self) -> bool:
        return False

    def get_position(self) -> Dict[str, Optional[float]]:
        return dict(self._pos)

    def move_to(
        self,
        x_um: Optional[float] = None,
        y_um: Optional[float] = None,
        z_um: Optional[float] = None,
        wait: bool = True,
        timeout_s: float = 30.0,
    ) -> Dict[str, Optional[float]]:
        if x_um is not None:
            self._pos["X"] = float(x_um)
        if y_um is not None:
            self._pos["Y"] = float(y_um)
        if z_um is not None:
            self._pos["Z"] = float(z_um)
        self._log(f"move_to(X={self._pos['X']}, Y={self._pos['Y']}, Z={self._pos['Z']}, wait={wait})")
        return dict(self._pos)

    def move_relative(
        self,
        dx_um: float = 0.0,
        dy_um: float = 0.0,
        dz_um: float = 0.0,
        wait: bool = True,
        timeout_s: float = 30.0,
    ) -> Dict[str, Optional[float]]:
        self._pos["X"] = float(self._pos["X"] or 0.0) + float(dx_um)
        self._pos["Y"] = float(self._pos["Y"] or 0.0) + float(dy_um)
        self._pos["Z"] = float(self._pos["Z"] or 0.0) + float(dz_um)
        self._log(f"move_relative(dx={dx_um}, dy={dy_um}, dz={dz_um}) -> {self._pos}")
        return dict(self._pos)

    def wait_for_stop(self, timeout_s: float = 30.0, poll_interval_s: float = 0.02) -> None:
        return

    def zero_position(self, axes=("X", "Y")) -> Dict[str, Optional[float]]:
        for ax in axes:
            if ax.upper() in self._pos:
                self._pos[ax.upper()] = 0.0
        self._log(f"zero_position({axes})")
        return dict(self._pos)

    def start_continuous_move(self, vx_um_s: float = 0.0, vy_um_s: float = 0.0) -> None:
        self._log(f"start_continuous_move(vx={vx_um_s}, vy={vy_um_s})")

    def halt(self) -> None:
        self._log("halt()")

    def get_status(self) -> Dict[str, object]:
        return {
            "connected": self._connected,
            "simulate": True,
            "backend": "dry-run",
            "port": "DRY_RUN",
            "baudrate": 0,
            "firmware": "DryRunStage",
            "has_z": True,
            "busy": False,
            "position_um": dict(self._pos),
        }


def move_and_settle(
    stage: StageProtocol,
    *,
    x_um: Optional[float] = None,
    y_um: Optional[float] = None,
    z_um: Optional[float] = None,
    dx_um: float = 0.0,
    dy_um: float = 0.0,
    dz_um: float = 0.0,
    relative: bool = False,
    settling_ms: float = 50.0,
) -> Dict[str, Optional[float]]:
    """
    Canonical stage movement + mechanical vibration settling sequence.
    Centralizes the settling delay previously duplicated across 4 GUI workers.
    """
    if relative:
        pos = stage.move_relative(dx_um=dx_um, dy_um=dy_um, dz_um=dz_um, wait=True)
    else:
        pos = stage.move_to(x_um=x_um, y_um=y_um, z_um=z_um, wait=True)

    settling_s = max(0.0, float(settling_ms)) / 1000.0
    if settling_s > 0.0:
        time.sleep(settling_s)
    return pos


class StageActionWorker(QtCore.QThread):
    """
    Canonical background worker for stage motion, single/averaged frame capture,
    and continuous live preview streaming across both Orca and Teledyne cameras.
    """

    sig_log = QtCore.Signal(str)
    sig_stage_updated = QtCore.Signal()
    sig_frame_acquired = QtCore.Signal(object, object, float, int, int)  # (frame, pos, exp_s, n_avg, binning)
    sig_moved = QtCore.Signal(object, object)  # (pos_dict, optional_snap_tuple)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        camera: CameraProtocol,
        stage: StageProtocol,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        brightfield_cam: Optional[LiveCameraProtocol] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.camera = camera
        self.brightfield_cam = brightfield_cam
        self.stage = stage
        self.action = action  # 'snap', 'move_rel', 'move_abs', 'live'
        self.params = params or {}
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _active_source(self) -> str:
        return str(self.params.get("camera_source", "orca"))

    def run(self) -> None:
        try:
            if self.action == "live":
                self._run_live_loop()
                return

            pos = None
            settling_ms = float(self.params.get("settling_ms", 50.0))

            if self.action == "move_rel":
                pos = move_and_settle(
                    self.stage,
                    dx_um=float(self.params.get("dx_um", 0.0)),
                    dy_um=float(self.params.get("dy_um", 0.0)),
                    dz_um=float(self.params.get("dz_um", 0.0)),
                    relative=True,
                    settling_ms=settling_ms,
                )
                self.sig_stage_updated.emit()

            elif self.action == "move_abs":
                pos = move_and_settle(
                    self.stage,
                    x_um=self.params.get("x_um"),
                    y_um=self.params.get("y_um"),
                    z_um=self.params.get("z_um"),
                    relative=False,
                    settling_ms=settling_ms,
                )
                self.sig_stage_updated.emit()

            label = self.params.get("label")
            if label and pos is not None:
                self.sig_log.emit(
                    f"Moved stage to {label}: "
                    f"X={pos.get('X', 0.0):.2f}, Y={pos.get('Y', 0.0):.2f}, "
                    f"Z={(pos.get('Z') or 0.0):.2f} µm."
                )

            should_snap = (self.action == "snap") or bool(self.params.get("auto_snap", True))
            snap_info = None
            if should_snap:
                if pos is None and self.stage.is_connected:
                    try:
                        pos = self.stage.get_position()
                    except Exception:
                        pos = None

                snap_n_frames = self.params.get("snap_n_frames")
                if self._active_source() == "teledyne":
                    bf = self.brightfield_cam
                    if bf is None or not bf.is_connected:
                        if self.action == "snap":
                            raise TeledyneCameraError("Teledyne Brightfield camera is not connected.")
                        self.sig_log.emit("Stage move finished (brightfield camera not connected -> skipped auto-snap).")
                        self.sig_moved.emit(pos, None)
                        return
                    exp_s = bf.get_exposure()
                    n_avg = int(snap_n_frames) if snap_n_frames is not None else bf.get_frame_averaging()
                    frame = bf.snap_frame(n_frames=n_avg, stage_pos_um=pos)
                    snap_info = (frame, pos, exp_s, n_avg, 1)
                    self.sig_frame_acquired.emit(*snap_info)
                else:
                    if not self.camera.is_connected:
                        if self.action == "snap":
                            raise OrcaCameraError("Camera is not connected.")
                        self.sig_log.emit("Stage move finished (camera not connected -> skipped auto-snap).")
                        self.sig_moved.emit(pos, None)
                        return
                    exp_s = self.camera.get_exposure()
                    n_avg = int(snap_n_frames) if snap_n_frames is not None else self.camera.get_frame_averaging()
                    binning = self.camera.get_binning() if hasattr(self.camera, "get_binning") else 1
                    frame = self.camera.snap_frame(n_frames=n_avg, stage_pos_um=pos)
                    snap_info = (frame, pos, exp_s, n_avg, binning)
                    self.sig_frame_acquired.emit(*snap_info)

            if self.action in ("move_abs", "move_rel"):
                self.sig_moved.emit(pos, snap_info)

        except Exception as exc:
            self.sig_error.emit("Hardware Action Error", str(exc))

    def _run_live_loop(self) -> None:
        pos = None
        if self.stage.is_connected:
            try:
                pos = self.stage.get_position()
            except Exception:
                pos = None

        if self._active_source() == "teledyne":
            bf = self.brightfield_cam
            if bf is None or not bf.is_connected:
                return
            self.sig_log.emit("Live Preview started (Teledyne Brightfield).")
            bf.start_live()
            try:
                while not self._stop_requested and bf.is_connected:
                    exp_s = bf.get_exposure()
                    frame = bf.read_live_frame(timeout_s=0.3)
                    if self._stop_requested:
                        break
                    if frame is not None:
                        self.sig_frame_acquired.emit(frame, pos, exp_s, 1, 1)
                    if bf.is_simulated:
                        time.sleep(min(max(exp_s, 0.015), 0.1))
            finally:
                try:
                    bf.stop_live()
                except Exception:
                    pass
                self.sig_log.emit("Live Preview stopped (Teledyne Brightfield).")
            return

        self.sig_log.emit("[Orca Flash 4.0 v3] Live view started")
        while not self._stop_requested and self.camera.is_connected:
            exp_s = self.camera.get_exposure()
            binning = self.camera.get_binning() if hasattr(self.camera, "get_binning") else 1
            frame = self.camera.snap_frame(n_frames=1, stage_pos_um=pos, log_acquisition=False)
            if self._stop_requested:
                break
            self.sig_frame_acquired.emit(frame, pos, exp_s, 1, binning)
            if self.camera.is_simulated:
                time.sleep(min(max(exp_s, 0.015), 0.1))
        self.sig_log.emit("[Orca Flash 4.0 v3] Live view stopped")


class PlateAutoCalibrateWorker(QtCore.QThread):
    """Background worker for 3-point brightfield pixel-to-stage auto-calibration."""

    sig_log = QtCore.Signal(str)
    sig_stage_updated = QtCore.Signal()
    sig_finished_calib = QtCore.Signal(object)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        brightfield_cam: CameraProtocol,
        stage: StageProtocol,
        step_um: float,
        settling_ms: float = 80.0,
        parent=None,
    ):
        super().__init__(parent)
        self.brightfield_cam = brightfield_cam
        self.stage = stage
        self.step_um = float(step_um)
        self.settling_ms = float(settling_ms)

    def run(self) -> None:
        try:
            if not self.brightfield_cam.is_connected:
                raise TeledyneCameraError("Brightfield camera is not connected.")
            if not self.stage.is_connected:
                raise MS2000Error("Stage is not connected.")

            pos0 = self.stage.get_position()
            x0, y0 = float(pos0["X"]), float(pos0["Y"])
            n_avg = self.brightfield_cam.get_frame_averaging()

            self.sig_log.emit(f"Plate auto-calibrate: capturing base frame at ({x0:.1f}, {y0:.1f}) µm...")
            img0 = self.brightfield_cam.snap_frame(n_frames=n_avg, stage_pos_um=pos0)

            self.sig_log.emit(f"Plate auto-calibrate: moving +{self.step_um:.1f} µm in X...")
            pos_x = move_and_settle(self.stage, x_um=x0 + self.step_um, y_um=y0, settling_ms=self.settling_ms)
            self.sig_stage_updated.emit()
            img_x = self.brightfield_cam.snap_frame(n_frames=n_avg, stage_pos_um=pos_x)

            self.sig_log.emit(f"Plate auto-calibrate: moving +{self.step_um:.1f} µm in Y...")
            pos_y = move_and_settle(self.stage, x_um=x0, y_um=y0 + self.step_um, settling_ms=self.settling_ms)
            self.sig_stage_updated.emit()
            img_y = self.brightfield_cam.snap_frame(n_frames=n_avg, stage_pos_um=pos_y)

            self.stage.move_to(x_um=x0, y_um=y0, wait=True)
            self.sig_stage_updated.emit()

            corr = compute_jog_correlations(img0=img0, img_x=img_x, img_y=img_y)
            calib = build_plate_calibration_from_jogs(
                dx_um=self.step_um,
                dy_um=self.step_um,
                dp_from_x_px=corr["dp_x"],
                dp_from_y_px=corr["dp_y"],
            )
            self.sig_finished_calib.emit(calib)
        except Exception as exc:
            self.sig_error.emit("Plate Auto-Calibrate Error", str(exc))


class PlateJogCaptureWorker(QtCore.QThread):
    """Executes one stage-move + brightfield-snap step for interactive 3-point plate calibration."""

    sig_log = QtCore.Signal(str)
    sig_captured = QtCore.Signal(object)
    sig_stage_updated = QtCore.Signal()
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        stage: StageProtocol,
        brightfield_cam: CameraProtocol,
        step_tag: str,
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
        return_pos: Optional[tuple] = None,
        settling_ms: float = 300.0,
        parent=None,
    ):
        super().__init__(parent)
        self.stage = stage
        self.brightfield_cam = brightfield_cam
        self.step_tag = step_tag
        self.target_x = target_x
        self.target_y = target_y
        self.return_pos = return_pos
        self.settling_ms = max(0.0, float(settling_ms))

    def run(self) -> None:
        try:
            if self.target_x is not None and self.target_y is not None:
                pos = move_and_settle(
                    self.stage,
                    x_um=float(self.target_x),
                    y_um=float(self.target_y),
                    settling_ms=self.settling_ms,
                )
                self.sig_stage_updated.emit()
            else:
                pos = self.stage.get_position()

            exp_s = self.brightfield_cam.get_exposure()
            frame = self.brightfield_cam.snap_frame(n_frames=1, stage_pos_um=pos)
            self.sig_log.emit(
                f"Plate Calib [{self.step_tag}] acquired ({frame.shape[1]}x{frame.shape[0]} px) "
                f"@ ({float(pos['X']):.2f}, {float(pos['Y']):.2f}) µm."
            )

            if self.return_pos is not None:
                self.stage.move_to(x_um=float(self.return_pos[0]), y_um=float(self.return_pos[1]), wait=True)
                self.sig_stage_updated.emit()

            self.sig_captured.emit(
                {
                    "step_tag": self.step_tag,
                    "frame": frame,
                    "pos": pos,
                    "exp_s": exp_s,
                }
            )
        except Exception as exc:
            if self.return_pos is not None:
                try:
                    self.stage.move_to(x_um=float(self.return_pos[0]), y_um=float(self.return_pos[1]), wait=True)
                    self.sig_stage_updated.emit()
                except Exception:
                    pass
            self.sig_error.emit(f"Plate Calibration Error ({self.step_tag})", str(exc))


class RigController(QtCore.QObject):
    """
    Centralized Control & State Machine coordinator for the arVIPAspectro rig.
    Owns the HAL device references and dispatches non-blocking motion/acquisition actions.
    """

    sig_state_changed = QtCore.Signal(str)
    sig_stage_updated = QtCore.Signal()

    def __init__(
        self,
        camera: CameraProtocol,
        stage: StageProtocol,
        brightfield_cam: Optional[LiveCameraProtocol] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.camera = camera
        self.stage = stage
        self.brightfield_cam = brightfield_cam
        self._state: RigState = RigState.IDLE
        self._active_worker: Optional[QtCore.QThread] = None

    @property
    def state(self) -> RigState:
        return self._state

    def set_state(self, new_state: RigState) -> None:
        if self._state != new_state:
            self._state = new_state
            self.sig_state_changed.emit(new_state.value)

    def emergency_halt(self) -> None:
        """Immediately halts stage motion and requests stop on any active worker."""
        if self.stage.is_connected:
            self.stage.halt()
        if self._active_worker is not None and hasattr(self._active_worker, "request_stop"):
            self._active_worker.request_stop()
        self.set_state(RigState.IDLE)
        self.sig_stage_updated.emit()

    def move_to_async(
        self,
        *,
        x_um: Optional[float] = None,
        y_um: Optional[float] = None,
        z_um: Optional[float] = None,
        settling_ms: float = 50.0,
        auto_snap: bool = False,
        label: Optional[str] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_moved: Optional[Callable[[Any, Any], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        parent=None,
    ) -> StageActionWorker:
        """Dispatches a non-blocking stage move (with optional snap) via `StageActionWorker`."""
        worker = StageActionWorker(
            camera=self.camera,
            brightfield_cam=self.brightfield_cam,
            stage=self.stage,
            action="move_abs",
            params={
                "x_um": x_um,
                "y_um": y_um,
                "z_um": z_um,
                "settling_ms": settling_ms,
                "auto_snap": auto_snap,
                "snap_n_frames": 1,
                "label": label,
            },
            parent=parent or self,
        )
        self._active_worker = worker
        self.set_state(RigState.MOVING)
        if on_log is not None:
            worker.sig_log.connect(on_log)
        worker.sig_stage_updated.connect(self.sig_stage_updated.emit)
        if on_moved is not None:
            worker.sig_moved.connect(on_moved)
        if on_error is not None:
            worker.sig_error.connect(on_error)

        def _done():
            if self._active_worker is worker:
                self._active_worker = None
            if self._state == RigState.MOVING:
                self.set_state(RigState.IDLE)
            if on_finished is not None:
                on_finished()

        worker.finished.connect(_done)
        worker.start()
        return worker
