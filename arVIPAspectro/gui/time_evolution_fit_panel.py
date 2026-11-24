"""
Time Evolution Fit Panel (`TimeEvolutionFitWidget`) & Background Worker (`TimeEvolutionFitWorker`).

Reuses the synchronized HDF5 & Calibration sections of `TimeEvolutionWidget`, and for each
periodic frame fits the theoretical cosine function `S(r) = S0 * cos(0.5 * arcsin(r / M_theta))`
keeping the ellipse center/geometry fixed from the calibration sample, saving & displaying:
  - M_theta(t) (1D vs time)
  - S0(t) (1D vs time)
  - Mean error to fitted curve (S_meas - S_fit) binned in n q-bins vs time
  - Variance w.r.t. fitted curve binned in n q-bins vs time
"""

import time
from typing import Optional, Callable, Dict, Any, List
import numpy as np
from PySide6 import QtWidgets, QtCore

from ..hardware import OrcaFlashCamera, MS2000Stage
from ..acquisition_pipeline import AcquisitionSession
from .time_evolution_panel import TimeEvolutionWidget


class TimeEvolutionFitWorker(QtCore.QThread):
    """
    Periodic acquisition worker that fits `(S0, M_theta)` per frame with fixed ellipse
    geometry, bins residuals in `n_q_bins` q-bins, persists to HDF5, and emits live updates.
    """

    sig_log = QtCore.Signal(str)
    sig_progress = QtCore.Signal(int, int, float)
    sig_time_evolution_fit_updated = QtCore.Signal(object, object, object, object, object, object)
    sig_finished_series = QtCore.Signal(bool, int, str)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        series_name: str,
        exposure_ms: float,
        n_avg: int,
        interval_s: float,
        n_points: Optional[int],
        n_q_bins: int,
        save_raw_data: bool,
        fit_params: Dict[str, Any],
        q_calib_result: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.camera = camera
        self.stage = stage
        self.series_name = series_name
        self.exposure_ms = float(exposure_ms)
        self.n_avg = max(1, int(n_avg))
        self.interval_s = max(0.05, float(interval_s))
        self.n_points = int(n_points) if n_points is not None else None
        self.n_q_bins = max(2, int(n_q_bins))
        self.save_raw_data = bool(save_raw_data)
        self.fit_params = dict(fit_params)
        self.q_calib_result = q_calib_result

        self._stop_requested = False
        self._paused = False

    def request_stop(self):
        self._stop_requested = True

    def set_paused(self, paused: bool):
        self._paused = bool(paused)

    def _sleep_interruptible(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self._stop_requested:
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return not self._stop_requested

    def run(self):
        group_path = ""
        n_done = 0
        try:
            if not self.camera.is_connected:
                raise RuntimeError("Spectrometer camera is not connected.")
            if not self.session.is_open:
                raise RuntimeError("No HDF5 session file is open.")

            self.camera.set_exposure(self.exposure_ms / 1000.0)
            self.camera.set_frame_averaging(self.n_avg)
            st_cam = self.camera.get_status()
            frame_shape = tuple(st_cam["output_shape"])

            pos = None
            if self.stage.is_connected:
                try:
                    pos = self.stage.get_position()
                except Exception:
                    pos = None

            group_path = self.session.initialize_time_evolution_fit_group(
                series_name=self.series_name,
                interval_s=self.interval_s,
                n_points=self.n_points,
                exposure_s=self.exposure_ms / 1000.0,
                n_avg=self.n_avg,
                n_q_bins=self.n_q_bins,
                save_raw_data=self.save_raw_data,
                frame_shape=frame_shape,
                stage_pos_um=pos,
            )

            bin_h = int(self.fit_params.get("bin_val_h", 4))
            bin_w = int(self.fit_params.get("bin_val_w", 2))
            n_orders = self.session.determine_n_orders(bin_val_h=bin_h, bin_val_w=bin_w)
            n_idx = float(self.fit_params.get("refractive_index", 1.33))

            mode_lbl = f"{self.n_points} pts" if self.n_points is not None else "continuous"
            self.sig_log.emit(
                f"Time Evolution Fit started -> '{group_path}' ({mode_lbl}, dt={self.interval_s:.2f} s, "
                f"exp={self.exposure_ms:.1f} ms × {self.n_avg} avg, {self.n_q_bins} q-bins)."
            )

            t_start = time.monotonic()
            times_list: List[float] = []
            m_list: List[float] = []
            s0_list: List[float] = []
            err_cols: List[np.ndarray] = []
            var_cols: List[np.ndarray] = []
            q_edges: Optional[np.ndarray] = None

            t_idx = 0
            while not self._stop_requested:
                if self.n_points is not None and t_idx >= self.n_points:
                    break

                while self._paused and not self._stop_requested:
                    time.sleep(0.05)
                if self._stop_requested:
                    break

                t_point_start = time.monotonic()
                elapsed_s = t_point_start - t_start

                frame = self.camera.snap_frame(
                    n_frames=self.n_avg,
                    stage_pos_um=pos,
                    log_acquisition=False,
                )
                if self._stop_requested:
                    break

                point_fit = self.session.treat_single_map_point(
                    frame=frame,
                    bin_val_h=bin_h,
                    bin_val_w=bin_w,
                    n_orders=n_orders,
                    expected_shift_ghz=float(self.fit_params.get("expected_shift_ghz", 5.1)),
                    shift_tol_ghz=float(self.fit_params.get("shift_tol_ghz", 2.0)),
                    expected_lw_ghz=float(self.fit_params.get("expected_lw_ghz", 0.8)),
                    lw_bounds_ghz=tuple(self.fit_params.get("lw_bounds_ghz", (0.4, 1.3))),
                    refractive_index=n_idx,
                    fast_drift_only=bool(self.fit_params.get("fast_drift_only", True)),
                    q_calib_result=self.q_calib_result,
                )

                cos_fit = self.session.fit_q_cosine_profile(
                    dho_results=point_fit["dho_results"],
                    q_calib_result=self.q_calib_result,
                    refractive_index=n_idx,
                    n_q_bins=self.n_q_bins,
                    q_edges=q_edges,
                )
                q_edges = cos_fit["q_edges"]
                q_centers = cos_fit["q_centers"]

                self.session.append_time_evolution_fit_point(
                    group_path=group_path,
                    t_idx=t_idx,
                    elapsed_s=elapsed_s,
                    fit_res=cos_fit,
                    raw_frame=frame if self.save_raw_data else None,
                )

                times_list.append(elapsed_s)
                m_list.append(float(cos_fit["M_theta_fit"]))
                s0_list.append(float(cos_fit["S0_fit"]))
                err_cols.append(cos_fit["mean_err_q"])
                var_cols.append(cos_fit["var_resid_q"])
                n_done = t_idx + 1

                self.sig_time_evolution_fit_updated.emit(
                    np.asarray(times_list, dtype=np.float64),
                    np.asarray(m_list, dtype=np.float64),
                    np.asarray(s0_list, dtype=np.float64),
                    q_centers,
                    np.column_stack(err_cols),
                    np.column_stack(var_cols),
                )
                self.sig_progress.emit(n_done, self.n_points or 0, elapsed_s)

                t_idx += 1
                if self.n_points is not None and t_idx >= self.n_points:
                    break

                remaining = self.interval_s - (time.monotonic() - t_point_start)
                if remaining > 0.0:
                    if not self._sleep_interruptible(remaining):
                        break

            completed_ok = (not self._stop_requested) or (self.n_points is None and n_done > 0)
            self.sig_log.emit(
                f"Time Evolution Fit finished: {n_done} point(s) saved to '{group_path}'."
            )
            self.sig_finished_series.emit(completed_ok, n_done, group_path)

        except Exception as exc:
            self.sig_error.emit("Time Evolution Fit Error", str(exc))
            self.sig_finished_series.emit(False, n_done, group_path)


class TimeEvolutionFitWidget(TimeEvolutionWidget):
    """
    Time Evolution Fit Tab: shares HDF5 & Calibration controls with `Mapping` and
    `Time Evolution`, and runs per-frame cosine fits `(M_theta, S0)` with fixed ellipse
    geometry from the calibration sample.
    """

    sig_update_time_evolution_fit = QtCore.Signal(object, object, object, object, object, object)

    def __init__(
        self,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        session: AcquisitionSession,
        stop_live_callback: Callable[[], None],
        parent=None,
    ):
        super().__init__(
            camera=camera,
            stage=stage,
            session=session,
            stop_live_callback=stop_live_callback,
            parent=parent,
        )
        self.sec_te.setTitle("4. Time Evolution Fit")
        self.txt_series_name.setText("TimeFit_01")
        self.btn_start_te.setText("Start Time Evolution Fit")

    def _on_start_time_evolution(self):
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the Spectrometer camera first.")
            return
        if not self.session.is_open or self.session.rb_calib_result is None or self._is_busy():
            return

        q_calib = self._active_ref_calib()
        if q_calib is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Reference Calibration Required",
                "Please fit a Reference Sample first before starting Time Evolution Fit.",
            )
            return

        self._stop_live_cb()
        is_fixed = (self.combo_te_stop_mode.currentData() == "fixed")
        n_points = int(self.spin_te_nt.value()) if is_fixed else None

        fit_params = {
            "bin_val_h": int(q_calib.get("bin_val_h", 4)),
            "bin_val_w": int(q_calib.get("bin_val_w", 2)),
            "expected_shift_ghz": float(self.spin_dho_shift.value()),
            "shift_tol_ghz": float(q_calib.get("shift_tol_ghz", 2.0)),
            "expected_lw_ghz": float(q_calib.get("expected_lw_ghz", 0.8)),
            "lw_bounds_ghz": tuple(q_calib.get("lw_bounds_ghz", (0.4, 1.3))),
            "refractive_index": float(self.spin_q_n.value()),
            "fast_drift_only": bool(self.chk_fast_drift.isChecked()),
        }

        self._te_paused = False
        self.btn_pause_te.setText("Pause")
        if is_fixed:
            self.progress_te.setRange(0, n_points)
            self.progress_te.setValue(0)
            self.progress_te.setFormat("0 / %m pts")
        else:
            self.progress_te.setRange(0, 0)
            self.progress_te.setFormat("Running (continuous)...")

        self._set_busy(True, te_running=True)
        self._te_worker = TimeEvolutionFitWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            series_name=self.txt_series_name.text().strip() or "TimeFit_01",
            exposure_ms=float(self.spin_te_exp.value()),
            n_avg=int(self.spin_te_avg.value()),
            interval_s=float(self.spin_te_interval.value()),
            n_points=n_points,
            n_q_bins=int(self.spin_te_nq.value()),
            save_raw_data=bool(self.chk_save_raw.isChecked()),
            fit_params=fit_params,
            q_calib_result=q_calib,
            parent=self,
        )
        self._te_worker.sig_log.connect(self.sig_log.emit)
        self._te_worker.sig_progress.connect(self._on_te_progress)
        self._te_worker.sig_time_evolution_fit_updated.connect(self.sig_update_time_evolution_fit.emit)
        self._te_worker.sig_finished_series.connect(self._on_te_finished)
        self._te_worker.sig_error.connect(self._on_worker_error)
        self._te_worker.start()
