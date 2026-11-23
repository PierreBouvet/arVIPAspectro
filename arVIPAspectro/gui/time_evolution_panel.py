"""
Time Evolution Acquisition & Live q-Dependent Shift Monitoring Panel (`TimeEvolutionWidget`).

Shares the HDF5_BLS `AcquisitionSession` and calibration controls bidirectionally with
`PipelineAcquisitionWidget` ('Mapping' tab), and runs periodic acquisitions while displaying
live stacked heatmaps of Mean Shift ν_B(q, t) and Shift Variance σ²(q, t).
"""

import os
import time
from typing import Optional, Callable, Dict, Any, List
import numpy as np
from PySide6 import QtWidgets, QtCore

from .. import config
from ..hardware import OrcaFlashCamera, MS2000Stage
from ..acquisition_pipeline import AcquisitionSession
from .pipeline_panel import (
    CalibrationCaptureWorker,
    LaserCalibrationWorker,
    RbCalibrationWorker,
    SampleTreatmentWorker,
)


class TimeEvolutionWorker(QtCore.QThread):
    """
    Background worker that acquires periodic spectrometer frames at a fixed interval,
    fits DHO Brillouin shifts on the fly, computes q-binned mean & variance profiles,
    appends each point to `Brillouin/<series_name>` in HDF5, and emits live heatmap updates.
    """

    sig_log = QtCore.Signal(str)
    sig_progress = QtCore.Signal(int, int, float)  # (done_count, total_or_0, elapsed_s)
    sig_time_evolution_updated = QtCore.Signal(object, object, object, object, object)
    sig_finished_series = QtCore.Signal(bool, int, str)  # (completed_ok, n_done, group_path)
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
        n_points: Optional[int],  # None = continuous until Stop
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
        """Sleeps up to `seconds`, waking early if stop is requested. Returns False if stopped."""
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

            group_path = self.session.initialize_time_evolution_group(
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
                f"Time Evolution started -> '{group_path}' ({mode_lbl}, dt={self.interval_s:.2f} s, "
                f"exp={self.exposure_ms:.1f} ms × {self.n_avg} avg, {self.n_q_bins} q-bins)."
            )

            t_start = time.monotonic()
            times_list: List[float] = []
            mean_cols: List[np.ndarray] = []
            var_cols: List[np.ndarray] = []
            q_edges: Optional[np.ndarray] = None
            q_centers: Optional[np.ndarray] = None

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

                q_edges, q_centers, mean_q, var_q = self.session.compute_q_shift_profile(
                    dho_results=point_fit["dho_results"],
                    q_calib_result=self.q_calib_result,
                    refractive_index=n_idx,
                    n_q_bins=self.n_q_bins,
                    q_edges=q_edges,
                )

                self.session.append_time_evolution_point(
                    group_path=group_path,
                    t_idx=t_idx,
                    elapsed_s=elapsed_s,
                    q_centers=q_centers,
                    mean_q=mean_q,
                    var_q=var_q,
                    point_fit=point_fit,
                    raw_frame=frame if self.save_raw_data else None,
                )

                times_list.append(elapsed_s)
                mean_cols.append(mean_q)
                var_cols.append(var_q)
                n_done = t_idx + 1

                time_arr = np.asarray(times_list, dtype=np.float64)
                mean_2d = np.column_stack(mean_cols)
                var_2d = np.column_stack(var_cols)

                self.sig_time_evolution_updated.emit(time_arr, q_centers, mean_2d, var_2d, point_fit)
                self.sig_progress.emit(n_done, self.n_points or 0, elapsed_s)

                t_idx += 1
                if self.n_points is not None and t_idx >= self.n_points:
                    break

                # Wait remaining interval time
                work_dur = time.monotonic() - t_point_start
                remaining = self.interval_s - work_dur
                if remaining > 0.0:
                    if not self._sleep_interruptible(remaining):
                        break

            completed_ok = (not self._stop_requested) or (self.n_points is None and n_done > 0)
            self.sig_log.emit(
                f"Time Evolution finished: {n_done} point(s) saved to '{group_path}'."
            )
            self.sig_finished_series.emit(completed_ok, n_done, group_path)

        except Exception as exc:
            self.sig_error.emit("Time Evolution Error", str(exc))
            self.sig_finished_series.emit(False, n_done, group_path)


class TimeEvolutionWidget(QtWidgets.QWidget):
    """
    Time Evolution Tab:
      - Section 1: HDF5 File (synchronized with Mapping tab)
      - Section 2: Calibration Spectra (Laser & Rb, synchronized with Mapping tab)
      - Section 3: Preview calibration (Laser & Rb processing, synchronized with Mapping tab)
      - Section 4: Time Evolution (Series Name, Reference Sample calibration, periodic parameters & execution)
    """

    sig_log = QtCore.Signal(str)
    sig_display_frame = QtCore.Signal(object, object, float, int, int)
    sig_show_laser_diagnostics = QtCore.Signal(object, str)
    sig_show_rb_diagnostics = QtCore.Signal(object, str)
    sig_show_sample_diagnostics = QtCore.Signal(object, str)
    sig_update_time_evolution = QtCore.Signal(object, object, object, object, object)
    sig_camera_updated = QtCore.Signal()
    sig_session_updated = QtCore.Signal()

    def __init__(
        self,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        session: AcquisitionSession,
        stop_live_callback: Callable[[], None],
        parent=None,
    ):
        super().__init__(parent)
        self.camera = camera
        self.stage = stage
        self.session = session
        self._stop_live_cb = stop_live_callback

        self._cal_worker: Optional[CalibrationCaptureWorker] = None
        self._laser_calib_worker: Optional[LaserCalibrationWorker] = None
        self._rb_calib_worker: Optional[RbCalibrationWorker] = None
        self._sample_fit_worker: Optional[SampleTreatmentWorker] = None
        self._te_worker: Optional[TimeEvolutionWorker] = None
        self._te_paused: bool = False
        self._last_fitted_ref_calib: Optional[Dict[str, Any]] = None
        self._syncing_from_session: bool = False

        self._build_ui()
        self._reload_sample_presets()
        self.refresh_from_session()

    # ==========================================================================
    # UI Construction
    # ==========================================================================

    @staticmethod
    def _make_divider(title: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 4, 0, 2)
        lbl = QtWidgets.QLabel(f"<b>{title}</b>")
        lbl.setStyleSheet("color: #90caf9; font-size: 11px;")
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        row.addWidget(lbl)
        row.addWidget(line, stretch=1)
        return w

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 6, 2)
        layout.setSpacing(8)

        self.sec_h5 = self._build_h5_card()
        self.sec_calib = self._build_calibration_card()
        self.sec_preview_calib = self._build_preview_calibration_card()
        self.sec_te = self._build_time_evolution_card()

        layout.addWidget(self.sec_h5)
        layout.addWidget(self.sec_calib)
        layout.addWidget(self.sec_preview_calib)
        layout.addWidget(self.sec_te)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_h5_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("1. HDF5 File")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        self.btn_open_h5 = QtWidgets.QPushButton("Choose a HDF5 file")
        self.btn_open_h5.setStyleSheet("font-weight: bold; text-align: left; padding: 5px 10px;")
        self.btn_open_h5.clicked.connect(self._on_browse_h5)
        layout.addWidget(self.btn_open_h5)

        self.lbl_h5_status = QtWidgets.QLabel("No HDF5 file opened yet. Click 'Choose a HDF5 file' to begin.")
        self.lbl_h5_status.setStyleSheet("font-size: 11px; color: #b0bec5;")
        layout.addWidget(self.lbl_h5_status)
        return box

    def _build_calibration_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("2. Calibration")
        grid = QtWidgets.QGridLayout(box)
        grid.setSpacing(6)

        grid.addWidget(QtWidgets.QLabel("<b>Spectrum Role</b>"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("<b>Exposure</b>"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("<b>Avg</b>"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("<b>Acquire</b>"), 0, 3)
        grid.addWidget(QtWidgets.QLabel("<b>Preview</b>"), 0, 4)
        grid.addWidget(QtWidgets.QLabel("<b>Status</b>"), 0, 5)

        grid.addWidget(QtWidgets.QLabel("1. Laser Reference:"), 1, 0)
        self.spin_laser_exp = QtWidgets.QDoubleSpinBox()
        self.spin_laser_exp.setDecimals(2)
        self.spin_laser_exp.setRange(0.02, 30000.0)
        self.spin_laser_exp.setValue(20.0)
        self.spin_laser_exp.setSuffix(" ms")
        grid.addWidget(self.spin_laser_exp, 1, 1)

        self.spin_laser_avg = QtWidgets.QSpinBox()
        self.spin_laser_avg.setRange(1, 500)
        self.spin_laser_avg.setValue(1)
        grid.addWidget(self.spin_laser_avg, 1, 2)

        self.btn_acq_laser = QtWidgets.QPushButton("📷 Acquire Laser")
        self.btn_acq_laser.clicked.connect(
            lambda: self._on_acquire_calibration(
                AcquisitionSession.ROLE_LASER, self.spin_laser_exp.value(), self.spin_laser_avg.value()
            )
        )
        grid.addWidget(self.btn_acq_laser, 1, 3)

        self.btn_view_laser = QtWidgets.QPushButton("👁 View")
        self.btn_view_laser.clicked.connect(lambda: self._on_view_calibration(AcquisitionSession.ROLE_LASER))
        grid.addWidget(self.btn_view_laser, 1, 4)

        self.lbl_status_laser = QtWidgets.QLabel("MISSING")
        self.lbl_status_laser.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status_laser.setMinimumWidth(95)
        grid.addWidget(self.lbl_status_laser, 1, 5)

        grid.addWidget(QtWidgets.QLabel("2. Rb Absorption:"), 2, 0)
        self.spin_rb_exp = QtWidgets.QDoubleSpinBox()
        self.spin_rb_exp.setDecimals(2)
        self.spin_rb_exp.setRange(0.02, 30000.0)
        self.spin_rb_exp.setValue(50.0)
        self.spin_rb_exp.setSuffix(" ms")
        grid.addWidget(self.spin_rb_exp, 2, 1)

        self.spin_rb_avg = QtWidgets.QSpinBox()
        self.spin_rb_avg.setRange(1, 500)
        self.spin_rb_avg.setValue(1)
        grid.addWidget(self.spin_rb_avg, 2, 2)

        self.btn_acq_rb = QtWidgets.QPushButton("📷 Acquire Rb")
        self.btn_acq_rb.clicked.connect(
            lambda: self._on_acquire_calibration(
                AcquisitionSession.ROLE_RB, self.spin_rb_exp.value(), self.spin_rb_avg.value()
            )
        )
        grid.addWidget(self.btn_acq_rb, 2, 3)

        self.btn_view_rb = QtWidgets.QPushButton("👁 View")
        self.btn_view_rb.clicked.connect(lambda: self._on_view_calibration(AcquisitionSession.ROLE_RB))
        grid.addWidget(self.btn_view_rb, 2, 4)

        self.lbl_status_rb = QtWidgets.QLabel("MISSING")
        self.lbl_status_rb.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status_rb.setMinimumWidth(95)
        grid.addWidget(self.lbl_status_rb, 2, 5)
        return box

    def _build_preview_calibration_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("3. Preview calibration")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        # Laser calibration row
        lc_row = QtWidgets.QHBoxLayout()
        self.btn_run_laser_calib = QtWidgets.QPushButton("⚙ Calibrate Laser")
        self.btn_run_laser_calib.setStyleSheet("font-weight: bold;")
        self.btn_run_laser_calib.clicked.connect(self._on_run_laser_calibration)
        lc_row.addWidget(self.btn_run_laser_calib)

        self.btn_diag_geom = QtWidgets.QPushButton("📊 Laser Geometry")
        self.btn_diag_geom.clicked.connect(lambda: self._on_show_laser_diag("fringes_geom"))
        lc_row.addWidget(self.btn_diag_geom)

        self.lbl_laser_calib_status = QtWidgets.QLabel("NOT CALIBRATED")
        self.lbl_laser_calib_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_laser_calib_status.setMinimumWidth(120)
        lc_row.addWidget(self.lbl_laser_calib_status)
        lc_row.addStretch()
        layout.addLayout(lc_row)

        # Rb calibration row
        rb_row = QtWidgets.QHBoxLayout()
        self.btn_run_rb_calib = QtWidgets.QPushButton("⚙ Calibrate Rb & Freq Map")
        self.btn_run_rb_calib.setStyleSheet("font-weight: bold;")
        self.btn_run_rb_calib.clicked.connect(self._on_run_rb_calibration)
        rb_row.addWidget(self.btn_run_rb_calib)

        self.btn_diag_rb_fsr = QtWidgets.QPushButton("📊 Rb FSR Fit")
        self.btn_diag_rb_fsr.clicked.connect(lambda: self._on_show_rb_diag("fsr_fit"))
        rb_row.addWidget(self.btn_diag_rb_fsr)

        self.btn_diag_rb_fmap = QtWidgets.QPushButton("📊 Freq Map")
        self.btn_diag_rb_fmap.clicked.connect(lambda: self._on_show_rb_diag("freq_map"))
        rb_row.addWidget(self.btn_diag_rb_fmap)

        self.lbl_rb_calib_status = QtWidgets.QLabel("NOT CALIBRATED")
        self.lbl_rb_calib_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_rb_calib_status.setMinimumWidth(120)
        rb_row.addWidget(self.lbl_rb_calib_status)
        rb_row.addStretch()
        layout.addLayout(rb_row)
        return box

    def _build_time_evolution_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("4. Time Evolution")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        # Series Name
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("<b>Series Name:</b>"))
        self.txt_series_name = QtWidgets.QLineEdit("TimeSeries_01")
        self.txt_series_name.textChanged.connect(lambda *_: self._refresh_ui_only())
        name_row.addWidget(self.txt_series_name, stretch=1)
        layout.addLayout(name_row)

        # Reference Sample Calibration (optional if already calibrated)
        layout.addWidget(self._make_divider("Reference Sample — Elliptical Shift Calibration"))

        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(QtWidgets.QLabel("Ref Sample:"))
        self.spin_ref_exp = QtWidgets.QDoubleSpinBox()
        self.spin_ref_exp.setDecimals(2)
        self.spin_ref_exp.setRange(0.02, 30000.0)
        self.spin_ref_exp.setValue(500.0)
        self.spin_ref_exp.setSuffix(" ms")
        ref_row.addWidget(self.spin_ref_exp)

        self.spin_ref_avg = QtWidgets.QSpinBox()
        self.spin_ref_avg.setRange(1, 500)
        self.spin_ref_avg.setValue(1)
        self.spin_ref_avg.setSuffix(" avg")
        ref_row.addWidget(self.spin_ref_avg)

        self.btn_acq_ref = QtWidgets.QPushButton("📷 Acquire Sample")
        self.btn_acq_ref.clicked.connect(
            lambda: self._on_acquire_calibration(
                self._current_ref_role(), self.spin_ref_exp.value(), self.spin_ref_avg.value()
            )
        )
        ref_row.addWidget(self.btn_acq_ref)

        self.btn_run_sample_fit = QtWidgets.QPushButton("⚙ Fit Sample")
        self.btn_run_sample_fit.setStyleSheet("font-weight: bold;")
        self.btn_run_sample_fit.clicked.connect(self._on_run_sample_fit)
        ref_row.addWidget(self.btn_run_sample_fit)

        self.lbl_sample_fit_status = QtWidgets.QLabel("NOT FITTED")
        self.lbl_sample_fit_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_sample_fit_status.setMinimumWidth(130)
        ref_row.addWidget(self.lbl_sample_fit_status)
        ref_row.addStretch()
        layout.addLayout(ref_row)

        dho_row = QtWidgets.QHBoxLayout()
        dho_row.addWidget(QtWidgets.QLabel("Sample:"))
        self.combo_ref_preset = QtWidgets.QComboBox()
        self.combo_ref_preset.setMinimumWidth(180)
        self.combo_ref_preset.currentIndexChanged.connect(self._on_sample_preset_selected)
        dho_row.addWidget(self.combo_ref_preset)

        self.btn_save_ref_preset = QtWidgets.QPushButton("+ Save Preset")
        self.btn_save_ref_preset.setToolTip("Save current Shift & n parameters to calibration_samples.json")
        self.btn_save_ref_preset.clicked.connect(self._on_save_sample_preset)
        dho_row.addWidget(self.btn_save_ref_preset)

        dho_row.addSpacing(8)
        dho_row.addWidget(QtWidgets.QLabel("Shift ν_B:"))
        self.spin_dho_shift = QtWidgets.QDoubleSpinBox()
        self.spin_dho_shift.setDecimals(2)
        self.spin_dho_shift.setRange(0.5, 25.0)
        self.spin_dho_shift.setValue(5.10)
        self.spin_dho_shift.setSuffix(" GHz")
        dho_row.addWidget(self.spin_dho_shift)

        dho_row.addWidget(QtWidgets.QLabel("n:"))
        self.spin_q_n = QtWidgets.QDoubleSpinBox()
        self.spin_q_n.setDecimals(3)
        self.spin_q_n.setRange(1.000, 3.000)
        self.spin_q_n.setValue(1.330)
        dho_row.addWidget(self.spin_q_n)
        dho_row.addStretch()
        layout.addLayout(dho_row)

        self._q_smin = 4.8
        self._q_smax = 5.3
        self._dho_tol = 2.0
        self._dho_lw = 0.8
        self._dho_lw_min = 0.4
        self._dho_lw_max = 1.3

        # Periodic Acquisition Parameters
        layout.addWidget(self._make_divider("Periodic Acquisition Parameters"))

        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addWidget(QtWidgets.QLabel("Exposure:"))
        self.spin_te_exp = QtWidgets.QDoubleSpinBox()
        self.spin_te_exp.setDecimals(2)
        self.spin_te_exp.setRange(0.02, 30000.0)
        self.spin_te_exp.setValue(100.0)
        self.spin_te_exp.setSuffix(" ms")
        exp_row.addWidget(self.spin_te_exp)
        exp_row.addSpacing(12)
        exp_row.addWidget(QtWidgets.QLabel("Frame average:"))
        self.spin_te_avg = QtWidgets.QSpinBox()
        self.spin_te_avg.setRange(1, 500)
        self.spin_te_avg.setValue(1)
        self.spin_te_avg.setSuffix(" frame(s)")
        exp_row.addWidget(self.spin_te_avg)
        exp_row.addStretch()
        layout.addLayout(exp_row)

        timing_row = QtWidgets.QHBoxLayout()
        timing_row.addWidget(QtWidgets.QLabel("Interval (dt):"))
        self.spin_te_interval = QtWidgets.QDoubleSpinBox()
        self.spin_te_interval.setDecimals(2)
        self.spin_te_interval.setRange(0.1, 86400.0)
        self.spin_te_interval.setValue(5.0)
        self.spin_te_interval.setSuffix(" s")
        timing_row.addWidget(self.spin_te_interval)

        timing_row.addSpacing(12)
        timing_row.addWidget(QtWidgets.QLabel("q bins:"))
        self.spin_te_nq = QtWidgets.QSpinBox()
        self.spin_te_nq.setRange(5, 200)
        self.spin_te_nq.setValue(40)
        timing_row.addWidget(self.spin_te_nq)
        timing_row.addStretch()
        layout.addLayout(timing_row)

        stop_row = QtWidgets.QHBoxLayout()
        stop_row.addWidget(QtWidgets.QLabel("Stop mode:"))
        self.combo_te_stop_mode = QtWidgets.QComboBox()
        self.combo_te_stop_mode.addItem("Fixed number of points", "fixed")
        self.combo_te_stop_mode.addItem("Continuous until Stop", "continuous")
        self.combo_te_stop_mode.currentIndexChanged.connect(self._on_stop_mode_changed)
        stop_row.addWidget(self.combo_te_stop_mode)

        stop_row.addSpacing(10)
        stop_row.addWidget(QtWidgets.QLabel("Points Nt:"))
        self.spin_te_nt = QtWidgets.QSpinBox()
        self.spin_te_nt.setRange(1, 100000)
        self.spin_te_nt.setValue(60)
        stop_row.addWidget(self.spin_te_nt)
        stop_row.addStretch()
        layout.addLayout(stop_row)

        self.chk_fast_drift = QtWidgets.QCheckBox("Fast elastic drift tracking per point")
        self.chk_fast_drift.setChecked(True)
        layout.addWidget(self.chk_fast_drift)

        self.chk_save_raw = QtWidgets.QCheckBox("Save raw frames in HDF5")
        self.chk_save_raw.setChecked(False)
        layout.addWidget(self.chk_save_raw)

        # Execution buttons + Progress
        exec_row = QtWidgets.QHBoxLayout()
        self.btn_start_te = QtWidgets.QPushButton("Start Time Evolution")
        self.btn_start_te.setMinimumHeight(32)
        self.btn_start_te.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 4px 12px;"
        )
        self.btn_start_te.clicked.connect(self._on_start_time_evolution)
        exec_row.addWidget(self.btn_start_te)

        self.btn_pause_te = QtWidgets.QPushButton("Pause")
        self.btn_pause_te.setMinimumHeight(32)
        self.btn_pause_te.setEnabled(False)
        self.btn_pause_te.clicked.connect(self._on_toggle_pause)
        exec_row.addWidget(self.btn_pause_te)

        self.btn_stop_te = QtWidgets.QPushButton("Stop")
        self.btn_stop_te.setMinimumHeight(32)
        self.btn_stop_te.setEnabled(False)
        self.btn_stop_te.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.btn_stop_te.clicked.connect(self._on_stop_time_evolution)
        exec_row.addWidget(self.btn_stop_te)
        layout.addLayout(exec_row)

        self.progress_te = QtWidgets.QProgressBar()
        self.progress_te.setRange(0, 100)
        self.progress_te.setValue(0)
        self.progress_te.setFormat("Idle")
        layout.addWidget(self.progress_te)

        return box

    # ==========================================================================
    # Shared Session Sync
    # ==========================================================================

    def _current_sanitized_label(self) -> str:
        name = self.txt_series_name.text().strip() if hasattr(self, "txt_series_name") else "TimeSeries_01"
        return AcquisitionSession.sanitize_angle_label(name or "TimeSeries_01")

    def _current_ref_role(self) -> str:
        name = self.txt_series_name.text().strip() if hasattr(self, "txt_series_name") else "TimeSeries_01"
        return AcquisitionSession.ref_role_for_angle(name or "TimeSeries_01")

    def _current_ref_image_and_role(self):
        ang = self._current_sanitized_label()
        if ang in self.session.multi_angle_ref_images:
            return self.session.multi_angle_ref_images[ang], self._current_ref_role()
        if self.session.ref_sample_image is not None:
            return self.session.ref_sample_image, AcquisitionSession.ROLE_REF_SAMPLE
        return None, self._current_ref_role()

    def _active_ref_calib(self) -> Optional[Dict[str, Any]]:
        ang = self._current_sanitized_label()
        if ang in self.session.multi_angle_ref_calibs:
            return self.session.multi_angle_ref_calibs[ang]
        if self._last_fitted_ref_calib is not None:
            return self._last_fitted_ref_calib
        if self.session.ref_sample_calib_result is not None:
            return self.session.ref_sample_calib_result
        if self.session.multi_angle_ref_calibs:
            return next(reversed(self.session.multi_angle_ref_calibs.values()))
        return None

    def refresh_from_session(self):
        """Synchronizes all UI widgets and badges from the shared `AcquisitionSession`."""
        self._syncing_from_session = True
        try:
            self._reload_sample_presets()
            self._refresh_ui_only()
        finally:
            self._syncing_from_session = False

    def _refresh_ui_only(self):
        def _style(lbl: QtWidgets.QLabel, ok: bool, ok_txt: str, bad_txt: str, warn_color: str = "#c62828"):
            lbl.setText(ok_txt if ok else bad_txt)
            bg = "#2e7d32" if ok else warn_color
            fg = "white" if ok else ("#cfd8dc" if warn_color == "#455a64" else "white")
            lbl.setStyleSheet(
                f"background-color: {bg}; color: {fg}; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )

        h5_open = self.session.is_open
        if h5_open and self.session.h5_path:
            self.btn_open_h5.setText(self.session.h5_path)
            self.btn_open_h5.setToolTip(self.session.h5_path)
            self.lbl_h5_status.setText(
                f"Opened: {os.path.basename(self.session.h5_path)}  |  "
                f"Laser: {'✓' if self.session.laser_image is not None else '—'}, "
                f"Rb: {'✓' if self.session.rb_image is not None else '—'}"
            )
        else:
            self.btn_open_h5.setText("Choose a HDF5 file")
            self.btn_open_h5.setToolTip("")
            self.lbl_h5_status.setText("No HDF5 file opened yet. Click 'Choose a HDF5 file' to begin.")

        has_laser = self.session.laser_image is not None
        has_rb = self.session.rb_image is not None
        has_lc = self.session.laser_calib_result is not None
        has_rb_cal = self.session.rb_calib_result is not None
        ref_img, _ = self._current_ref_image_and_role()
        sf_res = self._active_ref_calib()
        has_sf = sf_res is not None

        _style(self.lbl_status_laser, has_laser, "✓ READY (H5)", "MISSING")
        _style(self.lbl_status_rb, has_rb, "✓ READY (H5)", "MISSING")
        _style(self.lbl_laser_calib_status, has_lc, "✓ CALIBRATED", "NOT CALIBRATED", "#455a64")
        _style(self.lbl_rb_calib_status, has_rb_cal, "✓ CALIBRATED", "NOT CALIBRATED", "#455a64")

        ang = self._current_sanitized_label()
        sf_ok_txt = "✓ REUSING CALIB" if (has_sf and ang not in self.session.multi_angle_ref_calibs) else "✓ FITTED"
        _style(self.lbl_sample_fit_status, has_sf, sf_ok_txt, "NOT FITTED", "#455a64")

        self.btn_view_laser.setEnabled(has_laser)
        self.btn_view_rb.setEnabled(has_rb)
        self.btn_run_laser_calib.setEnabled(has_laser)
        self.btn_diag_geom.setEnabled(has_lc)
        self.btn_run_rb_calib.setEnabled(has_lc and has_rb)
        self.btn_diag_rb_fsr.setEnabled(has_rb_cal)
        self.btn_diag_rb_fmap.setEnabled(has_rb_cal)
        self.btn_run_sample_fit.setEnabled(has_rb_cal and (ref_img is not None))

        has_spectra = h5_open and has_laser and has_rb
        rb_ready = h5_open and has_rb and has_rb_cal
        self.sec_calib.setEnabled(h5_open)
        self.sec_preview_calib.setVisible(has_spectra)
        self.sec_preview_calib.setEnabled(has_spectra)
        self.sec_te.setEnabled(rb_ready)
        self.btn_start_te.setEnabled(rb_ready and has_sf and not self._is_busy())

    def _emit_session_changed(self):
        self._refresh_ui_only()
        if not self._syncing_from_session:
            self.sig_session_updated.emit()

    # ==========================================================================
    # Handlers: HDF5 & Shared Calibrations
    # ==========================================================================

    def _on_browse_h5(self):
        start_path = self.session.h5_path if (self.session.is_open and self.session.h5_path) else "Measures.h5"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select or Create HDF5_BLS Measures File",
            start_path,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
            options=QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if not path:
            return
        try:
            info = self.session.open_or_create_h5(path)
            self.sig_log.emit(f"Opened HDF5_BLS container: {info['h5_path']}")
            self._emit_session_changed()
        except Exception as exc:
            self.sig_log.emit(f"ERROR opening HDF5: {exc}")
            QtWidgets.QMessageBox.critical(self, "HDF5 Error", str(exc))

    def _on_acquire_calibration(self, role: str, exp_ms: float, n_avg: int):
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the Spectrometer camera first.")
            return
        if not self.session.is_open or self._is_busy():
            return
        self._stop_live_cb()
        self._set_busy(True)
        self._cal_worker = CalibrationCaptureWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            role=role,
            exposure_ms=exp_ms,
            n_avg=n_avg,
            parent=self,
        )
        self._cal_worker.sig_log.connect(self.sig_log.emit)
        self._cal_worker.sig_captured.connect(self._on_calib_captured)
        self._cal_worker.sig_error.connect(self._on_worker_error)
        self._cal_worker.finished.connect(lambda: self._set_busy(False))
        self._cal_worker.start()

    @QtCore.Slot(str, object, object, float, int, int)
    def _on_calib_captured(self, role: str, frame, pos, exp_s: float, n_avg: int, binning: int):
        self.sig_camera_updated.emit()
        self.sig_display_frame.emit(frame, pos, exp_s, n_avg, binning)
        self._emit_session_changed()

    def _on_view_calibration(self, role: str):
        img = self.session.laser_image if role == AcquisitionSession.ROLE_LASER else self.session.rb_image
        meta = self.session.laser_meta if role == AcquisitionSession.ROLE_LASER else self.session.rb_meta
        if img is None:
            return
        self.sig_display_frame.emit(
            img,
            None,
            float(meta.get("exposure_s", 0.05)),
            int(meta.get("frame_averaging", 1)),
            int(meta.get("hardware_binning", 1)),
        )

    def _on_run_laser_calibration(self):
        if not self.session.is_open or self.session.laser_image is None or self._is_busy():
            return
        params = {
            "nb_laser_modes": int(config.NB_LASER_MODES),
            "y_center_laser": int(config.Y_CENTER_LASER),
            "distance_peaks": int(config.DISTANCE_PEAKS_LASER),
            "scale_laser": int(config.SCALE_LASER),
            "poly_deg": 4,
        }
        self._set_busy(True)
        self._laser_calib_worker = LaserCalibrationWorker(session=self.session, params=params, parent=self)
        self._laser_calib_worker.sig_log.connect(self.sig_log.emit)
        self._laser_calib_worker.sig_calibrated.connect(self._on_laser_calibrated)
        self._laser_calib_worker.sig_error.connect(self._on_worker_error)
        self._laser_calib_worker.finished.connect(lambda: self._set_busy(False))
        self._laser_calib_worker.start()

    @QtCore.Slot(object)
    def _on_laser_calibrated(self, res: dict):
        self._emit_session_changed()
        self.sig_show_laser_diagnostics.emit(res, "fringes_geom")

    def _on_show_laser_diag(self, view_mode: str):
        if self.session.laser_calib_result is not None:
            self.sig_show_laser_diagnostics.emit(self.session.laser_calib_result, view_mode)

    def _on_run_rb_calibration(self):
        if not self.session.is_open or self.session.rb_image is None or self._is_busy():
            return
        params = {
            "T_cell_C": 70.0,
            "L_cell_m": 0.070,
            "rb_lock_offset_ghz": 1.323,
            "fit_window_ghz": 10.0,
            "scale_rb": 9,
        }
        self._set_busy(True)
        self._rb_calib_worker = RbCalibrationWorker(session=self.session, params=params, parent=self)
        self._rb_calib_worker.sig_log.connect(self.sig_log.emit)
        self._rb_calib_worker.sig_calibrated.connect(self._on_rb_calibrated)
        self._rb_calib_worker.sig_error.connect(self._on_worker_error)
        self._rb_calib_worker.finished.connect(lambda: self._set_busy(False))
        self._rb_calib_worker.start()

    @QtCore.Slot(object)
    def _on_rb_calibrated(self, res: dict):
        self._emit_session_changed()
        self.sig_show_rb_diagnostics.emit(res, "fsr_fit")

    def _on_show_rb_diag(self, view_mode: str):
        if self.session.rb_calib_result is not None:
            self.sig_show_rb_diagnostics.emit(self.session.rb_calib_result, view_mode)

    def _reload_sample_presets(self, select_name: Optional[str] = None):
        samples = config.load_calibration_samples()
        prev_name = select_name or self.combo_ref_preset.currentData()
        self.combo_ref_preset.blockSignals(True)
        self.combo_ref_preset.clear()
        for name, entry in samples.items():
            sh = float(entry.get("expected_shift_ghz", 5.1))
            n_val = float(entry.get("refractive_index", 1.33))
            self.combo_ref_preset.addItem(f"{name} ({sh:.2f} GHz, n={n_val:.2f})", name)
        if prev_name is not None:
            idx = self.combo_ref_preset.findData(prev_name)
            if idx >= 0:
                self.combo_ref_preset.setCurrentIndex(idx)
        self.combo_ref_preset.blockSignals(False)
        self._on_sample_preset_selected()

    def _on_sample_preset_selected(self):
        name = self.combo_ref_preset.currentData()
        if not name:
            return
        samples = config.load_calibration_samples()
        entry = samples.get(name)
        if not isinstance(entry, dict):
            return
        self.spin_dho_shift.setValue(float(entry.get("expected_shift_ghz", 5.1)))
        self.spin_q_n.setValue(float(entry.get("refractive_index", 1.33)))
        self._dho_tol = float(entry.get("shift_tol_ghz", 2.0))
        self._dho_lw = float(entry.get("expected_lw_ghz", 0.8))
        self._dho_lw_min = float(entry.get("lw_min_ghz", 0.4))
        self._dho_lw_max = float(entry.get("lw_max_ghz", 1.3))
        self._q_smin = float(entry.get("shift_filter_min_ghz", 4.8))
        self._q_smax = float(entry.get("shift_filter_max_ghz", 5.3))

    def _on_save_sample_preset(self):
        default_name = str(self.combo_ref_preset.currentData() or "")
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save Calibration Sample Preset",
            "Preset name (saved to calibration_samples.json):",
            text=default_name,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        shift_ghz = float(self.spin_dho_shift.value())
        n_idx = float(self.spin_q_n.value())
        vs_est = float((config.lambda_0 * (shift_ghz * 1e9)) / (2.0 * max(n_idx, 1e-6)))
        entry = {
            "expected_shift_ghz": shift_ghz,
            "shift_tol_ghz": float(self._dho_tol),
            "expected_lw_ghz": float(self._dho_lw),
            "lw_min_ghz": float(self._dho_lw_min),
            "lw_max_ghz": float(self._dho_lw_max),
            "refractive_index": n_idx,
            "shift_filter_min_ghz": round(max(0.5, shift_ghz - 0.35), 2),
            "shift_filter_max_ghz": round(shift_ghz + 0.35, 2),
            "vs_m_s": round(vs_est, 1),
        }
        try:
            config.save_calibration_sample(name, entry)
            self._reload_sample_presets(select_name=name)
            self.sig_log.emit(f"Saved calibration sample preset '{name}' to calibration_samples.json.")
            self._emit_session_changed()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save Preset Error", str(exc))

    def _on_run_sample_fit(self):
        if not self.session.is_open or self.session.rb_calib_result is None or self._is_busy():
            return
        ref_img, ref_role = self._current_ref_image_and_role()
        if ref_img is None:
            QtWidgets.QMessageBox.warning(self, "Reference Sample Missing", "Acquire a Reference Sample first.")
            return
        params = {
            "bin_val_h": 4,
            "bin_val_w": 2,
            "expected_shift_ghz": float(self.spin_dho_shift.value()),
            "shift_tol_ghz": float(self._dho_tol),
            "expected_lw_ghz": float(self._dho_lw),
            "lw_bounds_ghz": (float(self._dho_lw_min), float(self._dho_lw_max)),
            "run_q_calibration": True,
            "refractive_index": float(self.spin_q_n.value()),
            "shift_filter_bounds": (self._q_smin, self._q_smax),
            "ref_role": ref_role,
        }
        self._set_busy(True)
        self._sample_fit_worker = SampleTreatmentWorker(session=self.session, params=params, parent=self)
        self._sample_fit_worker.sig_log.connect(self.sig_log.emit)
        self._sample_fit_worker.sig_fitted.connect(self._on_sample_fitted)
        self._sample_fit_worker.sig_error.connect(self._on_worker_error)
        self._sample_fit_worker.finished.connect(lambda: self._set_busy(False))
        self._sample_fit_worker.start()

    @QtCore.Slot(object)
    def _on_sample_fitted(self, res: dict):
        self._last_fitted_ref_calib = res
        self._emit_session_changed()
        default_view = "q_dispersion" if res.get("q_results") is not None else "dho_maps"
        self.sig_show_sample_diagnostics.emit(res, default_view)

    # ==========================================================================
    # Handlers: Time Evolution Execution
    # ==========================================================================

    def _on_stop_mode_changed(self):
        is_fixed = (self.combo_te_stop_mode.currentData() == "fixed")
        self.spin_te_nt.setEnabled(is_fixed)

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
                "Please fit a Reference Sample first before starting Time Evolution.",
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
        self._te_worker = TimeEvolutionWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            series_name=self.txt_series_name.text().strip() or "TimeSeries_01",
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
        self._te_worker.sig_time_evolution_updated.connect(self.sig_update_time_evolution.emit)
        self._te_worker.sig_finished_series.connect(self._on_te_finished)
        self._te_worker.sig_error.connect(self._on_worker_error)
        self._te_worker.start()

    def _on_toggle_pause(self):
        if self._te_worker is None or not self._te_worker.isRunning():
            return
        self._te_paused = not self._te_paused
        self._te_worker.set_paused(self._te_paused)
        self.btn_pause_te.setText("Resume" if self._te_paused else "Pause")
        self.sig_log.emit("Time Evolution PAUSED." if self._te_paused else "Time Evolution RESUMED.")

    def _on_stop_time_evolution(self):
        if self._te_worker is not None and self._te_worker.isRunning():
            self._te_worker.request_stop()
            self.sig_log.emit("Stopping Time Evolution...")

    @QtCore.Slot(int, int, float)
    def _on_te_progress(self, done: int, total: int, elapsed_s: float):
        if total > 0:
            self.progress_te.setValue(done)
            self.progress_te.setFormat(f"{done} / {total} pts ({elapsed_s:.1f} s)")
        else:
            self.progress_te.setFormat(f"{done} pts ({elapsed_s:.1f} s)")

    @QtCore.Slot(bool, int, str)
    def _on_te_finished(self, completed_ok: bool, n_done: int, group_path: str):
        self._set_busy(False, te_running=False)
        self.progress_te.setRange(0, 100)
        self.progress_te.setValue(100 if completed_ok and n_done > 0 else 0)
        self.progress_te.setFormat(f"Done ({n_done} pts)" if n_done > 0 else "Stopped")
        self._emit_session_changed()

    def _on_worker_error(self, title: str, message: str):
        self.sig_log.emit(f"ERROR ({title}): {message}")
        QtWidgets.QMessageBox.warning(self, title, message)

    def _is_busy(self) -> bool:
        for w in (
            self._cal_worker,
            self._laser_calib_worker,
            self._rb_calib_worker,
            self._sample_fit_worker,
            self._te_worker,
        ):
            if w is not None and w.isRunning():
                return True
        return False

    def _set_busy(self, busy: bool, te_running: bool = False):
        for w in (
            self.btn_open_h5,
            self.btn_acq_laser,
            self.btn_acq_rb,
            self.btn_acq_ref,
            self.btn_run_laser_calib,
            self.btn_run_rb_calib,
            self.btn_run_sample_fit,
            self.btn_start_te,
        ):
            w.setEnabled(not busy)
        if not busy:
            self._refresh_ui_only()
        self.btn_pause_te.setEnabled(te_running)
        self.btn_stop_te.setEnabled(te_running)
