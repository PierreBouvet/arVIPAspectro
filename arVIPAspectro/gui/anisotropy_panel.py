"""
Single Point Anisotropy Panel (`SinglePointAnisotropyWidget`) for multi-VIPA-tilt
measurements across a set of saved sample points paired with a calibration sample point.

Workflow per VIPA tilt angle:
  1. Select/create HDF5 file (synchronized with Mapping, Time Evolution, Time Evolution Fit)
  2. Save the Calibration Sample stage position and a list of Sample Points `(Name, X, Y)`
  3. Add a VIPA tilt step (after manually tilting the VIPA in the spectrometer)
  4. Acquire & calibrate full sequence for that tilt angle:
       Laser -> Rb & Frequency Map -> Calibration Sample (at Calib Pos)
       -> Acquire & Fit ALL Saved Sample Points (using that tilt's calibration)
  5. Save tilt step under `Brillouin/<Study_name>/<Tilt_label>/` and update the 2x2
     detector-sensor maps (Left: Calib Shift & LW; Right: selected Sample Point Shift & LW).
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
)


class _AnisotropyCalibPointWorker(QtCore.QThread):
    """Moves to the saved Calibration Sample position, snaps a frame, and runs DHO + ellipse fit."""

    sig_log = QtCore.Signal(str)
    sig_stage_updated = QtCore.Signal()
    sig_done = QtCore.Signal(object, object)  # (raw_frame, fit_dict)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        target_pos_um: Optional[Dict[str, float]],
        exposure_ms: float,
        n_avg: int,
        fit_params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.camera = camera
        self.stage = stage
        self.target_pos_um = dict(target_pos_um) if target_pos_um else None
        self.exposure_ms = float(exposure_ms)
        self.n_avg = max(1, int(n_avg))
        self.fit_params = dict(fit_params)

    def run(self):
        try:
            if not self.camera.is_connected:
                raise RuntimeError("Spectrometer camera is not connected.")
            if self.session.rb_calib_result is None:
                raise RuntimeError("Rubidium Frequency Map must be calibrated for this VIPA tilt step first.")

            pos = None
            if self.target_pos_um is not None and self.stage.is_connected:
                tx, ty = self.target_pos_um.get("X"), self.target_pos_um.get("Y")
                if tx is not None and ty is not None:
                    self.sig_log.emit(f"Moving stage to Calibration Sample position ({tx:.2f}, {ty:.2f}) µm...")
                    pos = self.stage.move_to(x_um=float(tx), y_um=float(ty), wait=True)
                    self.sig_stage_updated.emit()
                    time.sleep(0.08)
            elif self.stage.is_connected:
                try:
                    pos = self.stage.get_position()
                except Exception:
                    pos = None

            self.camera.set_exposure(self.exposure_ms / 1000.0)
            self.camera.set_frame_averaging(self.n_avg)
            self.sig_log.emit(
                f"Acquiring Calibration Sample spectrum (exp={self.exposure_ms:.1f} ms × {self.n_avg} avg)..."
            )
            frame = self.camera.snap_frame(n_frames=self.n_avg, stage_pos_um=pos)

            fit_res = self.session.run_sample_treatment_and_q_calibration(
                sample_image=frame,
                bin_val_h=int(self.fit_params.get("bin_val_h", 4)),
                bin_val_w=int(self.fit_params.get("bin_val_w", 2)),
                expected_shift_ghz=float(self.fit_params.get("expected_shift_ghz", 5.1)),
                shift_tol_ghz=float(self.fit_params.get("shift_tol_ghz", 2.0)),
                expected_lw_ghz=float(self.fit_params.get("expected_lw_ghz", 0.8)),
                lw_bounds_ghz=tuple(self.fit_params.get("lw_bounds_ghz", (0.4, 1.3))),
                run_q_calibration=True,
                refractive_index=float(self.fit_params.get("refractive_index", 1.33)),
                shift_filter_bounds=tuple(self.fit_params.get("shift_filter_bounds", (4.8, 5.3))),
                persist_to_ref_group=False,
            )
            self.sig_done.emit(frame, fit_res)
        except Exception as exc:
            self.sig_error.emit("Anisotropy Calibration Sample Error", str(exc))


class _AnisotropySampleBatchWorker(QtCore.QThread):
    """Visits every saved sample point sequentially, snaps & fits DHO using the active tilt calibration."""

    sig_log = QtCore.Signal(str)
    sig_stage_updated = QtCore.Signal()
    sig_point_done = QtCore.Signal(str, object, object, object)  # (pt_name, frame, fit_dict, pos_um)
    sig_batch_finished = QtCore.Signal(int)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        sample_points: List[Dict[str, Any]],
        exposure_ms: float,
        n_avg: int,
        fit_params: Dict[str, Any],
        q_calib_result: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.camera = camera
        self.stage = stage
        self.sample_points = [dict(p) for p in sample_points]
        self.exposure_ms = float(exposure_ms)
        self.n_avg = max(1, int(n_avg))
        self.fit_params = dict(fit_params)
        self.q_calib_result = q_calib_result
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        n_done = 0
        try:
            if not self.camera.is_connected:
                raise RuntimeError("Spectrometer camera is not connected.")
            if self.session.rb_calib_result is None:
                raise RuntimeError("Rubidium Frequency Map must be calibrated first.")

            self.camera.set_exposure(self.exposure_ms / 1000.0)
            self.camera.set_frame_averaging(self.n_avg)
            bin_h = int(self.fit_params.get("bin_val_h", 4))
            bin_w = int(self.fit_params.get("bin_val_w", 2))
            n_orders = self.session.determine_n_orders(bin_val_h=bin_h, bin_val_w=bin_w)

            points_to_run = self.sample_points if self.sample_points else [{"name": "Point_01", "X": None, "Y": None}]

            for idx, pt in enumerate(points_to_run):
                if self._stop_requested:
                    break
                pt_name = str(pt["name"])
                pos = None
                if pt.get("X") is not None and pt.get("Y") is not None and self.stage.is_connected:
                    tx, ty = float(pt["X"]), float(pt["Y"])
                    self.sig_log.emit(
                        f"[{idx + 1}/{len(points_to_run)}] Moving to '{pt_name}' ({tx:.2f}, {ty:.2f}) µm..."
                    )
                    pos = self.stage.move_to(x_um=tx, y_um=ty, wait=True)
                    self.sig_stage_updated.emit()
                    time.sleep(0.08)
                elif self.stage.is_connected:
                    try:
                        pos = self.stage.get_position()
                    except Exception:
                        pos = None

                if self._stop_requested:
                    break

                self.sig_log.emit(
                    f"[{idx + 1}/{len(points_to_run)}] Acquiring '{pt_name}' (exp={self.exposure_ms:.1f} ms × {self.n_avg} avg)..."
                )
                frame = self.camera.snap_frame(n_frames=self.n_avg, stage_pos_um=pos)
                if self._stop_requested:
                    break

                fit_res = self.session.treat_single_map_point(
                    frame=frame,
                    bin_val_h=bin_h,
                    bin_val_w=bin_w,
                    n_orders=n_orders,
                    expected_shift_ghz=float(self.fit_params.get("expected_shift_ghz", 5.1)),
                    shift_tol_ghz=float(self.fit_params.get("shift_tol_ghz", 2.0)),
                    expected_lw_ghz=float(self.fit_params.get("expected_lw_ghz", 0.8)),
                    lw_bounds_ghz=tuple(self.fit_params.get("lw_bounds_ghz", (0.4, 1.3))),
                    refractive_index=float(self.fit_params.get("refractive_index", 1.33)),
                    fast_drift_only=bool(self.fit_params.get("fast_drift_only", True)),
                    q_calib_result=self.q_calib_result,
                )
                n_done += 1
                self.sig_point_done.emit(pt_name, frame, fit_res, pos)

            self.sig_batch_finished.emit(n_done)
        except Exception as exc:
            self.sig_error.emit("Anisotropy Sample Batch Error", str(exc))
            self.sig_batch_finished.emit(n_done)


class SinglePointAnisotropyWidget(QtWidgets.QWidget):
    """
    Single Point Anisotropy Tab:
      - Section 1: HDF5 File (synchronized across tabs)
      - Section 2: Study Name, Calibration Sample Position, and Sample Points List
      - Section 3: VIPA Tilt Steps List
      - Section 4: Active Tilt Step Acquisition (Laser -> Rb -> Calibration Sample -> All Sample Points -> Save & 4-Map View)
    """

    sig_log = QtCore.Signal(str)
    sig_display_frame = QtCore.Signal(object, object, float, int, int)
    sig_show_laser_diagnostics = QtCore.Signal(object, str)
    sig_show_rb_diagnostics = QtCore.Signal(object, str)
    sig_show_anisotropy_maps = QtCore.Signal(object, object, object)  # (tilt_steps_list, detector_shape, point_name)
    sig_stage_updated = QtCore.Signal()
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

        self._calib_pos_um: Optional[Dict[str, float]] = None
        self._sample_points: List[Dict[str, Any]] = []
        self._tilt_steps: List[Dict[str, Any]] = []
        self._active_step_idx: int = -1

        self._cal_worker: Optional[CalibrationCaptureWorker] = None
        self._laser_calib_worker: Optional[LaserCalibrationWorker] = None
        self._rb_calib_worker: Optional[RbCalibrationWorker] = None
        self._calib_pt_worker: Optional[_AnisotropyCalibPointWorker] = None
        self._sample_batch_worker: Optional[_AnisotropySampleBatchWorker] = None
        self._syncing_from_session: bool = False

        self._q_smin = 4.8
        self._q_smax = 5.3
        self._dho_tol = 2.0
        self._dho_lw = 0.8
        self._dho_lw_min = 0.4
        self._dho_lw_max = 1.3

        self._build_ui()
        self._reload_sample_presets()
        self.refresh_from_session()

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
        self.sec_positions = self._build_positions_card()
        self.sec_tilts = self._build_tilts_card()
        self.sec_active = self._build_active_tilt_card()

        layout.addWidget(self.sec_h5)
        layout.addWidget(self.sec_positions)
        layout.addWidget(self.sec_tilts)
        layout.addWidget(self.sec_active)
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

    def _build_positions_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("2. Study & Saved Stage Positions")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("<b>Study Name:</b>"))
        self.txt_study_name = QtWidgets.QLineEdit("Anisotropy_01")
        name_row.addWidget(self.txt_study_name, stretch=1)
        layout.addLayout(name_row)

        # Calibration sample position
        cal_row = QtWidgets.QHBoxLayout()
        cal_row.addWidget(QtWidgets.QLabel("Calibration Sample Pos:"))
        self.lbl_calib_pos = QtWidgets.QLabel("Not set")
        self.lbl_calib_pos.setStyleSheet("font-family: monospace; color: #cfd8dc;")
        cal_row.addWidget(self.lbl_calib_pos, stretch=1)

        self.btn_set_calib_pos = QtWidgets.QPushButton("Set Current")
        self.btn_set_calib_pos.clicked.connect(self._on_set_calib_position)
        cal_row.addWidget(self.btn_set_calib_pos)

        self.btn_goto_calib_pos = QtWidgets.QPushButton("Go To")
        self.btn_goto_calib_pos.clicked.connect(self._on_goto_calib_position)
        cal_row.addWidget(self.btn_goto_calib_pos)
        layout.addLayout(cal_row)

        # Sample points list
        layout.addWidget(self._make_divider("Saved Sample Points (measured at every VIPA tilt)"))

        pt_ctrl = QtWidgets.QHBoxLayout()
        pt_ctrl.addWidget(QtWidgets.QLabel("Point Name:"))
        self.txt_pt_name = QtWidgets.QLineEdit("Point_01")
        self.txt_pt_name.setMaximumWidth(120)
        pt_ctrl.addWidget(self.txt_pt_name)

        self.btn_add_sample_pt = QtWidgets.QPushButton("+ Add Current Pos")
        self.btn_add_sample_pt.setStyleSheet("font-weight: bold;")
        self.btn_add_sample_pt.clicked.connect(self._on_add_sample_point)
        pt_ctrl.addWidget(self.btn_add_sample_pt)

        self.btn_goto_sample_pt = QtWidgets.QPushButton("Go To Selected")
        self.btn_goto_sample_pt.clicked.connect(self._on_goto_selected_sample_point)
        pt_ctrl.addWidget(self.btn_goto_sample_pt)

        self.btn_rm_sample_pt = QtWidgets.QPushButton("Remove")
        self.btn_rm_sample_pt.clicked.connect(self._on_remove_sample_point)
        pt_ctrl.addWidget(self.btn_rm_sample_pt)
        pt_ctrl.addStretch()
        layout.addLayout(pt_ctrl)

        self.tbl_sample_pts = QtWidgets.QTableWidget(0, 3)
        self.tbl_sample_pts.setHorizontalHeaderLabels(["Point Name", "X (µm)", "Y (µm)"])
        self.tbl_sample_pts.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_sample_pts.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_sample_pts.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_sample_pts.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_sample_pts.setMaximumHeight(110)
        layout.addWidget(self.tbl_sample_pts)
        return box

    def _build_tilts_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("3. VIPA Tilt Angles")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        add_row = QtWidgets.QHBoxLayout()
        add_row.addWidget(QtWidgets.QLabel("Label:"))
        self.txt_tilt_label = QtWidgets.QLineEdit("Tilt_01")
        self.txt_tilt_label.setMaximumWidth(120)
        add_row.addWidget(self.txt_tilt_label)

        add_row.addWidget(QtWidgets.QLabel("VIPA Tilt:"))
        self.spin_tilt_deg = QtWidgets.QDoubleSpinBox()
        self.spin_tilt_deg.setRange(-45.0, 45.0)
        self.spin_tilt_deg.setDecimals(2)
        self.spin_tilt_deg.setValue(0.0)
        self.spin_tilt_deg.setSuffix(" °")
        add_row.addWidget(self.spin_tilt_deg)

        self.btn_add_tilt = QtWidgets.QPushButton("+ Add Tilt Step")
        self.btn_add_tilt.setStyleSheet("font-weight: bold;")
        self.btn_add_tilt.clicked.connect(self._on_add_tilt_step)
        add_row.addWidget(self.btn_add_tilt)

        self.btn_remove_tilt = QtWidgets.QPushButton("Remove")
        self.btn_remove_tilt.clicked.connect(self._on_remove_tilt_step)
        add_row.addWidget(self.btn_remove_tilt)
        add_row.addStretch()
        layout.addLayout(add_row)

        self.tbl_tilts = QtWidgets.QTableWidget(0, 7)
        self.tbl_tilts.setHorizontalHeaderLabels(
            ["Tilt Label", "Angle (°)", "Laser", "Rb", "Calib", "Samples", "Saved"]
        )
        self.tbl_tilts.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_tilts.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_tilts.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_tilts.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_tilts.setMaximumHeight(125)
        self.tbl_tilts.itemSelectionChanged.connect(self._on_tilt_selection_changed)
        layout.addWidget(self.tbl_tilts)
        return box

    def _build_active_tilt_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("4. Active Tilt Step — Laser → Rb → Calibration Sample → Sample Points")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        self.lbl_active_tilt = QtWidgets.QLabel("Select or add a VIPA tilt step above to begin.")
        self.lbl_active_tilt.setStyleSheet("font-weight: bold; color: #ffca28;")
        layout.addWidget(self.lbl_active_tilt)

        # A. Laser & Rb
        layout.addWidget(self._make_divider("A. Laser & Rubidium Calibration (for this VIPA tilt)"))

        las_row = QtWidgets.QHBoxLayout()
        las_row.addWidget(QtWidgets.QLabel("Laser:"))
        self.spin_laser_exp = QtWidgets.QDoubleSpinBox()
        self.spin_laser_exp.setDecimals(2)
        self.spin_laser_exp.setRange(0.02, 30000.0)
        self.spin_laser_exp.setValue(20.0)
        self.spin_laser_exp.setSuffix(" ms")
        las_row.addWidget(self.spin_laser_exp)
        self.spin_laser_avg = QtWidgets.QSpinBox()
        self.spin_laser_avg.setRange(1, 500)
        self.spin_laser_avg.setValue(1)
        las_row.addWidget(self.spin_laser_avg)
        self.btn_acq_laser = QtWidgets.QPushButton("📷 Acquire Laser")
        self.btn_acq_laser.clicked.connect(
            lambda: self._on_acquire_cal(AcquisitionSession.ROLE_LASER, self.spin_laser_exp.value(), self.spin_laser_avg.value())
        )
        las_row.addWidget(self.btn_acq_laser)
        self.btn_cal_laser = QtWidgets.QPushButton("⚙ Calibrate Laser")
        self.btn_cal_laser.clicked.connect(self._on_run_laser_cal)
        las_row.addWidget(self.btn_cal_laser)
        self.lbl_step_laser = QtWidgets.QLabel("PENDING")
        self.lbl_step_laser.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_step_laser.setMinimumWidth(90)
        las_row.addWidget(self.lbl_step_laser)
        las_row.addStretch()
        layout.addLayout(las_row)

        rb_row = QtWidgets.QHBoxLayout()
        rb_row.addWidget(QtWidgets.QLabel("Rb:"))
        self.spin_rb_exp = QtWidgets.QDoubleSpinBox()
        self.spin_rb_exp.setDecimals(2)
        self.spin_rb_exp.setRange(0.02, 30000.0)
        self.spin_rb_exp.setValue(50.0)
        self.spin_rb_exp.setSuffix(" ms")
        rb_row.addWidget(self.spin_rb_exp)
        self.spin_rb_avg = QtWidgets.QSpinBox()
        self.spin_rb_avg.setRange(1, 500)
        self.spin_rb_avg.setValue(1)
        rb_row.addWidget(self.spin_rb_avg)
        self.btn_acq_rb = QtWidgets.QPushButton("📷 Acquire Rb")
        self.btn_acq_rb.clicked.connect(
            lambda: self._on_acquire_cal(AcquisitionSession.ROLE_RB, self.spin_rb_exp.value(), self.spin_rb_avg.value())
        )
        rb_row.addWidget(self.btn_acq_rb)
        self.btn_cal_rb = QtWidgets.QPushButton("⚙ Calibrate Rb")
        self.btn_cal_rb.clicked.connect(self._on_run_rb_cal)
        rb_row.addWidget(self.btn_cal_rb)
        self.lbl_step_rb = QtWidgets.QLabel("PENDING")
        self.lbl_step_rb.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_step_rb.setMinimumWidth(90)
        rb_row.addWidget(self.lbl_step_rb)
        rb_row.addStretch()
        layout.addLayout(rb_row)

        # B. Calibration Sample (at Saved Calib Pos)
        layout.addWidget(self._make_divider("B. Calibration Sample (at Saved Calibration Position)"))

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Sample:"))
        self.combo_ref_preset = QtWidgets.QComboBox()
        self.combo_ref_preset.setMinimumWidth(180)
        self.combo_ref_preset.currentIndexChanged.connect(self._on_sample_preset_selected)
        preset_row.addWidget(self.combo_ref_preset)

        self.btn_save_ref_preset = QtWidgets.QPushButton("+ Save Preset")
        self.btn_save_ref_preset.clicked.connect(self._on_save_sample_preset)
        preset_row.addWidget(self.btn_save_ref_preset)

        preset_row.addSpacing(8)
        preset_row.addWidget(QtWidgets.QLabel("Shift:"))
        self.spin_calib_shift = QtWidgets.QDoubleSpinBox()
        self.spin_calib_shift.setDecimals(2)
        self.spin_calib_shift.setRange(0.5, 25.0)
        self.spin_calib_shift.setValue(5.10)
        self.spin_calib_shift.setSuffix(" GHz")
        preset_row.addWidget(self.spin_calib_shift)

        preset_row.addWidget(QtWidgets.QLabel("n:"))
        self.spin_calib_n = QtWidgets.QDoubleSpinBox()
        self.spin_calib_n.setDecimals(3)
        self.spin_calib_n.setRange(1.000, 3.000)
        self.spin_calib_n.setValue(1.330)
        preset_row.addWidget(self.spin_calib_n)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        cal_s_row = QtWidgets.QHBoxLayout()
        cal_s_row.addWidget(QtWidgets.QLabel("Exp:"))
        self.spin_calib_exp = QtWidgets.QDoubleSpinBox()
        self.spin_calib_exp.setDecimals(2)
        self.spin_calib_exp.setRange(0.02, 30000.0)
        self.spin_calib_exp.setValue(500.0)
        self.spin_calib_exp.setSuffix(" ms")
        cal_s_row.addWidget(self.spin_calib_exp)
        cal_s_row.addWidget(QtWidgets.QLabel("Avg:"))
        self.spin_calib_avg = QtWidgets.QSpinBox()
        self.spin_calib_avg.setRange(1, 500)
        self.spin_calib_avg.setValue(1)
        cal_s_row.addWidget(self.spin_calib_avg)

        self.btn_acq_calib_pt = QtWidgets.QPushButton("Go to Calib Pos & Acquire + Fit")
        self.btn_acq_calib_pt.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        self.btn_acq_calib_pt.clicked.connect(self._on_acquire_calib_point)
        cal_s_row.addWidget(self.btn_acq_calib_pt)

        self.lbl_step_calib = QtWidgets.QLabel("PENDING")
        self.lbl_step_calib.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_step_calib.setMinimumWidth(90)
        cal_s_row.addWidget(self.lbl_step_calib)
        cal_s_row.addStretch()
        layout.addLayout(cal_s_row)

        # C. Measured Sample Points (all saved points sequentially)
        layout.addWidget(self._make_divider("C. Measured Sample Points (shared calibration for this VIPA tilt)"))

        samp_row = QtWidgets.QHBoxLayout()
        samp_row.addWidget(QtWidgets.QLabel("Exp:"))
        self.spin_samp_exp = QtWidgets.QDoubleSpinBox()
        self.spin_samp_exp.setDecimals(2)
        self.spin_samp_exp.setRange(0.02, 30000.0)
        self.spin_samp_exp.setValue(500.0)
        self.spin_samp_exp.setSuffix(" ms")
        samp_row.addWidget(self.spin_samp_exp)

        samp_row.addWidget(QtWidgets.QLabel("Avg:"))
        self.spin_samp_avg = QtWidgets.QSpinBox()
        self.spin_samp_avg.setRange(1, 500)
        self.spin_samp_avg.setValue(1)
        samp_row.addWidget(self.spin_samp_avg)

        samp_row.addWidget(QtWidgets.QLabel("Shift:"))
        self.spin_samp_shift = QtWidgets.QDoubleSpinBox()
        self.spin_samp_shift.setDecimals(2)
        self.spin_samp_shift.setRange(0.5, 25.0)
        self.spin_samp_shift.setValue(5.10)
        self.spin_samp_shift.setSuffix(" GHz")
        samp_row.addWidget(self.spin_samp_shift)

        self.btn_acq_all_samples = QtWidgets.QPushButton("Acquire & Fit All Saved Sample Points")
        self.btn_acq_all_samples.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_acq_all_samples.clicked.connect(self._on_acquire_all_sample_points)
        samp_row.addWidget(self.btn_acq_all_samples)

        self.lbl_step_sample = QtWidgets.QLabel("0 / 0")
        self.lbl_step_sample.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_step_sample.setMinimumWidth(90)
        samp_row.addWidget(self.lbl_step_sample)
        samp_row.addStretch()
        layout.addLayout(samp_row)

        self.chk_fast_drift = QtWidgets.QCheckBox("Fast elastic drift tracking")
        self.chk_fast_drift.setChecked(True)
        layout.addWidget(self.chk_fast_drift)

        # D. Save Tilt Step & Combined 4-Map Sensor View with Display Point selector
        layout.addWidget(self._make_divider("D. Save Tilt Step & Combined 4-Map Sensor View"))

        disp_row = QtWidgets.QHBoxLayout()
        disp_row.addWidget(QtWidgets.QLabel("Display Point:"))
        self.combo_display_point = QtWidgets.QComboBox()
        self.combo_display_point.setMinimumWidth(140)
        self.combo_display_point.currentIndexChanged.connect(self._on_display_point_changed)
        disp_row.addWidget(self.combo_display_point)

        self.btn_show_4maps = QtWidgets.QPushButton("📊 Show Combined 4-Map View")
        self.btn_show_4maps.clicked.connect(self._on_show_combined_maps)
        disp_row.addWidget(self.btn_show_4maps)
        disp_row.addStretch()
        layout.addLayout(disp_row)

        self.btn_save_tilt = QtWidgets.QPushButton("💾 Save Tilt Step to HDF5 & Update 4-Map View")
        self.btn_save_tilt.setMinimumHeight(32)
        self.btn_save_tilt.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_save_tilt.clicked.connect(self._on_save_tilt_step)
        layout.addWidget(self.btn_save_tilt)

        return box

    # ==========================================================================
    # Session & Preset Sync
    # ==========================================================================

    def refresh_from_session(self):
        self._syncing_from_session = True
        try:
            self._reload_sample_presets()
            if self.session.is_open and self.session.h5_path:
                self.btn_open_h5.setText(self.session.h5_path)
                self.btn_open_h5.setToolTip(self.session.h5_path)
                self.lbl_h5_status.setText(f"Opened: {os.path.basename(self.session.h5_path)}")
                c_pos, pts = self.session.load_anisotropy_study_positions(self.txt_study_name.text().strip())
                if c_pos is not None and self._calib_pos_um is None:
                    self._calib_pos_um = c_pos
                if pts and not self._sample_points:
                    self._sample_points = pts
                    self._rebuild_sample_points_table()
            else:
                self.btn_open_h5.setText("Choose a HDF5 file")
                self.btn_open_h5.setToolTip("")
                self.lbl_h5_status.setText("No HDF5 file opened yet. Click 'Choose a HDF5 file' to begin.")
            self._refresh_ui_states()
        finally:
            self._syncing_from_session = False

    def _emit_session_changed(self):
        self._refresh_ui_states()
        if not self._syncing_from_session:
            self.sig_session_updated.emit()

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
        entry = config.load_calibration_samples().get(name)
        if not isinstance(entry, dict):
            return
        self.spin_calib_shift.setValue(float(entry.get("expected_shift_ghz", 5.1)))
        self.spin_calib_n.setValue(float(entry.get("refractive_index", 1.33)))
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
        shift_ghz = float(self.spin_calib_shift.value())
        n_idx = float(self.spin_calib_n.value())
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

    # ==========================================================================
    # Handlers: HDF5, Positions, Sample Points List, Tilt List
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

    def _persist_positions_to_h5(self):
        if self.session.is_open:
            try:
                self.session.save_anisotropy_study_positions(
                    self.txt_study_name.text().strip() or "Anisotropy_01",
                    calib_pos_um=self._calib_pos_um,
                    sample_points=self._sample_points,
                )
            except Exception:
                pass

    def _on_set_calib_position(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the Plate stage first.")
            return
        pos = self.stage.get_position()
        if pos.get("X") is None or pos.get("Y") is None:
            return
        self._calib_pos_um = {"X": float(pos["X"]), "Y": float(pos["Y"])}
        self.sig_log.emit(
            f"Saved Calibration Sample position: X={self._calib_pos_um['X']:.2f} µm, Y={self._calib_pos_um['Y']:.2f} µm."
        )
        self._persist_positions_to_h5()
        self._refresh_ui_states()

    def _on_goto_calib_position(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the Plate stage first.")
            return
        if self._calib_pos_um is None:
            QtWidgets.QMessageBox.information(self, "Position Not Set", "Please set the Calibration Sample position first.")
            return
        try:
            self._stop_live_cb()
            self.stage.move_to(x_um=self._calib_pos_um["X"], y_um=self._calib_pos_um["Y"], wait=True)
            self.sig_stage_updated.emit()
            self.sig_log.emit(
                f"Moved stage to Calibration Sample position ({self._calib_pos_um['X']:.2f}, {self._calib_pos_um['Y']:.2f}) µm."
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Stage Move Error", str(exc))

    def _on_add_sample_point(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the Plate stage first.")
            return
        pos = self.stage.get_position()
        if pos.get("X") is None or pos.get("Y") is None:
            return
        pt_name = self.txt_pt_name.text().strip() or f"Point_{len(self._sample_points) + 1:02d}"
        # Replace if same name exists, else append
        new_pt = {"name": pt_name, "X": float(pos["X"]), "Y": float(pos["Y"])}
        replaced = False
        for i, existing in enumerate(self._sample_points):
            if existing["name"] == pt_name:
                self._sample_points[i] = new_pt
                replaced = True
                break
        if not replaced:
            self._sample_points.append(new_pt)
        self.txt_pt_name.setText(f"Point_{len(self._sample_points) + 1:02d}")
        self.sig_log.emit(f"Saved Sample Point '{pt_name}' at ({new_pt['X']:.2f}, {new_pt['Y']:.2f}) µm.")
        self._persist_positions_to_h5()
        self._rebuild_sample_points_table()
        self._rebuild_tilt_table()
        self._refresh_ui_states()

    def _on_goto_selected_sample_point(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the Plate stage first.")
            return
        rows = self.tbl_sample_pts.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._sample_points):
            pt = self._sample_points[idx]
            try:
                self._stop_live_cb()
                self.stage.move_to(x_um=pt["X"], y_um=pt["Y"], wait=True)
                self.sig_stage_updated.emit()
                self.sig_log.emit(f"Moved stage to Sample Point '{pt['name']}' ({pt['X']:.2f}, {pt['Y']:.2f}) µm.")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Stage Move Error", str(exc))

    def _on_remove_sample_point(self):
        rows = self.tbl_sample_pts.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._sample_points):
            self._sample_points.pop(idx)
            self._persist_positions_to_h5()
            self._rebuild_sample_points_table()
            self._rebuild_tilt_table()
            self._refresh_ui_states()

    def _rebuild_sample_points_table(self):
        self.tbl_sample_pts.setRowCount(len(self._sample_points))
        for i, pt in enumerate(self._sample_points):
            for c, txt in enumerate((pt["name"], f"{pt['X']:.2f}", f"{pt['Y']:.2f}")):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.tbl_sample_pts.setItem(i, c, item)
        self._sync_display_point_combo()

    def _sync_display_point_combo(self):
        prev = self.combo_display_point.currentData()
        names = [p["name"] for p in self._sample_points]
        for st in self._tilt_steps:
            for k in st.get("sample_results", {}).keys():
                if k not in names:
                    names.append(k)
        self.combo_display_point.blockSignals(True)
        self.combo_display_point.clear()
        for nm in names:
            self.combo_display_point.addItem(nm, nm)
        if prev is not None:
            idx = self.combo_display_point.findData(prev)
            if idx >= 0:
                self.combo_display_point.setCurrentIndex(idx)
        self.combo_display_point.blockSignals(False)

    def _on_add_tilt_step(self):
        label = self.txt_tilt_label.text().strip() or f"Tilt_{len(self._tilt_steps) + 1:02d}"
        angle = float(self.spin_tilt_deg.value())
        step = {
            "label": label,
            "angle_deg": angle,
            "laser_img": None,
            "laser_res": None,
            "rb_img": None,
            "rb_res": None,
            "calib_img": None,
            "calib_fit": None,
            "sample_results": {},  # pt_name -> {"img": frame, "fit": fit_res, "pos_um": pos}
            "saved": False,
        }
        self._tilt_steps.append(step)
        self._active_step_idx = len(self._tilt_steps) - 1
        self.txt_tilt_label.setText(f"Tilt_{len(self._tilt_steps) + 1:02d}")
        self._rebuild_tilt_table()
        self.tbl_tilts.selectRow(self._active_step_idx)
        self._refresh_ui_states()

    def _on_remove_tilt_step(self):
        if 0 <= self._active_step_idx < len(self._tilt_steps):
            self._tilt_steps.pop(self._active_step_idx)
            self._active_step_idx = min(self._active_step_idx, len(self._tilt_steps) - 1)
            self._rebuild_tilt_table()
            if self._active_step_idx >= 0:
                self.tbl_tilts.selectRow(self._active_step_idx)
            self._refresh_ui_states()

    def _on_tilt_selection_changed(self):
        rows = self.tbl_tilts.selectionModel().selectedRows()
        self._active_step_idx = rows[0].row() if rows else -1
        self._refresh_ui_states()

    def _active_step(self) -> Optional[Dict[str, Any]]:
        if 0 <= self._active_step_idx < len(self._tilt_steps):
            return self._tilt_steps[self._active_step_idx]
        return None

    def _rebuild_tilt_table(self):
        self.tbl_tilts.blockSignals(True)
        self.tbl_tilts.setRowCount(len(self._tilt_steps))
        n_target = max(1, len(self._sample_points))
        for i, st in enumerate(self._tilt_steps):
            n_done = len(st.get("sample_results", {}))
            samp_txt = f"✓ {n_done}/{n_target}" if n_done >= n_target and n_done > 0 else f"{n_done}/{n_target}"
            vals = [
                st["label"],
                f"{st['angle_deg']:.2f}",
                "✓" if st["laser_res"] is not None else ("img" if st["laser_img"] is not None else "—"),
                "✓" if st["rb_res"] is not None else ("img" if st["rb_img"] is not None else "—"),
                "✓" if st["calib_fit"] is not None else "—",
                samp_txt,
                "✓" if st["saved"] else "—",
            ]
            for c, txt in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.tbl_tilts.setItem(i, c, item)
        self.tbl_tilts.blockSignals(False)

    # ==========================================================================
    # Handlers: Active Tilt Step Sequence
    # ==========================================================================

    def _on_acquire_cal(self, role: str, exp_ms: float, n_avg: int):
        step = self._active_step()
        if step is None or not self.session.is_open or self._is_busy():
            return
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the Spectrometer camera first.")
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
        self._cal_worker.sig_captured.connect(self._on_cal_captured)
        self._cal_worker.sig_error.connect(self._on_worker_error)
        self._cal_worker.finished.connect(lambda: self._set_busy(False))
        self._cal_worker.start()

    @QtCore.Slot(str, object, object, float, int, int)
    def _on_cal_captured(self, role: str, frame, pos, exp_s: float, n_avg: int, binning: int):
        step = self._active_step()
        if step is not None:
            if role == AcquisitionSession.ROLE_LASER:
                step["laser_img"] = np.array(frame, copy=True)
                step["laser_res"] = None
                step["rb_res"] = None
                step["saved"] = False
            elif role == AcquisitionSession.ROLE_RB:
                step["rb_img"] = np.array(frame, copy=True)
                step["rb_res"] = None
                step["saved"] = False
        self.sig_camera_updated.emit()
        self.sig_display_frame.emit(frame, pos, exp_s, n_avg, binning)
        self._rebuild_tilt_table()
        self._emit_session_changed()

    def _on_run_laser_cal(self):
        step = self._active_step()
        if step is None or step["laser_img"] is None or self._is_busy():
            return
        self.session.laser_image = step["laser_img"]
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
        step = self._active_step()
        if step is not None:
            step["laser_res"] = res
            step["saved"] = False
        self._rebuild_tilt_table()
        self._emit_session_changed()
        self.sig_show_laser_diagnostics.emit(res, "fringes_geom")

    def _on_run_rb_cal(self):
        step = self._active_step()
        if step is None or step["rb_img"] is None or step["laser_res"] is None or self._is_busy():
            return
        self.session.laser_calib_result = step["laser_res"]
        self.session.rb_image = step["rb_img"]
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
        step = self._active_step()
        if step is not None:
            step["rb_res"] = res
            step["saved"] = False
        self._rebuild_tilt_table()
        self._emit_session_changed()
        self.sig_show_rb_diagnostics.emit(res, "fsr_fit")

    def _on_acquire_calib_point(self):
        step = self._active_step()
        if step is None or step["rb_res"] is None or self._is_busy():
            return
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the Spectrometer camera first.")
            return

        self.session.laser_calib_result = step["laser_res"]
        self.session.rb_calib_result = step["rb_res"]

        fit_params = {
            "bin_val_h": 4,
            "bin_val_w": 2,
            "expected_shift_ghz": float(self.spin_calib_shift.value()),
            "shift_tol_ghz": float(self._dho_tol),
            "expected_lw_ghz": float(self._dho_lw),
            "lw_bounds_ghz": (float(self._dho_lw_min), float(self._dho_lw_max)),
            "refractive_index": float(self.spin_calib_n.value()),
            "shift_filter_bounds": (float(self._q_smin), float(self._q_smax)),
        }

        self._stop_live_cb()
        self._set_busy(True)
        self._calib_pt_worker = _AnisotropyCalibPointWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            target_pos_um=self._calib_pos_um,
            exposure_ms=self.spin_calib_exp.value(),
            n_avg=self.spin_calib_avg.value(),
            fit_params=fit_params,
            parent=self,
        )
        self._calib_pt_worker.sig_log.connect(self.sig_log.emit)
        self._calib_pt_worker.sig_stage_updated.connect(self.sig_stage_updated.emit)
        self._calib_pt_worker.sig_done.connect(self._on_calib_point_done)
        self._calib_pt_worker.sig_error.connect(self._on_worker_error)
        self._calib_pt_worker.finished.connect(lambda: self._set_busy(False))
        self._calib_pt_worker.start()

    @QtCore.Slot(object, object)
    def _on_calib_point_done(self, frame, fit_res: dict):
        step = self._active_step()
        if step is not None:
            step["calib_img"] = np.array(frame, copy=True)
            step["calib_fit"] = fit_res
            step["saved"] = False
        self.sig_camera_updated.emit()
        self._rebuild_tilt_table()
        self._refresh_ui_states()
        self._on_show_combined_maps()

    def _on_acquire_all_sample_points(self):
        step = self._active_step()
        if step is None or step["rb_res"] is None or step["calib_fit"] is None or self._is_busy():
            return
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the Spectrometer camera first.")
            return

        self.session.laser_calib_result = step["laser_res"]
        self.session.rb_calib_result = step["rb_res"]

        fit_params = {
            "bin_val_h": 4,
            "bin_val_w": 2,
            "expected_shift_ghz": float(self.spin_samp_shift.value()),
            "shift_tol_ghz": float(self._dho_tol),
            "expected_lw_ghz": float(self._dho_lw),
            "lw_bounds_ghz": (float(self._dho_lw_min), float(self._dho_lw_max)),
            "refractive_index": float(self.spin_calib_n.value()),
            "fast_drift_only": bool(self.chk_fast_drift.isChecked()),
        }

        self._stop_live_cb()
        self._set_busy(True)
        self._sample_batch_worker = _AnisotropySampleBatchWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            sample_points=self._sample_points,
            exposure_ms=self.spin_samp_exp.value(),
            n_avg=self.spin_samp_avg.value(),
            fit_params=fit_params,
            q_calib_result=step["calib_fit"],
            parent=self,
        )
        self._sample_batch_worker.sig_log.connect(self.sig_log.emit)
        self._sample_batch_worker.sig_stage_updated.connect(self.sig_stage_updated.emit)
        self._sample_batch_worker.sig_point_done.connect(self._on_sample_point_done)
        self._sample_batch_worker.sig_batch_finished.connect(self._on_sample_batch_finished)
        self._sample_batch_worker.sig_error.connect(self._on_worker_error)
        self._sample_batch_worker.start()

    @QtCore.Slot(str, object, object, object)
    def _on_sample_point_done(self, pt_name: str, frame, fit_res: dict, pos):
        step = self._active_step()
        if step is not None:
            step["sample_results"][pt_name] = {
                "img": np.array(frame, copy=True),
                "fit": fit_res,
                "pos_um": pos,
            }
            step["saved"] = False
        self._sync_display_point_combo()
        self._rebuild_tilt_table()
        self._refresh_ui_states()

    @QtCore.Slot(int)
    def _on_sample_batch_finished(self, n_done: int):
        self._set_busy(False)
        self.sig_camera_updated.emit()
        self._rebuild_tilt_table()
        self._refresh_ui_states()
        if n_done > 0:
            self._on_show_combined_maps()

    def _on_save_tilt_step(self):
        step = self._active_step()
        if step is None or not self.session.is_open:
            return
        if not (
            step["laser_res"] is not None
            and step["rb_res"] is not None
            and step["calib_fit"] is not None
            and bool(step["sample_results"])
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Tilt Step",
                "Complete Laser, Rb, Calibration Sample, and at least one Sample Point before saving.",
            )
            return
        try:
            path = self.session.save_anisotropy_tilt_step(
                study_name=self.txt_study_name.text().strip() or "Anisotropy_01",
                tilt_label=step["label"],
                tilt_angle_deg=step["angle_deg"],
                laser_img=step["laser_img"],
                rb_img=step["rb_img"],
                laser_res=step["laser_res"],
                rb_res=step["rb_res"],
                calib_img=step["calib_img"],
                calib_fit=step["calib_fit"],
                sample_results=step["sample_results"],
                calib_pos_um=self._calib_pos_um,
                sample_points=self._sample_points,
            )
            step["saved"] = True
            self.sig_log.emit(
                f"Saved VIPA tilt step '{step['label']}' ({len(step['sample_results'])} sample point(s)) -> {path}"
            )
            self._rebuild_tilt_table()
            self._refresh_ui_states()
            self._on_show_combined_maps()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(exc))

    def _on_display_point_changed(self):
        any_fit = any(
            (s["calib_fit"] is not None or bool(s.get("sample_results")))
            for s in self._tilt_steps
        )
        if any_fit:
            self._on_show_combined_maps()

    def _on_show_combined_maps(self):
        det_shape = (2048, 2048)
        for st in self._tilt_steps:
            if st["calib_img"] is not None:
                det_shape = tuple(st["calib_img"].shape[:2])
                break
            if st.get("sample_results"):
                first_res = next(iter(st["sample_results"].values()))
                if first_res.get("img") is not None:
                    det_shape = tuple(first_res["img"].shape[:2])
                    break
        selected_pt = self.combo_display_point.currentData()
        self.sig_show_anisotropy_maps.emit(self._tilt_steps, det_shape, selected_pt)

    # ==========================================================================
    # UI State Management
    # ==========================================================================

    def _refresh_ui_states(self):
        h5_open = self.session.is_open
        self.sec_positions.setEnabled(h5_open)
        self.sec_tilts.setEnabled(h5_open)

        if self._calib_pos_um is not None:
            self.lbl_calib_pos.setText(f"X={self._calib_pos_um['X']:.2f}, Y={self._calib_pos_um['Y']:.2f} µm")
        else:
            self.lbl_calib_pos.setText("Not set")

        step = self._active_step()
        self.sec_active.setEnabled(h5_open and step is not None)
        if step is None:
            self.lbl_active_tilt.setText("Select or add a VIPA tilt step above to begin.")
            return

        self.lbl_active_tilt.setText(f"Active Tilt Step: {step['label']}  ({step['angle_deg']:.2f} °)")

        def _badge(lbl: QtWidgets.QLabel, ok: bool, ok_txt: str = "✓ DONE", bad_txt: str = "PENDING"):
            lbl.setText(ok_txt if ok else bad_txt)
            bg = "#2e7d32" if ok else "#455a64"
            fg = "white" if ok else "#cfd8dc"
            lbl.setStyleSheet(
                f"background-color: {bg}; color: {fg}; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )

        has_l_img = step["laser_img"] is not None
        has_l_cal = step["laser_res"] is not None
        has_rb_img = step["rb_img"] is not None
        has_rb_cal = step["rb_res"] is not None
        has_calib = step["calib_fit"] is not None
        n_target = max(1, len(self._sample_points))
        n_done = len(step.get("sample_results", {}))
        has_samples = n_done > 0

        _badge(self.lbl_step_laser, has_l_cal, "✓ CALIB", "IMG ONLY" if has_l_img else "PENDING")
        _badge(self.lbl_step_rb, has_rb_cal, "✓ CALIB", "IMG ONLY" if has_rb_img else "PENDING")
        _badge(self.lbl_step_calib, has_calib, "✓ FITTED", "PENDING")
        _badge(self.lbl_step_sample, n_done >= n_target and has_samples, f"✓ {n_done}/{n_target}", f"{n_done}/{n_target}")

        busy = self._is_busy()
        self.btn_cal_laser.setEnabled(has_l_img and not busy)
        self.btn_cal_rb.setEnabled(has_l_cal and has_rb_img and not busy)
        self.btn_acq_calib_pt.setEnabled(has_rb_cal and not busy)
        self.btn_acq_all_samples.setEnabled(has_rb_cal and has_calib and not busy)
        self.btn_save_tilt.setEnabled(has_l_cal and has_rb_cal and has_calib and has_samples and not busy)
        any_fit = any(
            (s["calib_fit"] is not None or bool(s.get("sample_results")))
            for s in self._tilt_steps
        )
        self.btn_show_4maps.setEnabled(any_fit and not busy)

    def _on_worker_error(self, title: str, message: str):
        self.sig_log.emit(f"ERROR ({title}): {message}")
        QtWidgets.QMessageBox.warning(self, title, message)

    def _is_busy(self) -> bool:
        for w in (
            self._cal_worker,
            self._laser_calib_worker,
            self._rb_calib_worker,
            self._calib_pt_worker,
            self._sample_batch_worker,
        ):
            if w is not None and w.isRunning():
                return True
        return False

    def _set_busy(self, busy: bool):
        for w in (
            self.btn_open_h5,
            self.btn_acq_laser,
            self.btn_cal_laser,
            self.btn_acq_rb,
            self.btn_cal_rb,
            self.btn_acq_calib_pt,
            self.btn_acq_all_samples,
            self.btn_save_tilt,
        ):
            w.setEnabled(not busy)
        if not busy:
            self._refresh_ui_states()
