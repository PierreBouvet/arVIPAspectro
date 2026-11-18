"""
Multi-Angle Illumination Studies Tab (`arVIPAspectro/gui/multi_angle_panel.py`):
Manages sequential or arbitrary manual illumination angles sharing the spectrometer's
Laser Reference and Rubidium Absorption calibration, while acquiring and fitting a dedicated
Reference Sample (`Brillouin/Calibration/reference_sample_<angle>`) and Sample measurement
(Single Point or 2D Spatial Map at `Brillouin/<Study>_<angle>`) per illumination angle.
"""

import json
import re
import time
from typing import Optional, Dict, Any, Callable, List
import h5py
import numpy as np
from PySide6 import QtWidgets, QtCore

from ..hardware import MS2000Stage, OrcaFlashCamera, TeledyneBrightfieldCamera
from ..acquisition_pipeline import AcquisitionSession, MapGridConfig
from ..brightfield_align import (
    measure_image_shift_px,
    compute_jog_correlations,
    build_plate_calibration_from_jogs,
    build_manual_plate_calibration,
    pixel_shift_to_stage_delta_um,
)
from .pipeline_panel import (
    CalibrationCaptureWorker,
    BrightfieldCaptureWorker,
    SampleTreatmentWorker,
    MapScanWorker,
    _format_bytes,
    _format_duration,
)


class _PlateJogCaptureWorker(QtCore.QThread):
    """
    Executes one stage-move + brightfield-snap step for the interactive 3-point plate calibration.
    """

    sig_log = QtCore.Signal(str)
    sig_captured = QtCore.Signal(object)  # dict: step_tag, frame, pos, exp_s, target_x, target_y
    sig_stage_updated = QtCore.Signal()
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        stage: MS2000Stage,
        brightfield_cam: TeledyneBrightfieldCamera,
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

    def run(self):
        try:
            if self.target_x is not None and self.target_y is not None:
                pos = self.stage.move_to(x_um=float(self.target_x), y_um=float(self.target_y), wait=True)
                self.sig_stage_updated.emit()
                if self.settling_ms > 0:
                    time.sleep(self.settling_ms / 1000.0)
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


class _PlateCorrWorker(QtCore.QThread):
    """
    Step 2 of interactive Brightfield plate calibration:
    Computes autocorrelation & cross-correlation maps for the 3 jog frames.
    """

    sig_log = QtCore.Signal(str)
    sig_correlated = QtCore.Signal(object)  # corr_res dict
    sig_error = QtCore.Signal(str, str)

    def __init__(self, img0: np.ndarray, img_x: np.ndarray, img_y: np.ndarray, parent=None):
        super().__init__(parent)
        self.img0 = img0
        self.img_x = img_x
        self.img_y = img_y

    def run(self):
        try:
            self.sig_log.emit("Plate Calib Step 2/3 — Computing autocorrelation & 2D cross-correlations...")
            res = compute_jog_correlations(self.img0, self.img_x, self.img_y)
            dc_x, dr_x = res["dp_x"]
            dc_y, dr_y = res["dp_y"]
            self.sig_log.emit(
                f"Plate Calib Correlation Peaks: +X=({dc_x:+.2f}, {dr_x:+.2f}) px [pk={res['peak_x']:.3f}] | "
                f"+Y=({dc_y:+.2f}, {dr_y:+.2f}) px [pk={res['peak_y']:.3f}]."
            )
            self.sig_correlated.emit(res)
        except Exception as exc:
            self.sig_error.emit("Plate Calibration Correlation Error", str(exc))


class _StageMoveWorker(QtCore.QThread):
    """Background worker to move the ASI MS-2000 to a saved Reference or Sample (X, Y, Z) position."""

    sig_log = QtCore.Signal(str)
    sig_moved = QtCore.Signal(object, object)  # (pos_dict, optional_snap_frame_tuple)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        stage: MS2000Stage,
        camera: OrcaFlashCamera,
        target_pos: Dict[str, float],
        label: str,
        snap_preview: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.stage = stage
        self.camera = camera
        self.target_pos = target_pos
        self.label = label
        self.snap_preview = snap_preview

    def run(self):
        try:
            pos = self.stage.move_to(
                x_um=self.target_pos.get("X"),
                y_um=self.target_pos.get("Y"),
                z_um=self.target_pos.get("Z"),
                wait=True,
            )
            self.sig_log.emit(
                f"Moved stage to {self.label}: "
                f"X={pos.get('X', 0.0):.2f}, Y={pos.get('Y', 0.0):.2f}, Z={pos.get('Z', 0.0):.2f} µm."
            )
            snap_info = None
            if self.snap_preview and self.camera.is_connected:
                exp_s = self.camera.get_exposure()
                n_avg = self.camera.get_frame_averaging()
                binning = self.camera.get_binning()
                frame = self.camera.snap_frame(n_frames=1, stage_pos_um=pos)
                snap_info = (frame, pos, exp_s, 1, binning)
            self.sig_moved.emit(pos, snap_info)
        except Exception as exc:
            self.sig_error.emit(f"Stage Move Error ({self.label})", str(exc))


class MultiAngleAcquisitionWidget(QtWidgets.QWidget):
    """
    Tab 3 ('Multi angle studies') UI Widget:
      1. Shared Spectrometer Calibration Readiness (HDF5 container, Laser & Rb calibs)
      2. Optional Saved Reference vs. Sample Stage Positions (X, Y, Z)
      3. Interactive Illumination Angles List ('angle_1', '15deg', ...) & Next-Angle Stepper
      4. Per-Angle Reference Sample Acquisition & q-Vector Calibration
      5. Per-Angle Sample Acquisition (Single Point or 2D Spatial Map)
    """

    sig_log = QtCore.Signal(str)
    sig_display_frame = QtCore.Signal(object, object, float, int, int)
    sig_show_sample_diagnostics = QtCore.Signal(object, str)
    sig_show_brightfield_alignment = QtCore.Signal(object, object, str, str, float, float, float, float)
    sig_show_plate_calib_images = QtCore.Signal(object, object, object, float, float, float, float)
    sig_show_plate_calib_correlation = QtCore.Signal(object, object, object, float, float, float, float, object)
    sig_update_live_map = QtCore.Signal(
        object, object, object, object, object, object, object, int, int, object, object, object
    )
    sig_stage_updated = QtCore.Signal()
    sig_camera_updated = QtCore.Signal()
    sig_lock_geometry = QtCore.Signal(bool)
    sig_session_updated = QtCore.Signal()

    def __init__(
        self,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        session: AcquisitionSession,
        stop_live_callback: Callable[[], None],
        brightfield_cam: Optional[TeledyneBrightfieldCamera] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.camera = camera
        self.brightfield_cam = brightfield_cam
        self.stage = stage
        self.session = session
        self._stop_live_cb = stop_live_callback

        self._angles: List[str] = ["angle_1"]
        self._angle_measures: Dict[str, List[str]] = {}
        self._map_paused: bool = False

        self._move_worker: Optional[_StageMoveWorker] = None
        self._plate_calib_worker: Optional[_PlateJogCaptureWorker] = None
        self._plate_corr_worker: Optional[_PlateCorrWorker] = None
        self._plate_calib_pending: Optional[Dict[str, Any]] = None
        self._cal_worker: Optional[CalibrationCaptureWorker] = None
        self._bf_worker: Optional[BrightfieldCaptureWorker] = None
        self._ref_fit_worker: Optional[SampleTreatmentWorker] = None
        self._map_worker: Optional[MapScanWorker] = None

        self._load_bf_calib_from_settings()
        self._build_ui()
        self.refresh_from_session()

    # ==========================================================================
    # UI Construction
    # ==========================================================================

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        layout.addWidget(self._build_shared_calib_card())
        layout.addWidget(self._build_positions_card())
        layout.addWidget(self._build_bf_plate_calib_card())
        layout.addWidget(self._build_angles_card())
        layout.addWidget(self._build_angle_ref_card())
        layout.addWidget(self._build_angle_bf_card())
        layout.addWidget(self._build_angle_sample_card())
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_shared_calib_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("1. Shared Spectrometer Calibration Status (from 'Single angle study')")
        v = QtWidgets.QVBoxLayout(box)

        self.lbl_h5_banner = QtWidgets.QLabel("No HDF5 file open yet — open or create one in 'Single angle study'.")
        self.lbl_h5_banner.setWordWrap(True)
        self.lbl_h5_banner.setStyleSheet("color: #ffb74d; font-weight: bold;")
        v.addWidget(self.lbl_h5_banner)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Laser Calib:"))
        self.lbl_laser_badge = QtWidgets.QLabel("MISSING")
        row.addWidget(self.lbl_laser_badge)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Rb Frequency Calib:"))
        self.lbl_rb_badge = QtWidgets.QLabel("MISSING")
        row.addWidget(self.lbl_rb_badge)
        row.addStretch()
        self.btn_refresh_session = QtWidgets.QPushButton("🔄 Refresh Status")
        self.btn_refresh_session.clicked.connect(self.refresh_from_session)
        row.addWidget(self.btn_refresh_session)
        v.addLayout(row)
        return box

    def _build_positions_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("2. Saved Reference & Sample Stage Positions (Optional Bookmarks)")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)

        def _make_coord_spin(lo=-50000.0, hi=50000.0):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(2)
            s.setSingleStep(5.0)
            s.setSuffix(" µm")
            return s

        grid.addWidget(QtWidgets.QLabel("Reference Pos:"), 0, 0)
        self.spin_ref_x = _make_coord_spin()
        self.spin_ref_y = _make_coord_spin()
        self.spin_ref_z = _make_coord_spin()
        grid.addWidget(self.spin_ref_x, 0, 1)
        grid.addWidget(self.spin_ref_y, 0, 2)
        grid.addWidget(self.spin_ref_z, 0, 3)
        self.btn_set_ref_pos = QtWidgets.QPushButton("📍 Set from Stage")
        self.btn_set_ref_pos.clicked.connect(lambda: self._set_pos_from_stage("ref"))
        self.btn_go_ref_pos = QtWidgets.QPushButton("➡ Go to Ref")
        self.btn_go_ref_pos.clicked.connect(lambda: self._go_to_saved_pos("ref"))
        grid.addWidget(self.btn_set_ref_pos, 0, 4)
        grid.addWidget(self.btn_go_ref_pos, 0, 5)

        grid.addWidget(QtWidgets.QLabel("Sample Pos:"), 1, 0)
        self.spin_smp_x = _make_coord_spin()
        self.spin_smp_y = _make_coord_spin()
        self.spin_smp_z = _make_coord_spin()
        grid.addWidget(self.spin_smp_x, 1, 1)
        grid.addWidget(self.spin_smp_y, 1, 2)
        grid.addWidget(self.spin_smp_z, 1, 3)
        self.btn_set_smp_pos = QtWidgets.QPushButton("📍 Set from Stage")
        self.btn_set_smp_pos.clicked.connect(lambda: self._set_pos_from_stage("sample"))
        self.btn_go_smp_pos = QtWidgets.QPushButton("➡ Go to Sample")
        self.btn_go_smp_pos.clicked.connect(lambda: self._go_to_saved_pos("sample"))
        grid.addWidget(self.btn_set_smp_pos, 1, 4)
        grid.addWidget(self.btn_go_smp_pos, 1, 5)

        return box

    def _build_angles_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("3. Illumination Angles List (Manual Optics Adjustment)")
        v = QtWidgets.QVBoxLayout(box)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Angle Label:"))
        self.txt_new_angle = QtWidgets.QLineEdit("angle_2")
        self.txt_new_angle.setPlaceholderText("e.g. angle_2, 15deg, 45deg")
        ctrl.addWidget(self.txt_new_angle, stretch=1)

        self.btn_add_angle = QtWidgets.QPushButton("➕ Add Angle")
        self.btn_add_angle.clicked.connect(self._on_add_angle)
        ctrl.addWidget(self.btn_add_angle)

        self.btn_remove_angle = QtWidgets.QPushButton("🗑 Remove")
        self.btn_remove_angle.clicked.connect(self._on_remove_angle)
        ctrl.addWidget(self.btn_remove_angle)

        self.btn_next_angle = QtWidgets.QPushButton("⏭ Next Angle ▶")
        self.btn_next_angle.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;")
        self.btn_next_angle.clicked.connect(self._on_next_angle)
        ctrl.addWidget(self.btn_next_angle)
        v.addLayout(ctrl)

        self.table_angles = QtWidgets.QTableWidget(0, 5)
        self.table_angles.setHorizontalHeaderLabels(
            ["Angle Label", "Ref Image", "Ref Fit (S0 / Mθ / Vs)", "Brightfield", "Saved Measures"]
        )
        self.table_angles.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_angles.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table_angles.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_angles.verticalHeader().setVisible(False)
        hdr = self.table_angles.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.table_angles.setMinimumHeight(115)
        self.table_angles.setMaximumHeight(180)
        self.table_angles.currentCellChanged.connect(lambda *_: self._on_angle_selection_changed())
        v.addWidget(self.table_angles)

        self.lbl_active_angle_banner = QtWidgets.QLabel("Active Angle: angle_1")
        self.lbl_active_angle_banner.setStyleSheet(
            "background-color: #263238; color: #80cbc4; font-weight: bold; padding: 5px 8px; border-radius: 4px;"
        )
        v.addWidget(self.lbl_active_angle_banner)
        return box

    def _build_angle_ref_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("4. Active Angle — Reference Sample Acquisition & q-Vector Calibration")
        v = QtWidgets.QVBoxLayout(box)

        # Row 1: Acquire + View
        acq = QtWidgets.QHBoxLayout()
        acq.addWidget(QtWidgets.QLabel("Exp (ms):"))
        self.spin_ref_exp = QtWidgets.QDoubleSpinBox()
        self.spin_ref_exp.setRange(0.1, 10000.0)
        self.spin_ref_exp.setValue(100.0)
        self.spin_ref_exp.setDecimals(2)
        acq.addWidget(self.spin_ref_exp)

        acq.addWidget(QtWidgets.QLabel("Avg:"))
        self.spin_ref_avg = QtWidgets.QSpinBox()
        self.spin_ref_avg.setRange(1, 200)
        self.spin_ref_avg.setValue(3)
        acq.addWidget(self.spin_ref_avg)

        self.btn_acq_ref = QtWidgets.QPushButton("📷 Acquire Ref")
        self.btn_acq_ref.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        self.btn_acq_ref.clicked.connect(self._on_acquire_angle_ref)
        acq.addWidget(self.btn_acq_ref)

        self.btn_view_ref = QtWidgets.QPushButton("👁 View")
        self.btn_view_ref.clicked.connect(self._on_view_angle_ref)
        acq.addWidget(self.btn_view_ref)

        self.chk_auto_fit_ref = QtWidgets.QCheckBox("Auto-fit")
        self.chk_auto_fit_ref.setChecked(True)
        acq.addWidget(self.chk_auto_fit_ref)
        v.addLayout(acq)

        # Row 2: DHO & q-vector parameters
        pgrid = QtWidgets.QGridLayout()
        pgrid.setHorizontalSpacing(6)
        pgrid.setVerticalSpacing(4)

        pgrid.addWidget(QtWidgets.QLabel("Bin H×W:"), 0, 0)
        self.spin_bin_h = QtWidgets.QSpinBox()
        self.spin_bin_h.setRange(1, 32)
        self.spin_bin_h.setValue(4)
        self.spin_bin_w = QtWidgets.QSpinBox()
        self.spin_bin_w.setRange(1, 16)
        self.spin_bin_w.setValue(2)
        b_row = QtWidgets.QHBoxLayout()
        b_row.addWidget(self.spin_bin_h)
        b_row.addWidget(self.spin_bin_w)
        pgrid.addLayout(b_row, 0, 1)

        pgrid.addWidget(QtWidgets.QLabel("Shift Init / Tol (GHz):"), 0, 2)
        self.spin_dho_shift = QtWidgets.QDoubleSpinBox()
        self.spin_dho_shift.setRange(0.5, 30.0)
        self.spin_dho_shift.setDecimals(2)
        self.spin_dho_shift.setValue(5.10)
        self.spin_dho_tol = QtWidgets.QDoubleSpinBox()
        self.spin_dho_tol.setRange(0.2, 10.0)
        self.spin_dho_tol.setDecimals(2)
        self.spin_dho_tol.setValue(2.00)
        s_row = QtWidgets.QHBoxLayout()
        s_row.addWidget(self.spin_dho_shift)
        s_row.addWidget(self.spin_dho_tol)
        pgrid.addLayout(s_row, 0, 3)

        pgrid.addWidget(QtWidgets.QLabel("LW Init / Bounds (GHz):"), 1, 0)
        self.spin_dho_lw = QtWidgets.QDoubleSpinBox()
        self.spin_dho_lw.setRange(0.05, 5.0)
        self.spin_dho_lw.setDecimals(2)
        self.spin_dho_lw.setValue(0.80)
        self.spin_dho_lw_min = QtWidgets.QDoubleSpinBox()
        self.spin_dho_lw_min.setRange(0.01, 5.0)
        self.spin_dho_lw_min.setDecimals(2)
        self.spin_dho_lw_min.setValue(0.40)
        self.spin_dho_lw_max = QtWidgets.QDoubleSpinBox()
        self.spin_dho_lw_max.setRange(0.05, 10.0)
        self.spin_dho_lw_max.setDecimals(2)
        self.spin_dho_lw_max.setValue(1.30)
        lw_row = QtWidgets.QHBoxLayout()
        lw_row.addWidget(self.spin_dho_lw)
        lw_row.addWidget(self.spin_dho_lw_min)
        lw_row.addWidget(self.spin_dho_lw_max)
        pgrid.addLayout(lw_row, 1, 1)

        pgrid.addWidget(QtWidgets.QLabel("Index n / Filter (GHz):"), 1, 2)
        self.spin_q_n = QtWidgets.QDoubleSpinBox()
        self.spin_q_n.setRange(1.0, 2.5)
        self.spin_q_n.setDecimals(3)
        self.spin_q_n.setValue(1.333)
        self.spin_q_smin = QtWidgets.QDoubleSpinBox()
        self.spin_q_smin.setRange(0.5, 30.0)
        self.spin_q_smin.setDecimals(2)
        self.spin_q_smin.setValue(4.80)
        self.spin_q_smax = QtWidgets.QDoubleSpinBox()
        self.spin_q_smax.setRange(0.5, 30.0)
        self.spin_q_smax.setDecimals(2)
        self.spin_q_smax.setValue(5.30)
        q_row = QtWidgets.QHBoxLayout()
        q_row.addWidget(self.spin_q_n)
        q_row.addWidget(self.spin_q_smin)
        q_row.addWidget(self.spin_q_smax)
        pgrid.addLayout(q_row, 1, 3)

        v.addLayout(pgrid)

        self.chk_run_q_calib = QtWidgets.QCheckBox("Run q-Vector Concentric Ellipse Calibration (recommended for each angle)")
        self.chk_run_q_calib.setChecked(True)
        v.addWidget(self.chk_run_q_calib)

        # Row 3: Fit button + diagnostics
        btns = QtWidgets.QHBoxLayout()
        self.btn_fit_ref = QtWidgets.QPushButton("📐 Fit Ref & q-Vector")
        self.btn_fit_ref.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        self.btn_fit_ref.clicked.connect(self._on_fit_angle_ref)
        btns.addWidget(self.btn_fit_ref, stretch=1)

        self.btn_diag_freq = QtWidgets.QPushButton("🔬 In-Situ Freq")
        self.btn_diag_freq.clicked.connect(lambda: self._on_show_ref_diag("insitu_freq"))
        self.btn_diag_dho = QtWidgets.QPushButton("🗺 DHO Maps")
        self.btn_diag_dho.clicked.connect(lambda: self._on_show_ref_diag("dho_maps"))
        self.btn_diag_q = QtWidgets.QPushButton("📐 q-Dispersion")
        self.btn_diag_q.clicked.connect(lambda: self._on_show_ref_diag("q_dispersion"))
        btns.addWidget(self.btn_diag_freq)
        btns.addWidget(self.btn_diag_dho)
        btns.addWidget(self.btn_diag_q)
        v.addLayout(btns)

        self.lbl_angle_ref_summary = QtWidgets.QLabel("No Reference Sample fitted yet for this angle.")
        self.lbl_angle_ref_summary.setWordWrap(True)
        self.lbl_angle_ref_summary.setStyleSheet("font-size: 11px; color: #b0bec5;")
        v.addWidget(self.lbl_angle_ref_summary)
        return box

    def _build_bf_plate_calib_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("2B. Brightfield Plate Spatial Calibration (Pixel ↔ Stage µm/px)")
        v = QtWidgets.QVBoxLayout(box)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Plate Scale:"))
        self.lbl_bf_plate_status = QtWidgets.QLabel("NOT CALIBRATED")
        row1.addWidget(self.lbl_bf_plate_status)

        row1.addSpacing(8)
        row1.addWidget(QtWidgets.QLabel("Jog Step:"))
        self.spin_bf_jog_step = QtWidgets.QDoubleSpinBox()
        self.spin_bf_jog_step.setRange(1.0, 500.0)
        self.spin_bf_jog_step.setValue(20.0)
        self.spin_bf_jog_step.setDecimals(1)
        self.spin_bf_jog_step.setSuffix(" µm")
        row1.addWidget(self.spin_bf_jog_step)

        self.btn_auto_calib_plate = QtWidgets.QPushButton("🎯 Auto-Calibrate Plate (3-Pt Jog)")
        self.btn_auto_calib_plate.setStyleSheet("background-color: #00695c; color: white; font-weight: bold;")
        self.btn_auto_calib_plate.clicked.connect(self._on_auto_calibrate_plate)
        row1.addWidget(self.btn_auto_calib_plate)
        row1.addStretch()
        v.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Manual Fallback (µm/px):"))
        self.spin_bf_manual_scale = QtWidgets.QDoubleSpinBox()
        self.spin_bf_manual_scale.setRange(0.01, 50.0)
        self.spin_bf_manual_scale.setDecimals(4)
        self.spin_bf_manual_scale.setSingleStep(0.01)
        self.spin_bf_manual_scale.setValue(0.3450)
        row2.addWidget(self.spin_bf_manual_scale)

        self.chk_bf_flip_x = QtWidgets.QCheckBox("Flip X")
        self.chk_bf_flip_y = QtWidgets.QCheckBox("Flip Y")
        row2.addWidget(self.chk_bf_flip_x)
        row2.addWidget(self.chk_bf_flip_y)

        self.btn_apply_bf_manual = QtWidgets.QPushButton("Apply Manual")
        self.btn_apply_bf_manual.clicked.connect(self._on_apply_manual_plate_calib)
        row2.addWidget(self.btn_apply_bf_manual)
        row2.addStretch()
        v.addLayout(row2)
        return box

    def _build_angle_bf_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("4B. Active Angle — Brightfield Pre-Map Capture & Fine Alignment")
        v = QtWidgets.QVBoxLayout(box)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Angle Brightfield:"))
        self.lbl_angle_bf_status = QtWidgets.QLabel("MISSING")
        row1.addWidget(self.lbl_angle_bf_status)

        self.btn_acq_angle_bf = QtWidgets.QPushButton("📷 Acquire Brightfield")
        self.btn_acq_angle_bf.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        self.btn_acq_angle_bf.clicked.connect(self._on_acquire_angle_brightfield)
        row1.addWidget(self.btn_acq_angle_bf)

        self.btn_view_angle_bf = QtWidgets.QPushButton("👁 View BF")
        self.btn_view_angle_bf.clicked.connect(self._on_view_angle_brightfield)
        row1.addWidget(self.btn_view_angle_bf)
        row1.addStretch()
        v.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Align to Ref Angle:"))
        self.combo_bf_ref_angle = QtWidgets.QComboBox()
        row2.addWidget(self.combo_bf_ref_angle)

        self.btn_align_to_ref = QtWidgets.QPushButton("🎯 Align Map to Ref Angle")
        self.btn_align_to_ref.setStyleSheet("background-color: #ef6c00; color: white; font-weight: bold;")
        self.btn_align_to_ref.clicked.connect(self._on_align_to_ref_angle)
        row2.addWidget(self.btn_align_to_ref)
        row2.addStretch()
        v.addLayout(row2)

        self.lbl_angle_bf_summary = QtWidgets.QLabel(
            "Acquire a Brightfield image for this angle before starting a map."
        )
        self.lbl_angle_bf_summary.setWordWrap(True)
        self.lbl_angle_bf_summary.setStyleSheet("font-size: 11px; color: #b0bec5;")
        v.addWidget(self.lbl_angle_bf_summary)
        return box

    def _build_angle_sample_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("5. Active Angle — Sample Acquisition (Single Point or 2D Map)")
        v = QtWidgets.QVBoxLayout(box)

        # Target naming & mode selection
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Base Study Name:"))
        self.txt_study_name = QtWidgets.QLineEdit("Study")
        self.txt_study_name.textChanged.connect(self._update_sample_summary)
        top.addWidget(self.txt_study_name, stretch=1)

        self.rb_mode_single = QtWidgets.QRadioButton("Single Point (current stage)")
        self.rb_mode_map = QtWidgets.QRadioButton("2D Spatial Map")
        self.rb_mode_single.setChecked(True)
        self.rb_mode_single.toggled.connect(self._on_sample_mode_toggled)
        top.addWidget(self.rb_mode_single)
        top.addWidget(self.rb_mode_map)
        v.addLayout(top)

        # 2D map geometry sub-widget (hidden when Single Point is selected)
        self.widget_map_grid = QtWidgets.QWidget()
        mg = QtWidgets.QGridLayout(self.widget_map_grid)
        mg.setContentsMargins(0, 4, 0, 4)
        mg.setHorizontalSpacing(6)
        mg.setVerticalSpacing(4)

        mg.addWidget(QtWidgets.QLabel("Center Xc (µm):"), 0, 0)
        self.spin_xc = QtWidgets.QDoubleSpinBox()
        self.spin_xc.setRange(-50000.0, 50000.0)
        self.spin_xc.setDecimals(2)
        self.spin_xc.valueChanged.connect(self._update_sample_summary)
        mg.addWidget(self.spin_xc, 0, 1)

        mg.addWidget(QtWidgets.QLabel("Center Yc (µm):"), 0, 2)
        self.spin_yc = QtWidgets.QDoubleSpinBox()
        self.spin_yc.setRange(-50000.0, 50000.0)
        self.spin_yc.setDecimals(2)
        self.spin_yc.valueChanged.connect(self._update_sample_summary)
        mg.addWidget(self.spin_yc, 0, 3)

        self.btn_set_center = QtWidgets.QPushButton("📍 Center from Stage")
        self.btn_set_center.clicked.connect(self._on_set_center_from_stage)
        mg.addWidget(self.btn_set_center, 0, 4, 1, 2)

        mg.addWidget(QtWidgets.QLabel("Nx:"), 1, 0)
        self.spin_nx = QtWidgets.QSpinBox()
        self.spin_nx.setRange(1, 500)
        self.spin_nx.setValue(5)
        self.spin_nx.valueChanged.connect(self._update_sample_summary)
        mg.addWidget(self.spin_nx, 1, 1)

        mg.addWidget(QtWidgets.QLabel("Ny:"), 1, 2)
        self.spin_ny = QtWidgets.QSpinBox()
        self.spin_ny.setRange(1, 500)
        self.spin_ny.setValue(5)
        self.spin_ny.valueChanged.connect(self._update_sample_summary)
        mg.addWidget(self.spin_ny, 1, 3)

        mg.addWidget(QtWidgets.QLabel("Pattern:"), 1, 4)
        self.combo_pattern = QtWidgets.QComboBox()
        self.combo_pattern.addItems(["snake", "raster"])
        mg.addWidget(self.combo_pattern, 1, 5)

        mg.addWidget(QtWidgets.QLabel("ΔX (µm):"), 2, 0)
        self.spin_dx = QtWidgets.QDoubleSpinBox()
        self.spin_dx.setRange(0.05, 5000.0)
        self.spin_dx.setDecimals(2)
        self.spin_dx.setValue(5.0)
        self.spin_dx.valueChanged.connect(self._update_sample_summary)
        mg.addWidget(self.spin_dx, 2, 1)

        mg.addWidget(QtWidgets.QLabel("ΔY (µm):"), 2, 2)
        self.spin_dy = QtWidgets.QDoubleSpinBox()
        self.spin_dy.setRange(0.05, 5000.0)
        self.spin_dy.setDecimals(2)
        self.spin_dy.setValue(5.0)
        self.spin_dy.valueChanged.connect(self._update_sample_summary)
        mg.addWidget(self.spin_dy, 2, 3)

        self.widget_map_grid.setVisible(False)
        v.addWidget(self.widget_map_grid)

        # Exposure, avg, options
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("Exp (ms):"))
        self.spin_smp_exp = QtWidgets.QDoubleSpinBox()
        self.spin_smp_exp.setRange(0.1, 10000.0)
        self.spin_smp_exp.setValue(100.0)
        self.spin_smp_exp.setDecimals(2)
        self.spin_smp_exp.valueChanged.connect(self._update_sample_summary)
        opt_row.addWidget(self.spin_smp_exp)

        opt_row.addWidget(QtWidgets.QLabel("Avg:"))
        self.spin_smp_avg = QtWidgets.QSpinBox()
        self.spin_smp_avg.setRange(1, 200)
        self.spin_smp_avg.setValue(1)
        self.spin_smp_avg.valueChanged.connect(self._update_sample_summary)
        opt_row.addWidget(self.spin_smp_avg)

        opt_row.addWidget(QtWidgets.QLabel("Settling (ms):"))
        self.spin_smp_settling = QtWidgets.QDoubleSpinBox()
        self.spin_smp_settling.setRange(0.0, 5000.0)
        self.spin_smp_settling.setValue(50.0)
        self.spin_smp_settling.valueChanged.connect(self._update_sample_summary)
        opt_row.addWidget(self.spin_smp_settling)
        v.addLayout(opt_row)

        chk_row = QtWidgets.QHBoxLayout()
        self.chk_treat_on_the_fly = QtWidgets.QCheckBox("Treat on the fly using active angle's Ref calibration")
        self.chk_treat_on_the_fly.setChecked(True)
        chk_row.addWidget(self.chk_treat_on_the_fly)

        self.chk_save_raw = QtWidgets.QCheckBox("Save 4D Raw_data")
        self.chk_save_raw.setChecked(False)
        self.chk_save_raw.toggled.connect(self._update_sample_summary)
        chk_row.addWidget(self.chk_save_raw)
        chk_row.addStretch()
        v.addLayout(chk_row)

        self.lbl_sample_summary = QtWidgets.QLabel("")
        self.lbl_sample_summary.setWordWrap(True)
        self.lbl_sample_summary.setStyleSheet("font-size: 11px; color: #b0bec5;")
        v.addWidget(self.lbl_sample_summary)

        # Start / Pause / Abort + Progress
        run_row = QtWidgets.QHBoxLayout()
        self.btn_start_sample = QtWidgets.QPushButton("▶ Acquire Single Point")
        self.btn_start_sample.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_start_sample.clicked.connect(self._on_start_sample)
        run_row.addWidget(self.btn_start_sample, stretch=2)

        self.btn_pause_sample = QtWidgets.QPushButton("⏸ Pause")
        self.btn_pause_sample.setEnabled(False)
        self.btn_pause_sample.clicked.connect(self._on_pause_sample)
        run_row.addWidget(self.btn_pause_sample)

        self.btn_abort_sample = QtWidgets.QPushButton("⏹ Abort")
        self.btn_abort_sample.setEnabled(False)
        self.btn_abort_sample.clicked.connect(self._on_abort_sample)
        run_row.addWidget(self.btn_abort_sample)
        v.addLayout(run_row)

        self.progress_sample = QtWidgets.QProgressBar()
        self.progress_sample.setValue(0)
        self.progress_sample.setFormat("Idle")
        v.addWidget(self.progress_sample)

        self._update_sample_summary()
        return box

    # ==========================================================================
    # Session / Angle State Management
    # ==========================================================================

    @staticmethod
    def _style_status_badge(lbl: QtWidgets.QLabel, ok: bool, ok_text: str = "✓ READY", bad_text: str = "MISSING"):
        if ok:
            lbl.setText(ok_text)
            lbl.setStyleSheet(
                "background-color: #2e7d32; color: white; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
        else:
            lbl.setText(bad_text)
            lbl.setStyleSheet(
                "background-color: #c62828; color: white; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )

    def current_angle_label(self) -> Optional[str]:
        row = self.table_angles.currentRow()
        if 0 <= row < len(self._angles):
            return self._angles[row]
        if self._angles:
            return self._angles[0]
        return None

    def _next_suggested_angle_label(self) -> str:
        max_idx = 0
        for lbl in self._angles:
            m = re.match(r"^angle_(\d+)$", lbl)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        cand = max(len(self._angles) + 1, max_idx + 1)
        while f"angle_{cand}" in self._angles:
            cand += 1
        return f"angle_{cand}"

    def _scan_multi_angle_measures_from_h5(self):
        self._angle_measures.clear()
        if not self.session.is_open:
            return
        with self.session._h5_lock:
            with h5py.File(self.session.h5_path, "r") as f:
                if "Brillouin" not in f:
                    return
                for name, grp in f["Brillouin"].items():
                    if not isinstance(grp, h5py.Group) or name == "Calibration":
                        continue
                    ang = grp.attrs.get("illumination_angle_label")
                    if ang is not None:
                        ang_s = str(ang)
                        self._angle_measures.setdefault(ang_s, []).append(name)
                        if ang_s not in self._angles:
                            self._angles.append(ang_s)

    @QtCore.Slot()
    def refresh_from_session(self):
        if self.session.is_open:
            self.lbl_h5_banner.setText(f"Active HDF5: {self.session.h5_path}")
            self.lbl_h5_banner.setStyleSheet("color: #81c784; font-weight: bold;")
        else:
            self.lbl_h5_banner.setText("No HDF5 file open yet — open or create one in 'Single angle study'.")
            self.lbl_h5_banner.setStyleSheet("color: #ffb74d; font-weight: bold;")

        has_lc = self.session.laser_calib_result is not None
        has_rb = self.session.rb_calib_result is not None
        self._style_status_badge(self.lbl_laser_badge, has_lc)
        self._style_status_badge(self.lbl_rb_badge, has_rb)

        # Include any reference_sample_<angle> or brightfield_<angle> already stored in HDF5
        for ang in self.session.multi_angle_ref_images.keys():
            if ang not in self._angles:
                self._angles.append(ang)
        self._scan_multi_angle_measures_from_h5()

        self._rebuild_angles_table()
        self._refresh_bf_calib_ui()
        self._refresh_active_angle_ui()

    def _rebuild_angles_table(self):
        prev = self.current_angle_label()
        self.table_angles.blockSignals(True)
        try:
            self.table_angles.setRowCount(len(self._angles))
            for row_idx, ang in enumerate(self._angles):
                has_img = ang in self.session.multi_angle_ref_images
                fit_res = self.session.multi_angle_ref_calibs.get(ang)
                if fit_res is not None:
                    q_res = fit_res.get("q_results")
                    if q_res is not None:
                        fit_str = (
                            f"S0={q_res['Shift_0']:.3f} GHz, Mθ={q_res['M_theta']:.0f}, "
                            f"Vs={q_res['V_s']:.1f} m/s"
                        )
                    else:
                        dho = fit_res["dho_results"]
                        fit_str = f"DHO fitted ({dho['n_fringes']} orders)"
                else:
                    fit_str = "Not fitted"

                bf_role = AcquisitionSession.brightfield_role_for_target(ang)
                has_bf = bf_role in self.session.brightfield_images

                meas_list = self._angle_measures.get(ang, [])
                meas_str = ", ".join(meas_list) if meas_list else "—"

                self.table_angles.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(ang))
                self.table_angles.setItem(row_idx, 1, QtWidgets.QTableWidgetItem("✓" if has_img else "—"))
                self.table_angles.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(fit_str))
                self.table_angles.setItem(row_idx, 3, QtWidgets.QTableWidgetItem("✓" if has_bf else "—"))
                self.table_angles.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(meas_str))

            target_row = self._angles.index(prev) if (prev in self._angles) else 0
            if self._angles:
                self.table_angles.selectRow(target_row)
        finally:
            self.table_angles.blockSignals(False)

    def _refresh_active_angle_ui(self):
        ang = self.current_angle_label()
        if ang is None:
            self.lbl_active_angle_banner.setText("No angle selected — add an angle above.")
            self.btn_acq_ref.setEnabled(False)
            self.btn_view_ref.setEnabled(False)
            self.btn_fit_ref.setEnabled(False)
            self.btn_diag_freq.setEnabled(False)
            self.btn_diag_dho.setEnabled(False)
            self.btn_diag_q.setEnabled(False)
            self.btn_start_sample.setEnabled(False)
            self._update_sample_summary()
            return

        role = AcquisitionSession.ref_role_for_angle(ang)
        self.lbl_active_angle_banner.setText(
            f"Active Angle: '{ang}'   |   Ref Group: {AcquisitionSession.CALIBRATION_GROUP}/{role}"
        )

        has_ref_img = ang in self.session.multi_angle_ref_images
        fit_res = self.session.multi_angle_ref_calibs.get(ang)
        has_fit = fit_res is not None
        has_rb = self.session.rb_calib_result is not None

        busy = self._is_busy()
        self.btn_acq_ref.setEnabled(not busy)
        self.btn_view_ref.setEnabled(has_ref_img and not busy)
        self.btn_fit_ref.setEnabled(has_ref_img and has_rb and not busy)
        self.btn_diag_freq.setEnabled(has_fit)
        self.btn_diag_dho.setEnabled(has_fit)
        self.btn_diag_q.setEnabled(has_fit and (fit_res.get("q_results") is not None))
        self.btn_start_sample.setEnabled((self._map_worker is None) or (not self._map_worker.isRunning()))

        # Update Active Angle Brightfield & Alignment controls
        bf_role = AcquisitionSession.brightfield_role_for_target(ang)
        has_bf = bf_role in self.session.brightfield_images
        bf_conn = self.brightfield_cam is not None and self.brightfield_cam.is_connected
        self._style_status_badge(self.lbl_angle_bf_status, has_bf, ok_text="✓ READY", bad_text="MISSING")
        self.btn_acq_angle_bf.setEnabled(bf_conn and not busy)
        self.btn_view_angle_bf.setEnabled(has_bf and not busy)

        prev_ref_ang = self.combo_bf_ref_angle.currentData()
        self.combo_bf_ref_angle.clear()
        for a in self._angles:
            if AcquisitionSession.brightfield_role_for_target(a) in self.session.brightfield_images:
                self.combo_bf_ref_angle.addItem(a, a)
        if prev_ref_ang is not None:
            idx = self.combo_bf_ref_angle.findData(prev_ref_ang)
            if idx >= 0:
                self.combo_bf_ref_angle.setCurrentIndex(idx)

        can_align = (
            has_bf
            and (self.combo_bf_ref_angle.count() > 0)
            and (self.session.brightfield_stage_calib is not None)
            and not busy
        )
        self.btn_align_to_ref.setEnabled(can_align)
        if has_bf:
            bf_meta = self.session.brightfield_meta.get(bf_role, {})
            pos_txt = (
                f"@ ({float(bf_meta['stage_x_um']):.2f}, {float(bf_meta['stage_y_um']):.2f}) µm"
                if "stage_x_um" in bf_meta and "stage_y_um" in bf_meta
                else ""
            )
            self.lbl_angle_bf_summary.setText(
                f"Brightfield stored at Brillouin/Calibration/{bf_role}/image {pos_txt}."
            )
        else:
            self.lbl_angle_bf_summary.setText(
                f"Click 'Acquire Brightfield' at '{ang}' before starting a map or aligning to another angle."
            )
        self._refresh_bf_calib_ui()

        if has_fit:
            dho = fit_res["dho_results"]
            q_res = fit_res.get("q_results")
            txt = (
                f"[{ang}] DHO Fit ({dho['bin_val_h']}×{dho['bin_val_w']}): {dho['n_fringes']} orders  |  "
                f"S/AS RMS = {dho['stokes_antistokes_rms_mhz']:.1f} MHz"
            )
            if q_res is not None:
                opt = q_res["opt_params"]
                txt += (
                    f"\nCenter=({opt['xc']:.1f}, {opt['yc']:.1f}) px, γ={opt['gamma']:.3f}  |  "
                    f"S(0)={q_res['Shift_0']:.4f} GHz  |  M_θ={q_res['M_theta']:.1f} px  |  "
                    f"V_s={q_res['V_s']:.2f} m/s"
                )
            self.lbl_angle_ref_summary.setText(txt)
        elif has_ref_img:
            self.lbl_angle_ref_summary.setText(
                f"Reference image stored for '{ang}'. Click 'Fit Ref & q-Vector' to calibrate scattering geometry."
            )
        else:
            self.lbl_angle_ref_summary.setText(
                f"Adjust illumination optics for '{ang}', then click 'Acquire Ref' to record its Reference Sample."
            )
        self._update_sample_summary()

    def _on_angle_selection_changed(self):
        self._refresh_active_angle_ui()

    def _on_add_angle(self):
        raw = self.txt_new_angle.text().strip()
        if not raw:
            raw = self._next_suggested_angle_label()
        try:
            ang = AcquisitionSession.sanitize_angle_label(raw)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Angle Label", str(exc))
            return

        if ang not in self._angles:
            self._angles.append(ang)
        self._rebuild_angles_table()
        self.table_angles.selectRow(self._angles.index(ang))
        self.txt_new_angle.setText(self._next_suggested_angle_label())
        self._refresh_active_angle_ui()

    def _on_remove_angle(self):
        row = self.table_angles.currentRow()
        if not (0 <= row < len(self._angles)):
            return
        if len(self._angles) <= 1:
            QtWidgets.QMessageBox.information(self, "Keep At Least One Angle", "At least one angle must remain in the list.")
            return
        self._angles.pop(row)
        self._rebuild_angles_table()
        self._refresh_active_angle_ui()

    def _on_next_angle(self):
        cur_row = self.table_angles.currentRow()
        if cur_row < 0:
            cur_row = 0
        next_row = cur_row + 1
        if next_row >= len(self._angles):
            new_lbl = self._next_suggested_angle_label()
            self._angles.append(new_lbl)
            self._rebuild_angles_table()
            next_row = len(self._angles) - 1
            self.txt_new_angle.setText(self._next_suggested_angle_label())
        self.table_angles.selectRow(next_row)
        self._refresh_active_angle_ui()
        ang = self.current_angle_label()
        self.sig_log.emit(
            f"[Multi-Angle] Switched active angle to '{ang}'. Adjust illumination optics before acquiring Reference Sample."
        )

    # ==========================================================================
    # Stage Bookmarks & Helpers
    # ==========================================================================

    def _is_busy(self) -> bool:
        for w in (
            self._move_worker,
            self._plate_calib_worker,
            self._plate_corr_worker,
            self._cal_worker,
            self._bf_worker,
            self._ref_fit_worker,
            self._map_worker,
        ):
            if w is not None and w.isRunning():
                return True
        return False

    def _set_controls_busy(self, busy: bool, map_running: bool = False):
        self.btn_add_angle.setEnabled(not busy)
        self.btn_remove_angle.setEnabled(not busy)
        self.btn_next_angle.setEnabled(not busy)
        self.table_angles.setEnabled(not busy)
        self.btn_go_ref_pos.setEnabled(not busy)
        self.btn_go_smp_pos.setEnabled(not busy)
        self.btn_auto_calib_plate.setEnabled(not busy)
        self.btn_apply_bf_manual.setEnabled(not busy)
        self.btn_acq_ref.setEnabled(not busy)
        self.btn_fit_ref.setEnabled(not busy)
        self.btn_acq_angle_bf.setEnabled(not busy)
        self.btn_align_to_ref.setEnabled(not busy)
        self.btn_start_sample.setEnabled(not busy)
        self.btn_pause_sample.setEnabled(map_running)
        self.btn_abort_sample.setEnabled(map_running)

    def _ensure_ready_for_multi_angle(self) -> bool:
        if not self.session.is_open:
            QtWidgets.QMessageBox.warning(
                self,
                "HDF5 File Not Open",
                "Please open or create an HDF5 container in the 'Single angle study' tab first.",
            )
            return False
        if self.session.laser_calib_result is None or self.session.rb_calib_result is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Shared Calibrations Missing",
                "Please acquire and calibrate both Laser Reference and Rubidium Absorption in 'Single angle study' first.",
            )
            return False
        return True

    def _set_pos_from_stage(self, which: str):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the ASI MS-2000 stage first.")
            return
        try:
            pos = self.stage.get_position()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Stage Read Error", str(exc))
            return
        if which == "ref":
            if pos.get("X") is not None:
                self.spin_ref_x.setValue(float(pos["X"]))
            if pos.get("Y") is not None:
                self.spin_ref_y.setValue(float(pos["Y"]))
            if pos.get("Z") is not None:
                self.spin_ref_z.setValue(float(pos["Z"]))
        else:
            if pos.get("X") is not None:
                self.spin_smp_x.setValue(float(pos["X"]))
            if pos.get("Y") is not None:
                self.spin_smp_y.setValue(float(pos["Y"]))
            if pos.get("Z") is not None:
                self.spin_smp_z.setValue(float(pos["Z"]))

    def _go_to_saved_pos(self, which: str):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the ASI MS-2000 stage first.")
            return
        if self._is_busy():
            return
        self._stop_live_cb()
        if which == "ref":
            target = {"X": float(self.spin_ref_x.value()), "Y": float(self.spin_ref_y.value()), "Z": float(self.spin_ref_z.value())}
            label = "Reference Position"
        else:
            target = {"X": float(self.spin_smp_x.value()), "Y": float(self.spin_smp_y.value()), "Z": float(self.spin_smp_z.value())}
            label = "Sample Position"

        self._set_controls_busy(True)
        self._move_worker = _StageMoveWorker(
            stage=self.stage,
            camera=self.camera,
            target_pos=target,
            label=label,
            snap_preview=True,
            parent=self,
        )
        self._move_worker.sig_log.connect(self.sig_log.emit)
        self._move_worker.sig_moved.connect(self._on_stage_moved)
        self._move_worker.sig_error.connect(self._on_worker_error)
        self._move_worker.finished.connect(lambda: self._on_worker_done())
        self._move_worker.start()

    @QtCore.Slot(object, object)
    def _on_stage_moved(self, pos: dict, snap_info):
        self.sig_stage_updated.emit()
        if snap_info is not None:
            frame, p, exp_s, n_avg, binning = snap_info
            self.sig_display_frame.emit(frame, p, exp_s, n_avg, binning)

    def _on_set_center_from_stage(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the ASI MS-2000 stage first.")
            return
        pos = self.stage.get_position()
        if pos.get("X") is not None:
            self.spin_xc.setValue(float(pos["X"]))
        if pos.get("Y") is not None:
            self.spin_yc.setValue(float(pos["Y"]))

    # ==========================================================================
    # Active Angle Reference Sample Acquisition & q-Vector Fit
    # ==========================================================================

    def _on_acquire_angle_ref(self):
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the Orca camera first.")
            return
        if not self._ensure_ready_for_multi_angle():
            return
        ang = self.current_angle_label()
        if ang is None or self._is_busy():
            return

        self._stop_live_cb()
        role = AcquisitionSession.ref_role_for_angle(ang)
        self._set_controls_busy(True)
        self._cal_worker = CalibrationCaptureWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            role=role,
            exposure_ms=float(self.spin_ref_exp.value()),
            n_avg=int(self.spin_ref_avg.value()),
            parent=self,
        )
        self._cal_worker.sig_log.connect(self.sig_log.emit)
        self._cal_worker.sig_captured.connect(self._on_ref_captured)
        self._cal_worker.sig_error.connect(self._on_worker_error)
        self._cal_worker.finished.connect(self._on_worker_done)
        self._cal_worker.start()

    @QtCore.Slot(str, object, object, float, int, int)
    def _on_ref_captured(self, role: str, frame, pos, exp_s: float, n_avg: int, binning: int):
        self.sig_lock_geometry.emit(True)
        self.sig_camera_updated.emit()
        self.sig_display_frame.emit(frame, pos, exp_s, n_avg, binning)
        self._rebuild_angles_table()
        self._refresh_active_angle_ui()

        if self.chk_auto_fit_ref.isChecked() and self.session.rb_calib_result is not None:
            QtCore.QTimer.singleShot(50, self._on_fit_angle_ref)

    def _on_view_angle_ref(self):
        ang = self.current_angle_label()
        if ang is None or ang not in self.session.multi_angle_ref_images:
            return
        img = self.session.multi_angle_ref_images[ang]
        meta = self.session.multi_angle_ref_meta.get(ang, {})
        exp_s = float(meta.get("exposure_s", 0.05))
        n_avg = int(meta.get("frame_averaging", 1))
        binning = int(meta.get("hardware_binning", 1))
        self.sig_display_frame.emit(img, None, exp_s, n_avg, binning)

    def _on_fit_angle_ref(self):
        if not self._ensure_ready_for_multi_angle():
            return
        ang = self.current_angle_label()
        if ang is None or ang not in self.session.multi_angle_ref_images:
            QtWidgets.QMessageBox.warning(
                self,
                "Reference Image Missing",
                f"Acquire a Reference Sample image for angle '{ang}' first.",
            )
            return
        if self._is_busy():
            return

        role = AcquisitionSession.ref_role_for_angle(ang)
        params = {
            "ref_role": role,
            "bin_val_h": int(self.spin_bin_h.value()),
            "bin_val_w": int(self.spin_bin_w.value()),
            "expected_shift_ghz": float(self.spin_dho_shift.value()),
            "shift_tol_ghz": float(self.spin_dho_tol.value()),
            "expected_lw_ghz": float(self.spin_dho_lw.value()),
            "lw_bounds_ghz": (float(self.spin_dho_lw_min.value()), float(self.spin_dho_lw_max.value())),
            "run_q_calibration": bool(self.chk_run_q_calib.isChecked()),
            "refractive_index": float(self.spin_q_n.value()),
            "shift_filter_bounds": (float(self.spin_q_smin.value()), float(self.spin_q_smax.value())),
        }
        self._set_controls_busy(True)
        self._ref_fit_worker = SampleTreatmentWorker(
            session=self.session,
            params=params,
            parent=self,
        )
        self._ref_fit_worker.sig_log.connect(self.sig_log.emit)
        self._ref_fit_worker.sig_fitted.connect(self._on_ref_fitted)
        self._ref_fit_worker.sig_error.connect(self._on_worker_error)
        self._ref_fit_worker.finished.connect(self._on_worker_done)
        self._ref_fit_worker.start()

    @QtCore.Slot(object)
    def _on_ref_fitted(self, res: dict):
        self._rebuild_angles_table()
        self._refresh_active_angle_ui()
        default_view = "q_dispersion" if res.get("q_results") is not None else "dho_maps"
        self.sig_show_sample_diagnostics.emit(res, default_view)

    def _on_show_ref_diag(self, view_mode: str):
        ang = self.current_angle_label()
        res = self.session.multi_angle_ref_calibs.get(ang) if ang else None
        if res is None:
            return
        self.sig_show_sample_diagnostics.emit(res, view_mode)

    # ==========================================================================
    # Brightfield Plate Calibration & Fine Multi-Angle Alignment
    # ==========================================================================

    def _load_bf_calib_from_settings(self):
        if self.session.brightfield_stage_calib is not None:
            return
        settings = QtCore.QSettings("arVIPAspectro", "BrightfieldCalibration")
        raw = settings.value("calib_json", "")
        if not raw:
            return
        try:
            data = json.loads(str(raw))
            data["M_um_per_px"] = np.asarray(data["M_um_per_px"], dtype=np.float64)
            data["P_px_per_um"] = np.asarray(data["P_px_per_um"], dtype=np.float64)
            self.session.brightfield_stage_calib = data
        except Exception:
            pass

    def _save_bf_calib_to_settings(self, calib: Dict[str, Any]):
        settings = QtCore.QSettings("arVIPAspectro", "BrightfieldCalibration")
        payload = {
            "method": str(calib.get("method", "unknown")),
            "um_per_px": float(calib["um_per_px"]),
            "um_per_px_x": float(calib.get("um_per_px_x", calib["um_per_px"])),
            "um_per_px_y": float(calib.get("um_per_px_y", calib["um_per_px"])),
            "rotation_deg": float(calib.get("rotation_deg", 0.0)),
            "flip_x": bool(calib.get("flip_x", False)),
            "flip_y": bool(calib.get("flip_y", False)),
            "M_um_per_px": np.asarray(calib["M_um_per_px"], dtype=np.float64).tolist(),
            "P_px_per_um": np.asarray(calib["P_px_per_um"], dtype=np.float64).tolist(),
        }
        settings.setValue("calib_json", json.dumps(payload))

    def _refresh_bf_calib_ui(self):
        cal = self.session.brightfield_stage_calib
        if cal is not None:
            self._style_status_badge(
                self.lbl_bf_plate_status,
                True,
                ok_text=f"✓ {cal['um_per_px']:.4f} µm/px (rot={cal.get('rotation_deg', 0.0):+.1f}°)",
            )
            self.spin_bf_manual_scale.blockSignals(True)
            self.spin_bf_manual_scale.setValue(float(cal["um_per_px"]))
            self.spin_bf_manual_scale.blockSignals(False)
        else:
            self._style_status_badge(self.lbl_bf_plate_status, False, bad_text="NOT CALIBRATED")

        ready_hw = (
            self.brightfield_cam is not None
            and self.brightfield_cam.is_connected
            and self.stage.is_connected
            and not self._is_busy()
        )
        self.btn_auto_calib_plate.setEnabled(ready_hw)
        self.btn_apply_bf_manual.setEnabled(not self._is_busy())

    def _ask_calib_continue(self, title: str, text: str) -> bool:
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(text)
        btn_cont = msg.addButton("Continue", QtWidgets.QMessageBox.AcceptRole)
        msg.addButton(QtWidgets.QMessageBox.Cancel)
        msg.exec()
        return msg.clickedButton() == btn_cont

    def _cancel_plate_calib(self, reason: str, return_to_base: bool = False):
        pending = self._plate_calib_pending
        self._plate_calib_pending = None
        if return_to_base and pending is not None and self.stage.is_connected:
            try:
                self.stage.move_to(x_um=float(pending["x0"]), y_um=float(pending["y0"]), wait=True)
                self.sig_stage_updated.emit()
            except Exception:
                pass
        self.sig_log.emit(reason)
        self._on_worker_done()

    def _launch_plate_jog_step(self, step_tag: str, target_x=None, target_y=None, return_pos=None):
        if self._plate_calib_worker is not None and self._plate_calib_worker.isRunning():
            self._plate_calib_worker.wait(2000)
        self._set_controls_busy(True)
        self._plate_calib_worker = _PlateJogCaptureWorker(
            stage=self.stage,
            brightfield_cam=self.brightfield_cam,
            step_tag=step_tag,
            target_x=target_x,
            target_y=target_y,
            return_pos=return_pos,
            settling_ms=300.0,
            parent=self,
        )
        self._plate_calib_worker.sig_log.connect(self.sig_log.emit)
        self._plate_calib_worker.sig_stage_updated.connect(self.sig_stage_updated.emit)
        self._plate_calib_worker.sig_captured.connect(self._on_plate_jog_step_done)
        self._plate_calib_worker.sig_error.connect(self._on_worker_error)
        self._plate_calib_worker.finished.connect(self._on_worker_done)
        self._plate_calib_worker.start()

    def _on_auto_calibrate_plate(self):
        if self.brightfield_cam is None or not self.brightfield_cam.is_connected:
            QtWidgets.QMessageBox.warning(
                self,
                "Brightfield Camera Not Connected",
                "Please connect the Teledyne Brightfield camera in 'Live preview' first.",
            )
            return
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the ASI MS-2000 stage first.")
            return
        if self._is_busy():
            return

        self._stop_live_cb()
        try:
            base_pos = self.stage.get_position()
            x0 = float(base_pos["X"])
            y0 = float(base_pos["Y"])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Stage Position Error", str(exc))
            return

        step = float(self.spin_bf_jog_step.value())
        self._plate_calib_pending = {
            "x0": x0,
            "y0": y0,
            "step": step,
            "dx_um": step,
            "dy_um": step,
        }
        self._launch_plate_jog_step("img0")

    @QtCore.Slot(object)
    def _on_plate_jog_step_done(self, res: dict):
        data = self._plate_calib_pending
        if data is None:
            return

        tag = res["step_tag"]
        frame = res["frame"]
        pos = res["pos"]
        exp_s = float(res["exp_s"])
        data[tag] = frame
        self.sig_display_frame.emit(frame, pos, exp_s, 1, 1)

        px_act = float(pos.get("X", data["x0"]))
        py_act = float(pos.get("Y", data["y0"]))
        x0, y0, step = float(data["x0"]), float(data["y0"]), float(data["step"])

        if tag == "img0":
            ok = self._ask_calib_continue(
                "Plate Calibration — Image 1/3 (Base)",
                f"Image 1/3 captured at Base ({px_act:.2f}, {py_act:.2f}) µm.\n"
                f"Check the frame on the left (Min={int(np.min(frame))}, Max={int(np.max(frame))}).\n\n"
                f"Is this image OK? Click 'Continue' to move +X ({step:+.1f} µm) and acquire Image 2.",
            )
            if not ok:
                self._cancel_plate_calib("Plate calibration canceled after Image 1 (Base).")
                return
            self._launch_plate_jog_step("img_x", target_x=x0 + step, target_y=y0)
            return

        if tag == "img_x":
            dx_act = px_act - x0
            data["dx_um"] = dx_act if abs(dx_act) > 0.1 else step
            diff_x = float(np.mean(np.abs(frame.astype(np.float64) - data["img0"].astype(np.float64))))
            ok = self._ask_calib_continue(
                "Plate Calibration — Image 2/3 (+X)",
                f"Image 2/3 captured at +X ({px_act:.2f}, {py_act:.2f}) µm [ΔX={data['dx_um']:+.2f} µm].\n"
                f"Mean abs pixel difference vs Base: {diff_x:.1f} ADU.\n\n"
                f"Is this image OK? Click 'Continue' to move +Y ({step:+.1f} µm) and acquire Image 3.",
            )
            if not ok:
                self._cancel_plate_calib("Plate calibration canceled after Image 2 (+X).", return_to_base=True)
                return
            self._launch_plate_jog_step(
                "img_y",
                target_x=x0,
                target_y=y0 + step,
                return_pos=(x0, y0),
            )
            return

        if tag == "img_y":
            dy_act = py_act - y0
            data["dy_um"] = dy_act if abs(dy_act) > 0.1 else step
            diff_y = float(np.mean(np.abs(frame.astype(np.float64) - data["img0"].astype(np.float64))))
            ok = self._ask_calib_continue(
                "Plate Calibration — Image 3/3 (+Y)",
                f"Image 3/3 captured at +Y ({px_act:.2f}, {py_act:.2f}) µm [ΔY={data['dy_um']:+.2f} µm].\n"
                f"Stage returned to Base ({x0:.2f}, {y0:.2f}) µm.\n"
                f"Mean abs pixel difference vs Base: {diff_y:.1f} ADU.\n\n"
                "Is this image OK? Click 'Continue' to review all 3 images together before correlation.",
            )
            if not ok:
                self._cancel_plate_calib("Plate calibration canceled after Image 3 (+Y).")
                return

            # Pre-correlation 3-image validation view
            self.sig_show_plate_calib_images.emit(
                data["img0"],
                data["img_x"],
                data["img_y"],
                x0,
                y0,
                float(data["dx_um"]),
                float(data["dy_um"]),
            )
            diff_x = float(np.mean(np.abs(data["img_x"].astype(np.float64) - data["img0"].astype(np.float64))))
            ok_corr = self._ask_calib_continue(
                "Plate Calibration — Pre-Correlation Validation",
                f"All 3 Brightfield images displayed together on the left frame.\n"
                f"• +X vs Base mean diff: {diff_x:.1f} ADU\n"
                f"• +Y vs Base mean diff: {diff_y:.1f} ADU\n\n"
                "Click 'Continue' to run autocorrelation & 2D cross-correlation.",
            )
            if not ok_corr:
                self._cancel_plate_calib("Plate calibration canceled before correlation.")
                return

            self._set_controls_busy(True)
            self._plate_corr_worker = _PlateCorrWorker(
                img0=data["img0"],
                img_x=data["img_x"],
                img_y=data["img_y"],
                parent=self,
            )
            self._plate_corr_worker.sig_log.connect(self.sig_log.emit)
            self._plate_corr_worker.sig_correlated.connect(self._on_plate_corr_ready)
            self._plate_corr_worker.sig_error.connect(self._on_worker_error)
            self._plate_corr_worker.finished.connect(self._on_worker_done)
            self._plate_corr_worker.start()

    @QtCore.Slot(object)
    def _on_plate_corr_ready(self, corr_res: dict):
        data = self._plate_calib_pending
        if data is None:
            return
        self.sig_show_plate_calib_correlation.emit(
            data["img0"],
            data["img_x"],
            data["img_y"],
            float(data["x0"]),
            float(data["y0"]),
            float(data["dx_um"]),
            float(data["dy_um"]),
            corr_res,
        )

        dc_x, dr_x = corr_res["dp_x"]
        dc_y, dr_y = corr_res["dp_y"]
        try:
            calib_preview = build_plate_calibration_from_jogs(
                dx_um=float(data["dx_um"]),
                dy_um=float(data["dy_um"]),
                dp_from_x_px=(dc_x, dr_x),
                dp_from_y_px=(dc_y, dr_y),
            )
            preview_str = (
                f"Deduced scale: {calib_preview['um_per_px']:.4f} µm/px "
                f"(X={calib_preview['um_per_px_x']:.4f}, Y={calib_preview['um_per_px_y']:.4f}, "
                f"rot={calib_preview['rotation_deg']:+.2f}°)"
            )
        except Exception as exc:
            preview_str = f"Deduction error: {exc}"

        ok = self._ask_calib_continue(
            "Plate Calibration — Final Check",
            f"Autocorrelation & 2D Cross-Correlation RGB map displayed on panel 4 (R=+X, G=+Y, B=Auto).\n\n"
            f"• +X jog ({data['dx_um']:+.1f} µm): ΔX = {dc_x:+.2f} px, ΔY = {dr_x:+.2f} px (peak = {corr_res['peak_x']:.3f})\n"
            f"• +Y jog ({data['dy_um']:+.1f} µm): ΔX = {dc_y:+.2f} px, ΔY = {dr_y:+.2f} px (peak = {corr_res['peak_y']:.3f})\n"
            f"• {preview_str}\n\n"
            "Click 'Continue' to apply calibration and update fields.",
        )
        if not ok:
            self._cancel_plate_calib("Plate calibration canceled at final check.")
            return

        self._finalize_plate_calibration(data, corr_res)

    def _finalize_plate_calibration(self, step1_data: dict, corr_res: dict):
        try:
            calib = build_plate_calibration_from_jogs(
                dx_um=float(step1_data["dx_um"]),
                dy_um=float(step1_data["dy_um"]),
                dp_from_x_px=corr_res["dp_x"],
                dp_from_y_px=corr_res["dp_y"],
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Plate Calibration Deduction Error", str(exc))
            return
        finally:
            self._plate_calib_pending = None
            self._on_worker_done()

        P = calib["P_px_per_um"]
        self.chk_bf_flip_x.setChecked(bool(P[0, 0] < 0))
        self.chk_bf_flip_y.setChecked(bool(P[1, 1] < 0))
        self._on_plate_calibrated(calib)
        self.sig_log.emit(
            f"Plate Calib Step 3/3 — Deduced scale = {calib['um_per_px']:.4f} µm/px "
            f"(X={calib['um_per_px_x']:.4f}, Y={calib['um_per_px_y']:.4f} µm/px, rot={calib['rotation_deg']:+.2f}°)."
        )

    @QtCore.Slot(object)
    def _on_plate_calibrated(self, calib: dict):
        self.session.save_brightfield_stage_calibration(calib)
        self._save_bf_calib_to_settings(calib)
        self._refresh_bf_calib_ui()
        self._refresh_active_angle_ui()

    def _on_apply_manual_plate_calib(self):
        calib = build_manual_plate_calibration(
            um_per_px=float(self.spin_bf_manual_scale.value()),
            flip_x=bool(self.chk_bf_flip_x.isChecked()),
            flip_y=bool(self.chk_bf_flip_y.isChecked()),
            rotation_deg=0.0,
        )
        self.session.save_brightfield_stage_calibration(calib)
        self._save_bf_calib_to_settings(calib)
        self.sig_log.emit(
            f"Applied manual brightfield plate calibration: {calib['um_per_px']:.4f} µm/px "
            f"(flip_x={calib['flip_x']}, flip_y={calib['flip_y']})."
        )
        self._refresh_bf_calib_ui()
        self._refresh_active_angle_ui()

    def _on_acquire_angle_brightfield(self):
        if self.brightfield_cam is None or not self.brightfield_cam.is_connected:
            QtWidgets.QMessageBox.warning(
                self,
                "Brightfield Camera Not Connected",
                "Please connect the Teledyne Brightfield camera in 'Live preview' first.",
            )
            return
        if not self._ensure_ready_for_multi_angle():
            return
        ang = self.current_angle_label()
        if ang is None or self._is_busy():
            return

        self._stop_live_cb()
        role = AcquisitionSession.brightfield_role_for_target(ang)
        self._set_controls_busy(True)
        self._bf_worker = BrightfieldCaptureWorker(
            session=self.session,
            brightfield_cam=self.brightfield_cam,
            stage=self.stage,
            role=role,
            extra_attrs={"illumination_angle_label": str(ang)},
            parent=self,
        )
        self._bf_worker.sig_log.connect(self.sig_log.emit)
        self._bf_worker.sig_captured.connect(self._on_angle_bf_captured)
        self._bf_worker.sig_error.connect(self._on_worker_error)
        self._bf_worker.finished.connect(self._on_worker_done)
        self._bf_worker.start()

    @QtCore.Slot(str, object, object, float, int, int)
    def _on_angle_bf_captured(self, role: str, frame, pos, exp_s: float, n_avg: int, binning: int):
        self.sig_display_frame.emit(frame, pos, exp_s, n_avg, binning)
        self._rebuild_angles_table()
        self._refresh_active_angle_ui()
        self.sig_session_updated.emit()

    def _on_view_angle_brightfield(self):
        ang = self.current_angle_label()
        if not ang:
            return
        role = AcquisitionSession.brightfield_role_for_target(ang)
        img = self.session.brightfield_images.get(role)
        if img is None:
            return
        meta = self.session.brightfield_meta.get(role, {})
        exp_s = float(meta.get("exposure_s", 0.02))
        n_avg = int(meta.get("frame_averaging", 1))
        pos = None
        if "stage_x_um" in meta and "stage_y_um" in meta:
            pos = {"X": float(meta["stage_x_um"]), "Y": float(meta["stage_y_um"]), "Z": meta.get("stage_z_um")}
        self.sig_display_frame.emit(img, pos, exp_s, n_avg, 1)

    def compute_alignment_to_ref(self, ref_ang: str, cur_ang: str) -> Dict[str, Any]:
        """
        Computes sub-pixel brightfield offset between `ref_ang` and `cur_ang`, converts it
        to `(dx_um, dy_um)` via the calibrated 2x2 Jacobian, and updates the preview viewer.
        """
        calib = self.session.brightfield_stage_calib
        if calib is None:
            raise RuntimeError("Brightfield plate calibration (µm/px) must be run or applied first.")
        ref_role = AcquisitionSession.brightfield_role_for_target(ref_ang)
        cur_role = AcquisitionSession.brightfield_role_for_target(cur_ang)
        ref_img = self.session.brightfield_images.get(ref_role)
        cur_img = self.session.brightfield_images.get(cur_role)
        if ref_img is None or cur_img is None:
            raise RuntimeError("Both Reference Angle and Active Angle Brightfield images are required.")

        dcol_px, drow_px, peak_val = measure_image_shift_px(ref_img, cur_img)
        dx_um, dy_um = pixel_shift_to_stage_delta_um(calib, dcol_px, drow_px)

        cur_meta = self.session.brightfield_meta.get(cur_role, {})
        if "stage_x_um" in cur_meta and "stage_y_um" in cur_meta:
            base_x = float(cur_meta["stage_x_um"])
            base_y = float(cur_meta["stage_y_um"])
        elif self.stage.is_connected:
            pos = self.stage.get_position()
            base_x = float(pos.get("X") or self.spin_xc.value())
            base_y = float(pos.get("Y") or self.spin_yc.value())
        else:
            base_x = float(self.spin_xc.value())
            base_y = float(self.spin_yc.value())

        self.sig_show_brightfield_alignment.emit(
            ref_img, cur_img, ref_ang, cur_ang, dcol_px, drow_px, dx_um, dy_um
        )
        return {
            "ref_ang": ref_ang,
            "cur_ang": cur_ang,
            "dcol_px": dcol_px,
            "drow_px": drow_px,
            "peak": peak_val,
            "dx_um": dx_um,
            "dy_um": dy_um,
            "target_x_um": base_x + dx_um,
            "target_y_um": base_y + dy_um,
        }

    def apply_alignment_coordinates(self, target_x_um: float, target_y_um: float, move_stage: bool = False):
        self.spin_xc.setValue(float(target_x_um))
        self.spin_yc.setValue(float(target_y_um))
        self.spin_smp_x.setValue(float(target_x_um))
        self.spin_smp_y.setValue(float(target_y_um))
        self.sig_log.emit(
            f"[Multi-Angle] Updated Map Center & Sample Pos to ({target_x_um:.2f}, {target_y_um:.2f}) µm."
        )
        if move_stage and self.stage.is_connected and not self._is_busy():
            self._stop_live_cb()
            self._set_controls_busy(True)
            self._move_worker = _StageMoveWorker(
                stage=self.stage,
                camera=self.camera,
                target_pos={"X": float(target_x_um), "Y": float(target_y_um), "Z": None},
                label="Aligned Map Center",
                snap_preview=False,
                parent=self,
            )
            self._move_worker.sig_log.connect(self.sig_log.emit)
            self._move_worker.sig_moved.connect(self._on_stage_moved)
            self._move_worker.sig_error.connect(self._on_worker_error)
            self._move_worker.finished.connect(self._on_worker_done)
            self._move_worker.start()

    def _on_align_to_ref_angle(self):
        cur_ang = self.current_angle_label()
        ref_ang = self.combo_bf_ref_angle.currentData()
        if not cur_ang or not ref_ang:
            return
        try:
            res = self.compute_alignment_to_ref(ref_ang, cur_ang)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Brightfield Alignment Error", str(exc))
            return

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(f"Align '{cur_ang}' → '{ref_ang}'")
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText(
            f"Measured pixel shift ({cur_ang} vs {ref_ang}): "
            f"ΔX = {res['dcol_px']:+.2f} px, ΔY = {res['drow_px']:+.2f} px (CC={res['peak']:.3f})\n"
            f"Stage correction: ΔX = {res['dx_um']:+.2f} µm, ΔY = {res['dy_um']:+.2f} µm\n\n"
            f"Aligned Map Center / Sample Pos: ({res['target_x_um']:.2f}, {res['target_y_um']:.2f}) µm"
        )
        btn_move = msg.addButton("Apply Shift && Move Stage", QtWidgets.QMessageBox.AcceptRole)
        btn_center = msg.addButton("Apply to Center Only", QtWidgets.QMessageBox.ActionRole)
        msg.addButton(QtWidgets.QMessageBox.Cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_move:
            self.apply_alignment_coordinates(res["target_x_um"], res["target_y_um"], move_stage=True)
        elif clicked == btn_center:
            self.apply_alignment_coordinates(res["target_x_um"], res["target_y_um"], move_stage=False)

    # ==========================================================================
    # Active Angle Sample Acquisition (Single Point or 2D Spatial Map)
    # ==========================================================================

    def _on_sample_mode_toggled(self):
        is_map = self.rb_mode_map.isChecked()
        self.widget_map_grid.setVisible(is_map)
        self.btn_start_sample.setText("▶ Start Angle 2D Map" if is_map else "▶ Acquire Single Point")
        self._update_sample_summary()

    def _target_measure_name(self, angle_label: str) -> str:
        base = (self.txt_study_name.text().strip() or "Study").replace("/", "_")
        return f"{base}_{AcquisitionSession.sanitize_angle_label(angle_label)}"

    def _update_sample_summary(self):
        ang = self.current_angle_label() or "angle_1"
        target_name = self._target_measure_name(ang)
        if self.rb_mode_map.isChecked():
            cfg = self.get_current_grid_config()
            exp_s = float(self.spin_smp_exp.value()) / 1000.0
            n_avg = int(self.spin_smp_avg.value())
            settling_s = float(self.spin_smp_settling.value()) / 1000.0
            dur_s = cfg.estimate_duration_seconds(exp_s, n_avg, settling_s)
            frame_shape = self._current_frame_shape()
            raw_str = (
                f"Raw_data ~{_format_bytes(cfg.estimate_raw_size_bytes(frame_shape))}"
                if self.chk_save_raw.isChecked()
                else "Raw_data omitted"
            )
            self.lbl_sample_summary.setText(
                f"Target: Brillouin/{target_name}  |  {cfg.nx}×{cfg.ny} = {cfg.total_points} pts  |  "
                f"Est. {_format_duration(dur_s)}  |  {raw_str}"
            )
        else:
            self.lbl_sample_summary.setText(
                f"Target: Brillouin/{target_name} (1×1 Single Point at current stage X, Y)"
            )

    def _current_frame_shape(self):
        if self.camera.is_connected:
            roi = self.camera.get_roi()
            b = max(1, int(roi[4]))
            return (roi[3] - roi[2]) // b, (roi[1] - roi[0]) // b
        return (2048, 2048)

    def get_current_grid_config(self) -> MapGridConfig:
        if self.rb_mode_map.isChecked():
            return MapGridConfig.from_center(
                x_center_um=float(self.spin_xc.value()),
                y_center_um=float(self.spin_yc.value()),
                dx_um=float(self.spin_dx.value()),
                dy_um=float(self.spin_dy.value()),
                nx=int(self.spin_nx.value()),
                ny=int(self.spin_ny.value()),
                scan_pattern=self.combo_pattern.currentText(),
            )
        pos = self.stage.get_position() if self.stage.is_connected else {}
        x_cur = float(pos.get("X") if pos.get("X") is not None else self.spin_smp_x.value())
        y_cur = float(pos.get("Y") if pos.get("Y") is not None else self.spin_smp_y.value())
        return MapGridConfig.from_center(
            x_center_um=x_cur,
            y_center_um=y_cur,
            dx_um=1.0,
            dy_um=1.0,
            nx=1,
            ny=1,
            scan_pattern="raster",
        )

    def _on_start_sample(self):
        if not self.camera.is_connected or not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(
                self,
                "Hardware Not Connected",
                "Please connect both the Camera (Orca Flash 4.0 v3) and the Stage (ASI MS-2000) first.",
            )
            return
        if not self._ensure_ready_for_multi_angle():
            return
        ang = self.current_angle_label()
        if ang is None or self._is_busy():
            return

        ref_calib = self.session.multi_angle_ref_calibs.get(ang)
        if self.chk_treat_on_the_fly.isChecked() and ref_calib is None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Angle Reference Not Fitted",
                f"No fitted Reference Sample calibration was found for angle '{ang}'.\n"
                "Recommended: Cancel and fit the Reference Sample for this angle first.\n"
                "Proceed anyway using default/Rb geometry?",
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Cancel,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        bf_role = AcquisitionSession.brightfield_role_for_target(ang)
        if bf_role not in self.session.brightfield_images:
            if self.brightfield_cam is not None and self.brightfield_cam.is_connected:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Brightfield Image Required",
                    f"No Brightfield image has been recorded for angle '{ang}'.\n"
                    "Please click '📷 Acquire Brightfield' in Section 4B before starting the sample/map.",
                )
                return
            reply_bf = QtWidgets.QMessageBox.question(
                self,
                "Brightfield Camera Disconnected",
                f"The Teledyne Brightfield camera is not connected and no Brightfield image exists for '{ang}'.\n"
                "Cancel to connect & acquire Brightfield first (Recommended), or proceed without Brightfield?",
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Cancel,
            )
            if reply_bf != QtWidgets.QMessageBox.Yes:
                return

        self._stop_live_cb()
        self.sig_lock_geometry.emit(True)
        self._set_controls_busy(True, map_running=True)

        cfg = self.get_current_grid_config()
        measure_name = self._target_measure_name(ang)
        role = AcquisitionSession.ref_role_for_angle(ang)
        mode_tag = "map_2d" if self.rb_mode_map.isChecked() else "single_point"

        self.progress_sample.setRange(0, cfg.total_points)
        self.progress_sample.setValue(0)
        self.progress_sample.setFormat(f"[{ang}] Starting (0 / {cfg.total_points})...")

        self._map_paused = False
        self.btn_pause_sample.setText("⏸ Pause")

        fit_params = {
            "bin_val_h": int(self.spin_bin_h.value()),
            "bin_val_w": int(self.spin_bin_w.value()),
            "expected_shift_ghz": float(self.spin_dho_shift.value()),
            "shift_tol_ghz": float(self.spin_dho_tol.value()),
            "expected_lw_ghz": float(self.spin_dho_lw.value()),
            "lw_bounds_ghz": (float(self.spin_dho_lw_min.value()), float(self.spin_dho_lw_max.value())),
            "refractive_index": float(self.spin_q_n.value()),
            "fast_drift_only": True,
        }
        extra_attrs = {
            "illumination_angle_label": str(ang),
            "reference_sample_group": f"{AcquisitionSession.CALIBRATION_GROUP}/{role}",
            "measurement_mode": mode_tag,
        }

        self._map_worker = MapScanWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            map_name=measure_name,
            grid=cfg,
            exposure_ms=float(self.spin_smp_exp.value()),
            n_avg=int(self.spin_smp_avg.value()),
            settling_ms=float(self.spin_smp_settling.value()),
            save_raw_data=bool(self.chk_save_raw.isChecked()),
            on_the_fly_treatment=bool(self.chk_treat_on_the_fly.isChecked()),
            fit_params=fit_params,
            q_calib_result=ref_calib,
            extra_attrs=extra_attrs,
            parent=self,
        )
        self._map_worker.sig_log.connect(self.sig_log.emit)
        self._map_worker.sig_progress.connect(self._on_sample_progress)
        self._map_worker.sig_stage_updated.connect(self.sig_stage_updated.emit)
        self._map_worker.sig_point_acquired.connect(self.sig_display_frame.emit)
        self._map_worker.sig_live_map_updated.connect(self.sig_update_live_map.emit)
        self._map_worker.sig_finished_map.connect(self._on_sample_finished)
        self._map_worker.sig_error.connect(self._on_worker_error)
        self._map_worker.finished.connect(self._on_worker_done)
        self._map_worker.start()

    def _on_pause_sample(self):
        if self._map_worker is None or not self._map_worker.isRunning():
            return
        self._map_paused = not self._map_paused
        self._map_worker.request_pause(self._map_paused)
        self.btn_pause_sample.setText("▶ Resume" if self._map_paused else "⏸ Pause")

    def _on_abort_sample(self):
        if self._map_worker is not None and self._map_worker.isRunning():
            self._map_worker.request_abort()
            self.btn_abort_sample.setEnabled(False)

    @QtCore.Slot(int, int, int, int, float, float)
    def _on_sample_progress(self, step_idx: int, total: int, iy: int, ix: int, x_um: float, y_um: float):
        ang = self.current_angle_label() or ""
        self.progress_sample.setValue(step_idx)
        self.progress_sample.setFormat(f"[{ang}] {step_idx}/{total} pts — [{iy},{ix}] ({x_um:.1f}, {y_um:.1f}) µm")

    @QtCore.Slot(str, int)
    def _on_sample_finished(self, map_group_path: str, completed: int):
        self.sig_log.emit(f"[Multi-Angle] Finished '{map_group_path}' ({completed} point(s) saved).")
        self.sig_camera_updated.emit()
        self.sig_stage_updated.emit()
        self.refresh_from_session()
        self.sig_session_updated.emit()

    @QtCore.Slot(str, str)
    def _on_worker_error(self, title: str, message: str):
        self._plate_calib_pending = None
        self.sig_log.emit(f"ERROR [{title}]: {message}")
        QtWidgets.QMessageBox.critical(self, title, message)
        self._on_worker_done()

    def _on_worker_done(self):
        if self._plate_calib_pending is not None:
            return
        self._set_controls_busy(False, map_running=False)
        self._refresh_active_angle_ui()
