"""
Step 1, Step 2 & Step 3 GUI:
  - Step 1: Hardware Connection & Status Panel (Hamamatsu Orca Flash 4.0 v3 & ASI MS-2000)
  - Step 2: Stage Movement (Jog / Absolute Go-To) + Automatic Image Acquisition & Live Display
  - Step 3: Drivable Camera Acquisition Parameters (Exposure, Frame Averaging, Hardware Binning,
            Sensor ROI, Readout Speed, Defect Pixel Correction, Geometry Lock, and Live Preview)
"""

import json
import os
import time
import datetime
from typing import Optional, Dict, Any
from PySide6 import QtWidgets, QtCore, QtGui

_USER_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "user_config.json")
)
_DEFAULT_USER_CONFIG: Dict[str, Any] = {
    "plate_fast_speed_um_s": 500.0,
    "plate_slow_speed_um_s": 50.0,
}


def _load_user_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULT_USER_CONFIG)
    if os.path.exists(_USER_CONFIG_PATH):
        try:
            with open(_USER_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
        except Exception:
            pass
    return cfg


def _save_user_config(cfg: Dict[str, Any]):
    try:
        with open(_USER_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

from ..hardware import (
    MS2000Stage,
    MS2000Error,
    list_serial_ports,
    OrcaFlashCamera,
    OrcaCameraError,
    count_dcam_cameras,
    TeledyneBrightfieldCamera,
    TeledyneCameraError,
    count_teledyne_cameras,
)
from ..control import RigController, StageActionWorker, PlateAutoCalibrateWorker
from .image_viewer import ImageViewerWidget
from .pipeline_panel import PipelineAcquisitionWidget


class CollapsibleSection(QtWidgets.QWidget):
    """Collapsible section with a clickable header bar and toggleable content area."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._expanded = expanded

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self.toggle_btn = QtWidgets.QToolButton()
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.toggle_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.toggle_btn.setStyleSheet(
            "QToolButton { font-weight: bold; padding: 5px 8px; text-align: left; "
            "background-color: #d6eaf8; color: #1b4f72; border: 1px solid #85c1e9; border-radius: 4px; }"
            "QToolButton:hover { background-color: #ebf5fb; }"
        )
        self.toggle_btn.toggled.connect(self._on_toggled)
        outer.addWidget(self.toggle_btn)

        self.content_area = QtWidgets.QFrame()
        self.content_area.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.content_layout = QtWidgets.QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(6)
        self.content_area.setVisible(expanded)
        outer.addWidget(self.content_area)

    def _on_toggled(self, checked: bool):
        self._expanded = checked
        self.toggle_btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.content_area.setVisible(checked)


class MainWindow(QtWidgets.QMainWindow):
    """Step 1–3 Application Window: Connection, Stage Motion, Image Capture & Acquisition Parameters."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "arVIPAspectro — Step 4: Acquisition Pipeline & Spatial Mapping (Orca Flash 4.0 v3 & ASI MS-2000)"
        )
        self.resize(1460, 900)

        self._worker: Optional[StageActionWorker] = None
        self._plate_calib_worker: Optional[PlateAutoCalibrateWorker] = None
        self._live_running: bool = False
        self._active_live_source: str = "orca"
        self._syncing_cam_widgets: bool = False
        self._syncing_bf_widgets: bool = False
        self._pending_frame: Optional[tuple] = None
        self._frame_flush_scheduled: bool = False

        # Hardware driver instances & centralized RigController
        self.camera = OrcaFlashCamera(logger=self._threadsafe_log)
        self.brightfield_cam = TeledyneBrightfieldCamera(logger=self._threadsafe_log)
        self.stage = MS2000Stage(logger=self._threadsafe_log)
        self.rig = RigController(
            camera=self.camera,
            stage=self.stage,
            brightfield_cam=self.brightfield_cam,
            parent=self,
        )

        self._user_config = _load_user_config()
        self._build_ui()
        saved_bf_scale = self._user_config.get("brightfield_um_per_px")
        if saved_bf_scale is not None:
            try:
                scale_val = float(saved_bf_scale)
                if scale_val > 0:
                    self.image_viewer.set_brightfield_scale(scale_val)
                    self.lbl_bf_calib_scale.setText(f"{scale_val:.4f} µm/px")
            except (TypeError, ValueError):
                pass
        self._refresh_serial_ports()
        self._scan_dcam_cameras()
        self._scan_bf_cameras()
        self._update_camera_ui()
        self._update_brightfield_ui()
        self._update_stage_ui()

    # ==========================================================================
    # UI Construction
    # ==========================================================================

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Top Header Bar
        top_bar = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel(
            "Step 4 — Hardware Control, Calibration Spectra & Spatial Map Acquisition"
        )
        title_font = title_lbl.font()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title_lbl.setFont(title_font)

        self.btn_connect_all = QtWidgets.QPushButton("Connect All Hardware")
        self.btn_connect_all.setMinimumHeight(30)
        self.btn_connect_all.clicked.connect(self._on_connect_all)

        self.btn_disconnect_all = QtWidgets.QPushButton("Disconnect All")
        self.btn_disconnect_all.setMinimumHeight(30)
        self.btn_disconnect_all.clicked.connect(self._on_disconnect_all)

        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_connect_all)
        top_bar.addWidget(self.btn_disconnect_all)
        main_layout.addLayout(top_bar)

        # Horizontal Splitter: Left = Viewer, Right = Tabbed Controls
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=4)

        # --- LEFT PANEL: Viewer ---
        viewer_box = QtWidgets.QGroupBox("Viewer")
        viewer_layout = QtWidgets.QVBoxLayout(viewer_box)
        viewer_layout.setContentsMargins(6, 6, 6, 6)

        quick_live_row = QtWidgets.QHBoxLayout()
        quick_live_row.setSpacing(6)
        self.btn_quick_live_orca = QtWidgets.QPushButton("Spectrometer")
        self.btn_quick_live_orca.setMinimumHeight(28)
        self.btn_quick_live_orca.clicked.connect(lambda: self._start_quick_live("orca"))
        quick_live_row.addWidget(self.btn_quick_live_orca)

        self.btn_quick_live_bf = QtWidgets.QPushButton("Brightfield")
        self.btn_quick_live_bf.setMinimumHeight(28)
        self.btn_quick_live_bf.clicked.connect(lambda: self._start_quick_live("teledyne"))
        quick_live_row.addWidget(self.btn_quick_live_bf)

        self.btn_stop_live = QtWidgets.QPushButton("Stop acquisition")
        self.btn_stop_live.setMinimumHeight(28)
        self.btn_stop_live.setEnabled(False)
        self.btn_stop_live.clicked.connect(self._stop_live_if_running)
        quick_live_row.addWidget(self.btn_stop_live)

        quick_live_row.addStretch()
        viewer_layout.addLayout(quick_live_row)

        self.image_viewer = ImageViewerWidget(self)
        viewer_layout.addWidget(self.image_viewer)
        splitter.addWidget(viewer_box)

        # --- RIGHT PANEL: Tabbed Controls (Hardware, Live preview, Mapping) ---
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(8)

        self.tabs_right = QtWidgets.QTabWidget()

        # Tab 1: Hardware (Spectrometer Camera, Brightfield Camera, Plate)
        hw_scroll = QtWidgets.QScrollArea()
        hw_scroll.setWidgetResizable(True)
        hw_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        hw_content = QtWidgets.QWidget()
        hw_vbox = QtWidgets.QVBoxLayout(hw_content)
        hw_vbox.setContentsMargins(4, 4, 4, 4)
        hw_vbox.setSpacing(8)
        hw_vbox.addWidget(self._build_camera_card())
        hw_vbox.addWidget(self._build_brightfield_card())
        hw_vbox.addWidget(self._build_stage_card())
        hw_vbox.addStretch()
        hw_scroll.setWidget(hw_content)
        self.tabs_right.addTab(hw_scroll, "Hardware")

        # Tab 2: Live preview (3 collapsible sections: Spectrometer Camera, Brightfield Camera, Plate)
        lp_scroll = QtWidgets.QScrollArea()
        lp_scroll.setWidgetResizable(True)
        lp_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        lp_content = QtWidgets.QWidget()
        lp_vbox = QtWidgets.QVBoxLayout(lp_content)
        lp_vbox.setContentsMargins(4, 4, 4, 4)
        lp_vbox.setSpacing(8)

        sec_spec = CollapsibleSection("Spectrometer Camera", expanded=True)
        sec_spec.content_layout.addWidget(self._build_step3_camera_parameters_card())
        lp_vbox.addWidget(sec_spec)

        sec_bf = CollapsibleSection("Brightfield Camera", expanded=True)
        sec_bf.content_layout.addWidget(self._build_brightfield_preview_card())
        lp_vbox.addWidget(sec_bf)

        sec_plate = CollapsibleSection("Plate", expanded=True)
        sec_plate.content_layout.addWidget(self._build_step2_motion_and_capture_card())
        lp_vbox.addWidget(sec_plate)

        lp_vbox.addStretch()
        lp_scroll.setWidget(lp_content)
        self.tabs_right.addTab(lp_scroll, "Live preview")

        # Tab 3: Mapping (merged Single + Multi angle study)
        self.pipeline_panel = PipelineAcquisitionWidget(
            camera=self.camera,
            stage=self.stage,
            stop_live_callback=self._stop_live_if_running,
            brightfield_cam=self.brightfield_cam,
            parent=self,
        )
        self.pipeline_panel.sig_log.connect(self._append_log_slot)
        self.pipeline_panel.sig_display_frame.connect(self._on_frame_acquired)
        self.pipeline_panel.sig_show_laser_diagnostics.connect(
            self.image_viewer.show_laser_calibration_diagnostics
        )
        self.pipeline_panel.sig_show_rb_diagnostics.connect(
            self.image_viewer.show_rb_calibration_diagnostics
        )
        self.pipeline_panel.sig_show_sample_diagnostics.connect(
            self.image_viewer.show_sample_treatment_diagnostics
        )
        self.pipeline_panel.sig_update_live_map.connect(
            self.image_viewer.update_live_map_view
        )
        self.pipeline_panel.sig_stage_updated.connect(self._update_stage_ui)
        self.pipeline_panel.sig_camera_updated.connect(self._update_camera_ui)
        self.pipeline_panel.sig_mapping_ready_changed.connect(self._on_mapping_ready_changed)
        self.pipeline_panel.sig_set_bf_reference.connect(self._on_set_bf_reference)
        self.tabs_right.addTab(self.pipeline_panel, "Mapping")
        self.tabs_right.currentChanged.connect(self._on_right_tab_changed)

        right_layout.addWidget(self.tabs_right)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 5)

        # Bottom Hardware Log (spanning full width under both left & right frames)
        log_group = QtWidgets.QGroupBox("Hardware Communication Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QtGui.QFont("Courier", 10))
        self.txt_log.setMinimumHeight(90)
        self.txt_log.setMaximumHeight(160)

        btn_clear_log = QtWidgets.QPushButton("Clear Log")
        btn_clear_log.setFixedWidth(85)
        btn_clear_log.clicked.connect(self.txt_log.clear)

        log_hdr = QtWidgets.QHBoxLayout()
        log_hdr.addStretch()
        log_hdr.addWidget(btn_clear_log)
        log_layout.addWidget(self.txt_log)
        log_layout.addLayout(log_hdr)
        main_layout.addWidget(log_group, stretch=1)

    def _build_camera_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Spectrometer Camera — Orca Flash 4.0 v3")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(5)

        self.lbl_cam_status = QtWidgets.QLabel("DISCONNECTED")
        self.lbl_cam_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_cam_status.setMinimumHeight(24)
        layout.addWidget(self.lbl_cam_status)

        form = QtWidgets.QFormLayout()
        self.chk_cam_sim = QtWidgets.QCheckBox("Simulation Mode")
        form.addRow("Mode:", self.chk_cam_sim)

        cam_idx_row = QtWidgets.QHBoxLayout()
        self.spin_cam_idx = QtWidgets.QSpinBox()
        self.spin_cam_idx.setRange(0, 7)
        self.btn_scan_dcam = QtWidgets.QPushButton("Scan")
        self.btn_scan_dcam.clicked.connect(self._scan_dcam_cameras)
        self.lbl_dcam_count = QtWidgets.QLabel("0 found")
        cam_idx_row.addWidget(self.spin_cam_idx)
        cam_idx_row.addWidget(self.btn_scan_dcam)
        cam_idx_row.addWidget(self.lbl_dcam_count)
        form.addRow("Index:", cam_idx_row)
        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_cam_connect = QtWidgets.QPushButton("Connect Camera")
        self.btn_cam_connect.clicked.connect(self._on_toggle_camera)
        self.btn_cam_refresh = QtWidgets.QPushButton("Status")
        self.btn_cam_refresh.clicked.connect(self._update_camera_ui)
        btn_row.addWidget(self.btn_cam_connect)
        btn_row.addWidget(self.btn_cam_refresh)
        layout.addLayout(btn_row)

        info_form = QtWidgets.QFormLayout()
        self.lbl_cam_model = QtWidgets.QLabel("—")
        self.lbl_cam_sensor = QtWidgets.QLabel("—")
        self.lbl_cam_exposure = QtWidgets.QLabel("—")
        info_form.addRow("Model:", self.lbl_cam_model)
        info_form.addRow("Frame:", self.lbl_cam_sensor)
        info_form.addRow("Exp / T:", self.lbl_cam_exposure)
        layout.addLayout(info_form)
        return box

    def _build_stage_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Plate — ASI MS-2000")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(5)

        self.lbl_stage_status = QtWidgets.QLabel("DISCONNECTED")
        self.lbl_stage_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_stage_status.setMinimumHeight(24)
        layout.addWidget(self.lbl_stage_status)

        form = QtWidgets.QFormLayout()
        self.chk_stage_sim = QtWidgets.QCheckBox("Simulation Mode")
        form.addRow("Mode:", self.chk_stage_sim)

        port_row = QtWidgets.QHBoxLayout()
        self.combo_stage_port = QtWidgets.QComboBox()
        self.combo_stage_port.setEditable(True)
        self.btn_refresh_ports = QtWidgets.QPushButton("Scan")
        self.btn_refresh_ports.clicked.connect(self._refresh_serial_ports)
        port_row.addWidget(self.combo_stage_port, stretch=1)
        port_row.addWidget(self.btn_refresh_ports)
        form.addRow("Port:", port_row)

        self.combo_stage_baud = QtWidgets.QComboBox()
        self.combo_stage_baud.addItem("115200 (USB)", 115200)
        self.combo_stage_baud.addItem("9600 (RS-232)", 9600)
        self.combo_stage_baud.addItem("28800", 28800)
        form.addRow("Baud:", self.combo_stage_baud)
        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_stage_connect = QtWidgets.QPushButton("Connect Stage")
        self.btn_stage_connect.clicked.connect(self._on_toggle_stage)
        self.btn_stage_refresh = QtWidgets.QPushButton("Read Pos")
        self.btn_stage_refresh.clicked.connect(self._update_stage_ui)
        self.btn_stage_zero_xy = QtWidgets.QPushButton("Zero XY")
        self.btn_stage_zero_xy.setToolTip("Set current physical X, Y stage position as (0.0, 0.0) µm")
        self.btn_stage_zero_xy.clicked.connect(lambda: self._on_zero_stage(("X", "Y")))
        self.btn_stage_zero_z = QtWidgets.QPushButton("Zero Z")
        self.btn_stage_zero_z.setToolTip("Set current physical Z stage position as 0.0 µm")
        self.btn_stage_zero_z.clicked.connect(lambda: self._on_zero_stage(("Z",)))
        btn_row.addWidget(self.btn_stage_connect)
        btn_row.addWidget(self.btn_stage_refresh)
        btn_row.addWidget(self.btn_stage_zero_xy)
        btn_row.addWidget(self.btn_stage_zero_z)
        layout.addLayout(btn_row)

        info_form = QtWidgets.QFormLayout()
        self.lbl_stage_fw = QtWidgets.QLabel("—")
        self.lbl_stage_pos_xy = QtWidgets.QLabel("—")
        self.lbl_stage_pos_z = QtWidgets.QLabel("—")
        info_form.addRow("Info:", self.lbl_stage_fw)
        info_form.addRow("X / Y:", self.lbl_stage_pos_xy)
        info_form.addRow("Z:", self.lbl_stage_pos_z)
        layout.addLayout(info_form)

        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel("Fast Speed:"))
        self.spin_plate_fast_speed = QtWidgets.QDoubleSpinBox()
        self.spin_plate_fast_speed.setRange(1.0, 20000.0)
        self.spin_plate_fast_speed.setDecimals(1)
        self.spin_plate_fast_speed.setValue(float(self._user_config.get("plate_fast_speed_um_s", 500.0)))
        self.spin_plate_fast_speed.setSuffix(" µm/s")
        self.spin_plate_fast_speed.setKeyboardTracking(False)
        self.spin_plate_fast_speed.valueChanged.connect(self._on_plate_speeds_changed)
        speed_row.addWidget(self.spin_plate_fast_speed)

        speed_row.addSpacing(12)
        speed_row.addWidget(QtWidgets.QLabel("Slow Speed:"))
        self.spin_plate_slow_speed = QtWidgets.QDoubleSpinBox()
        self.spin_plate_slow_speed.setRange(0.1, 10000.0)
        self.spin_plate_slow_speed.setDecimals(1)
        self.spin_plate_slow_speed.setValue(float(self._user_config.get("plate_slow_speed_um_s", 50.0)))
        self.spin_plate_slow_speed.setSuffix(" µm/s")
        self.spin_plate_slow_speed.setKeyboardTracking(False)
        self.spin_plate_slow_speed.valueChanged.connect(self._on_plate_speeds_changed)
        speed_row.addWidget(self.spin_plate_slow_speed)
        speed_row.addStretch()
        layout.addLayout(speed_row)

        return box

    def _build_step3_camera_parameters_card(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 1. Exposure Time (with quick presets)
        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addWidget(QtWidgets.QLabel("Exposure:"))
        self.spin_exposure_ms = QtWidgets.QDoubleSpinBox()
        self.spin_exposure_ms.setDecimals(2)
        self.spin_exposure_ms.setRange(0.02, 30000.0)
        self.spin_exposure_ms.setValue(50.0)
        self.spin_exposure_ms.setSuffix(" ms")
        self.spin_exposure_ms.setMinimumWidth(110)
        self.spin_exposure_ms.editingFinished.connect(self._on_apply_exposure)
        exp_row.addWidget(self.spin_exposure_ms)

        for ms_val, lbl in [(5.0, "5ms"), (20.0, "20ms"), (50.0, "50ms"), (200.0, "200ms"), (1000.0, "1s"), (5000.0, "5s")]:
            btn_e = QtWidgets.QPushButton(lbl)
            btn_e.setFixedWidth(46)
            btn_e.clicked.connect(lambda checked=False, v=ms_val: self._set_exposure_preset(v))
            exp_row.addWidget(btn_e)
        exp_row.addStretch()
        layout.addLayout(exp_row)

        # 2. Frame Average
        avg_row = QtWidgets.QHBoxLayout()
        avg_row.addWidget(QtWidgets.QLabel("Frame average:"))
        self.spin_frame_avg = QtWidgets.QSpinBox()
        self.spin_frame_avg.setRange(1, 500)
        self.spin_frame_avg.setValue(1)
        self.spin_frame_avg.setSuffix(" frame(s)")
        self.spin_frame_avg.valueChanged.connect(self._on_apply_frame_averaging)
        avg_row.addWidget(self.spin_frame_avg)
        avg_row.addStretch()
        layout.addLayout(avg_row)

        # 3. Readout Speed
        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel("Readout Speed:"))
        self.combo_readout_speed = QtWidgets.QComboBox()
        self.combo_readout_speed.addItem("Fast (Standard Scan)", "fast")
        self.combo_readout_speed.addItem("Slow (Ultra-Quiet Low-Noise Scan)", "slow")
        self.combo_readout_speed.currentIndexChanged.connect(self._on_apply_readout_speed)
        speed_row.addWidget(self.combo_readout_speed)
        speed_row.addStretch()
        layout.addLayout(speed_row)

        # 4. On-camera Defect Pixel Correction
        defect_row = QtWidgets.QHBoxLayout()
        self.chk_defect_correct = QtWidgets.QCheckBox("On camera Defect Pixel Correction")
        self.chk_defect_correct.setChecked(True)
        self.chk_defect_correct.toggled.connect(self._on_apply_defect_correction)
        defect_row.addWidget(self.chk_defect_correct)
        defect_row.addStretch()
        layout.addLayout(defect_row)

        return box

    def _build_brightfield_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Brightfield Camera — Teledyne CM3-U3-31S4M-CS")
        v = QtWidgets.QVBoxLayout(box)
        v.setSpacing(6)

        self.lbl_bf_status = QtWidgets.QLabel("Unconnected")
        self.lbl_bf_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bf_status.setMinimumHeight(24)
        v.addWidget(self.lbl_bf_status)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Idx:"))
        self.spin_bf_idx = QtWidgets.QSpinBox()
        self.spin_bf_idx.setRange(0, 16)
        row1.addWidget(self.spin_bf_idx)

        self.btn_bf_scan = QtWidgets.QPushButton("Scan")
        self.btn_bf_scan.clicked.connect(self._scan_bf_cameras)
        row1.addWidget(self.btn_bf_scan)

        self.lbl_bf_count = QtWidgets.QLabel("—")
        row1.addWidget(self.lbl_bf_count)

        self.chk_bf_sim = QtWidgets.QCheckBox("Simulation Mode")
        row1.addWidget(self.chk_bf_sim)
        row1.addStretch()

        self.btn_bf_connect = QtWidgets.QPushButton("Connect Brightfield")
        self.btn_bf_connect.setMinimumHeight(28)
        self.btn_bf_connect.clicked.connect(self._on_toggle_brightfield)
        row1.addWidget(self.btn_bf_connect)
        v.addLayout(row1)

        self.lbl_bf_info = QtWidgets.QLabel("—")
        self.lbl_bf_info.setStyleSheet("font-size: 11px; color: #b0bec5;")
        v.addWidget(self.lbl_bf_info)
        return box

    def _build_brightfield_preview_card(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 1. Exposure
        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addWidget(QtWidgets.QLabel("Exposure:"))
        self.spin_bf_exp_ms = QtWidgets.QDoubleSpinBox()
        self.spin_bf_exp_ms.setRange(0.05, 5000.0)
        self.spin_bf_exp_ms.setDecimals(2)
        self.spin_bf_exp_ms.setValue(20.0)
        self.spin_bf_exp_ms.setSuffix(" ms")
        self.spin_bf_exp_ms.setKeyboardTracking(False)
        self.spin_bf_exp_ms.valueChanged.connect(self._on_apply_bf_exposure)
        exp_row.addWidget(self.spin_bf_exp_ms)
        exp_row.addStretch()
        layout.addLayout(exp_row)

        # 2. Frame average
        avg_row = QtWidgets.QHBoxLayout()
        avg_row.addWidget(QtWidgets.QLabel("Frame average:"))
        self.spin_bf_avg = QtWidgets.QSpinBox()
        self.spin_bf_avg.setRange(1, 100)
        self.spin_bf_avg.setValue(1)
        self.spin_bf_avg.setSuffix(" frame(s)")
        self.spin_bf_avg.valueChanged.connect(self._on_apply_bf_avg)
        avg_row.addWidget(self.spin_bf_avg)
        avg_row.addStretch()
        layout.addLayout(avg_row)

        # 3. Gain
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(QtWidgets.QLabel("Gain:"))
        self.spin_bf_gain_db = QtWidgets.QDoubleSpinBox()
        self.spin_bf_gain_db.setRange(0.0, 24.0)
        self.spin_bf_gain_db.setDecimals(1)
        self.spin_bf_gain_db.setValue(0.0)
        self.spin_bf_gain_db.setSuffix(" dB")
        self.spin_bf_gain_db.setKeyboardTracking(False)
        self.spin_bf_gain_db.valueChanged.connect(self._on_apply_bf_gain)
        gain_row.addWidget(self.spin_bf_gain_db)
        gain_row.addStretch()
        layout.addLayout(gain_row)

        # 4. Auto-calibrate + step spinbox placed right of the button
        cal_row = QtWidgets.QHBoxLayout()
        self.btn_bf_auto_calib = QtWidgets.QPushButton("Auto-calibrate")
        self.btn_bf_auto_calib.setStyleSheet("background-color: #00695c; color: white; font-weight: bold;")
        self.btn_bf_auto_calib.clicked.connect(self._on_bf_auto_calibrate)
        cal_row.addWidget(self.btn_bf_auto_calib)

        self.spin_bf_calib_step = QtWidgets.QDoubleSpinBox()
        self.spin_bf_calib_step.setRange(1.0, 500.0)
        self.spin_bf_calib_step.setDecimals(1)
        self.spin_bf_calib_step.setValue(20.0)
        self.spin_bf_calib_step.setSuffix(" µm")
        cal_row.addWidget(self.spin_bf_calib_step)

        self.lbl_bf_calib_scale = QtWidgets.QLabel("Not calibrated")
        self.lbl_bf_calib_scale.setStyleSheet("font-size: 11px; color: #b0bec5;")
        cal_row.addWidget(self.lbl_bf_calib_scale)
        cal_row.addStretch()
        layout.addLayout(cal_row)

        return box

    def _build_step2_motion_and_capture_card(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Cross pattern of 4 high-speed (100 µm/s) + 4 low-speed (10 µm/s) press-and-hold buttons
        pad_grid = QtWidgets.QGridLayout()
        pad_grid.setSpacing(4)

        self.btn_jog_yp_fast = QtWidgets.QPushButton("▲▲ +Y")
        self.btn_jog_yp_slow = QtWidgets.QPushButton("▲ +Y")
        self.btn_jog_ym_slow = QtWidgets.QPushButton("▼ -Y")
        self.btn_jog_ym_fast = QtWidgets.QPushButton("▼▼ -Y")
        self.btn_jog_xm_fast = QtWidgets.QPushButton("◀◀ -X")
        self.btn_jog_xm_slow = QtWidgets.QPushButton("◀ -X")
        self.btn_jog_xp_slow = QtWidgets.QPushButton("▶ +X")
        self.btn_jog_xp_fast = QtWidgets.QPushButton("▶▶ +X")

        jog_specs = [
            (self.btn_jog_yp_fast, 0.0, +1.0, True, 0, 2),
            (self.btn_jog_yp_slow, 0.0, +1.0, False, 1, 2),
            (self.btn_jog_xm_fast, -1.0, 0.0, True, 2, 0),
            (self.btn_jog_xm_slow, -1.0, 0.0, False, 2, 1),
            (self.btn_jog_xp_slow, +1.0, 0.0, False, 2, 3),
            (self.btn_jog_xp_fast, +1.0, 0.0, True, 2, 4),
            (self.btn_jog_ym_slow, 0.0, -1.0, False, 3, 2),
            (self.btn_jog_ym_fast, 0.0, -1.0, True, 4, 2),
        ]
        for btn, dx_sign, dy_sign, is_fast, r, c in jog_specs:
            btn.setMinimumSize(64, 30)
            btn.pressed.connect(
                lambda dx=dx_sign, dy=dy_sign, fast=is_fast: self._on_jog_hold_start(dx, dy, fast)
            )
            btn.released.connect(self._on_jog_hold_stop)
            pad_grid.addWidget(btn, r, c)

        self._update_jog_tooltips()

        layout.addLayout(pad_grid)

        ctrl_col = QtWidgets.QVBoxLayout()
        ctrl_col.setSpacing(8)
        self.btn_zero_xy_nav = QtWidgets.QPushButton("Zero")
        self.btn_zero_xy_nav.setMinimumHeight(32)
        self.btn_zero_xy_nav.setToolTip("Set current physical X, Y position as (0.0, 0.0) µm")
        self.btn_zero_xy_nav.clicked.connect(lambda: self._on_zero_stage(("X", "Y")))
        ctrl_col.addWidget(self.btn_zero_xy_nav)

        self.btn_halt_stage = QtWidgets.QPushButton("Emergency Stop")
        self.btn_halt_stage.setMinimumHeight(36)
        self.btn_halt_stage.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.btn_halt_stage.clicked.connect(self._on_halt_stage)
        ctrl_col.addWidget(self.btn_halt_stage)
        ctrl_col.addStretch()
        layout.addLayout(ctrl_col)
        layout.addStretch()

        return box

    # ==========================================================================
    # Thread-Safe Logging
    # ==========================================================================

    def _threadsafe_log(self, message: str):
        QtCore.QMetaObject.invokeMethod(
            self,
            "_append_log_slot",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, message),
        )

    @QtCore.Slot(str)
    def _append_log_slot(self, message: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "txt_log"):
            self.txt_log.appendPlainText(f"[{ts}] {message}")

    # ==========================================================================
    # Step 1: Connection Actions
    # ==========================================================================

    def _scan_dcam_cameras(self):
        n_cams = count_dcam_cameras()
        self.lbl_dcam_count.setText(f"{n_cams} found")
        self._threadsafe_log(f"DCAM scan: {n_cams} Hamamatsu camera(s) detected.")

    def _scan_bf_cameras(self):
        n_cams = count_teledyne_cameras()
        self.lbl_bf_count.setText(f"{n_cams} found")
        self._threadsafe_log(f"Spinnaker scan: {n_cams} Teledyne camera(s) detected.")

    def _on_toggle_brightfield(self):
        if self._live_running and self._current_preview_source() == "teledyne":
            self._stop_live_if_running()
        if self.brightfield_cam.is_connected:
            self.brightfield_cam.disconnect()
        else:
            simulate = self.chk_bf_sim.isChecked()
            idx = int(self.spin_bf_idx.value())
            try:
                self.brightfield_cam.connect(camera_idx=idx, simulate=simulate)
            except TeledyneCameraError as exc:
                self._threadsafe_log(f"ERROR (Brightfield): {exc}")
                QtWidgets.QMessageBox.warning(self, "Brightfield Connection Error", str(exc))
        self._update_brightfield_ui()

    def _on_apply_bf_exposure(self):
        if self._syncing_bf_widgets or not self.brightfield_cam.is_connected:
            return
        try:
            actual_s = self.brightfield_cam.set_exposure(float(self.spin_bf_exp_ms.value()) / 1000.0)
            self._syncing_bf_widgets = True
            self.spin_bf_exp_ms.setValue(actual_s * 1000.0)
            self._syncing_bf_widgets = False
            self._update_brightfield_ui()
        except Exception as exc:
            self._threadsafe_log(f"Brightfield exposure error: {exc}")

    def _on_apply_bf_gain(self):
        if self._syncing_bf_widgets or not self.brightfield_cam.is_connected:
            return
        try:
            actual_g = self.brightfield_cam.set_gain(float(self.spin_bf_gain_db.value()))
            if actual_g is not None:
                self._syncing_bf_widgets = True
                self.spin_bf_gain_db.setValue(float(actual_g))
                self._syncing_bf_widgets = False
            self._update_brightfield_ui()
        except Exception as exc:
            self._threadsafe_log(f"Brightfield gain error: {exc}")

    def _on_apply_bf_avg(self, val: int):
        if self._syncing_bf_widgets:
            return
        self.brightfield_cam.set_frame_averaging(int(val))

    def _on_bf_reset_roi(self):
        if not self.brightfield_cam.is_connected:
            return
        was_live = self._live_running and self._current_preview_source() == "teledyne"
        if was_live:
            self._stop_live_if_running()
        try:
            self.brightfield_cam.reset_full_roi()
            self._update_brightfield_ui()
        except Exception as exc:
            self._threadsafe_log(f"Brightfield ROI reset error: {exc}")
        if was_live:
            self._on_toggle_live_preview()

    def _update_brightfield_ui(self):
        st = self.brightfield_cam.get_status()
        connected = bool(st["connected"])
        simulated = bool(st["simulate"])
        self._set_badge_style(self.lbl_bf_status, connected, simulated)
        self.btn_bf_connect.setText("Disconnect" if connected else "Connect Brightfield")
        self.chk_bf_sim.setEnabled(not connected)
        self.spin_bf_idx.setEnabled(not connected)

        self.spin_bf_exp_ms.setEnabled(connected)
        self.spin_bf_gain_db.setEnabled(connected)
        self.spin_bf_avg.setEnabled(connected)
        self.btn_bf_auto_calib.setEnabled(connected and self.stage.is_connected)

        if connected:
            self._syncing_bf_widgets = True
            if st["exposure_s"] is not None:
                self.spin_bf_exp_ms.setValue(float(st["exposure_s"]) * 1000.0)
            if st["gain_db"] is not None:
                self.spin_bf_gain_db.setValue(float(st["gain_db"]))
            self.spin_bf_avg.setValue(int(st["frame_averaging"]))
            self._syncing_bf_widgets = False
            out_h, out_w = st["output_shape"]
            temp_s = f" | {st['temperature_c']:.1f}°C" if st["temperature_c"] is not None else ""
            self.lbl_bf_info.setText(f"{st['model']} | {out_w}×{out_h} px{temp_s}")
        else:
            self.lbl_bf_info.setText("Unconnected")

        busy = bool(self._worker and self._worker.isRunning() and not self._live_running)
        self._set_motion_controls_enabled(not busy)
        if hasattr(self, "pipeline_panel"):
            self.pipeline_panel._update_brightfield_ui()

    def _refresh_serial_ports(self):
        self.combo_stage_port.clear()
        self.combo_stage_port.addItem("auto (Scan COM)", "auto")
        ports = list_serial_ports()
        for p in ports:
            label = f"{p['port']} — {p['description']}"
            self.combo_stage_port.addItem(label, p["port"])
        self._threadsafe_log(f"Serial scan: {len(ports)} COM port(s) found.")

    def _on_toggle_camera(self):
        self._stop_live_if_running()
        if self.camera.is_connected:
            self.camera.disconnect()
        else:
            simulate = self.chk_cam_sim.isChecked()
            idx = int(self.spin_cam_idx.value())
            try:
                self.camera.connect(camera_idx=idx, simulate=simulate)
            except OrcaCameraError as exc:
                self._threadsafe_log(f"ERROR (Camera): {exc}")
                QtWidgets.QMessageBox.warning(self, "Camera Connection Error", str(exc))
        self._update_camera_ui()

    def _on_toggle_stage(self):
        if self.stage.is_connected:
            self.stage.disconnect()
        else:
            simulate = self.chk_stage_sim.isChecked()
            port_data = self.combo_stage_port.currentData()
            port = port_data if port_data is not None else self.combo_stage_port.currentText().strip()
            baudrate = int(self.combo_stage_baud.currentData() or 115200)
            try:
                self.stage.connect(port=port, baudrate=baudrate, simulate=simulate)
            except MS2000Error as exc:
                self._threadsafe_log(f"ERROR (Stage): {exc}")
                QtWidgets.QMessageBox.warning(self, "Stage Connection Error", str(exc))
        self._update_stage_ui()

    def _on_connect_all(self):
        if not self.camera.is_connected:
            self._on_toggle_camera()
        if not self.brightfield_cam.is_connected:
            self._on_toggle_brightfield()
        if not self.stage.is_connected:
            self._on_toggle_stage()

    def _on_disconnect_all(self):
        self._stop_live_if_running()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        if self.camera.is_connected:
            self.camera.disconnect()
            self._update_camera_ui()
        if self.brightfield_cam.is_connected:
            self.brightfield_cam.disconnect()
            self._update_brightfield_ui()
        if self.stage.is_connected:
            self.stage.disconnect()
            self._update_stage_ui()

    # ==========================================================================
    # Step 3: Drivable Camera Parameter Handlers
    # ==========================================================================

    def _set_exposure_preset(self, ms_val: float):
        self.spin_exposure_ms.setValue(ms_val)
        self._on_apply_exposure()

    def _on_apply_exposure(self):
        if self._syncing_cam_widgets or not self.camera.is_connected:
            return
        try:
            exp_s = float(self.spin_exposure_ms.value()) / 1000.0
            actual_s = self.camera.set_exposure(exp_s)
            self._syncing_cam_widgets = True
            self.spin_exposure_ms.setValue(actual_s * 1000.0)
            self._syncing_cam_widgets = False
            self._update_camera_ui()
        except Exception as exc:
            self._threadsafe_log(f"Exposure error: {exc}")

    def _on_apply_frame_averaging(self, val: int):
        if self._syncing_cam_widgets:
            return
        self.camera.set_frame_averaging(int(val))
        self._update_camera_ui()

    def _on_apply_readout_speed(self):
        if self._syncing_cam_widgets or not self.camera.is_connected:
            return
        was_live = self._live_running
        if was_live:
            self._stop_live_if_running()
        try:
            speed = str(self.combo_readout_speed.currentData() or "fast")
            self.camera.set_readout_speed(speed)
            self._update_camera_ui()
        except Exception as exc:
            self._threadsafe_log(f"Readout speed error: {exc}")
        if was_live:
            self._on_toggle_live_preview()

    def _on_apply_defect_correction(self, checked: bool):
        if self._syncing_cam_widgets or not self.camera.is_connected:
            return
        was_live = self._live_running
        if was_live:
            self._stop_live_if_running()
        try:
            self.camera.set_defect_correct_mode(checked)
            self._update_camera_ui()
        except Exception as exc:
            self._threadsafe_log(f"Defect pixel correction error: {exc}")
        if was_live:
            self._on_toggle_live_preview()

    # ==========================================================================
    # Brightfield Auto-Calibrate & Tab / Dual-Viewer Coordination
    # ==========================================================================

    def _on_bf_auto_calibrate(self):
        if not self.brightfield_cam.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the Brightfield camera first.")
            return
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the Plate stage first.")
            return
        if self._plate_calib_worker is not None and self._plate_calib_worker.isRunning():
            return
        self._stop_live_if_running()
        step_um = float(self.spin_bf_calib_step.value())
        self.btn_bf_auto_calib.setEnabled(False)
        self._plate_calib_worker = PlateAutoCalibrateWorker(
            brightfield_cam=self.brightfield_cam,
            stage=self.stage,
            step_um=step_um,
            parent=self,
        )
        self._plate_calib_worker.sig_log.connect(self._append_log_slot)
        self._plate_calib_worker.sig_stage_updated.connect(self._update_stage_ui)
        self._plate_calib_worker.sig_finished_calib.connect(self._on_bf_auto_calib_done)
        self._plate_calib_worker.sig_error.connect(self._on_worker_error)
        self._plate_calib_worker.finished.connect(
            lambda: self.btn_bf_auto_calib.setEnabled(
                self.brightfield_cam.is_connected and self.stage.is_connected
            )
        )
        self._plate_calib_worker.start()

    @QtCore.Slot(object)
    def _on_bf_auto_calib_done(self, calib: Dict[str, Any]):
        scale = calib.get("um_per_px", calib.get("scale_um_per_px"))
        if scale is not None:
            scale_f = float(scale)
            self.lbl_bf_calib_scale.setText(f"{scale_f:.4f} µm/px")
            self.image_viewer.set_brightfield_scale(scale_f)
            self._user_config["brightfield_um_per_px"] = scale_f
            self._user_config["brightfield_calib"] = {
                "method": calib.get("method", "auto_3pt_jog"),
                "um_per_px": scale_f,
                "um_per_px_x": float(calib["um_per_px_x"]) if "um_per_px_x" in calib else scale_f,
                "um_per_px_y": float(calib["um_per_px_y"]) if "um_per_px_y" in calib else scale_f,
                "rotation_deg": float(calib.get("rotation_deg", 0.0)),
                "P_px_per_um": calib["P_px_per_um"].tolist() if hasattr(calib.get("P_px_per_um"), "tolist") else None,
                "M_um_per_px": calib["M_um_per_px"].tolist() if hasattr(calib.get("M_um_per_px"), "tolist") else None,
            }
            _save_user_config(self._user_config)
        else:
            self.lbl_bf_calib_scale.setText("Calibrated")
        if hasattr(self, "pipeline_panel") and self.pipeline_panel.session.is_open:
            try:
                self.pipeline_panel.session.save_brightfield_stage_calibration(calib)
            except Exception as exc:
                self._threadsafe_log(f"Warning saving plate calibration to HDF5: {exc}")
        self._threadsafe_log(f"Brightfield plate auto-calibration complete: {self.lbl_bf_calib_scale.text()}")

    def _on_right_tab_changed(self, idx: int):
        widget = self.tabs_right.widget(idx)
        if widget is self.pipeline_panel and self.pipeline_panel.is_mapping_ready():
            self._enter_mapping_dual_brightfield()
        elif widget is not self.pipeline_panel and self.image_viewer._view_mode == "dual_brightfield":
            self.image_viewer._build_single_plot()

    def _on_mapping_ready_changed(self, ready: bool):
        if ready and self.tabs_right.currentWidget() is self.pipeline_panel:
            self._enter_mapping_dual_brightfield()

    def _enter_mapping_dual_brightfield(self):
        self.image_viewer.show_dual_brightfield_view()
        if self.brightfield_cam.is_connected and not (self._live_running and self._active_live_source == "teledyne"):
            self._start_quick_live("teledyne")

    @QtCore.Slot(object, str)
    def _on_set_bf_reference(self, img, title: str):
        self.image_viewer.set_dual_reference_image(img, ref_title=title)

    # ==========================================================================
    # Step 2: Stage Motion, Single Snap & Live Preview Actions
    # ==========================================================================

    def _sync_live_buttons(self):
        src = self._current_preview_source()
        active_style = "background-color: #ef6c00; color: white; font-weight: bold;"
        orca_live = self._live_running and (src == "orca")
        bf_live = self._live_running and (src == "teledyne")

        self.btn_quick_live_orca.setText("Spectrometer")
        self.btn_quick_live_orca.setStyleSheet(active_style if orca_live else "")
        self.btn_quick_live_orca.setEnabled(self.camera.is_connected and not orca_live)

        self.btn_quick_live_bf.setText("Brightfield")
        self.btn_quick_live_bf.setStyleSheet(active_style if bf_live else "")
        self.btn_quick_live_bf.setEnabled(self.brightfield_cam.is_connected and not bf_live)

        self.btn_stop_live.setEnabled(self._live_running)

    def _start_quick_live(self, source: str):
        already_running_this = self._live_running and (self._current_preview_source() == source)
        if already_running_this:
            return
        if self._live_running:
            self._stop_live_if_running()
        self._active_live_source = source
        self._on_toggle_live_preview()

    def _stop_live_if_running(self):
        if self._live_running and self._worker is not None and self._worker.isRunning():
            try:
                self._worker.sig_frame_acquired.disconnect(self._on_frame_acquired)
            except (TypeError, RuntimeError):
                pass
            try:
                self._worker.finished.disconnect(self._on_live_worker_finished)
            except (TypeError, RuntimeError):
                pass
            self._worker.request_stop()
            self._worker.wait(10000)
        self._live_running = False
        self._sync_live_buttons()

    def _current_preview_source(self) -> str:
        return self._active_live_source

    def _preview_camera_connected(self) -> bool:
        if self._current_preview_source() == "teledyne":
            return self.brightfield_cam.is_connected
        return self.camera.is_connected

    def _on_toggle_live_preview(self):
        if self._live_running:
            self._stop_live_if_running()
            self._set_motion_controls_enabled(True)
            return

        src = self._current_preview_source()
        if not self._preview_camera_connected():
            cam_label = "Teledyne Brightfield camera" if src == "teledyne" else "Orca camera"
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", f"Please connect the {cam_label} first.")
            return

        if self._worker is not None and self._worker.isRunning():
            return

        self._live_running = True
        self._sync_live_buttons()
        self._worker = StageActionWorker(
            camera=self.camera,
            brightfield_cam=self.brightfield_cam,
            stage=self.stage,
            action="live",
            params={"camera_source": src},
            parent=self,
        )
        self._worker.sig_log.connect(self._append_log_slot)
        self._worker.sig_frame_acquired.connect(self._on_frame_acquired)
        self._worker.sig_error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_live_worker_finished)
        self._worker.start()

    def _on_live_worker_finished(self):
        if self._worker is not None and self.sender() is not None and self.sender() is not self._worker:
            return
        self._live_running = False
        self._sync_live_buttons()
        map_running = (
            hasattr(self, "pipeline_panel")
            and self.pipeline_panel._map_worker is not None
            and self.pipeline_panel._map_worker.isRunning()
        )
        if not map_running:
            self._set_motion_controls_enabled(True)

    def _on_plate_speeds_changed(self):
        self._user_config["plate_fast_speed_um_s"] = float(self.spin_plate_fast_speed.value())
        self._user_config["plate_slow_speed_um_s"] = float(self.spin_plate_slow_speed.value())
        _save_user_config(self._user_config)
        self._update_jog_tooltips()

    def _update_jog_tooltips(self):
        if not hasattr(self, "btn_jog_yp_fast"):
            return
        v_fast = float(self.spin_plate_fast_speed.value())
        v_slow = float(self.spin_plate_slow_speed.value())
        self.btn_jog_yp_fast.setToolTip(f"Fast +Y ({v_fast:.1f} µm/s)")
        self.btn_jog_ym_fast.setToolTip(f"Fast -Y ({v_fast:.1f} µm/s)")
        self.btn_jog_xm_fast.setToolTip(f"Fast -X ({v_fast:.1f} µm/s)")
        self.btn_jog_xp_fast.setToolTip(f"Fast +X ({v_fast:.1f} µm/s)")
        self.btn_jog_yp_slow.setToolTip(f"Slow +Y ({v_slow:.1f} µm/s)")
        self.btn_jog_ym_slow.setToolTip(f"Slow -Y ({v_slow:.1f} µm/s)")
        self.btn_jog_xm_slow.setToolTip(f"Slow -X ({v_slow:.1f} µm/s)")
        self.btn_jog_xp_slow.setToolTip(f"Slow +X ({v_slow:.1f} µm/s)")

    def _on_jog_hold_start(self, dx_sign: float, dy_sign: float, is_fast: bool):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the Plate stage first.")
            return
        speed = float(self.spin_plate_fast_speed.value() if is_fast else self.spin_plate_slow_speed.value())
        try:
            self.stage.start_continuous_move(vx_um_s=dx_sign * speed, vy_um_s=dy_sign * speed)
        except Exception as exc:
            self._append_log_slot(f"ERROR (Continuous Jog): {exc}")

    def _on_jog_hold_stop(self):
        if self.stage.is_connected:
            self.stage.stop_continuous_move()
        self._update_stage_ui()

    def _on_halt_stage(self):
        self.rig.emergency_halt()
        self._update_stage_ui()

    def _on_zero_stage(self, axes=("X", "Y")):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the ASI MS-2000 stage first.")
            return
        if self._worker is not None and self._worker.isRunning() and not self._live_running:
            self._threadsafe_log("Cannot zero stage while a move/acquisition action is running.")
            return
        try:
            self.stage.zero_position(axes=axes)
            self._update_stage_ui()
        except Exception as exc:
            self._append_log_slot(f"ERROR (Zero Stage): {exc}")
            QtWidgets.QMessageBox.warning(self, "Zero Stage Error", str(exc))

    @QtCore.Slot(object, object, float, int, int)
    def _on_frame_acquired(self, frame, pos, exposure_s: float, n_avg: int, binning: int):
        # Coalesce rapid live frames so only the newest frame is rendered per GUI cycle.
        self._pending_frame = (frame, pos, exposure_s, n_avg, binning)
        if not self._frame_flush_scheduled:
            self._frame_flush_scheduled = True
            QtCore.QTimer.singleShot(0, self._flush_pending_frame)

    def _flush_pending_frame(self):
        self._frame_flush_scheduled = False
        payload = self._pending_frame
        self._pending_frame = None
        if payload is None:
            return
        frame, pos, exposure_s, n_avg, binning = payload
        if (
            self.image_viewer._view_mode == "dual_brightfield"
            and self._live_running
            and self._active_live_source == "teledyne"
        ):
            self.image_viewer.set_dual_live_image(frame)
            return
        is_bf = bool(self._live_running and self._active_live_source == "teledyne")
        if not is_bf and self.brightfield_cam.is_connected and hasattr(frame, "shape"):
            bf_shape = tuple(self.brightfield_cam.output_shape)
            cam_shape = tuple(self.camera.output_shape) if self.camera.is_connected else None
            if tuple(frame.shape[:2]) == bf_shape and tuple(frame.shape[:2]) != cam_shape:
                is_bf = True

        self.image_viewer.update_image(
            frame,
            stage_pos_um=pos,
            exposure_s=exposure_s,
            n_avg=n_avg,
            binning=binning,
            is_brightfield=is_bf,
        )

    @QtCore.Slot(str, str)
    def _on_worker_error(self, title: str, message: str):
        self._append_log_slot(f"ERROR: {message}")
        QtWidgets.QMessageBox.warning(self, title, message)

    # ==========================================================================
    # UI State Updates
    # ==========================================================================

    def _set_motion_controls_enabled(self, enabled: bool):
        cam_ok = self.camera.is_connected and enabled
        stage_ok = self.stage.is_connected and enabled
        has_z = self.stage.has_z_axis and stage_ok

        self._sync_live_buttons()
        self.spin_exposure_ms.setEnabled(cam_ok or self._live_running)
        self.spin_frame_avg.setEnabled(cam_ok)
        self.combo_readout_speed.setEnabled(cam_ok)
        self.chk_defect_correct.setEnabled(cam_ok)

        for b in (
            self.btn_jog_yp_fast,
            self.btn_jog_yp_slow,
            self.btn_jog_ym_slow,
            self.btn_jog_ym_fast,
            self.btn_jog_xm_fast,
            self.btn_jog_xm_slow,
            self.btn_jog_xp_slow,
            self.btn_jog_xp_fast,
            self.btn_stage_zero_xy,
            self.btn_zero_xy_nav,
        ):
            b.setEnabled(stage_ok)
        self.btn_stage_zero_z.setEnabled(has_z)
        self.btn_bf_auto_calib.setEnabled(self.brightfield_cam.is_connected and stage_ok)

    def _set_badge_style(self, label: QtWidgets.QLabel, connected: bool, simulated: bool):
        if not connected:
            label.setText("Unconnected")
            label.setStyleSheet(
                "background-color: #c62828; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 3px;"
            )
        elif simulated:
            label.setText("Connected (Simulation)")
            label.setStyleSheet(
                "background-color: #2e7d32; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 3px;"
            )
        else:
            label.setText("Connected")
            label.setStyleSheet(
                "background-color: #2e7d32; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 3px;"
            )

    def _update_camera_ui(self):
        try:
            st = self.camera.get_status()
        except Exception as exc:
            self._threadsafe_log(f"Camera status error: {exc}")
            return

        connected = bool(st["connected"])
        simulated = bool(st["simulate"])
        self._set_badge_style(self.lbl_cam_status, connected, simulated)

        self.btn_cam_connect.setText("Disconnect" if connected else "Connect Camera")
        self.btn_cam_refresh.setEnabled(connected)
        self.chk_cam_sim.setEnabled(not connected)
        self.spin_cam_idx.setEnabled(not connected)

        if connected:
            self.lbl_cam_model.setText(f"{st['model']} (S/N: {st['serial_number']})")
            roi = st["roi"]
            out_h, out_w = st["output_shape"]
            self.lbl_cam_sensor.setText(
                f"{out_w}×{out_h} px (ROI [{roi[0]}:{roi[1]}, {roi[2]}:{roi[3]}], Bin {roi[4]}×{roi[5]})"
            )
            exp_ms = (st["exposure_s"] or 0.0) * 1e3
            n_avg = st["frame_averaging"]
            temp = st["temperature_c"]
            temp_str = f"{temp:.1f} °C" if temp is not None else "N/A"
            self.lbl_cam_exposure.setText(f"{exp_ms:.2f} ms × {n_avg} avg  |  {st['readout_speed']}  |  {temp_str}")

            # Sync Live Preview controls with actual hardware values
            self._syncing_cam_widgets = True
            self.spin_exposure_ms.setValue(exp_ms)
            self.spin_frame_avg.setValue(int(n_avg))
            idx_spd = self.combo_readout_speed.findData(st["readout_speed"])
            if idx_spd >= 0:
                self.combo_readout_speed.setCurrentIndex(idx_spd)
            if st["defect_correct"] is not None:
                self.chk_defect_correct.setChecked(bool(st["defect_correct"]))
            self._syncing_cam_widgets = False
        else:
            self.lbl_cam_model.setText("—")
            self.lbl_cam_sensor.setText("—")
            self.lbl_cam_exposure.setText("—")

        busy = bool(self._worker and self._worker.isRunning() and not self._live_running)
        self._set_motion_controls_enabled(not busy)
        if hasattr(self, "pipeline_panel"):
            self.pipeline_panel._update_map_summary()

    def _update_stage_ui(self):
        try:
            st = self.stage.get_status()
        except Exception as exc:
            self._threadsafe_log(f"Stage status error: {exc}")
            return

        connected = bool(st["connected"])
        simulated = bool(st["simulate"])
        self._set_badge_style(self.lbl_stage_status, connected, simulated)

        self.btn_stage_connect.setText("Disconnect" if connected else "Connect Stage")
        self.btn_stage_refresh.setEnabled(connected)
        self.chk_stage_sim.setEnabled(not connected)
        self.combo_stage_port.setEnabled(not connected)
        self.combo_stage_baud.setEnabled(not connected)

        if connected:
            self.lbl_stage_fw.setText(f"{st['port']} | {st['firmware']}")
            pos = st["position_um"]
            self.lbl_stage_pos_xy.setText(f"X = {pos['X']:.2f} µm,  Y = {pos['Y']:.2f} µm")
            self.lbl_stage_pos_z.setText(f"Z = {pos['Z']:.2f} µm" if pos["Z"] is not None else "Not present")
        else:
            self.lbl_stage_fw.setText("—")
            self.lbl_stage_pos_xy.setText("—")
            self.lbl_stage_pos_z.setText("—")

        busy = bool(self._worker and self._worker.isRunning() and not self._live_running)
        self._set_motion_controls_enabled(not busy)
        if hasattr(self, "pipeline_panel"):
            self.pipeline_panel._update_map_summary()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Ensure hardware connections are cleanly closed when exiting the GUI."""
        self._on_disconnect_all()
        super().closeEvent(event)
