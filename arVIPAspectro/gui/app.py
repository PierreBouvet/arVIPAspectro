"""
Step 1 & Step 2 GUI:
  - Step 1: Hardware Connection & Status Panel (Hamamatsu Orca Flash 4.0 v3 & ASI MS-2000)
  - Step 2: Stage Movement (Jog / Absolute Go-To) + Automatic Image Acquisition & Live Display
"""

import time
import datetime
from typing import Optional, Dict, Any
from PySide6 import QtWidgets, QtCore, QtGui

from ..hardware import (
    MS2000Stage,
    MS2000Error,
    list_serial_ports,
    OrcaFlashCamera,
    OrcaCameraError,
    count_dcam_cameras,
)
from .image_viewer import ImageViewerWidget


class HardwareActionWorker(QtCore.QThread):
    """
    Background worker thread that executes stage movement and/or camera frame acquisition
    without blocking the GUI event loop.
    """

    sig_log = QtCore.Signal(str)
    sig_stage_updated = QtCore.Signal()
    sig_frame_acquired = QtCore.Signal(object, object, float)  # (np.ndarray, pos_dict, exposure_s)
    sig_error = QtCore.Signal(str, str)                        # (title, message)

    def __init__(
        self,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.camera = camera
        self.stage = stage
        self.action = action  # 'snap', 'move_rel', 'move_abs'
        self.params = params or {}

    def run(self):
        try:
            pos = None
            # 1. Execute stage movement if requested
            if self.action == "move_rel":
                dx = float(self.params.get("dx_um", 0.0))
                dy = float(self.params.get("dy_um", 0.0))
                dz = float(self.params.get("dz_um", 0.0))
                pos = self.stage.move_relative(dx_um=dx, dy_um=dy, dz_um=dz, wait=True)
                self.sig_stage_updated.emit()

                settling_s = float(self.params.get("settling_ms", 50.0)) / 1000.0
                if settling_s > 0:
                    time.sleep(settling_s)

            elif self.action == "move_abs":
                x_um = self.params.get("x_um")
                y_um = self.params.get("y_um")
                z_um = self.params.get("z_um")
                pos = self.stage.move_to(x_um=x_um, y_um=y_um, z_um=z_um, wait=True)
                self.sig_stage_updated.emit()

                settling_s = float(self.params.get("settling_ms", 50.0)) / 1000.0
                if settling_s > 0:
                    time.sleep(settling_s)

            # 2. Acquire image if requested ('snap' or auto_snap after move)
            should_snap = (self.action == "snap") or bool(self.params.get("auto_snap", True))
            if should_snap:
                if not self.camera.is_connected:
                    if self.action == "snap":
                        raise OrcaCameraError("Camera is not connected.")
                    self.sig_log.emit("Stage move finished (camera not connected -> skipped auto-snap).")
                    return

                if pos is None and self.stage.is_connected:
                    try:
                        pos = self.stage.get_position()
                    except Exception:
                        pos = None

                exp_s = self.camera.get_exposure()
                frame = self.camera.snap_frame(stage_pos_um=pos)
                self.sig_frame_acquired.emit(frame, pos, exp_s)

        except Exception as exc:
            self.sig_error.emit("Hardware Action Error", str(exc))


class MainWindow(QtWidgets.QMainWindow):
    """Step 1 & Step 2 Application Window: Hardware Connection, Stage Motion & Image Capture."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "arVIPAspectro — Step 2: Stage Motion & Image Acquisition (Orca Flash 4.0 v3 & ASI MS-2000)"
        )
        self.resize(1360, 820)

        self._log_mutex = QtCore.QMutex()
        self._worker: Optional[HardwareActionWorker] = None

        # Hardware driver instances
        self.camera = OrcaFlashCamera(logger=self._threadsafe_log)
        self.stage = MS2000Stage(logger=self._threadsafe_log)

        self._build_ui()
        self._refresh_serial_ports()
        self._scan_dcam_cameras()
        self._update_camera_ui()
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
        title_lbl = QtWidgets.QLabel("Step 2 — Hardware Connection, Stage Motion & Frame Capture")
        title_font = title_lbl.font()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title_lbl.setFont(title_font)

        self.btn_connect_all = QtWidgets.QPushButton("Connect Both (Camera + Stage)")
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

        # Horizontal Splitter: Left = 2D Image Viewer, Right = Controls & Log
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # --- LEFT PANEL: 2D Camera Image Viewer ---
        viewer_box = QtWidgets.QGroupBox("Acquired Camera Frame (Hamamatsu Orca Flash 4.0 v3)")
        viewer_layout = QtWidgets.QVBoxLayout(viewer_box)
        viewer_layout.setContentsMargins(6, 6, 6, 6)
        self.image_viewer = ImageViewerWidget(self)
        viewer_layout.addWidget(self.image_viewer)
        splitter.addWidget(viewer_box)

        # --- RIGHT PANEL: Step 1 Connections + Step 2 Stage & Snap Controls + Log ---
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(8)

        # Step 1: Side-by-side compact Camera & Stage Connection cards
        conn_row = QtWidgets.QHBoxLayout()
        conn_row.setSpacing(8)
        conn_row.addWidget(self._build_camera_card(), stretch=1)
        conn_row.addWidget(self._build_stage_card(), stretch=1)
        right_layout.addLayout(conn_row)

        # Step 2: Stage Motion & Image Capture Card
        right_layout.addWidget(self._build_step2_motion_and_capture_card())

        # Bottom Hardware Log
        log_group = QtWidgets.QGroupBox("Hardware Communication Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 6, 6, 6)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QtGui.QFont("Courier", 10))
        self.txt_log.setMinimumHeight(110)

        btn_clear_log = QtWidgets.QPushButton("Clear Log")
        btn_clear_log.setFixedWidth(85)
        btn_clear_log.clicked.connect(self.txt_log.clear)

        log_hdr = QtWidgets.QHBoxLayout()
        log_hdr.addStretch()
        log_hdr.addWidget(btn_clear_log)
        log_layout.addWidget(self.txt_log)
        log_layout.addLayout(log_hdr)
        right_layout.addWidget(log_group, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 5)

    def _build_camera_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("1A. Camera — Orca Flash 4.0 v3")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

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
        info_form.addRow("Sensor:", self.lbl_cam_sensor)
        info_form.addRow("Exp / T:", self.lbl_cam_exposure)
        layout.addLayout(info_form)
        return box

    def _build_stage_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("1B. Stage — ASI MS-2000")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

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
        btn_row.addWidget(self.btn_stage_connect)
        btn_row.addWidget(self.btn_stage_refresh)
        layout.addLayout(btn_row)

        info_form = QtWidgets.QFormLayout()
        self.lbl_stage_fw = QtWidgets.QLabel("—")
        self.lbl_stage_pos_xy = QtWidgets.QLabel("—")
        self.lbl_stage_pos_z = QtWidgets.QLabel("—")
        info_form.addRow("Info:", self.lbl_stage_fw)
        info_form.addRow("X / Y:", self.lbl_stage_pos_xy)
        info_form.addRow("Z:", self.lbl_stage_pos_z)
        layout.addLayout(info_form)
        return box

    def _build_step2_motion_and_capture_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Step 2 — Stage Navigation & Automatic Image Capture After Move")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(10)

        # A. Capture trigger & settling time row
        snap_row = QtWidgets.QHBoxLayout()
        self.chk_auto_snap = QtWidgets.QCheckBox("Auto-acquire image after every move")
        self.chk_auto_snap.setChecked(True)

        snap_row.addWidget(self.chk_auto_snap)
        snap_row.addSpacing(12)
        snap_row.addWidget(QtWidgets.QLabel("Settling delay:"))
        self.spin_settling_ms = QtWidgets.QSpinBox()
        self.spin_settling_ms.setRange(0, 5000)
        self.spin_settling_ms.setValue(50)
        self.spin_settling_ms.setSuffix(" ms")
        snap_row.addWidget(self.spin_settling_ms)
        snap_row.addStretch()

        self.btn_snap_now = QtWidgets.QPushButton("Snap Single Image Now")
        self.btn_snap_now.setMinimumHeight(30)
        self.btn_snap_now.setStyleSheet("font-weight: bold;")
        self.btn_snap_now.clicked.connect(self._on_snap_now)
        snap_row.addWidget(self.btn_snap_now)
        layout.addLayout(snap_row)

        # Separator
        line1 = QtWidgets.QFrame()
        line1.setFrameShape(QtWidgets.QFrame.HLine)
        line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line1)

        # B. Relative Jog Pad + Step Size Presets
        jog_outer = QtWidgets.QHBoxLayout()
        jog_outer.setSpacing(16)

        # Left side of Jog section: Step Size & Presets
        step_col = QtWidgets.QVBoxLayout()
        step_form = QtWidgets.QFormLayout()
        self.spin_jog_step = QtWidgets.QDoubleSpinBox()
        self.spin_jog_step.setDecimals(2)
        self.spin_jog_step.setRange(0.1, 50000.0)
        self.spin_jog_step.setValue(50.0)
        self.spin_jog_step.setSuffix(" µm")
        step_form.addRow("Jog Step (Δ):", self.spin_jog_step)
        step_col.addLayout(step_form)

        preset_row = QtWidgets.QHBoxLayout()
        for val in (1.0, 10.0, 50.0, 100.0, 500.0):
            btn_p = QtWidgets.QPushButton(f"{val:g} µm")
            btn_p.setFixedWidth(56)
            btn_p.clicked.connect(lambda checked=False, v=val: self.spin_jog_step.setValue(v))
            preset_row.addWidget(btn_p)
        step_col.addLayout(preset_row)

        self.btn_halt_stage = QtWidgets.QPushButton("HALT / Stop Stage")
        self.btn_halt_stage.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.btn_halt_stage.clicked.connect(self._on_halt_stage)
        step_col.addWidget(self.btn_halt_stage)
        step_col.addStretch()
        jog_outer.addLayout(step_col)

        # Right side of Jog section: 2D XY Directional Pad + Z buttons
        pad_grid = QtWidgets.QGridLayout()
        pad_grid.setSpacing(4)

        self.btn_jog_yp = QtWidgets.QPushButton("▲ +Y")
        self.btn_jog_ym = QtWidgets.QPushButton("▼ -Y")
        self.btn_jog_xm = QtWidgets.QPushButton("◀ -X")
        self.btn_jog_xp = QtWidgets.QPushButton("▶ +X")
        self.btn_jog_zp = QtWidgets.QPushButton("▲ +Z")
        self.btn_jog_zm = QtWidgets.QPushButton("▼ -Z")

        for b in (self.btn_jog_yp, self.btn_jog_ym, self.btn_jog_xm, self.btn_jog_xp, self.btn_jog_zp, self.btn_jog_zm):
            b.setMinimumSize(68, 30)

        self.btn_jog_yp.clicked.connect(lambda: self._on_jog(0.0, +1.0, 0.0))
        self.btn_jog_ym.clicked.connect(lambda: self._on_jog(0.0, -1.0, 0.0))
        self.btn_jog_xm.clicked.connect(lambda: self._on_jog(-1.0, 0.0, 0.0))
        self.btn_jog_xp.clicked.connect(lambda: self._on_jog(+1.0, 0.0, 0.0))
        self.btn_jog_zp.clicked.connect(lambda: self._on_jog(0.0, 0.0, +1.0))
        self.btn_jog_zm.clicked.connect(lambda: self._on_jog(0.0, 0.0, -1.0))

        pad_grid.addWidget(self.btn_jog_yp, 0, 1)
        pad_grid.addWidget(self.btn_jog_xm, 1, 0)
        pad_grid.addWidget(self.btn_jog_xp, 1, 2)
        pad_grid.addWidget(self.btn_jog_ym, 2, 1)
        pad_grid.addWidget(self.btn_jog_zp, 0, 3)
        pad_grid.addWidget(self.btn_jog_zm, 2, 3)

        jog_outer.addLayout(pad_grid)
        layout.addLayout(jog_outer)

        # Separator
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line2)

        # C. Absolute Move (Go To X, Y)
        abs_row = QtWidgets.QHBoxLayout()
        abs_row.addWidget(QtWidgets.QLabel("Go To X:"))
        self.spin_target_x = QtWidgets.QDoubleSpinBox()
        self.spin_target_x.setDecimals(2)
        self.spin_target_x.setRange(-100000.0, 100000.0)
        self.spin_target_x.setSuffix(" µm")
        abs_row.addWidget(self.spin_target_x)

        abs_row.addWidget(QtWidgets.QLabel("Y:"))
        self.spin_target_y = QtWidgets.QDoubleSpinBox()
        self.spin_target_y.setDecimals(2)
        self.spin_target_y.setRange(-100000.0, 100000.0)
        self.spin_target_y.setSuffix(" µm")
        abs_row.addWidget(self.spin_target_y)

        self.btn_copy_pos = QtWidgets.QPushButton("Use Current")
        self.btn_copy_pos.setToolTip("Copy current stage X, Y into target fields")
        self.btn_copy_pos.clicked.connect(self._on_copy_current_pos)
        abs_row.addWidget(self.btn_copy_pos)

        self.btn_move_abs = QtWidgets.QPushButton("Move to (X, Y)")
        self.btn_move_abs.setStyleSheet("font-weight: bold;")
        self.btn_move_abs.clicked.connect(self._on_move_absolute)
        abs_row.addWidget(self.btn_move_abs)
        layout.addLayout(abs_row)

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

    def _refresh_serial_ports(self):
        self.combo_stage_port.clear()
        self.combo_stage_port.addItem("auto (Scan COM)", "auto")
        ports = list_serial_ports()
        for p in ports:
            label = f"{p['port']} — {p['description']}"
            self.combo_stage_port.addItem(label, p["port"])
        self._threadsafe_log(f"Serial scan: {len(ports)} COM port(s) found.")

    def _on_toggle_camera(self):
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
                self._on_copy_current_pos()
            except MS2000Error as exc:
                self._threadsafe_log(f"ERROR (Stage): {exc}")
                QtWidgets.QMessageBox.warning(self, "Stage Connection Error", str(exc))
        self._update_stage_ui()

    def _on_connect_all(self):
        if not self.camera.is_connected:
            self._on_toggle_camera()
        if not self.stage.is_connected:
            self._on_toggle_stage()

    def _on_disconnect_all(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2000)
        if self.camera.is_connected:
            self.camera.disconnect()
            self._update_camera_ui()
        if self.stage.is_connected:
            self.stage.disconnect()
            self._update_stage_ui()

    # ==========================================================================
    # Step 2: Stage Motion & Image Capture Actions
    # ==========================================================================

    def _is_worker_busy(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            self._threadsafe_log("Hardware action already in progress; please wait...")
            return True
        return False

    def _start_hardware_worker(self, action: str, params: Optional[Dict[str, Any]] = None):
        if self._is_worker_busy():
            return
        self._set_motion_controls_enabled(False)
        self._worker = HardwareActionWorker(
            camera=self.camera,
            stage=self.stage,
            action=action,
            params=params,
            parent=self,
        )
        self._worker.sig_log.connect(self._append_log_slot)
        self._worker.sig_stage_updated.connect(self._update_stage_ui)
        self._worker.sig_frame_acquired.connect(self._on_frame_acquired)
        self._worker.sig_error.connect(self._on_worker_error)
        self._worker.finished.connect(lambda: self._set_motion_controls_enabled(True))
        self._worker.start()

    def _on_snap_now(self):
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the camera first.")
            return
        self._start_hardware_worker(action="snap")

    def _on_jog(self, dir_x: float, dir_y: float, dir_z: float):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the ASI MS-2000 stage first.")
            return
        step = float(self.spin_jog_step.value())
        params = {
            "dx_um": dir_x * step,
            "dy_um": dir_y * step,
            "dz_um": dir_z * step,
            "settling_ms": self.spin_settling_ms.value(),
            "auto_snap": self.chk_auto_snap.isChecked(),
        }
        self._start_hardware_worker(action="move_rel", params=params)

    def _on_move_absolute(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Please connect the ASI MS-2000 stage first.")
            return
        params = {
            "x_um": float(self.spin_target_x.value()),
            "y_um": float(self.spin_target_y.value()),
            "z_um": None,
            "settling_ms": self.spin_settling_ms.value(),
            "auto_snap": self.chk_auto_snap.isChecked(),
        }
        self._start_hardware_worker(action="move_abs", params=params)

    def _on_copy_current_pos(self):
        if not self.stage.is_connected:
            return
        try:
            pos = self.stage.get_position()
            if pos.get("X") is not None:
                self.spin_target_x.setValue(float(pos["X"]))
            if pos.get("Y") is not None:
                self.spin_target_y.setValue(float(pos["Y"]))
        except Exception:
            pass

    def _on_halt_stage(self):
        if self.stage.is_connected:
            self.stage.halt()
            self._update_stage_ui()

    @QtCore.Slot(object, object, float)
    def _on_frame_acquired(self, frame, pos, exposure_s: float):
        self.image_viewer.update_image(frame, stage_pos_um=pos, exposure_s=exposure_s)

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

        self.btn_snap_now.setEnabled(cam_ok)
        for b in (self.btn_jog_xp, self.btn_jog_xm, self.btn_jog_yp, self.btn_jog_ym, self.btn_move_abs, self.btn_copy_pos):
            b.setEnabled(stage_ok)
        self.btn_jog_zp.setEnabled(has_z)
        self.btn_jog_zm.setEnabled(has_z)

    def _set_badge_style(self, label: QtWidgets.QLabel, connected: bool, simulated: bool):
        if not connected:
            label.setText("DISCONNECTED")
            label.setStyleSheet(
                "background-color: #424242; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 3px;"
            )
        elif simulated:
            label.setText("CONNECTED (SIMULATION)")
            label.setStyleSheet(
                "background-color: #ef6c00; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 3px;"
            )
        else:
            label.setText("CONNECTED (HARDWARE)")
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
            det_w, det_h = st["detector_size"]
            roi = st["roi"]
            self.lbl_cam_sensor.setText(f"{det_w}×{det_h} px (Bin {roi[4]}×{roi[5]})")
            exp_ms = (st["exposure_s"] or 0.0) * 1e3
            temp = st["temperature_c"]
            temp_str = f"{temp:.1f} °C" if temp is not None else "N/A"
            self.lbl_cam_exposure.setText(f"{exp_ms:.2f} ms  |  {temp_str}")
        else:
            self.lbl_cam_model.setText("—")
            self.lbl_cam_sensor.setText("—")
            self.lbl_cam_exposure.setText("—")

        self._set_motion_controls_enabled(not (self._worker and self._worker.isRunning()))

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

        self._set_motion_controls_enabled(not (self._worker and self._worker.isRunning()))

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Ensure hardware connections are cleanly closed when exiting the GUI."""
        self._on_disconnect_all()
        super().closeEvent(event)
