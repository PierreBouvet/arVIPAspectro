"""
Step 1 GUI: Hardware Connection & Status Panel for
  - Hamamatsu Orca Flash 4.0 v3 sCMOS Camera (DCAM-API / `pylablib`)
  - ASI MS-2000 Motorized XY(Z) Stage (`pyserial`)
"""

import datetime
from PySide6 import QtWidgets, QtCore, QtGui

from ..hardware import (
    MS2000Stage,
    MS2000Error,
    list_serial_ports,
    OrcaFlashCamera,
    OrcaCameraError,
    count_dcam_cameras,
)


class MainWindow(QtWidgets.QMainWindow):
    """Step 1 Application Window: Hardware Connection & Verification."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("arVIPAspectro — Step 1: Hardware Connection (Orca Flash 4.0 v3 & ASI MS-2000)")
        self.resize(920, 620)

        # Hardware driver instances
        self.camera = OrcaFlashCamera(logger=self._append_log)
        self.stage = MS2000Stage(logger=self._append_log)

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
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # 1. Top Toolbar: Global Connect / Disconnect
        top_bar = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel("Step 1 — Hardware Connection & Diagnostics")
        title_font = title_lbl.font()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title_lbl.setFont(title_font)

        self.btn_connect_all = QtWidgets.QPushButton("Connect Both (Camera + Stage)")
        self.btn_connect_all.setMinimumHeight(32)
        self.btn_connect_all.clicked.connect(self._on_connect_all)

        self.btn_disconnect_all = QtWidgets.QPushButton("Disconnect All")
        self.btn_disconnect_all.setMinimumHeight(32)
        self.btn_disconnect_all.clicked.connect(self._on_disconnect_all)

        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_connect_all)
        top_bar.addWidget(self.btn_disconnect_all)
        root_layout.addLayout(top_bar)

        # 2. Middle Section: Side-by-Side Camera & Stage Cards
        cards_layout = QtWidgets.QHBoxLayout()
        cards_layout.setSpacing(12)
        cards_layout.addWidget(self._build_camera_card(), stretch=1)
        cards_layout.addWidget(self._build_stage_card(), stretch=1)
        root_layout.addLayout(cards_layout, stretch=1)

        # 3. Bottom Section: Hardware Communication Log
        log_group = QtWidgets.QGroupBox("Hardware Communication Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QtGui.QFont("Courier", 10))
        self.txt_log.setMinimumHeight(150)

        btn_clear_log = QtWidgets.QPushButton("Clear Log")
        btn_clear_log.setFixedWidth(90)
        btn_clear_log.clicked.connect(self.txt_log.clear)

        log_header = QtWidgets.QHBoxLayout()
        log_header.addStretch()
        log_header.addWidget(btn_clear_log)

        log_layout.addWidget(self.txt_log)
        log_layout.addLayout(log_header)
        root_layout.addWidget(log_group)

    def _build_camera_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Camera — Hamamatsu Orca Flash 4.0 v3")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(8)

        # Status badge
        self.lbl_cam_status = QtWidgets.QLabel("DISCONNECTED")
        self.lbl_cam_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_cam_status.setMinimumHeight(28)
        layout.addWidget(self.lbl_cam_status)

        # Connection settings form
        form = QtWidgets.QFormLayout()
        self.chk_cam_sim = QtWidgets.QCheckBox("Simulation / Mock Mode (offline testing)")
        form.addRow("Mode:", self.chk_cam_sim)

        cam_idx_row = QtWidgets.QHBoxLayout()
        self.spin_cam_idx = QtWidgets.QSpinBox()
        self.spin_cam_idx.setRange(0, 7)
        self.spin_cam_idx.setValue(0)
        self.btn_scan_dcam = QtWidgets.QPushButton("Scan DCAM")
        self.btn_scan_dcam.clicked.connect(self._scan_dcam_cameras)
        self.lbl_dcam_count = QtWidgets.QLabel("0 detected")
        cam_idx_row.addWidget(self.spin_cam_idx)
        cam_idx_row.addWidget(self.btn_scan_dcam)
        cam_idx_row.addWidget(self.lbl_dcam_count)
        form.addRow("Camera Index:", cam_idx_row)
        layout.addLayout(form)

        # Action buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_cam_connect = QtWidgets.QPushButton("Connect Camera")
        self.btn_cam_connect.clicked.connect(self._on_toggle_camera)
        self.btn_cam_refresh = QtWidgets.QPushButton("Query Status")
        self.btn_cam_refresh.clicked.connect(self._update_camera_ui)
        btn_row.addWidget(self.btn_cam_connect)
        btn_row.addWidget(self.btn_cam_refresh)
        layout.addLayout(btn_row)

        # Readout fields
        info_form = QtWidgets.QFormLayout()
        self.lbl_cam_model = QtWidgets.QLabel("—")
        self.lbl_cam_serial = QtWidgets.QLabel("—")
        self.lbl_cam_sensor = QtWidgets.QLabel("—")
        self.lbl_cam_exposure = QtWidgets.QLabel("—")
        self.lbl_cam_temp = QtWidgets.QLabel("—")

        info_form.addRow("Model:", self.lbl_cam_model)
        info_form.addRow("Serial / FW:", self.lbl_cam_serial)
        info_form.addRow("Sensor / ROI:", self.lbl_cam_sensor)
        info_form.addRow("Exposure:", self.lbl_cam_exposure)
        info_form.addRow("Temperature:", self.lbl_cam_temp)
        layout.addLayout(info_form)
        layout.addStretch()
        return box

    def _build_stage_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Motorized Stage — ASI MS-2000")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(8)

        # Status badge
        self.lbl_stage_status = QtWidgets.QLabel("DISCONNECTED")
        self.lbl_stage_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_stage_status.setMinimumHeight(28)
        layout.addWidget(self.lbl_stage_status)

        # Connection settings form
        form = QtWidgets.QFormLayout()
        self.chk_stage_sim = QtWidgets.QCheckBox("Simulation / Mock Mode (offline testing)")
        form.addRow("Mode:", self.chk_stage_sim)

        port_row = QtWidgets.QHBoxLayout()
        self.combo_stage_port = QtWidgets.QComboBox()
        self.combo_stage_port.setEditable(True)
        self.btn_refresh_ports = QtWidgets.QPushButton("Refresh Ports")
        self.btn_refresh_ports.clicked.connect(self._refresh_serial_ports)
        port_row.addWidget(self.combo_stage_port, stretch=1)
        port_row.addWidget(self.btn_refresh_ports)
        form.addRow("Serial COM Port:", port_row)

        self.combo_stage_baud = QtWidgets.QComboBox()
        self.combo_stage_baud.addItem("115200 (USB Default)", 115200)
        self.combo_stage_baud.addItem("9600 (RS-232 Default)", 9600)
        self.combo_stage_baud.addItem("28800", 28800)
        form.addRow("Baudrate:", self.combo_stage_baud)
        layout.addLayout(form)

        # Action buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_stage_connect = QtWidgets.QPushButton("Connect Stage")
        self.btn_stage_connect.clicked.connect(self._on_toggle_stage)
        self.btn_stage_refresh = QtWidgets.QPushButton("Read Position")
        self.btn_stage_refresh.clicked.connect(self._update_stage_ui)
        btn_row.addWidget(self.btn_stage_connect)
        btn_row.addWidget(self.btn_stage_refresh)
        layout.addLayout(btn_row)

        # Readout fields
        info_form = QtWidgets.QFormLayout()
        self.lbl_stage_port = QtWidgets.QLabel("—")
        self.lbl_stage_fw = QtWidgets.QLabel("—")
        self.lbl_stage_axes = QtWidgets.QLabel("—")
        self.lbl_stage_pos_xy = QtWidgets.QLabel("—")
        self.lbl_stage_pos_z = QtWidgets.QLabel("—")

        info_form.addRow("Active Port:", self.lbl_stage_port)
        info_form.addRow("Controller:", self.lbl_stage_fw)
        info_form.addRow("Axes / State:", self.lbl_stage_axes)
        info_form.addRow("X / Y Position:", self.lbl_stage_pos_xy)
        info_form.addRow("Z Position:", self.lbl_stage_pos_z)
        layout.addLayout(info_form)
        layout.addStretch()
        return box

    # ==========================================================================
    # Hardware Actions & Event Handlers
    # ==========================================================================

    def _append_log(self, message: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "txt_log"):
            self.txt_log.appendPlainText(f"[{ts}] {message}")

    def _scan_dcam_cameras(self):
        n_cams = count_dcam_cameras()
        self.lbl_dcam_count.setText(f"{n_cams} detected")
        self._append_log(f"DCAM scan: {n_cams} Hamamatsu camera(s) detected on system.")

    def _refresh_serial_ports(self):
        self.combo_stage_port.clear()
        self.combo_stage_port.addItem("auto (Scan COM Ports)", "auto")
        ports = list_serial_ports()
        for p in ports:
            label = f"{p['port']} — {p['description']}"
            self.combo_stage_port.addItem(label, p["port"])
        self._append_log(f"Serial scan: {len(ports)} COM port(s) found.")

    def _on_toggle_camera(self):
        if self.camera.is_connected:
            self.camera.disconnect()
        else:
            simulate = self.chk_cam_sim.isChecked()
            idx = int(self.spin_cam_idx.value())
            try:
                self.camera.connect(camera_idx=idx, simulate=simulate)
            except OrcaCameraError as exc:
                self._append_log(f"ERROR (Camera): {exc}")
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
                self._append_log(f"ERROR (Stage): {exc}")
                QtWidgets.QMessageBox.warning(self, "Stage Connection Error", str(exc))
        self._update_stage_ui()

    def _on_connect_all(self):
        if not self.camera.is_connected:
            self._on_toggle_camera()
        if not self.stage.is_connected:
            self._on_toggle_stage()

    def _on_disconnect_all(self):
        if self.camera.is_connected:
            self.camera.disconnect()
            self._update_camera_ui()
        if self.stage.is_connected:
            self.stage.disconnect()
            self._update_stage_ui()

    # ==========================================================================
    # UI State Updates
    # ==========================================================================

    def _set_badge_style(self, label: QtWidgets.QLabel, connected: bool, simulated: bool):
        if not connected:
            label.setText("DISCONNECTED")
            label.setStyleSheet(
                "background-color: #424242; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 4px;"
            )
        elif simulated:
            label.setText("CONNECTED (SIMULATION MODE)")
            label.setStyleSheet(
                "background-color: #ef6c00; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 4px;"
            )
        else:
            label.setText("CONNECTED (HARDWARE)")
            label.setStyleSheet(
                "background-color: #2e7d32; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 4px;"
            )

    def _update_camera_ui(self):
        try:
            st = self.camera.get_status()
        except Exception as exc:
            self._append_log(f"Camera status error: {exc}")
            return

        connected = bool(st["connected"])
        simulated = bool(st["simulate"])
        self._set_badge_style(self.lbl_cam_status, connected, simulated)

        self.btn_cam_connect.setText("Disconnect Camera" if connected else "Connect Camera")
        self.btn_cam_refresh.setEnabled(connected)
        self.chk_cam_sim.setEnabled(not connected)
        self.spin_cam_idx.setEnabled(not connected)

        if connected:
            self.lbl_cam_model.setText(f"{st['vendor']} {st['model']}")
            self.lbl_cam_serial.setText(f"S/N: {st['serial_number']}  |  Ver: {st['camera_version']}")
            det_w, det_h = st["detector_size"]
            roi = st["roi"]
            self.lbl_cam_sensor.setText(
                f"{det_w} × {det_h} px  (ROI: {roi[0]}..{roi[1]} × {roi[2]}..{roi[3]}, Bin {roi[4]}×{roi[5]})"
            )
            exp_ms = (st["exposure_s"] or 0.0) * 1e3
            self.lbl_cam_exposure.setText(f"{exp_ms:.2f} ms")
            temp = st["temperature_c"]
            self.lbl_cam_temp.setText(f"{temp:.1f} °C" if temp is not None else "N/A")
        else:
            self.lbl_cam_model.setText("—")
            self.lbl_cam_serial.setText("—")
            self.lbl_cam_sensor.setText("—")
            self.lbl_cam_exposure.setText("—")
            self.lbl_cam_temp.setText("—")

    def _update_stage_ui(self):
        try:
            st = self.stage.get_status()
        except Exception as exc:
            self._append_log(f"Stage status error: {exc}")
            return

        connected = bool(st["connected"])
        simulated = bool(st["simulate"])
        self._set_badge_style(self.lbl_stage_status, connected, simulated)

        self.btn_stage_connect.setText("Disconnect Stage" if connected else "Connect Stage")
        self.btn_stage_refresh.setEnabled(connected)
        self.chk_stage_sim.setEnabled(not connected)
        self.combo_stage_port.setEnabled(not connected)
        self.combo_stage_baud.setEnabled(not connected)

        if connected:
            self.lbl_stage_port.setText(f"{st['port']} @ {st['baudrate']} baud ({st['backend']})")
            self.lbl_stage_fw.setText(str(st["firmware"]))
            axes_str = "X, Y, Z" if st["has_z"] else "X, Y"
            state_str = "BUSY (Moving)" if st["busy"] else "IDLE (Ready)"
            self.lbl_stage_axes.setText(f"Axes: {axes_str}  |  State: {state_str}")
            pos = st["position_um"]
            self.lbl_stage_pos_xy.setText(f"X = {pos['X']:.2f} µm,   Y = {pos['Y']:.2f} µm")
            self.lbl_stage_pos_z.setText(f"Z = {pos['Z']:.2f} µm" if pos["Z"] is not None else "Not present")
        else:
            self.lbl_stage_port.setText("—")
            self.lbl_stage_fw.setText("—")
            self.lbl_stage_axes.setText("—")
            self.lbl_stage_pos_xy.setText("—")
            self.lbl_stage_pos_z.setText("—")

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Ensure hardware connections are cleanly closed when exiting the GUI."""
        self._on_disconnect_all()
        super().closeEvent(event)
