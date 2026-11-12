"""
app.py: Main Window and GUI Orchestrator for arVIPAspectro Suite.
"""

import os
import sys
from typing import Optional, Dict, Any, List
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

from .canvas import PlotVisualizer
from .h5_explorer import H5ExplorerWidget
from .step_widgets import (
    LaserCalibrationWidget,
    RbCalibrationWidget,
    PeakFittingWidget,
    WavevectorDispersionWidget,
)
from .workers import PipelineWorker
from .theme import DARK_QSS, LIGHT_QSS, get_system_theme


class MainWindow(QtWidgets.QMainWindow):
    """Main Application Window for arVIPAspectro."""

    def __init__(self, default_h5: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("arVIPAspectro — Angle-Resolved VIPA Brillouin Spectrometry Suite")
        self.resize(1400, 900)

        self.theme_mode: str = "system"  # 'system', 'dark', or 'light'
        self.current_theme: str = get_system_theme()
        self.current_worker: Optional[PipelineWorker] = None

        self._setup_ui()
        self._setup_menus()
        self.apply_theme(self.current_theme)

        # Connect system appearance day/night changes
        hints = QtGui.QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._on_system_color_scheme_changed)

        if default_h5 and os.path.exists(default_h5):
            self.h5_explorer.set_h5_file(default_h5)

    def _setup_ui(self):
        # Main central splitter (Horizontal: Left Visualizer | Right Controls)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # ----------------------------------------------------------------------
        # LEFT FRAME: Visualizer & Canvas
        # ----------------------------------------------------------------------
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # Left/Bottom Vertical Splitter (Canvas on top, Console on bottom)
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Graph Visualizer
        self.visualizer = PlotVisualizer(self)
        self.left_splitter.addWidget(self.visualizer)

        # Collapsible Bottom Console Drawer
        self.console_drawer = self._create_console_drawer()
        self.left_splitter.addWidget(self.console_drawer)
        self.left_splitter.setStretchFactor(0, 4)
        self.left_splitter.setStretchFactor(1, 1)

        left_layout.addWidget(self.left_splitter)
        self.main_splitter.addWidget(left_container)

        # ----------------------------------------------------------------------
        # RIGHT FRAME: Dual Tab (Tab 1: H5 Explorer, Tab 2: Processing Steps)
        # ----------------------------------------------------------------------
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.tabs_right = QtWidgets.QTabWidget()

        # Tab 1: HDF5 File Explorer
        self.h5_explorer = H5ExplorerWidget(self)
        self.h5_explorer.sig_dataset_preview.connect(self._on_h5_dataset_preview)
        self.h5_explorer.sig_sample_selected.connect(self._on_sample_selected_from_h5)
        self.h5_explorer.sig_laser_selected.connect(self._on_laser_selected_from_h5)
        self.h5_explorer.sig_rb_selected.connect(self._on_rb_selected_from_h5)
        self.h5_explorer.sig_file_loaded.connect(self._on_file_loaded)
        self.tabs_right.addTab(self.h5_explorer, "📁 HDF5 Data Explorer")

        # Tab 2: Processing & Calibration Steps
        steps_container = self._create_steps_container()
        self.tabs_right.addTab(steps_container, "⚙ Processing & Calibration Steps")

        right_layout.addWidget(self.tabs_right)
        self.main_splitter.addWidget(right_container)

        # Proportions: 60% Left, 40% Right
        self.main_splitter.setStretchFactor(0, 6)
        self.main_splitter.setStretchFactor(1, 4)

        # Status Bar
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load an HDF5 container to begin.")

    def _create_console_drawer(self) -> QtWidgets.QWidget:
        drawer = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(drawer)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Header bar
        h_bar = QtWidgets.QHBoxLayout()
        lbl_console = QtWidgets.QLabel("Execution Console & Logs")
        lbl_console.setStyleSheet("font-weight: bold; font-size: 11px; color: #aaaaaa;")

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.setStyleSheet("font-size: 10px; padding: 1px 6px;")
        btn_clear.clicked.connect(self._clear_console)

        self.btn_toggle_console = QtWidgets.QPushButton("Hide Logs")
        self.btn_toggle_console.setStyleSheet("font-size: 10px; padding: 1px 6px;")
        self.btn_toggle_console.clicked.connect(self._toggle_console)

        h_bar.addWidget(lbl_console)
        h_bar.addStretch()
        h_bar.addWidget(self.progress_bar)
        h_bar.addWidget(btn_clear)
        h_bar.addWidget(self.btn_toggle_console)
        layout.addLayout(h_bar)

        self.txt_console = QtWidgets.QPlainTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumBlockCount(2000)
        layout.addWidget(self.txt_console)

        return drawer

    def _create_steps_container(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(6)

        # Active Sample Selector Bar
        sample_bar = QtWidgets.QHBoxLayout()
        lbl_sample = QtWidgets.QLabel("Active Sample:")
        lbl_sample.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        self.combo_sample = QtWidgets.QComboBox()
        self.combo_sample.setMinimumWidth(180)
        self.combo_sample.currentTextChanged.connect(self._on_active_sample_changed)

        btn_refresh_samples = QtWidgets.QPushButton("↻")
        btn_refresh_samples.setToolTip("Refresh sample list from HDF5")
        btn_refresh_samples.setFixedWidth(28)
        btn_refresh_samples.clicked.connect(self._refresh_samples_list)

        sample_bar.addWidget(lbl_sample)
        sample_bar.addWidget(self.combo_sample, stretch=1)
        sample_bar.addWidget(btn_refresh_samples)
        layout.addLayout(sample_bar)

        # Scroll Area for Collapsible Sections
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(2, 2, 6, 2)
        scroll_layout.setSpacing(8)

        # Section 1: Laser Calibration
        self.sec_laser = LaserCalibrationWidget(self)
        self.sec_laser.btn_action.clicked.connect(self._run_laser_all)
        self.sec_laser.card_step2.btn_action.clicked.connect(self._run_laser_all)
        self.sec_laser.card_step3.btn_action.clicked.connect(self._run_laser_all)
        self.sec_laser.card_step4.btn_action.clicked.connect(self._run_laser_all)
        self.sec_laser.card_step5.btn_action.clicked.connect(self._run_laser_all)
        self.sec_laser.card_step6.btn_action.clicked.connect(self._run_laser_all)
        scroll_layout.addWidget(self.sec_laser)

        # Section 2: Rubidium Calibration
        self.sec_rb = RbCalibrationWidget(self)
        self.sec_rb.btn_action.clicked.connect(self._run_rb_all)
        self.sec_rb.card_step7.btn_action.clicked.connect(self._run_rb_all)
        self.sec_rb.card_step8.btn_action.clicked.connect(self._run_rb_all)
        self.sec_rb.card_step9.btn_action.clicked.connect(self._run_rb_all)
        self.sec_rb.card_step10a.btn_action.clicked.connect(self._run_rb_all)
        self.sec_rb.card_step10b.btn_action.clicked.connect(self._run_rb_all)
        scroll_layout.addWidget(self.sec_rb)

        # Section 3: Peak Fitting
        self.sec_peak = PeakFittingWidget(self)
        self.sec_peak.btn_action.clicked.connect(self._run_peak_fit)
        scroll_layout.addWidget(self.sec_peak)

        # Section 4: Wavevector & Dispersion
        self.sec_wavevector = WavevectorDispersionWidget(self)
        self.sec_wavevector.btn_action.clicked.connect(self._run_q_vector)
        scroll_layout.addWidget(self.sec_wavevector)

        self._cards = [
            self.sec_laser, self.sec_laser.card_step2, self.sec_laser.card_step3,
            self.sec_laser.card_step4, self.sec_laser.card_step5, self.sec_laser.card_step6,
            self.sec_rb, self.sec_rb.card_step7, self.sec_rb.card_step8,
            self.sec_rb.card_step9, self.sec_rb.card_step10a, self.sec_rb.card_step10b,
            self.sec_peak, self.sec_wavevector
        ]

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        return container

    def _setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        act_open = QtGui.QAction("Open HDF5 Measures File...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.h5_explorer.browse_h5_file)
        file_menu.addAction(act_open)

        act_convert = QtGui.QAction("Convert Folder of .dat to H5...", self)
        act_convert.triggered.connect(self.h5_explorer._on_convert_dat_folder)
        file_menu.addAction(act_convert)

        file_menu.addSeparator()
        act_exit = QtGui.QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        view_menu = menubar.addMenu("View")
        act_toggle_log = QtGui.QAction("Toggle Bottom Log Drawer", self)
        act_toggle_log.setShortcut("Ctrl+L")
        act_toggle_log.triggered.connect(self._toggle_console)
        view_menu.addAction(act_toggle_log)

        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme / Appearance")
        theme_group = QtGui.QActionGroup(self)

        act_auto = QtGui.QAction("Auto (Match System Day/Night)", self, checkable=True)
        act_auto.setChecked(True)
        act_auto.triggered.connect(lambda: self._set_theme_mode("system"))
        theme_group.addAction(act_auto)
        theme_menu.addAction(act_auto)

        act_light = QtGui.QAction("Light Mode (Day)", self, checkable=True)
        act_light.triggered.connect(lambda: self._set_theme_mode("light"))
        theme_group.addAction(act_light)
        theme_menu.addAction(act_light)

        act_dark = QtGui.QAction("Dark Mode (Night)", self, checkable=True)
        act_dark.triggered.connect(lambda: self._set_theme_mode("dark"))
        theme_group.addAction(act_dark)
        theme_menu.addAction(act_dark)

    # --------------------------------------------------------------------------
    # Theme Management
    # --------------------------------------------------------------------------

    def apply_theme(self, theme_name: str):
        """Applies dark or light theme across all widgets and canvases."""
        self.current_theme = theme_name
        qss = DARK_QSS if theme_name == "dark" else LIGHT_QSS
        self.setStyleSheet(qss)
        if hasattr(self, "visualizer"):
            self.visualizer.set_theme(theme_name)
        if hasattr(self, "h5_explorer"):
            self.h5_explorer.set_theme(theme_name)
        for card in getattr(self, "_cards", []):
            card.style().unpolish(card)
            card.style().polish(card)
            card.update()

    def _on_system_color_scheme_changed(self, scheme):
        if self.theme_mode == "system":
            resolved = "dark" if scheme == QtCore.Qt.ColorScheme.Dark else "light"
            self.apply_theme(resolved)
            self.status_bar.showMessage(f"Appearance updated to system {resolved.capitalize()} Mode.")

    def _set_theme_mode(self, mode: str):
        self.theme_mode = mode
        resolved = get_system_theme() if mode == "system" else mode
        self.apply_theme(resolved)
        mode_str = f"System Default ({resolved.capitalize()})" if mode == "system" else f"{mode.capitalize()} Mode"
        self.status_bar.showMessage(f"Theme set to {mode_str}.")

    # --------------------------------------------------------------------------
    # Inter-widget Event Callbacks
    # --------------------------------------------------------------------------

    def _on_h5_dataset_preview(self, path: str, data: Any, attrs: dict):
        """Displays selected H5 dataset on Left Visualizer."""
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                self.visualizer.show_2d_image(data, title=f"H5 Preview: {path}")
                self.status_bar.showMessage(f"Previewing 2D Matrix: {path} ({data.shape[0]}x{data.shape[1]})")
            elif data.ndim == 1:
                x = np.arange(len(data))
                self.visualizer.show_1d_curve(x, data, title=f"H5 Preview: {path}", xlabel="Index", ylabel="Intensity")
                self.status_bar.showMessage(f"Previewing 1D Vector: {path} (length {len(data)})")
            else:
                self.status_bar.showMessage(f"Dataset {path} has {data.ndim} dimensions; cannot preview directly.")

    def _on_sample_selected_from_h5(self, sample_name: str):
        idx = self.combo_sample.findText(sample_name)
        if idx >= 0:
            self.combo_sample.setCurrentIndex(idx)
        else:
            self.combo_sample.addItem(sample_name)
            self.combo_sample.setCurrentText(sample_name)
        self.tabs_right.setCurrentIndex(1)
        self.status_bar.showMessage(f"Set active sample to: '{sample_name}'")

    def _on_laser_selected_from_h5(self, laser_path: str):
        # Strip parent if 'Brillouin/' prefix is present
        clean = laser_path.replace("Brillouin/", "")
        self.sec_laser.txt_laser_path.setText(clean)
        self.tabs_right.setCurrentIndex(1)
        self.status_bar.showMessage(f"Set laser reference group to: '{clean}'")

    def _on_rb_selected_from_h5(self, rb_path: str):
        clean = rb_path.replace("Brillouin/", "")
        self.sec_rb.txt_rb_path.setText(clean)
        self.tabs_right.setCurrentIndex(1)
        self.status_bar.showMessage(f"Set Rb reference group to: '{clean}'")

    def _on_file_loaded(self, h5_path: str):
        self._refresh_samples_list()
        self.status_bar.showMessage(f"Loaded HDF5 container: {os.path.basename(h5_path)}")

    def _refresh_samples_list(self):
        samples = self.h5_explorer.list_available_samples()
        current = self.combo_sample.currentText()
        self.combo_sample.blockSignals(True)
        self.combo_sample.clear()
        for s in samples:
            self.combo_sample.addItem(s)
        if current in samples:
            self.combo_sample.setCurrentText(current)
        elif self.combo_sample.count() > 0:
            self.combo_sample.setCurrentIndex(0)
        self.combo_sample.blockSignals(False)

        self.sec_peak.populate_batch_samples(samples)

    def _on_active_sample_changed(self, sample_name: str):
        if sample_name:
            self.status_bar.showMessage(f"Active Sample: '{sample_name}'")

    def _clear_console(self):
        self.txt_console.clear()

    def _toggle_console(self):
        is_visible = self.txt_console.isVisible()
        self.txt_console.setVisible(not is_visible)
        self.btn_toggle_console.setText("Show Logs" if is_visible else "Hide Logs")

    # --------------------------------------------------------------------------
    # Pipeline Worker Execution
    # --------------------------------------------------------------------------

    def _start_worker(self, task_type: str, params: Dict[str, Any]):
        h5_path = self.h5_explorer.current_h5_path
        if not h5_path or not os.path.exists(h5_path):
            QtWidgets.QMessageBox.warning(self, "No HDF5 File", "Please select or load an HDF5 Measures file first!")
            return

        if self.current_worker and self.current_worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "Task Running", "A pipeline task is already running. Please wait.")
            return

        self.current_worker = PipelineWorker(task_type, h5_path, params, parent=self)
        self.current_worker.signals.started.connect(self._on_task_started)
        self.current_worker.signals.progress.connect(self._on_task_progress)
        self.current_worker.signals.log.connect(self._on_task_log)
        self.current_worker.signals.view_ready.connect(self._on_task_view_ready)
        self.current_worker.signals.finished.connect(self._on_task_finished)
        self.current_worker.signals.error.connect(self._on_task_error)
        self.current_worker.start()

    def _on_task_started(self, task_name: str):
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"Running '{task_name}'...")
        if not self.txt_console.isVisible():
            self._toggle_console()

    def _on_task_progress(self, current: int, total: int, msg: str):
        pct = int(100 * current / max(1, total))
        self.progress_bar.setValue(pct)
        self.status_bar.showMessage(f"{msg} ({pct}%)")

    def _on_task_log(self, text: str):
        self.txt_console.moveCursor(QtGui.QTextCursor.End)
        self.txt_console.insertPlainText(text)
        self.txt_console.moveCursor(QtGui.QTextCursor.End)

    def _on_task_view_ready(self, task_name: str, views: dict):
        self.visualizer.set_views(views)

    def _on_task_finished(self, task_name: str, results: dict):
        self.progress_bar.setValue(100)
        self.status_bar.showMessage(f"✓ Task '{task_name}' completed successfully.")

        # Update section status labels
        if task_name == "laser_all":
            self.sec_laser.set_status("✓ Calibrated", "#4caf50")
            if "xc_gnom" in results:
                self.sec_laser.lbl_vipa_geom_result.setText(
                    f"Center: ({results['xc_gnom']:.1f}, {results['yc_gnom']:.1f}) | Tilt y0: {results['y0_fit']:.1f}"
                )
        elif task_name == "rb_all":
            self.sec_rb.set_status("✓ Calibrated", "#4caf50")
            if "FSR_cal" in results:
                self.sec_rb.lbl_fsr_result.setText(f"Calibrated FSR: {results['FSR_cal']:.4f} GHz")
        elif task_name == "peak_fit":
            self.sec_peak.set_status("✓ Fitted", "#4caf50")
        elif task_name == "q_vector":
            self.sec_wavevector.set_status("✓ Extracted", "#4caf50")
            if "V_s" in results:
                vs = results["V_s"]
                vs_err = results.get("V_s_err", 0.0)
                self.sec_wavevector.lbl_results_vs.setText(f"Speed of Sound V_s = {vs:.2f} ± {vs_err:.2f} m/s")

        self.h5_explorer.reload_tree()

    def _on_task_error(self, task_name: str, tb: str):
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"❌ Error during '{task_name}'.")
        QtWidgets.QMessageBox.critical(self, f"Pipeline Error ({task_name})", f"An error occurred during execution:\n\n{tb}")

    # --------------------------------------------------------------------------
    # Step Execution Triggers
    # --------------------------------------------------------------------------

    def _run_laser_all(self):
        params = {
            "laser_group": self.sec_laser.txt_laser_path.text().strip() or "laser_reference",
            "scale_laser": self.sec_laser.spin_scale_laser.value(),
            "ksize_se_laser": self.sec_laser.spin_ksize_se_laser.value(),
            "ksize_blur_laser": self.sec_laser.spin_ksize_blur_laser.value(),
            "poly_deg": self.sec_laser.spin_poly_deg.value(),
        }
        self.sec_laser.set_status("Running...", "#00bcd4")
        self._start_worker("laser_all", params)

    def _run_rb_all(self):
        params = {
            "rb_group": self.sec_rb.txt_rb_path.text().strip() or "Rb_absorption",
            "laser_group": self.sec_laser.txt_laser_path.text().strip() or "laser_reference",
            "t_cell": self.sec_rb.spin_t_cell.value(),
            "l_cell": self.sec_rb.spin_l_cell.value(),
            "scale_rb": self.sec_rb.spin_scale_rb.value(),
            "ksize_se_rb": self.sec_rb.spin_ksize_se_rb.value(),
            "ksize_blur_rb": self.sec_rb.spin_ksize_blur_rb.value(),
            "laser_offset_guess": self.sec_rb.spin_laser_offset_guess.value(),
            "fit_window_ghz": self.sec_rb.spin_fit_window_ghz.value(),
            "rb_lock_offset": self.sec_rb.spin_rb_lock_offset.value(),
        }
        self.sec_rb.set_status("Running...", "#00bcd4")
        self._start_worker("rb_all", params)

    def _run_peak_fit(self):
        sample = self.combo_sample.currentText()
        if not sample:
            QtWidgets.QMessageBox.warning(self, "No Sample Selected", "Please select a sample from the dropdown first!")
            return

        params = {
            "sample_name": sample,
            "rb_group": self.sec_rb.txt_rb_path.text().strip() or "Rb_absorption",
            "bin_val_h": self.sec_peak.spin_bin_h.value(),
            "bin_val_w": self.sec_peak.spin_bin_w.value(),
            "expected_shift": self.sec_peak.spin_expected_shift.value(),
            "shift_tol": self.sec_peak.spin_shift_tol.value(),
            "expected_lw": self.sec_peak.spin_expected_lw.value(),
            "lw_min": self.sec_peak.spin_lw_min.value(),
            "lw_max": self.sec_peak.spin_lw_max.value(),
        }
        self.sec_peak.set_status("Running...", "#00bcd4")
        self._start_worker("peak_fit", params)

    def _run_q_vector(self):
        sample = self.combo_sample.currentText()
        if not sample:
            QtWidgets.QMessageBox.warning(self, "No Sample Selected", "Please select a sample from the dropdown first!")
            return

        fixed_angle = self.sec_wavevector.spin_fixed_angle.value() if self.sec_wavevector.chk_fixed_angle.isChecked() else None

        params = {
            "sample_name": sample,
            "bin_val_h": self.sec_peak.spin_bin_h.value(),
            "bin_val_w": self.sec_peak.spin_bin_w.value(),
            "refractive_index": self.sec_wavevector.spin_refractive_index.value(),
            "shift_min": self.sec_wavevector.spin_shift_min.value(),
            "shift_max": self.sec_wavevector.spin_shift_max.value(),
            "lw_min": self.sec_wavevector.spin_lw_filter_min.value(),
            "lw_max": self.sec_wavevector.spin_lw_filter_max.value(),
            "use_errors": self.sec_wavevector.chk_use_errors.isChecked(),
            "error_percentile_floor": self.sec_wavevector.spin_error_percentile_floor.value(),
            "average_pairs": self.sec_wavevector.chk_average_pairs.isChecked(),
            "initial_m_theta": self.sec_wavevector.spin_init_m_theta.value(),
            "fixed_ellipse_angle_deg": fixed_angle,
        }
        self.sec_wavevector.set_status("Running...", "#00bcd4")
        self._start_worker("q_vector", params)
