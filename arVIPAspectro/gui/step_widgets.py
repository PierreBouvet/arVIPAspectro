"""
step_widgets.py: Collapsible Section & Step Configuration Panels for arVIPAspectro GUI.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui


# ==============================================================================
# Reusable Collapsible Card Widget
# ==============================================================================

class CollapsibleCard(QtWidgets.QFrame):
    """
    A collapsible UI card featuring a clickable header with arrow indicator,
    title, status indicator, action button, and a collapsible content body.
    """

    def __init__(
        self,
        title: str,
        parent=None,
        is_expanded: bool = False,
        action_button_text: Optional[str] = None,
        is_major_section: bool = False,
    ):
        super().__init__(parent)
        self.is_expanded = is_expanded
        self.is_major_section = is_major_section

        # Frame Styling
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setProperty("isMajor", "true" if self.is_major_section else "false")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Header Bar
        self.header = QtWidgets.QWidget()
        self.header.setCursor(QtCore.Qt.PointingHandCursor)
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(6)

        self.lbl_arrow = QtWidgets.QLabel("▼" if self.is_expanded else "▶")
        self.lbl_arrow.setStyleSheet("font-size: 11px;")

        self.lbl_title = QtWidgets.QLabel(title)
        if self.is_major_section:
            self.lbl_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        else:
            self.lbl_title.setStyleSheet("font-weight: 500; font-size: 12px;")

        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("font-size: 11px; font-style: italic;")

        header_layout.addWidget(self.lbl_arrow)
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_status)

        self.btn_action = None
        if action_button_text:
            self.btn_action = QtWidgets.QPushButton(action_button_text)
            if self.is_major_section:
                self.btn_action.setStyleSheet(
                    "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; font-size: 11px; padding: 4px 10px; border-radius: 3px; }"
                    "QPushButton:hover { background-color: #388e3c; }"
                )
            else:
                self.btn_action.setStyleSheet(
                    "QPushButton { background-color: #0288d1; color: white; font-size: 11px; padding: 2px 8px; border-radius: 3px; }"
                    "QPushButton:hover { background-color: #039be5; }"
                )
            header_layout.addWidget(self.btn_action)

        main_layout.addWidget(self.header)

        # 2. Content Body
        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 6, 10, 8)
        self.content_layout.setSpacing(6)
        main_layout.addWidget(self.content_widget)

        self.content_widget.setVisible(self.is_expanded)
        self.header.mousePressEvent = self._toggle_expanded

    def _toggle_expanded(self, event):
        # Ignore clicks directly on the action button
        if self.btn_action and self.btn_action.geometry().contains(event.pos()):
            return
        self.is_expanded = not self.is_expanded
        self.content_widget.setVisible(self.is_expanded)
        self.lbl_arrow.setText("▼" if self.is_expanded else "▶")

    def set_status(self, text: str, color: str = "#757575"):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: bold;")


# ==============================================================================
# Global System & Optics Settings Section
# ==============================================================================

class SystemOpticsWidget(CollapsibleCard):
    """Global hardware, VIPA optical geometry, and atomic reference constants."""

    def __init__(self, parent=None):
        super().__init__(
            title="⚙ System & Optics Settings (VIPA, Laser, Cell)",
            parent=parent,
            is_expanded=False,
            action_button_text=None,
            is_major_section=True,
        )
        self._build_ui()

    def _build_ui(self):
        form = QtWidgets.QFormLayout()

        # Focal length in pixels (f_px)
        self.spin_f_px = QtWidgets.QDoubleSpinBox()
        self.spin_f_px.setRange(100.0, 1000000.0)
        self.spin_f_px.setDecimals(1)
        self.spin_f_px.setValue(133333.3)
        self.spin_f_px.setToolTip("Effective focal length in pixels (f_px = f / pixel_size, e.g. 200 mm / 1.5 um)")
        form.addRow("VIPA Focal Length f (px):", self.spin_f_px)

        # VIPA Tilt Angle in degrees
        self.spin_tilt_deg = QtWidgets.QDoubleSpinBox()
        self.spin_tilt_deg.setRange(0.01, 45.0)
        self.spin_tilt_deg.setDecimals(4)
        self.spin_tilt_deg.setValue(3.0000)
        self.spin_tilt_deg.setToolTip("Nominal internal/external VIPA tilt angle phi (degrees)")
        form.addRow("VIPA Tilt Angle phi (°):", self.spin_tilt_deg)

        # VIPA Refractive Index n_r
        self.spin_n_r = QtWidgets.QDoubleSpinBox()
        self.spin_n_r.setRange(1.0, 3.0)
        self.spin_n_r.setDecimals(5)
        self.spin_n_r.setValue(1.45340)
        self.spin_n_r.setToolTip("Refractive index of fused silica VIPA etalon at 780 nm")
        form.addRow("VIPA Etalon Index n_r:", self.spin_n_r)

        # VIPA Nominal FSR (GHz)
        self.spin_fsr_vipa = QtWidgets.QDoubleSpinBox()
        self.spin_fsr_vipa.setRange(1.0, 500.0)
        self.spin_fsr_vipa.setDecimals(2)
        self.spin_fsr_vipa.setValue(40.00)
        self.spin_fsr_vipa.setToolTip("Nominal VIPA Free Spectral Range in GHz")
        form.addRow("VIPA Nominal FSR (GHz):", self.spin_fsr_vipa)

        # Laser Wavelength lambda_0 (nm)
        self.spin_lambda_0 = QtWidgets.QDoubleSpinBox()
        self.spin_lambda_0.setRange(200.0, 2000.0)
        self.spin_lambda_0.setDecimals(6)
        self.spin_lambda_0.setValue(780.241368)
        self.spin_lambda_0.setToolTip("Vacuum laser wavelength in nm (Rb D2 line: 780.241368 nm)")
        form.addRow("Laser Wavelength (nm):", self.spin_lambda_0)

        # Rubidium D2 transition frequency nu_0 (THz)
        self.spin_nu_0 = QtWidgets.QDoubleSpinBox()
        self.spin_nu_0.setRange(100.0, 1000.0)
        self.spin_nu_0.setDecimals(7)
        self.spin_nu_0.setValue(384.2304845)
        self.spin_nu_0.setToolTip("Rubidium D2 line center frequency in THz")
        form.addRow("Rb D2 Center Freq (THz):", self.spin_nu_0)

        # Rubidium Vapor Cell Temp & Path Length
        h_cell = QtWidgets.QHBoxLayout()
        self.spin_t_cell = QtWidgets.QDoubleSpinBox()
        self.spin_t_cell.setRange(20.0, 150.0)
        self.spin_t_cell.setValue(70.0)
        self.spin_l_cell = QtWidgets.QDoubleSpinBox()
        self.spin_l_cell.setRange(0.001, 1.0)
        self.spin_l_cell.setDecimals(3)
        self.spin_l_cell.setValue(0.070)
        h_cell.addWidget(QtWidgets.QLabel("Temp (°C):"))
        h_cell.addWidget(self.spin_t_cell)
        h_cell.addWidget(QtWidgets.QLabel("Length (m):"))
        h_cell.addWidget(self.spin_l_cell)
        form.addRow("Rb Cell Parameters:", h_cell)

        # Rubidium 85Rb F=2 Lock Line Offset (GHz)
        self.spin_rb_lock_offset = QtWidgets.QDoubleSpinBox()
        self.spin_rb_lock_offset.setRange(-10.0, 10.0)
        self.spin_rb_lock_offset.setDecimals(3)
        self.spin_rb_lock_offset.setValue(1.323)
        self.spin_rb_lock_offset.setToolTip("85Rb F=2 absorption line offset from D2 center (GHz)")
        form.addRow("85Rb F=2 Lock Offset (GHz):", self.spin_rb_lock_offset)

        self.content_layout.addLayout(form)

    def get_params(self) -> Dict[str, Any]:
        """Returns the dictionary of optical and hardware constants."""
        return {
            "f_px": float(self.spin_f_px.value()),
            "phi_deg": float(self.spin_tilt_deg.value()),
            "phi_rad": float(np.deg2rad(self.spin_tilt_deg.value())),
            "n_r": float(self.spin_n_r.value()),
            "fsr_vipa": float(self.spin_fsr_vipa.value()),
            "laser_wavelength": float(self.spin_lambda_0.value()) * 1e-9,  # in meters
            "nu_0": float(self.spin_nu_0.value()) * 1e12,                # in Hz
            "t_cell": float(self.spin_t_cell.value()),
            "l_cell": float(self.spin_l_cell.value()),
            "rb_lock_offset_ghz": float(self.spin_rb_lock_offset.value()),
        }


# ==============================================================================
# Step 1-6: Laser Calibration Section
# ==============================================================================

class LaserCalibrationWidget(CollapsibleCard):
    """Section 1: Laser Reference Calibration (Steps 1 to 6)."""

    def __init__(self, parent=None):
        super().__init__(
            title="⚡ Section 1: Laser Reference Calibration (Steps 1-6)",
            parent=parent,
            is_expanded=True,
            action_button_text="Run All Laser Steps (1-6)",
            is_major_section=True,
        )
        self._build_steps()

    def _build_steps(self):
        # Target Laser Group Selector
        grp_box = QtWidgets.QHBoxLayout()
        grp_box.addWidget(QtWidgets.QLabel("Target Group in H5:"))
        self.txt_laser_path = QtWidgets.QLineEdit("laser_reference")
        self.txt_laser_path.setPlaceholderText("laser_reference")
        grp_box.addWidget(self.txt_laser_path)
        self.content_layout.addLayout(grp_box)

        # Step 2: Denoising
        self.card_step2 = CollapsibleCard("Step 2: Laser Image Denoising", action_button_text="Run Step 2")
        form2 = QtWidgets.QFormLayout()
        self.spin_scale_laser = QtWidgets.QSpinBox()
        self.spin_scale_laser.setRange(1, 30)
        self.spin_scale_laser.setValue(9)
        self.spin_ksize_se_laser = QtWidgets.QSpinBox()
        self.spin_ksize_se_laser.setRange(1, 51)
        self.spin_ksize_se_laser.setSingleStep(2)
        self.spin_ksize_se_laser.setValue(3)
        self.spin_ksize_blur_laser = QtWidgets.QSpinBox()
        self.spin_ksize_blur_laser.setRange(1, 51)
        self.spin_ksize_blur_laser.setSingleStep(2)
        self.spin_ksize_blur_laser.setValue(11)
        form2.addRow("Downsample Scale:", self.spin_scale_laser)
        form2.addRow("Structuring Element Size:", self.spin_ksize_se_laser)
        form2.addRow("Gaussian Blur Kernel Size:", self.spin_ksize_blur_laser)
        self.card_step2.content_layout.addLayout(form2)
        self.content_layout.addWidget(self.card_step2)

        # Step 3: Fringe Extraction
        self.card_step3 = CollapsibleCard("Step 3: Fringe Detection & Tracing", action_button_text="Run Step 3")
        form3 = QtWidgets.QFormLayout()
        self.spin_y_center_laser = QtWidgets.QSpinBox()
        self.spin_y_center_laser.setRange(0, 5000)
        self.spin_y_center_laser.setValue(512)
        self.spin_dy_laser = QtWidgets.QSpinBox()
        self.spin_dy_laser.setRange(5, 1000)
        self.spin_dy_laser.setValue(50)
        self.spin_nb_laser_modes = QtWidgets.QSpinBox()
        self.spin_nb_laser_modes.setRange(1, 30)
        self.spin_nb_laser_modes.setValue(3)
        self.spin_distance_peaks = QtWidgets.QSpinBox()
        self.spin_distance_peaks.setRange(1, 500)
        self.spin_distance_peaks.setValue(15)
        self.spin_dy_trace = QtWidgets.QSpinBox()
        self.spin_dy_trace.setRange(1, 50)
        self.spin_dy_trace.setValue(5)
        self.spin_var_trace = QtWidgets.QSpinBox()
        self.spin_var_trace.setRange(1, 50)
        self.spin_var_trace.setValue(5)

        form3.addRow("Y Center (px):", self.spin_y_center_laser)
        form3.addRow("Vertical Window Half-Height dy (px):", self.spin_dy_laser)
        form3.addRow("Expected Number of Modes:", self.spin_nb_laser_modes)
        form3.addRow("Min Distance Between Peaks (px):", self.spin_distance_peaks)
        form3.addRow("Vertical Tracing Step dy_trace (px):", self.spin_dy_trace)
        form3.addRow("Search Window Half-Width var_trace (px):", self.spin_var_trace)
        self.card_step3.content_layout.addLayout(form3)
        self.content_layout.addWidget(self.card_step3)

        # Step 4: Sub-pixel Refinement
        self.card_step4 = CollapsibleCard("Step 4: Sub-Pixel Refinement & Outlier Filtering", action_button_text="Run Step 4")
        form4 = QtWidgets.QFormLayout()
        self.spin_w_sub = QtWidgets.QSpinBox()
        self.spin_w_sub.setRange(1, 50)
        self.spin_w_sub.setValue(5)
        self.spin_saturation_limit = QtWidgets.QDoubleSpinBox()
        self.spin_saturation_limit.setRange(100.0, 100000.0)
        self.spin_saturation_limit.setValue(60000.0)
        self.spin_min_net_amp = QtWidgets.QDoubleSpinBox()
        self.spin_min_net_amp.setRange(1.0, 10000.0)
        self.spin_min_net_amp.setValue(50.0)

        form4.addRow("Sub-Pixel Window Width (w_sub px):", self.spin_w_sub)
        form4.addRow("Camera Saturation Limit:", self.spin_saturation_limit)
        form4.addRow("Min Net Peak Amplitude:", self.spin_min_net_amp)
        self.card_step4.content_layout.addLayout(form4)
        self.content_layout.addWidget(self.card_step4)

        # Step 5: Exact 2D Gnomonic VIPA Geometry Fit
        self.card_step5 = CollapsibleCard("Step 5: Exact 2D Gnomonic VIPA Geometry Fit", action_button_text="Run Step 5")
        form5 = QtWidgets.QFormLayout()
        self.lbl_vipa_geom_result = QtWidgets.QLabel("Optical Axis: Not fitted | Gnomonic Center: Not fitted")
        self.lbl_vipa_geom_result.setStyleSheet("color: #00bcd4; font-family: monospace; font-size: 11px;")
        form5.addRow("Calibrated Geometry:", self.lbl_vipa_geom_result)
        self.card_step5.content_layout.addLayout(form5)
        self.content_layout.addWidget(self.card_step5)

        # Step 6: 2D Anamorphic Polynomial Surface Calibration
        self.card_step6 = CollapsibleCard("Step 6: 2D Anamorphic Order Surface Calibration", action_button_text="Run Step 6")
        form6 = QtWidgets.QFormLayout()
        self.spin_poly_deg = QtWidgets.QSpinBox()
        self.spin_poly_deg.setRange(1, 8)
        self.spin_poly_deg.setValue(4)
        self.chk_lower_order_study = QtWidgets.QCheckBox("Compare Lower-Order Fits (1D Linear & Quadratic)")
        self.chk_lower_order_study.setChecked(False)
        self.lbl_surface_rms = QtWidgets.QLabel("Surface RMS: Not fitted")
        self.lbl_surface_rms.setStyleSheet("color: #00bcd4; font-family: monospace; font-size: 11px;")

        form6.addRow("Polynomial Degree (u, v):", self.spin_poly_deg)
        form6.addRow("Options:", self.chk_lower_order_study)
        form6.addRow("Residuals:", self.lbl_surface_rms)
        self.card_step6.content_layout.addLayout(form6)
        self.content_layout.addWidget(self.card_step6)


# ==============================================================================
# Step 7-10B: Rubidium Calibration Section
# ==============================================================================

class RbCalibrationWidget(CollapsibleCard):
    """Section 2: Rubidium Absorption Calibration (Steps 7 to 10B)."""

    def __init__(self, parent=None):
        super().__init__(
            title="🔬 Section 2: Rubidium Calibration (Steps 7-10B)",
            parent=parent,
            is_expanded=False,
            action_button_text="Run All Rb Steps (7-10B)",
            is_major_section=True,
        )
        self._build_steps()

    def _build_steps(self):
        # Target Groups
        grp_box = QtWidgets.QHBoxLayout()
        grp_box.addWidget(QtWidgets.QLabel("Target Rb Group:"))
        self.txt_rb_path = QtWidgets.QLineEdit("Rb_absorption")
        self.txt_rb_path.setPlaceholderText("Rb_absorption")
        grp_box.addWidget(self.txt_rb_path)

        grp_box.addWidget(QtWidgets.QLabel("Laser Ref Group:"))
        self.txt_laser_path = QtWidgets.QLineEdit("laser_reference")
        self.txt_laser_path.setPlaceholderText("laser_reference")
        grp_box.addWidget(self.txt_laser_path)
        self.content_layout.addLayout(grp_box)

        # Step 7: Theoretical Rb D2 Model
        self.card_step7 = CollapsibleCard("Step 7: Theoretical Rb D2 Model", action_button_text="Run Step 7")
        form7 = QtWidgets.QFormLayout()
        lbl_info7 = QtWidgets.QLabel("Uses T_cell and L_cell from 'System & Optics Settings'.")
        lbl_info7.setStyleSheet("color: #888888; font-size: 11px;")
        form7.addRow("Info:", lbl_info7)
        self.card_step7.content_layout.addLayout(form7)
        self.content_layout.addWidget(self.card_step7)

        # Step 8: Preprocessing & Denoising
        self.card_step8 = CollapsibleCard("Step 8: Rb Absorption Denoising", action_button_text="Run Step 8")
        form8 = QtWidgets.QFormLayout()
        self.spin_scale_rb = QtWidgets.QSpinBox()
        self.spin_scale_rb.setRange(1, 30)
        self.spin_scale_rb.setValue(9)
        self.spin_ksize_se_rb = QtWidgets.QSpinBox()
        self.spin_ksize_se_rb.setRange(1, 51)
        self.spin_ksize_se_rb.setSingleStep(2)
        self.spin_ksize_se_rb.setValue(3)
        self.spin_ksize_blur_rb = QtWidgets.QSpinBox()
        self.spin_ksize_blur_rb.setRange(1, 51)
        self.spin_ksize_blur_rb.setSingleStep(2)
        self.spin_ksize_blur_rb.setValue(11)
        form8.addRow("Downsample Scale:", self.spin_scale_rb)
        form8.addRow("Structuring Element Size:", self.spin_ksize_se_rb)
        form8.addRow("Gaussian Blur Kernel Size:", self.spin_ksize_blur_rb)
        self.card_step8.content_layout.addLayout(form8)
        self.content_layout.addWidget(self.card_step8)

        # Step 9: Radial Profile Extraction
        self.card_step9 = CollapsibleCard("Step 9: Radial Profile Extraction", action_button_text="Run Step 9")
        form9 = QtWidgets.QFormLayout()
        self.spin_num_angles = QtWidgets.QSpinBox()
        self.spin_num_angles.setRange(50, 5000)
        self.spin_num_angles.setSingleStep(100)
        self.spin_num_angles.setValue(1000)
        form9.addRow("Number of Radial Ray Angles:", self.spin_num_angles)
        self.card_step9.content_layout.addLayout(form9)
        self.content_layout.addWidget(self.card_step9)

        # Step 10A: Calibrate FSR & Offset
        self.card_step10a = CollapsibleCard("Step 10A: FSR & Laser Detuning Calibration", action_button_text="Run Step 10A")
        form10a = QtWidgets.QFormLayout()
        self.spin_fit_window_ghz = QtWidgets.QDoubleSpinBox()
        self.spin_fit_window_ghz.setRange(1.0, 50.0)
        self.spin_fit_window_ghz.setValue(10.0)
        self.spin_delta_nu_laser_offset = QtWidgets.QDoubleSpinBox()
        self.spin_delta_nu_laser_offset.setRange(-30.0, 30.0)
        self.spin_delta_nu_laser_offset.setValue(-2.0)
        self.lbl_fsr_result = QtWidgets.QLabel("Calibrated FSR: Not fitted | delta_nu: Not fitted")
        self.lbl_fsr_result.setStyleSheet("color: #00bcd4; font-family: monospace; font-size: 11px;")
        form10a.addRow("Fit Window Around D2 (GHz):", self.spin_fit_window_ghz)
        form10a.addRow("Initial Laser Offset Guess (GHz):", self.spin_delta_nu_laser_offset)
        form10a.addRow("Calibration Result:", self.lbl_fsr_result)
        self.card_step10a.content_layout.addLayout(form10a)
        self.content_layout.addWidget(self.card_step10a)

        # Step 10B: 2D Calibrated Frequency Map
        self.card_step10b = CollapsibleCard("Step 10B: Calibrated 2D Frequency Map", action_button_text="Run Step 10B")
        form10b = QtWidgets.QFormLayout()
        self.lbl_map_status = QtWidgets.QLabel("Map Status: Not generated")
        self.lbl_map_status.setStyleSheet("color: #00bcd4; font-family: monospace; font-size: 11px;")
        form10b.addRow("Frequency Map:", self.lbl_map_status)
        self.card_step10b.content_layout.addLayout(form10b)
        self.content_layout.addWidget(self.card_step10b)


# ==============================================================================
# Step 11-13: Sample DHO Peak Fitting Section
# ==============================================================================

class PeakFittingWidget(CollapsibleCard):
    """Section 3: Sample In-Situ Calibration & DHO Peak Fitting (Steps 11 to 13)."""

    def __init__(self, parent=None):
        super().__init__(
            title="📈 Section 3: Sample Spectrum Fitting (DHO) (Steps 11-13)",
            parent=parent,
            is_expanded=False,
            action_button_text="Run Peak Fitting (11-13)",
            is_major_section=True,
        )
        self._build_steps()

    def _build_steps(self):
        # Sample selection & Batch Mode
        h_sample = QtWidgets.QHBoxLayout()
        h_sample.addWidget(QtWidgets.QLabel("Sample Dataset:"))
        self.txt_sample_path = QtWidgets.QLineEdit("water_5s")
        self.txt_sample_path.setPlaceholderText("water_5s or Brillouin/water_5s")
        h_sample.addWidget(self.txt_sample_path)
        self.chk_batch_all = QtWidgets.QCheckBox("Batch: Process all samples in group")
        self.chk_batch_all.setChecked(False)
        h_sample.addWidget(self.chk_batch_all)
        self.content_layout.addLayout(h_sample)

        # Material Presets
        h_presets = QtWidgets.QHBoxLayout()
        h_presets.addWidget(QtWidgets.QLabel("Material Preset:"))
        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItems(["Water", "Glycerol", "Polystyrene (Plastic)", "Custom"])
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        h_presets.addWidget(self.combo_preset)
        self.content_layout.addLayout(h_presets)

        # Step 11: In-Situ Calibration
        self.card_step11 = CollapsibleCard("Step 11: In-Situ Frequency Map Self-Calibration", action_button_text="Run Step 11")
        form11 = QtWidgets.QFormLayout()

        # Spatial Binning
        h_bin = QtWidgets.QHBoxLayout()
        self.spin_bin_h = QtWidgets.QSpinBox()
        self.spin_bin_h.setRange(1, 32)
        self.spin_bin_h.setValue(4)
        self.spin_bin_w = QtWidgets.QSpinBox()
        self.spin_bin_w.setRange(1, 32)
        self.spin_bin_w.setValue(2)
        h_bin.addWidget(QtWidgets.QLabel("H (vert px):"))
        h_bin.addWidget(self.spin_bin_h)
        h_bin.addWidget(QtWidgets.QLabel("W (horiz px):"))
        h_bin.addWidget(self.spin_bin_w)
        form11.addRow("Spatial Binning:", h_bin)

        self.spin_sigma_y_bins = QtWidgets.QDoubleSpinBox()
        self.spin_sigma_y_bins.setRange(0.5, 30.0)
        self.spin_sigma_y_bins.setValue(6.0)
        form11.addRow("Smoothing Length (sigma_y bins):", self.spin_sigma_y_bins)

        self.spin_d_freq = QtWidgets.QDoubleSpinBox()
        self.spin_d_freq.setRange(-10.0, 10.0)
        self.spin_d_freq.setValue(0.0)
        form11.addRow("Frequency Map Global Offset (GHz):", self.spin_d_freq)

        self.card_step11.content_layout.addLayout(form11)
        self.content_layout.addWidget(self.card_step11)

        # Step 12: DHO Fitting
        self.card_step12 = CollapsibleCard("Step 12: Vectorized DHO Peak Fitting", action_button_text="Run Step 12")
        form12 = QtWidgets.QFormLayout()

        self.spin_expected_shift = QtWidgets.QDoubleSpinBox()
        self.spin_expected_shift.setRange(0.1, 40.0)
        self.spin_expected_shift.setSingleStep(0.1)
        self.spin_expected_shift.setValue(5.1)
        form12.addRow("Expected Brillouin Shift nu_B (GHz):", self.spin_expected_shift)

        self.spin_shift_tol = QtWidgets.QDoubleSpinBox()
        self.spin_shift_tol.setRange(0.1, 10.0)
        self.spin_shift_tol.setValue(2.0)
        form12.addRow("Shift Search Tolerance (± GHz):", self.spin_shift_tol)

        self.spin_expected_lw = QtWidgets.QDoubleSpinBox()
        self.spin_expected_lw.setRange(0.05, 10.0)
        self.spin_expected_lw.setValue(0.8)
        form12.addRow("Expected Linewidth Gamma_B (GHz):", self.spin_expected_lw)

        h_lw_bounds = QtWidgets.QHBoxLayout()
        self.spin_lw_min = QtWidgets.QDoubleSpinBox()
        self.spin_lw_min.setRange(0.01, 10.0)
        self.spin_lw_min.setValue(0.4)
        self.spin_lw_max = QtWidgets.QDoubleSpinBox()
        self.spin_lw_max.setRange(0.05, 20.0)
        self.spin_lw_max.setValue(1.3)
        h_lw_bounds.addWidget(QtWidgets.QLabel("Min:"))
        h_lw_bounds.addWidget(self.spin_lw_min)
        h_lw_bounds.addWidget(QtWidgets.QLabel("Max:"))
        h_lw_bounds.addWidget(self.spin_lw_max)
        form12.addRow("Linewidth Bounds (GHz):", h_lw_bounds)

        self.spin_dho_hw_px = QtWidgets.QSpinBox()
        self.spin_dho_hw_px.setRange(3, 50)
        self.spin_dho_hw_px.setValue(20)
        form12.addRow("Fitting Window Half-Width (px):", self.spin_dho_hw_px)

        self.spin_dho_min_amp = QtWidgets.QDoubleSpinBox()
        self.spin_dho_min_amp.setRange(0.1, 1000.0)
        self.spin_dho_min_amp.setValue(10.0)
        form12.addRow("Min Net Peak Amplitude Above Chord:", self.spin_dho_min_amp)

        self.card_step12.content_layout.addLayout(form12)
        self.content_layout.addWidget(self.card_step12)

        # Step 13: Plotting & Diagnostics Options
        self.card_step13 = CollapsibleCard("Step 13: Diagnostics & Figure Options", is_expanded=False)
        form13 = QtWidgets.QFormLayout()
        self.chk_plot_overview = QtWidgets.QCheckBox("Generate Overview Heatmaps (Figure 13)")
        self.chk_plot_overview.setChecked(True)
        self.chk_plot_profiles = QtWidgets.QCheckBox("Generate Fringe Profiles (Figure 13B)")
        self.chk_plot_profiles.setChecked(False)
        form13.addRow("Plots:", self.chk_plot_overview)
        form13.addRow("", self.chk_plot_profiles)
        self.card_step13.content_layout.addLayout(form13)
        self.content_layout.addWidget(self.card_step13)

    def _on_preset_changed(self, index: int):
        preset = self.combo_preset.currentText()
        if preset == "Water":
            self.spin_expected_shift.setValue(5.1)
            self.spin_shift_tol.setValue(2.0)
            self.spin_expected_lw.setValue(0.8)
            self.spin_lw_min.setValue(0.4)
            self.spin_lw_max.setValue(1.3)
        elif preset == "Glycerol":
            self.spin_expected_shift.setValue(10.8)
            self.spin_shift_tol.setValue(3.0)
            self.spin_expected_lw.setValue(1.8)
            self.spin_lw_min.setValue(1.5)
            self.spin_lw_max.setValue(3.0)
        elif preset == "Polystyrene (Plastic)":
            self.spin_expected_shift.setValue(9.7)
            self.spin_shift_tol.setValue(2.0)
            self.spin_expected_lw.setValue(0.8)
            self.spin_lw_min.setValue(0.4)
            self.spin_lw_max.setValue(1.3)


# ==============================================================================
# Step 14-15: Acoustic Wavevector (q-vector) & Dispersion Section
# ==============================================================================

class WavevectorDispersionWidget(CollapsibleCard):
    """Section 4: Acoustic Wavevector (q-vector) & Dispersion Extraction (Steps 14 & 15)."""

    def __init__(self, parent=None):
        super().__init__(
            title="📐 Section 4: Wavevector (q) & Dispersion (Steps 14-15)",
            parent=parent,
            is_expanded=False,
            action_button_text="Run Dispersion Analysis (14-15)",
            is_major_section=True,
        )
        self._build_steps()

    def _build_steps(self):
        # Sample Target & Batch
        h_sample = QtWidgets.QHBoxLayout()
        h_sample.addWidget(QtWidgets.QLabel("Sample Dataset:"))
        self.txt_sample_path = QtWidgets.QLineEdit("water_5s")
        self.txt_sample_path.setPlaceholderText("water_5s or Brillouin/water_5s")
        h_sample.addWidget(self.txt_sample_path)
        self.chk_batch_all = QtWidgets.QCheckBox("Batch: Process all samples in group")
        self.chk_batch_all.setChecked(False)
        h_sample.addWidget(self.chk_batch_all)
        self.content_layout.addLayout(h_sample)

        # Material Presets
        h_presets = QtWidgets.QHBoxLayout()
        h_presets.addWidget(QtWidgets.QLabel("Material Preset:"))
        self.combo_preset_q = QtWidgets.QComboBox()
        self.combo_preset_q.addItems(["Water", "Glycerol", "Polystyrene (Plastic)", "Custom"])
        self.combo_preset_q.currentIndexChanged.connect(self._on_preset_q_changed)
        h_presets.addWidget(self.combo_preset_q)
        self.content_layout.addLayout(h_presets)

        form = QtWidgets.QFormLayout()

        # Refractive Index
        self.spin_refractive_index = QtWidgets.QDoubleSpinBox()
        self.spin_refractive_index.setRange(1.0, 3.0)
        self.spin_refractive_index.setDecimals(4)
        self.spin_refractive_index.setValue(1.3300)
        form.addRow("Sample Refractive Index (n):", self.spin_refractive_index)

        # Shift Range Filtering
        h_shift_range = QtWidgets.QHBoxLayout()
        self.spin_shift_min = QtWidgets.QDoubleSpinBox()
        self.spin_shift_min.setRange(0.0, 30.0)
        self.spin_shift_min.setValue(4.8)
        self.spin_shift_max = QtWidgets.QDoubleSpinBox()
        self.spin_shift_max.setRange(0.0, 30.0)
        self.spin_shift_max.setValue(5.3)
        h_shift_range.addWidget(QtWidgets.QLabel("Min:"))
        h_shift_range.addWidget(self.spin_shift_min)
        h_shift_range.addWidget(QtWidgets.QLabel("Max:"))
        h_shift_range.addWidget(self.spin_shift_max)
        form.addRow("Shift Filter Range (GHz):", h_shift_range)

        # Linewidth Range Filtering
        h_lw_range = QtWidgets.QHBoxLayout()
        self.spin_lw_filter_min = QtWidgets.QDoubleSpinBox()
        self.spin_lw_filter_min.setRange(0.0, 10.0)
        self.spin_lw_filter_min.setValue(0.5)
        self.spin_lw_filter_max = QtWidgets.QDoubleSpinBox()
        self.spin_lw_filter_max.setRange(0.0, 10.0)
        self.spin_lw_filter_max.setValue(1.2)
        h_lw_range.addWidget(QtWidgets.QLabel("Min:"))
        h_lw_range.addWidget(self.spin_lw_filter_min)
        h_lw_range.addWidget(QtWidgets.QLabel("Max:"))
        h_lw_range.addWidget(self.spin_lw_filter_max)
        form.addRow("Linewidth Filter Range (GHz):", h_lw_range)

        # Error Weighting & Pair Averaging
        h_flags = QtWidgets.QHBoxLayout()
        self.chk_use_errors = QtWidgets.QCheckBox("Error Weighting (w = 1/σ)")
        self.chk_use_errors.setChecked(True)
        self.chk_average_pairs = QtWidgets.QCheckBox("Average Stokes/Anti-Stokes")
        self.chk_average_pairs.setChecked(True)
        h_flags.addWidget(self.chk_use_errors)
        h_flags.addWidget(self.chk_average_pairs)
        form.addRow("Weights & Pairing:", h_flags)

        self.spin_error_percentile_floor = QtWidgets.QDoubleSpinBox()
        self.spin_error_percentile_floor.setRange(0.0, 20.0)
        self.spin_error_percentile_floor.setValue(1.0)
        form.addRow("Error Floor Percentile (%):", self.spin_error_percentile_floor)

        self.spin_init_m_theta = QtWidgets.QDoubleSpinBox()
        self.spin_init_m_theta.setRange(100.0, 50000.0)
        self.spin_init_m_theta.setValue(3000.0)
        form.addRow("Initial M_theta Guess (px):", self.spin_init_m_theta)

        # Fixed Ellipse Angle Parameter
        h_angle = QtWidgets.QHBoxLayout()
        self.chk_fixed_angle = QtWidgets.QCheckBox("Impose Ellipse Angle:")
        self.chk_fixed_angle.setChecked(True)
        self.spin_fixed_angle = QtWidgets.QDoubleSpinBox()
        self.spin_fixed_angle.setRange(-180.0, 180.0)
        self.spin_fixed_angle.setValue(90.0)
        self.chk_fixed_angle.toggled.connect(self.spin_fixed_angle.setEnabled)
        h_angle.addWidget(self.chk_fixed_angle)
        h_angle.addWidget(self.spin_fixed_angle)
        form.addRow("Ellipse Rotation:", h_angle)

        self.content_layout.addLayout(form)

        # Results Summary Banner
        self.lbl_results_vs = QtWidgets.QLabel("Speed of Sound V_s: Awaiting fit")
        self.lbl_results_vs.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 13px; padding: 4px 0;")
        self.content_layout.addWidget(self.lbl_results_vs)

    def _on_preset_q_changed(self, index: int):
        preset = self.combo_preset_q.currentText()
        if preset == "Water":
            self.spin_refractive_index.setValue(1.3300)
            self.spin_shift_min.setValue(4.8)
            self.spin_shift_max.setValue(5.3)
            self.spin_lw_filter_min.setValue(0.5)
            self.spin_lw_filter_max.setValue(1.2)
        elif preset == "Glycerol":
            self.spin_refractive_index.setValue(1.4670)
            self.spin_shift_min.setValue(10.0)
            self.spin_shift_max.setValue(11.5)
            self.spin_lw_filter_min.setValue(1.5)
            self.spin_lw_filter_max.setValue(3.0)
        elif preset == "Polystyrene (Plastic)":
            self.spin_refractive_index.setValue(1.5900)
            self.spin_shift_min.setValue(9.3)
            self.spin_shift_max.setValue(9.9)
            self.spin_lw_filter_min.setValue(0.5)
            self.spin_lw_filter_max.setValue(1.2)
