"""
2D Camera Frame & Calibration Diagnostic Viewer Widget (`ImageViewerWidget`)
for displaying acquired Hamamatsu Orca Flash 4.0 v3 interferograms and
step-by-step calibration diagnostics.
"""

from typing import Optional, Dict, Any
import numpy as np
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class ImageViewerWidget(QtWidgets.QWidget):
    """
    Interactive 2D image canvas with colormap/contrast controls, frame statistics,
    live cursor pixel/intensity readout, and calibration diagnostic plot modes.
    """

    COLORMAPS = ["inferno", "viridis", "magma", "plasma", "gray", "turbo"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: Optional[np.ndarray] = None
        self._current_title: str = "No image acquired yet"
        self._im_artist = None
        self._cbar = None
        self._in_diagnostic_view: bool = False
        self._last_live_map_state: Optional[Dict[str, Any]] = None

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Top Display Controls Bar
        ctrl_bar = QtWidgets.QHBoxLayout()

        ctrl_bar.addWidget(QtWidgets.QLabel("Colormap:"))
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(self.COLORMAPS)
        self.combo_cmap.currentTextChanged.connect(self._on_display_settings_changed)
        ctrl_bar.addWidget(self.combo_cmap)

        ctrl_bar.addSpacing(12)
        ctrl_bar.addWidget(QtWidgets.QLabel("Contrast:"))
        self.combo_contrast = QtWidgets.QComboBox()
        self.combo_contrast.addItem("Auto (1% – 99.8%)", (1.0, 99.8))
        self.combo_contrast.addItem("5% – 95%", (5.0, 95.0))
        self.combo_contrast.addItem("10% – 90%", (10.0, 90.0))
        self.combo_contrast.addItem("0% – 50%", (0.0, 50.0))
        self.combo_contrast.addItem("Min – Max", "minmax")
        self.combo_contrast.addItem("Full 16-bit (0 – 65535)", "uint16")
        self.combo_contrast.currentIndexChanged.connect(self._on_display_settings_changed)
        ctrl_bar.addWidget(self.combo_contrast)

        ctrl_bar.addSpacing(12)
        ctrl_bar.addWidget(QtWidgets.QLabel("Map Metric:"))
        self.combo_map_metric = QtWidgets.QComboBox()
        self.combo_map_metric.addItem("Mean Shift ν_B (GHz)", "mean_shift")
        self.combo_map_metric.addItem("Velocity V_s (m/s)", "v_s")
        self.combo_map_metric.addItem("Mean Linewidth Γ_B (GHz)", "mean_lw")
        self.combo_map_metric.addItem("Normal Shift S(0) (GHz)", "shift_0")
        self.combo_map_metric.currentIndexChanged.connect(self._on_map_metric_changed)
        ctrl_bar.addWidget(self.combo_map_metric)

        ctrl_bar.addStretch()
        self.lbl_stats = QtWidgets.QLabel("No frame loaded")
        self.lbl_stats.setStyleSheet("font-family: monospace; font-size: 11px;")
        ctrl_bar.addWidget(self.lbl_stats)
        layout.addLayout(ctrl_bar)

        # 2. Matplotlib Figure & Navigation Toolbar
        self.figure = Figure(figsize=(6, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(self._current_title, fontsize=11)
        self.ax.set_xlabel("Detector X (px)")
        self.ax.set_ylabel("Detector Y (px)")

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # 3. Bottom Cursor Readout Bar
        self.lbl_cursor = QtWidgets.QLabel("Cursor: —")
        self.lbl_cursor.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.lbl_cursor)

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def update_image(
        self,
        image: np.ndarray,
        stage_pos_um: Optional[Dict[str, Optional[float]]] = None,
        exposure_s: Optional[float] = None,
        n_avg: Optional[int] = None,
        binning: Optional[int] = None,
    ):
        """
        Displays a newly acquired 2D image and updates statistics and stage coordinates in the title.
        """
        self._current_image = np.asarray(image)
        h, w = self._current_image.shape[:2]

        # Build title string
        bin_str = f" (Bin {binning}×{binning})" if binning else ""
        parts = [f"Frame {w}×{h} px{bin_str}"]
        if stage_pos_um is not None and stage_pos_um.get("X") is not None:
            pos_str = f"X={stage_pos_um['X']:.1f} µm, Y={stage_pos_um['Y']:.1f} µm"
            if stage_pos_um.get("Z") is not None:
                pos_str += f", Z={stage_pos_um['Z']:.1f} µm"
            parts.append(f"Stage: ({pos_str})")
        if exposure_s is not None:
            exp_str = f"Exp: {exposure_s * 1e3:.2f} ms"
            if n_avg and n_avg > 1:
                exp_str += f" × {n_avg} avg"
            parts.append(exp_str)
        self._current_title = "  |  ".join(parts)

        # Compute statistics
        vmin_raw = int(np.min(self._current_image))
        vmax_raw = int(np.max(self._current_image))
        vmean = float(np.mean(self._current_image))
        n_sat = int(np.sum(self._current_image >= 65000))
        self.lbl_stats.setText(
            f"Min: {vmin_raw}  |  Max: {vmax_raw}  |  Mean: {vmean:.1f}  |  Sat(≥65k): {n_sat} px"
        )

        if self._in_diagnostic_view:
            self._im_artist = None
            self._in_diagnostic_view = False
        self._last_live_map_state = None
        self._render_image()

    def _compute_clim(self, img: np.ndarray):
        mode = self.combo_contrast.currentData()
        if mode == "uint16":
            return 0.0, 65535.0
        if mode == "minmax":
            vmin, vmax = float(np.min(img)), float(np.max(img))
            return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)
        low_pct, high_pct = mode if isinstance(mode, tuple) else (1.0, 99.8)
        sample = img[::4, ::4] if img.size > 65536 else img
        vmin, vmax = np.percentile(sample, [low_pct, high_pct])
        if vmax <= vmin:
            vmin, vmax = float(np.min(img)), float(np.max(img))
            if vmax <= vmin:
                vmax = vmin + 1.0
        return float(vmin), float(vmax)

    def _render_image(self):
        if self._current_image is None:
            return

        cmap = self.combo_cmap.currentText()
        vmin, vmax = self._compute_clim(self._current_image)
        h, w = self._current_image.shape[:2]

        if (
            not self._in_diagnostic_view
            and self._im_artist is not None
            and self._im_artist.get_array().shape == (h, w)
        ):
            self._im_artist.set_data(self._current_image)
            self._im_artist.set_cmap(cmap)
            self._im_artist.set_clim(vmin, vmax)
            self.ax.set_title(self._current_title, fontsize=10)
        else:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self._im_artist = self.ax.imshow(
                self._current_image,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="upper",
                aspect="equal",
            )
            self._cbar = self.figure.colorbar(self._im_artist, ax=self.ax, fraction=0.046, pad=0.04)
            self._cbar.set_label("Counts (ADU)")
            self.ax.set_title(self._current_title, fontsize=10)
            self.ax.set_xlabel("Detector X (px)")
            self.ax.set_ylabel("Detector Y (px)")

        self.canvas.draw_idle()

    def _on_display_settings_changed(self):
        if self._in_diagnostic_view:
            if self._last_live_map_state is not None:
                self._render_live_map_state()
        else:
            self._render_image()

    # ==========================================================================
    # Step 5.1: Laser Reference Calibration Diagnostic Views
    # ==========================================================================

    def show_laser_calibration_diagnostics(
        self,
        calib_res: Dict[str, Any],
        view_mode: str = "fringes_geom",
    ):
        """
        Renders diagnostic plots for Step 5.1 (Laser Reference Calibration):
          - `'denoise_seeds'`: Step 2–3 Denoised laser image + central row profile & seed peaks.
          - `'fringes_geom'`: Step 4–5 Traced & refined sub-pixel fringes + gnomonic optical axis.
          - `'anamorphic_res'`: Step 6 2D anamorphic polynomial surface residuals (MHz).
        """
        self._in_diagnostic_view = True
        self._last_live_map_state = None
        self._im_artist = None
        self.figure.clear()

        cmap = self.combo_cmap.currentText()
        data_sub = calib_res["data_subtracted"]
        vmin, vmax = self._compute_clim(data_sub)

        if view_mode == "denoise_seeds":
            gs = self.figure.add_gridspec(2, 1, height_ratios=[2.2, 1.0])
            ax_img = self.figure.add_subplot(gs[0])
            ax_prof = self.figure.add_subplot(gs[1])

            im = ax_img.imshow(data_sub, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
            self.figure.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, label="Net ADU")

            yc = calib_res["y_center_laser"]
            dy = calib_res["dy_laser"]
            peaks = np.asarray(calib_res["peaks"]).ravel()
            ax_img.axhline(yc - dy, color="cyan", linestyle="--", linewidth=1.0, label="Row-avg band")
            ax_img.axhline(yc + dy, color="cyan", linestyle="--", linewidth=1.0)
            ax_img.plot(peaks, np.full_like(peaks, yc), "r*", markersize=8, label=f"{len(peaks)} seed peaks")
            ax_img.set_title("Step 5.1A — Denoised Laser Interferogram & Central Seed Band", fontsize=10)
            ax_img.set_ylabel("Detector Y (px)")
            ax_img.legend(loc="upper right", fontsize=8)

            avg_row = calib_res["average_row"]
            ax_prof.plot(np.arange(len(avg_row)), avg_row, color="#0288d1", linewidth=1.2, label="Row average")
            ax_prof.plot(peaks, avg_row[peaks], "ro", markersize=5, label="Detected modes")
            ax_prof.set_title("Central Row-Averaged Profile & Detected Laser Fringes", fontsize=10)
            ax_prof.set_xlabel("Detector X (px)")
            ax_prof.set_ylabel("Net Intensity (ADU)")
            ax_prof.grid(True, alpha=0.3)
            ax_prof.legend(loc="upper right", fontsize=8)
            self.ax = ax_img

        elif view_mode == "anamorphic_res":
            ax1 = self.figure.add_subplot(1, 2, 1)
            ax2 = self.figure.add_subplot(1, 2, 2)

            all_x = calib_res["all_x_laser"]
            all_y = calib_res["all_y_laser"]
            err_raw = calib_res["err_2d_mhz"]
            err_sm = calib_res["err_2d_smooth_mhz"]
            rms_raw = calib_res["rms_raw_mhz"]
            rms_sm = calib_res["rms_smooth_mhz"]

            lim = max(5.0, float(np.percentile(np.abs(err_raw), 98)))
            sc = ax1.scatter(all_x, all_y, c=err_raw, s=2, cmap="coolwarm", vmin=-lim, vmax=lim)
            self.figure.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="Residual (MHz)")
            ax1.invert_yaxis()
            ax1.set_title(f"2D Surface Residuals (Raw RMS = {rms_raw:.2f} MHz)", fontsize=10)
            ax1.set_xlabel("Detector X (px)")
            ax1.set_ylabel("Detector Y (px)")
            ax1.grid(True, alpha=0.25)

            ax2.scatter(err_raw, all_y, s=2, alpha=0.3, color="#64b5f6", label=f"Raw ({rms_raw:.2f} MHz)")
            ax2.scatter(err_sm, all_y, s=2, alpha=0.7, color="#e53935", label=f"Smoothed ({rms_sm:.2f} MHz)")
            ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            ax2.invert_yaxis()
            ax2.set_title("Anamorphic Residual vs. Detector Y", fontsize=10)
            ax2.set_xlabel("Frequency Residual (MHz)")
            ax2.set_ylabel("Detector Y (px)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right", fontsize=8)
            self.ax = ax1

        else:
            # Default: 'fringes_geom' — Refined fringes + fitted optical axis
            ax_img = self.figure.add_subplot(111)
            im = ax_img.imshow(data_sub, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
            self.figure.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, label="Net ADU")

            ref_fringes = calib_res["refined_fringes"]
            for k, (fx, fy) in enumerate(ref_fringes):
                lbl = "Refined fringes (k)" if k == 0 else None
                ax_img.plot(fx, fy, color="#00e5ff", linewidth=1.1, alpha=0.9, label=lbl)

            x0_fit = calib_res["x0_fit"]
            y0_fit = calib_res["y0_fit"]
            xc_gnom = calib_res["xc_gnom"]
            ax_img.axhline(y0_fit, color="#ffeb3b", linestyle="--", linewidth=1.2, label=f"Symmetry y₀ = {y0_fit:.1f} px")
            ax_img.plot([x0_fit], [y0_fit], "r+", markersize=12, markeredgewidth=2.0, label=f"Optical axis x₀ = {x0_fit:.1f} px")

            ax_img.set_title(
                f"Step 5.1B — {len(ref_fringes)} Refined Fringes | "
                f"Optical Axis ({x0_fit:.1f}, {y0_fit:.1f}) px | Gnomonic Xc = {xc_gnom:.1f} px",
                fontsize=10,
            )
            ax_img.set_xlabel("Detector X (px)")
            ax_img.set_ylabel("Detector Y (px)")
            ax_img.legend(loc="upper right", fontsize=8)
            self.ax = ax_img

        self.lbl_stats.setText(
            f"Laser Calib: {len(calib_res['refined_fringes'])} modes  |  "
            f"(x₀, y₀)=({calib_res['x0_fit']:.1f}, {calib_res['y0_fit']:.1f}) px  |  "
            f"RMS: {calib_res['rms_raw_mhz']:.2f} MHz raw, {calib_res['rms_smooth_mhz']:.2f} MHz smooth"
        )
        self.canvas.draw_idle()

    # ==========================================================================
    # Step 5.2: Rubidium Absorption & 2D Frequency Map Diagnostic Views
    # ==========================================================================

    def show_rb_calibration_diagnostics(
        self,
        rb_res: Dict[str, Any],
        view_mode: str = "fsr_fit",
    ):
        """
        Renders diagnostic plots for Step 5.2 (Rubidium Calibration & 2D Frequency Map):
          - `'rb_denoise_radial'`: Step 8–9 Denoised Rb image + radial order profile.
          - `'fsr_fit'`: Step 10 Folded multi-order Rb spectrum & hyperfine FSR fit.
          - `'freq_map'`: Step 10B Full 2D calibrated frequency map & intra-order detuning map.
        """
        self._in_diagnostic_view = True
        self._last_live_map_state = None
        self._im_artist = None
        self.figure.clear()

        cmap = self.combo_cmap.currentText()
        fsr_cal = rb_res["FSR_cal"]
        dnu = rb_res["delta_nu_opt"]

        if view_mode == "rb_denoise_radial":
            gs = self.figure.add_gridspec(2, 1, height_ratios=[2.0, 1.0])
            ax_img = self.figure.add_subplot(gs[0])
            ax_rad = self.figure.add_subplot(gs[1])

            data_abs = rb_res["data_absorption"]
            vmin, vmax = self._compute_clim(data_abs)
            im = ax_img.imshow(data_abs, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
            self.figure.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, label="Net ADU")
            ax_img.set_title("Step 5.2A — Denoised Rubidium Absorption Interferogram", fontsize=10)
            ax_img.set_ylabel("Detector Y (px)")

            phi_r = rb_res["phi_order_r"]
            spec_r = rb_res["avg_spectrum_radius"]
            ax_rad.plot(phi_r, spec_r, color="#ab47bc", linewidth=1.2)
            ax_rad.set_title("Radially Averaged Rb Transmission vs. Continuous Interference Order φ(r)", fontsize=10)
            ax_rad.set_xlabel("Interference Order φ")
            ax_rad.set_ylabel("Radial Mean Intensity (ADU)")
            ax_rad.grid(True, alpha=0.3)
            self.ax = ax_img

        elif view_mode == "freq_map":
            ax1 = self.figure.add_subplot(1, 2, 1)
            ax2 = self.figure.add_subplot(1, 2, 2)

            freq_map = rb_res["freq_map"]
            im1 = ax1.imshow(freq_map, cmap="viridis", origin="upper", aspect="equal")
            self.figure.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Frequency (GHz)")
            ax1.set_title(
                f"Step 5.2C — 2D Frequency Map (FSR = {fsr_cal:.4f} GHz)",
                fontsize=10,
            )
            ax1.set_xlabel("Detector X (px)")
            ax1.set_ylabel("Detector Y (px)")

            # Intra-order detuning in [-FSR/2, +FSR/2] around each elastic Rayleigh order
            intra_ghz = ((freq_map + 0.5 * fsr_cal) % fsr_cal) - 0.5 * fsr_cal
            im2 = ax2.imshow(
                intra_ghz,
                cmap="coolwarm",
                vmin=-0.5 * fsr_cal,
                vmax=+0.5 * fsr_cal,
                origin="upper",
                aspect="equal",
            )
            self.figure.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Δν from Elastic Order (GHz)")
            ax2.set_title(
                f"Intra-Order Detuning (Elastic Zero ξ = {rb_res['xi_elastic']:.4f})",
                fontsize=10,
            )
            ax2.set_xlabel("Detector X (px)")
            ax2.set_ylabel("Detector Y (px)")
            self.ax = ax1

        else:
            # Default: 'fsr_fit' — Coherently folded multi-order Rb spectrum & hyperfine match
            gs = self.figure.add_gridspec(2, 1, height_ratios=[1.1, 1.1])
            ax_fold = self.figure.add_subplot(gs[0])
            ax_fit = self.figure.add_subplot(gs[1])

            f_grid = rb_res["f_grid"]
            order_spectra = rb_res["order_spectra"]
            orders_used = rb_res["orders_used"]
            for idx_o, k_ord in enumerate(orders_used):
                lbl = f"Orders k={orders_used[0]}..{orders_used[-1]}" if idx_o == 0 else None
                ax_fold.plot(f_grid, order_spectra[idx_o], alpha=0.35, linewidth=0.9, label=lbl)

            f_valid = rb_res["f_valid"]
            sig_valid = rb_res["sig_valid"]
            base_env = rb_res["baseline_env"]
            ax_fold.plot(f_valid, sig_valid, color="#1e88e5", linewidth=1.6, label="Order-averaged spectrum")
            ax_fold.plot(f_valid, base_env, color="#e53935", linestyle="--", linewidth=1.3, label="Envelope baseline")
            ax_fold.set_title(
                f"Step 5.2B — Coherently Folded Rb Orders (Calibrated FSR = {fsr_cal:.4f} GHz)",
                fontsize=10,
            )
            ax_fold.set_ylabel("Intensity (ADU)")
            ax_fold.grid(True, alpha=0.3)
            ax_fold.legend(loc="upper right", fontsize=8)

            f_fit = rb_res["f_fit_rb"]
            meas_abs = rb_res["meas_abs_fit"]
            theo_abs = rb_res["theo_fit_rb"]
            meas_norm = meas_abs / max(float(np.max(meas_abs)), 1e-9)
            theo_norm = theo_abs / max(float(np.max(theo_abs)), 1e-9)

            ax_fit.plot(f_fit, meas_norm, color="#00897b", linewidth=1.5, label="Measured Rb absorption (norm.)")
            ax_fit.plot(
                f_fit,
                theo_norm,
                color="#d81b60",
                linestyle="--",
                linewidth=1.5,
                label=f"Theoretical Rb D2 (T={rb_res['T_cell']:.0f}°C, Δν={dnu:+.3f} GHz)",
            )
            ax_fit.set_title(
                f"Hyperfine Cross-Correlation Fit: Δν = {dnu:+.3f} GHz  |  FSR = {fsr_cal:.4f} GHz",
                fontsize=10,
            )
            ax_fit.set_xlabel("Relative Frequency within Order (GHz)")
            ax_fit.set_ylabel("Normalized Absorption")
            ax_fit.grid(True, alpha=0.3)
            ax_fit.legend(loc="upper right", fontsize=8)
            self.ax = ax_fold

        self.lbl_stats.setText(
            f"Rb Calib: FSR = {fsr_cal:.4f} GHz  |  "
            f"Δν = {dnu:+.3f} GHz  |  "
            f"ξ_elastic = {rb_res['xi_elastic']:.4f}  |  "
            f"Freq Map: {rb_res['freq_map'].shape[1]}×{rb_res['freq_map'].shape[0]} px"
        )
        self.canvas.draw_idle()

    # ==========================================================================
    # Step 5.3: Sample In-Situ Calibration, DHO Fit & q-Vector Diagnostic Views
    # ==========================================================================

    def show_sample_treatment_diagnostics(
        self,
        sample_res: Dict[str, Any],
        view_mode: str = "dho_maps",
    ):
        """
        Renders diagnostic plots for Step 5.3 (Sample DHO Fitting & Reference Geometry):
          - `'insitu_freq'`: Step 11 In-situ frequency self-calibration & Brillouin windows.
          - `'dho_maps'`: Step 12–13 2D Fitted Brillouin Shift & Linewidth maps.
          - `'q_dispersion'`: Step 14 2D Coaxial Ellipse Geometry & Acoustic Dispersion V_s.
        """
        self._in_diagnostic_view = True
        self._last_live_map_state = None
        self._im_artist = None
        self.figure.clear()

        dho = sample_res["dho_results"]
        q_res = sample_res.get("q_results")

        if view_mode == "insitu_freq":
            gs = self.figure.add_gridspec(2, 2, height_ratios=[1.1, 1.1])
            ax_prof = self.figure.add_subplot(gs[0, :])
            ax_dz = self.figure.add_subplot(gs[1, 0])
            ax_di = self.figure.add_subplot(gs[1, 1])

            f_avg = sample_res["freq_avg_data_sample"]
            d_avg = sample_res["avg_data_sample"]
            ax_prof.plot(f_avg, d_avg, color="#1e88e5", linewidth=1.2, label="Central row-averaged spectrum")
            for idx_p, (f_as, f_s) in enumerate(sample_res["bls_peaks"]):
                lbl_as = "Expected Anti-Stokes / Stokes" if idx_p == 0 else None
                ax_prof.axvline(f_as, color="#e53935", linestyle="--", alpha=0.6, linewidth=0.9, label=lbl_as)
                ax_prof.axvline(f_s, color="#43a047", linestyle="--", alpha=0.6, linewidth=0.9)
            ax_prof.set_title(
                f"Step 5.3A — Central Spectrum & Brillouin Search Windows (Coarse df = {sample_res['df_coarse']*1e3:+.1f} MHz)",
                fontsize=10,
            )
            ax_prof.set_xlabel("Calibrated Frequency (GHz)")
            ax_prof.set_ylabel("Intensity (ADU)")
            ax_prof.grid(True, alpha=0.3)
            ax_prof.legend(loc="upper right", fontsize=8)

            dz_mhz = sample_res["dz_map_2d"] * 1e3
            im_dz = ax_dz.imshow(dz_mhz, cmap="coolwarm", origin="upper", aspect="auto")
            self.figure.colorbar(im_dz, ax=ax_dz, fraction=0.046, pad=0.04, label="MHz")
            ax_dz.set_title(f"2D Zero-Offset Map (RMS={np.std(dz_mhz):.1f} MHz)", fontsize=9)
            ax_dz.set_xlabel("Detector X (px)")
            ax_dz.set_ylabel("Detector Y (px)")

            di_mhz = sample_res["dinst_map_2d"] * 1e3
            im_di = ax_di.imshow(di_mhz, cmap="coolwarm", origin="upper", aspect="auto")
            self.figure.colorbar(im_di, ax=ax_di, fraction=0.046, pad=0.04, label="MHz")
            ax_di.set_title(f"Higher-Order Dispersion Residual (RMS={np.std(di_mhz):.1f} MHz)", fontsize=9)
            ax_di.set_xlabel("Detector X (px)")
            ax_di.set_ylabel("Detector Y (px)")
            self.ax = ax_prof

        elif view_mode == "q_dispersion" and q_res is not None:
            ax1 = self.figure.add_subplot(1, 2, 1)
            ax2 = self.figure.add_subplot(1, 2, 2)

            x_f = q_res["x_coords"]
            y_f = q_res["y_coords"]
            s_f = q_res["shift"]
            opt = q_res["opt_params"]

            sc1 = ax1.scatter(x_f, y_f, c=s_f, s=6, cmap="viridis")
            self.figure.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="Shift ν_B (GHz)")

            # Overlay fitted concentric coaxial ellipses
            theta_t = np.linspace(0.0, 2.0 * np.pi, 200)
            ct = np.cos(opt["phi_rad"])
            st = np.sin(opt["phi_rad"])
            for r_iso in (300.0, 600.0, 900.0, 1200.0):
                xp = r_iso * np.cos(theta_t)
                yp = opt["gamma"] * r_iso * np.sin(theta_t)
                xe = opt["xc"] + xp * ct - yp * st
                ye = opt["yc"] + xp * st + yp * ct
                ax1.plot(xe, ye, "r--", linewidth=1.0, alpha=0.8)
            ax1.plot([opt["xc"]], [opt["yc"]], "r+", markersize=12, markeredgewidth=2.0)
            ax1.invert_yaxis()
            ax1.set_title(
                f"Step 5.3C — Coaxial Ellipses: Center=({opt['xc']:.1f}, {opt['yc']:.1f}), γ={opt['gamma']:.3f}",
                fontsize=9,
            )
            ax1.set_xlabel("Detector X (px)")
            ax1.set_ylabel("Detector Y (px)")
            ax1.grid(True, alpha=0.25)

            theta_deg = q_res["theta_deg"]
            ax2.scatter(theta_deg, s_f, s=5, alpha=0.4, color="#1e88e5", label="Measured DHO Shift")
            th_grid = np.linspace(0.0, max(float(np.max(theta_deg)), 5.0), 200)
            s_curve = q_res["Shift_0"] * np.cos(0.5 * np.deg2rad(th_grid))
            ax2.plot(
                th_grid,
                s_curve,
                "r-",
                linewidth=1.8,
                label=f"Fit: S(0)={q_res['Shift_0']:.3f} GHz, M_θ={q_res['M_theta']:.0f} px",
            )
            vs = q_res["V_s"]
            vs_err = q_res.get("V_s_err") or 0.0
            ax2.set_title(
                f"Acoustic Dispersion: V_s = {vs:.1f} ± {vs_err:.1f} m/s",
                fontsize=10,
            )
            ax2.set_xlabel("Internal Scattering Angle θ (deg)")
            ax2.set_ylabel("Brillouin Shift ν_B (GHz)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right", fontsize=8)
            self.ax = ax1

        else:
            # Default: 'dho_maps' — 2D Fitted Brillouin Shift & Linewidth maps
            ax1 = self.figure.add_subplot(1, 2, 1)
            ax2 = self.figure.add_subplot(1, 2, 2)

            xa = dho["x_coords_averaged"]
            ya = dho["y_coords_averaged"]
            sa = dho["shift_clipped_averaged"]
            la = dho["linewidth_clipped_averaged"]
            valid = np.isfinite(sa) & np.isfinite(la) & np.isfinite(xa) & np.isfinite(ya)

            if np.any(valid):
                sc1 = ax1.scatter(xa[valid], ya[valid], c=sa[valid], s=6, cmap="viridis")
                self.figure.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="Shift ν_B (GHz)")
                sc2 = ax2.scatter(xa[valid], ya[valid], c=la[valid], s=6, cmap="magma")
                self.figure.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04, label="Linewidth Γ_B (GHz)")

            ax1.invert_yaxis()
            ax1.set_title(
                f"Step 5.3B — Order-Averaged Brillouin Shift ({dho['bin_val_h']}×{dho['bin_val_w']} bin)",
                fontsize=9,
            )
            ax1.set_xlabel("Detector X (px)")
            ax1.set_ylabel("Detector Y (px)")
            ax1.grid(True, alpha=0.25)

            ax2.invert_yaxis()
            ax2.set_title(
                f"Brillouin Linewidth (S/AS Splitting RMS = {dho['stokes_antistokes_rms_mhz']:.1f} MHz)",
                fontsize=9,
            )
            ax2.set_xlabel("Detector X (px)")
            ax2.set_ylabel("Detector Y (px)")
            ax2.grid(True, alpha=0.25)
            self.ax = ax1

        vs_str = (
            f"  |  V_s = {q_res['V_s']:.1f} ± {(q_res.get('V_s_err') or 0.0):.1f} m/s"
            if q_res is not None and "V_s" in q_res
            else ""
        )
        self.lbl_stats.setText(
            f"DHO Fit ({dho['bin_val_h']}×{dho['bin_val_w']}): {dho['n_fringes']} orders  |  "
            f"Mean Shift = {float(np.nanmean(dho['shift_clipped_averaged'])):.3f} GHz  |  "
            f"Mean LW = {float(np.nanmean(dho['linewidth_clipped_averaged'])):.3f} GHz{vs_str}"
        )
        self.canvas.draw_idle()

    # ==========================================================================
    # Step 5.4: Live 2D Spatial Map Heatmap + Interferogram View
    # ==========================================================================

    def update_live_map_view(
        self,
        frame: Optional[np.ndarray],
        x_coords_um: np.ndarray,
        y_coords_um: np.ndarray,
        map_mean_shift: np.ndarray,
        map_mean_lw: np.ndarray,
        map_shift_0: np.ndarray,
        map_vs: np.ndarray,
        iy: int,
        ix: int,
        pos_um: Optional[Dict[str, Optional[float]]] = None,
        point_fit: Optional[Dict[str, Any]] = None,
        step_um: Optional[tuple] = None,
    ):
        """
        Renders a side-by-side live view during spatial mapping:
          - Left subplot: Latest acquired 2D camera interferogram at `(iy, ix)`.
          - Right subplot: Live 2D spatial heatmap `(X_um, Y_um)` of the selected Brillouin metric
            with proportional physical step aspect ratio (`aspect="equal"`).
        """
        self._last_live_map_state = {
            "frame": None if frame is None else np.asarray(frame),
            "x_coords_um": np.asarray(x_coords_um, dtype=np.float64),
            "y_coords_um": np.asarray(y_coords_um, dtype=np.float64),
            "map_mean_shift": np.array(map_mean_shift, dtype=np.float64, copy=True),
            "map_mean_lw": np.array(map_mean_lw, dtype=np.float64, copy=True),
            "map_shift_0": np.array(map_shift_0, dtype=np.float64, copy=True),
            "map_vs": np.array(map_vs, dtype=np.float64, copy=True),
            "iy": int(iy),
            "ix": int(ix),
            "pos_um": pos_um,
            "point_fit": point_fit,
            "step_um": tuple(float(v) for v in step_um) if step_um is not None else None,
        }
        self._render_live_map_state()

    def _on_map_metric_changed(self):
        if self._in_diagnostic_view and self._last_live_map_state is not None:
            self._render_live_map_state()

    def _render_live_map_state(self):
        st = self._last_live_map_state
        if st is None:
            return

        self._in_diagnostic_view = True
        self._im_artist = None
        self.figure.clear()

        frame = st["frame"]
        xs = st["x_coords_um"]
        ys = st["y_coords_um"]
        iy, ix = st["iy"], st["ix"]
        pos_um = st["pos_um"] or {}
        point_fit = st["point_fit"]

        ax_img = self.figure.add_subplot(1, 2, 1)
        ax_map = self.figure.add_subplot(1, 2, 2)

        x_val = pos_um.get("X")
        if x_val is None:
            x_val = float(xs[min(ix, len(xs) - 1)])
        y_val = pos_um.get("Y")
        if y_val is None:
            y_val = float(ys[min(iy, len(ys) - 1)])

        if frame is not None:
            cmap_img = self.combo_cmap.currentText()
            vmin, vmax = self._compute_clim(frame)
            im1 = ax_img.imshow(frame, cmap=cmap_img, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
            self.figure.colorbar(im1, ax=ax_img, fraction=0.046, pad=0.04, label="Counts (ADU)")
            ax_img.set_title(
                f"Point [row={iy}, col={ix}] @ ({x_val:.1f}, {y_val:.1f}) µm",
                fontsize=10,
            )
            ax_img.set_xlabel("Detector X (px)")
            ax_img.set_ylabel("Detector Y (px)")
        else:
            ax_img.set_facecolor("#1e1e1e")
            ax_img.text(
                0.5,
                0.5,
                f"Map scan started\nAcquiring [row={iy}, col={ix}] @ ({x_val:.1f}, {y_val:.1f}) µm...",
                color="#cfd8dc",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
                fontsize=10,
            )
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title("Detector Frame (waiting for point 1)", fontsize=10)
            self.lbl_stats.setText("Map initialized — acquiring first point...")

        metric_key = self.combo_map_metric.currentData() or "mean_shift"
        if metric_key == "v_s":
            data_2d = st["map_vs"]
            metric_title = "Live 2D Map — Velocity V_s (m/s)"
            cbar_lbl = "V_s (m/s)"
            cmap_m = "plasma"
        elif metric_key == "mean_lw":
            data_2d = st["map_mean_lw"]
            metric_title = "Live 2D Map — Mean Linewidth Γ_B (GHz)"
            cbar_lbl = "Γ_B (GHz)"
            cmap_m = "magma"
        elif metric_key == "shift_0":
            data_2d = st["map_shift_0"]
            metric_title = "Live 2D Map — Normal Shift S(0) (GHz)"
            cbar_lbl = "S(0) (GHz)"
            cmap_m = "viridis"
        else:
            data_2d = st["map_mean_shift"]
            metric_title = "Live 2D Map — Mean Shift ν_B (GHz)"
            cbar_lbl = "ν_B (GHz)"
            cmap_m = "viridis"

        # Compute extent in µm using configured step sizes so pixel sizes are proportional
        step_um = st.get("step_um")
        if step_um is not None and len(step_um) == 2:
            dx_step = float(abs(step_um[0])) if step_um[0] else 5.0
            dy_step = float(abs(step_um[1])) if step_um[1] else 5.0
        else:
            dx_step = float(abs(xs[1] - xs[0])) if len(xs) > 1 else None
            dy_step = float(abs(ys[1] - ys[0])) if len(ys) > 1 else None
            if dx_step is None and dy_step is None:
                dx_step = dy_step = 5.0
            elif dx_step is None:
                dx_step = dy_step
            elif dy_step is None:
                dy_step = dx_step
        dx_half = 0.5 * dx_step
        dy_half = 0.5 * dy_step
        x_min, x_max = float(np.min(xs)) - dx_half, float(np.max(xs)) + dx_half
        y_min, y_max = float(np.min(ys)) - dy_half, float(np.max(ys)) + dy_half

        im2 = ax_map.imshow(
            data_2d,
            cmap=cmap_m,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
            interpolation="nearest",
        )
        ax_map.set_xlim(x_min, x_max)
        ax_map.set_ylim(y_min, y_max)
        self.figure.colorbar(im2, ax=ax_map, fraction=0.046, pad=0.04, label=cbar_lbl)
        ax_map.plot([x_val], [y_val], "r+", markersize=10, markeredgewidth=2.0)
        ax_map.set_title(metric_title, fontsize=10)
        ax_map.set_xlabel("Stage X (µm)")
        ax_map.set_ylabel("Stage Y (µm)")
        self.ax = ax_img

        if point_fit is not None:
            self.lbl_stats.setText(
                f"Map [{iy},{ix}]: ν_B = {point_fit['mean_shift_ghz']:.3f} GHz  |  "
                f"Γ_B = {point_fit['mean_lw_ghz']:.3f} GHz  |  "
                f"S(0) = {point_fit['Shift_0_ghz']:.3f} GHz  |  "
                f"V_s = {point_fit['V_s_ms']:.1f} m/s"
            )
        self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self.lbl_cursor.setText("Cursor: —")
            return
        if self._in_diagnostic_view:
            self.lbl_cursor.setText(f"Cursor: X = {event.xdata:.2f}, Y = {event.ydata:.2f}")
            return
        if self._current_image is None or event.inaxes != self.ax:
            self.lbl_cursor.setText("Cursor: —")
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        h, w = self._current_image.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            val = int(self._current_image[row, col])
            self.lbl_cursor.setText(f"Cursor: X = {col:4d} px, Y = {row:4d} px   |   Intensity = {val:5d} ADU")
        else:
            self.lbl_cursor.setText("Cursor: —")
