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
        self.combo_contrast.addItem("Auto (1% – 99.8%)", "percentile")
        self.combo_contrast.addItem("Min – Max", "minmax")
        self.combo_contrast.addItem("Full 16-bit (0 – 65535)", "uint16")
        self.combo_contrast.currentIndexChanged.connect(self._on_display_settings_changed)
        ctrl_bar.addWidget(self.combo_contrast)

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
        self._render_image()

    def _compute_clim(self, img: np.ndarray):
        mode = self.combo_contrast.currentData()
        if mode == "uint16":
            return 0.0, 65535.0
        if mode == "minmax":
            vmin, vmax = float(np.min(img)), float(np.max(img))
            return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)
        sample = img[::4, ::4] if img.size > 65536 else img
        vmin, vmax = np.percentile(sample, [1.0, 99.8])
        if vmax <= vmin:
            vmin, vmax = float(np.min(img)), float(np.max(img)) + 1.0
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
        if not self._in_diagnostic_view:
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
