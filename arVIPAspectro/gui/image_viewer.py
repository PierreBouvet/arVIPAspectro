"""
2D Camera Frame & Calibration Diagnostic Viewer Widget (`ImageViewerWidget`)
for displaying acquired Hamamatsu Orca Flash 4.0 v3 interferograms and
step-by-step calibration diagnostics.
"""

import time
from typing import Optional, Dict, Any
import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg
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
        self._syncing_map_range: bool = False
        self._last_pg_shape: Optional[tuple] = None
        self._last_pg_cmap: Optional[str] = None
        self._pg_cmap_cache: Dict[str, Any] = {}
        self._last_stats_ts: float = 0.0
        self._map_range_settings: Dict[str, Dict[str, Any]] = {
            "mean_shift": {
                "state_key": "map_mean_shift",
                "title": "Live 2D Map — Mean Shift ν_B (GHz)",
                "cbar_label": "ν_B (GHz)",
                "cmap": "viridis",
                "auto": True,
                "vmin": 4.0,
                "vmax": 7.0,
                "decimals": 3,
                "step": 0.05,
                "bounds": (-100.0, 100.0),
                "suffix": " GHz",
            },
            "mean_lw": {
                "state_key": "map_mean_lw",
                "title": "Live 2D Map — Mean Linewidth Γ_B (GHz)",
                "cbar_label": "Γ_B (GHz)",
                "cmap": "magma",
                "auto": True,
                "vmin": 0.2,
                "vmax": 1.5,
                "decimals": 3,
                "step": 0.05,
                "bounds": (-100.0, 100.0),
                "suffix": " GHz",
            },
            "shift_0": {
                "state_key": "map_shift_0",
                "title": "Live 2D Map — Normal Shift S(0) (GHz)",
                "cbar_label": "S(0) (GHz)",
                "cmap": "viridis",
                "auto": True,
                "vmin": 4.0,
                "vmax": 7.0,
                "decimals": 3,
                "step": 0.05,
                "bounds": (-100.0, 100.0),
                "suffix": " GHz",
            },
            "v_s": {
                "state_key": "map_vs",
                "title": "Live 2D Map — Velocity V_s (m/s)",
                "cbar_label": "V_s (m/s)",
                "cmap": "plasma",
                "auto": True,
                "vmin": 1300.0,
                "vmax": 2000.0,
                "decimals": 1,
                "step": 10.0,
                "bounds": (-1000.0, 50000.0),
                "suffix": " m/s",
            },
        }

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Top Detector Display Controls Bar
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

        ctrl_bar.addStretch()
        self.lbl_stats = QtWidgets.QLabel("No frame loaded")
        self.lbl_stats.setStyleSheet("font-family: monospace; font-size: 11px;")
        ctrl_bar.addWidget(self.lbl_stats)
        layout.addLayout(ctrl_bar)

        # 1b. Live Spatial Map Display & Colorscale Range Bar
        map_bar = QtWidgets.QHBoxLayout()
        map_bar.addWidget(QtWidgets.QLabel("Map Metric:"))
        self.combo_map_metric = QtWidgets.QComboBox()
        self.combo_map_metric.addItem("Mean Shift ν_B (GHz)", "mean_shift")
        self.combo_map_metric.addItem("Velocity V_s (m/s)", "v_s")
        self.combo_map_metric.addItem("Mean Linewidth Γ_B (GHz)", "mean_lw")
        self.combo_map_metric.addItem("Normal Shift S(0) (GHz)", "shift_0")
        self.combo_map_metric.currentIndexChanged.connect(self._on_map_metric_changed)
        map_bar.addWidget(self.combo_map_metric)

        map_bar.addSpacing(12)
        self.chk_map_auto_range = QtWidgets.QCheckBox("Auto Range")
        self.chk_map_auto_range.setChecked(True)
        self.chk_map_auto_range.toggled.connect(self._on_map_auto_toggled)
        map_bar.addWidget(self.chk_map_auto_range)

        map_bar.addSpacing(6)
        map_bar.addWidget(QtWidgets.QLabel("Min:"))
        self.spin_map_vmin = QtWidgets.QDoubleSpinBox()
        self.spin_map_vmin.setKeyboardTracking(False)
        self.spin_map_vmin.setMinimumWidth(105)
        self.spin_map_vmin.valueChanged.connect(self._on_map_bounds_changed)
        map_bar.addWidget(self.spin_map_vmin)

        map_bar.addSpacing(4)
        map_bar.addWidget(QtWidgets.QLabel("Max:"))
        self.spin_map_vmax = QtWidgets.QDoubleSpinBox()
        self.spin_map_vmax.setKeyboardTracking(False)
        self.spin_map_vmax.setMinimumWidth(105)
        self.spin_map_vmax.valueChanged.connect(self._on_map_bounds_changed)
        map_bar.addWidget(self.spin_map_vmax)

        map_bar.addStretch()
        layout.addLayout(map_bar)
        self._sync_map_range_widgets()

        # 2. Stacked Display Area: PyQtGraph (live/2D frames) & Matplotlib (diagnostics)
        self._view_stack = QtWidgets.QStackedWidget()

        # Page 0: PyQtGraph high-FPS 2D image canvas
        self._pg_widget = pg.GraphicsLayoutWidget()
        self._pg_widget.setBackground("#1e1e1e")
        self._pg_plot = self._pg_widget.addPlot(title=self._current_title)
        self._pg_plot.invertY(True)
        self._pg_plot.setAspectLocked(True)
        self._pg_plot.setLabel("bottom", "Detector X (px)")
        self._pg_plot.setLabel("left", "Detector Y (px)")
        self._pg_img_item = pg.ImageItem(axisOrder="row-major")
        self._pg_plot.addItem(self._pg_img_item)

        init_cmap = self._get_pg_colormap(self.combo_cmap.currentText())
        self._pg_cbar = pg.ColorBarItem(
            values=(0, 65535),
            colorMap=init_cmap,
            label="Counts (ADU)",
            interactive=True,
        )
        self._pg_cbar.setImageItem(self._pg_img_item, insert_in=self._pg_plot)
        self._last_pg_cmap = self.combo_cmap.currentText()
        self._view_stack.addWidget(self._pg_widget)

        # Page 1: Matplotlib Figure & Navigation Toolbar (for multi-panel diagnostics)
        self._mpl_page = QtWidgets.QWidget()
        mpl_layout = QtWidgets.QVBoxLayout(self._mpl_page)
        mpl_layout.setContentsMargins(0, 0, 0, 0)
        mpl_layout.setSpacing(4)
        self.figure = Figure(figsize=(6, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(self._current_title, fontsize=11)
        self.ax.set_xlabel("Detector X (px)")
        self.ax.set_ylabel("Detector Y (px)")
        self.toolbar = NavigationToolbar(self.canvas, self._mpl_page)
        mpl_layout.addWidget(self.toolbar)
        mpl_layout.addWidget(self.canvas, stretch=1)
        self._view_stack.addWidget(self._mpl_page)

        self._view_stack.setCurrentWidget(self._pg_widget)
        layout.addWidget(self._view_stack, stretch=1)

        # 3. Bottom Cursor Readout Bar
        self.lbl_cursor = QtWidgets.QLabel("Cursor: —")
        self.lbl_cursor.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.lbl_cursor)

        self._pg_plot.scene().sigMouseMoved.connect(self._on_pg_mouse_move)
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

        # Compute statistics (throttled to ~10 Hz during rapid live streaming)
        now = time.monotonic()
        if (now - self._last_stats_ts) >= 0.1 or self._last_pg_shape != (h, w):
            self._last_stats_ts = now
            vmin_raw = int(np.min(self._current_image))
            vmax_raw = int(np.max(self._current_image))
            vmean = float(np.mean(self._current_image))
            n_sat = int(np.sum(self._current_image >= 65000))
            self.lbl_stats.setText(
                f"Min: {vmin_raw}  |  Max: {vmax_raw}  |  Mean: {vmean:.1f}  |  Sat(≥65k): {n_sat} px"
            )

        self._in_diagnostic_view = False
        self._last_live_map_state = None
        self._render_image()

    def _get_pg_colormap(self, cmap_name: str):
        if cmap_name not in self._pg_cmap_cache:
            self._pg_cmap_cache[cmap_name] = pg.colormap.getFromMatplotlib(cmap_name)
        return self._pg_cmap_cache[cmap_name]

    def _enter_diagnostic_mode(self):
        self._in_diagnostic_view = True
        self._im_artist = None
        self.figure.clear()
        self._view_stack.setCurrentWidget(self._mpl_page)

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

        self._view_stack.setCurrentWidget(self._pg_widget)

        cmap_name = self.combo_cmap.currentText()
        vmin, vmax = self._compute_clim(self._current_image)
        h, w = self._current_image.shape[:2]

        if cmap_name != self._last_pg_cmap:
            self._pg_cbar.setColorMap(self._get_pg_colormap(cmap_name))
            self._last_pg_cmap = cmap_name

        self._pg_img_item.setImage(self._current_image, autoLevels=False, levels=(vmin, vmax))
        self._pg_cbar.setLevels((vmin, vmax))
        self._pg_plot.setTitle(self._current_title, size="10pt")

        if self._last_pg_shape != (h, w):
            self._last_pg_shape = (h, w)
            self._pg_plot.autoRange()

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
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

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
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

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
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

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

    @staticmethod
    def _compute_finite_minmax(data_2d: np.ndarray):
        finite_mask = np.isfinite(data_2d)
        if not np.any(finite_mask):
            return None
        vals = data_2d[finite_mask]
        return float(np.min(vals)), float(np.max(vals))

    def _current_metric_key(self) -> str:
        return self.combo_map_metric.currentData() or "mean_shift"

    def _sync_map_range_widgets(self, live_bounds=None):
        """Synchronizes Auto checkbox and unit-aware Min/Max spinboxes for the active metric."""
        metric_key = self._current_metric_key()
        cfg = self._map_range_settings[metric_key]
        is_auto = bool(cfg["auto"])

        if live_bounds is None and is_auto and self._last_live_map_state is not None:
            data_2d = self._last_live_map_state[cfg["state_key"]]
            live_bounds = self._compute_finite_minmax(data_2d)

        vmin_disp, vmax_disp = (
            live_bounds if (is_auto and live_bounds is not None) else (cfg["vmin"], cfg["vmax"])
        )

        self._syncing_map_range = True
        try:
            self.chk_map_auto_range.setChecked(is_auto)
            lo_b, hi_b = cfg["bounds"]
            for spin, val in ((self.spin_map_vmin, vmin_disp), (self.spin_map_vmax, vmax_disp)):
                spin.setDecimals(cfg["decimals"])
                spin.setSingleStep(cfg["step"])
                spin.setRange(lo_b, hi_b)
                spin.setSuffix(cfg["suffix"])
                spin.setValue(float(val))
                spin.setEnabled(not is_auto)
        finally:
            self._syncing_map_range = False

    def _on_map_metric_changed(self):
        self._sync_map_range_widgets()
        if self._in_diagnostic_view and self._last_live_map_state is not None:
            self._render_live_map_state()

    def _on_map_auto_toggled(self, checked: bool):
        if self._syncing_map_range:
            return
        metric_key = self._current_metric_key()
        cfg = self._map_range_settings[metric_key]
        cfg["auto"] = bool(checked)

        # Pre-fill Manual Min/Max with the current finite Auto bounds when switching Auto -> Manual
        if not checked and self._last_live_map_state is not None:
            minmax = self._compute_finite_minmax(self._last_live_map_state[cfg["state_key"]])
            if minmax is not None:
                d_min, d_max = minmax
                eps = 1.0 if metric_key == "v_s" else 0.001
                cfg["vmin"] = d_min
                cfg["vmax"] = d_max if d_max > d_min else d_min + eps

        self._sync_map_range_widgets()
        if self._in_diagnostic_view and self._last_live_map_state is not None:
            self._render_live_map_state()

    def _on_map_bounds_changed(self):
        if self._syncing_map_range:
            return
        metric_key = self._current_metric_key()
        cfg = self._map_range_settings[metric_key]
        cfg["vmin"] = float(self.spin_map_vmin.value())
        cfg["vmax"] = float(self.spin_map_vmax.value())
        if not cfg["auto"] and self._in_diagnostic_view and self._last_live_map_state is not None:
            self._render_live_map_state()

    def _render_live_map_state(self):
        st = self._last_live_map_state
        if st is None:
            return

        self._enter_diagnostic_mode()

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

        metric_key = self._current_metric_key()
        cfg = self._map_range_settings[metric_key]
        data_2d = st[cfg["state_key"]]
        metric_title = cfg["title"]
        cbar_lbl = cfg["cbar_label"]
        cmap_m = cfg["cmap"]

        live_bounds = self._compute_finite_minmax(data_2d)
        if cfg["auto"]:
            if live_bounds is not None:
                self._sync_map_range_widgets(live_bounds=live_bounds)
                d_min, d_max = live_bounds
                map_vmin, map_vmax = (d_min, d_max) if d_max > d_min else (None, None)
            else:
                map_vmin, map_vmax = None, None
        else:
            map_vmin = float(cfg["vmin"])
            map_vmax = float(cfg["vmax"])
            if map_vmax <= map_vmin:
                eps = 1.0 if metric_key == "v_s" else 1e-3
                map_vmax = map_vmin + eps

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
            vmin=map_vmin,
            vmax=map_vmax,
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

    def show_brightfield_alignment(
        self,
        ref_img: np.ndarray,
        mov_img: np.ndarray,
        ref_label: str,
        cur_label: str,
        dcol_px: float,
        drow_px: float,
        dx_um: float,
        dy_um: float,
    ):
        """
        Displays a before/after RGB false-color overlay comparing `ref_img` (magenta)
        vs `mov_img` (green) before and after compensating `(dcol_px, drow_px)`.
        """
        from scipy.ndimage import shift as nd_shift

        self._last_live_map_state = None
        self._enter_diagnostic_mode()

        def _norm01(a: np.ndarray) -> np.ndarray:
            arr = np.asarray(a, dtype=np.float64)
            lo, hi = np.percentile(arr, [1.0, 99.5])
            if hi <= lo:
                return np.zeros_like(arr)
            return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        ref_n = _norm01(ref_img)
        mov_n = _norm01(mov_img)
        mov_aligned = np.clip(
            nd_shift(mov_n, shift=(-float(drow_px), -float(dcol_px)), order=1, mode="nearest"),
            0.0,
            1.0,
        )

        # Magenta (R=ref, B=ref) vs Green (G=mov): aligned overlapping features appear neutral gray/white
        rgb_before = np.stack([ref_n, mov_n, ref_n], axis=-1)
        rgb_after = np.stack([ref_n, mov_aligned, ref_n], axis=-1)

        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        ax1.imshow(rgb_before, origin="upper", aspect="equal")
        h, w = ref_n.shape[:2]
        cx, cy = 0.5 * w, 0.5 * h
        if abs(dcol_px) > 0.5 or abs(drow_px) > 0.5:
            ax1.annotate(
                "",
                xy=(cx + dcol_px, cy + drow_px),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2.0),
            )
        ax1.set_title(
            f"Before Alignment: {ref_label} (Mag) vs {cur_label} (Grn)\n"
            f"Shift = ({dcol_px:+.2f}, {drow_px:+.2f}) px",
            fontsize=9,
        )
        ax1.set_xlabel("X (px)")
        ax1.set_ylabel("Y (px)")

        ax2.imshow(rgb_after, origin="upper", aspect="equal")
        ax2.set_title(
            f"After Stage Correction: ΔX = {dx_um:+.2f} µm, ΔY = {dy_um:+.2f} µm\n"
            "(Aligned Overlay Preview)",
            fontsize=9,
        )
        ax2.set_xlabel("X (px)")
        ax2.set_ylabel("Y (px)")

        self.ax = ax1
        self.lbl_stats.setText(
            f"BF Align [{cur_label} → {ref_label}]: Δpx=({dcol_px:+.2f}, {drow_px:+.2f}) | "
            f"Stage Δ=({dx_um:+.2f}, {dy_um:+.2f}) µm"
        )
        self.canvas.draw_idle()

    def _render_plate_jog_panels(
        self,
        img0: np.ndarray,
        img_x: np.ndarray,
        img_y: np.ndarray,
        x0: float,
        y0: float,
        dx_um: float,
        dy_um: float,
    ):
        """Renders the 3 brightfield jog images in subplots (1..3) of a 2x2 grid and returns ax4."""
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

        cmap = self.combo_cmap.currentText()
        specs = [
            (1, img0, f"1. Base ({x0:.1f}, {y0:.1f}) µm"),
            (2, img_x, f"2. +X ({x0 + dx_um:.1f}, {y0:.1f}) µm [ΔX={dx_um:+.1f}]"),
            (3, img_y, f"3. +Y ({x0:.1f}, {y0 + dy_um:.1f}) µm [ΔY={dy_um:+.1f}]"),
        ]
        first_ax = None
        for idx, arr, title in specs:
            ax = self.figure.add_subplot(2, 2, idx)
            if first_ax is None:
                first_ax = ax
            vmin, vmax = np.percentile(arr, [1.0, 99.5])
            if vmax <= vmin:
                vmin, vmax = float(np.min(arr)), float(np.max(arr)) + 1.0
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")

        self.ax = first_ax
        return self.figure.add_subplot(2, 2, 4)

    def show_plate_calib_images(
        self,
        img0: np.ndarray,
        img_x: np.ndarray,
        img_y: np.ndarray,
        x0: float,
        y0: float,
        dx_um: float,
        dy_um: float,
    ):
        """Step 1 diagnostic view: displays the 3 brightfield jog images before correlation."""
        ax4 = self._render_plate_jog_panels(img0, img_x, img_y, x0, y0, dx_um, dy_um)
        ax4.set_facecolor("#1e1e1e")
        ax4.text(
            0.5,
            0.5,
            "3 jog frames captured.\nClick 'Continue' to compute\nautocorrelation & cross-correlation.",
            color="#cfd8dc",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=10,
        )
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title("4. Correlation Result (Pending)", fontsize=9)
        self.lbl_stats.setText(
            f"Plate Calib Step 1: 3 frames captured at Base ({x0:.1f}, {y0:.1f}), +{dx_um:.1f} X, +{dy_um:.1f} Y µm"
        )
        self.canvas.draw_idle()

    def show_plate_calib_correlation(
        self,
        img0: np.ndarray,
        img_x: np.ndarray,
        img_y: np.ndarray,
        x0: float,
        y0: float,
        dx_um: float,
        dy_um: float,
        corr_res: Dict[str, Any],
    ):
        """
        Step 2 diagnostic view: shows the 3 jog images plus subplot 4 as an RGB composite of
        the correlation maps centered at (0, 0): Red=CC(+X), Green=CC(+Y), Blue=AutoCorr(Base).
        """
        ax4 = self._render_plate_jog_panels(img0, img_x, img_y, x0, y0, dx_um, dy_um)

        cc_auto = np.fft.fftshift(corr_res["cc_auto"])
        cc_x = np.fft.fftshift(corr_res["cc_x"])
        cc_y = np.fft.fftshift(corr_res["cc_y"])
        dc_x, dr_x = corr_res["dp_x"]
        dc_y, dr_y = corr_res["dp_y"]
        pk_x = float(corr_res["peak_x"])
        pk_y = float(corr_res["peak_y"])

        h, w = cc_auto.shape
        cy, cx = h // 2, w // 2
        half_span = int(min(max(120.0, 2.0 * max(abs(dc_x), abs(dr_x), abs(dc_y), abs(dr_y)) + 40.0), cy, cx))
        y1, y2 = cy - half_span, cy + half_span + 1
        x1, x2 = cx - half_span, cx + half_span + 1

        def _pos_norm(a: np.ndarray) -> np.ndarray:
            crop = np.maximum(a[y1:y2, x1:x2], 0.0)
            m = float(np.max(crop))
            return np.clip(crop / m, 0.0, 1.0) if m > 1e-12 else np.zeros_like(crop)

        rgb_corr = np.stack([_pos_norm(cc_x), _pos_norm(cc_y), _pos_norm(cc_auto)], axis=-1)
        ax4.imshow(
            rgb_corr,
            origin="upper",
            extent=[-half_span - 0.5, half_span + 0.5, half_span + 0.5, -half_span - 0.5],
            aspect="equal",
        )
        ax4.axhline(0.0, color="white", ls=":", lw=0.7, alpha=0.5)
        ax4.axvline(0.0, color="white", ls=":", lw=0.7, alpha=0.5)
        ax4.plot([0.0], [0.0], "wo", markersize=4, fillstyle="none")
        ax4.annotate("", xy=(dc_x, dr_x), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="->", color="#ff5252", lw=1.8))
        ax4.annotate("", xy=(dc_y, dr_y), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="->", color="#69f0ae", lw=1.8))
        ax4.set_title(
            f"4. RGB Correlation (R=+X, G=+Y, B=Auto)\n"
            f"+X=({dc_x:+.1f},{dr_x:+.1f}) px [{pk_x:.2f}] | +Y=({dc_y:+.1f},{dr_y:+.1f}) px [{pk_y:.2f}]",
            fontsize=8.5,
        )
        ax4.set_xlabel("Shift ΔX (px)")
        ax4.set_ylabel("Shift ΔY (px)")
        self.lbl_stats.setText(
            f"Plate Calib Step 2: +X=({dc_x:+.2f}, {dr_x:+.2f}) px [pk={pk_x:.3f}] | "
            f"+Y=({dc_y:+.2f}, {dr_y:+.2f}) px [pk={pk_y:.3f}]"
        )
        self.canvas.draw_idle()

    def _on_pg_mouse_move(self, scene_pos):
        if self._in_diagnostic_view or self._current_image is None:
            return
        if not self._pg_plot.sceneBoundingRect().contains(scene_pos):
            self.lbl_cursor.setText("Cursor: —")
            return
        pt = self._pg_plot.vb.mapSceneToView(scene_pos)
        col = int(round(pt.x()))
        row = int(round(pt.y()))
        h, w = self._current_image.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            val = int(self._current_image[row, col])
            self.lbl_cursor.setText(f"Cursor: X = {col:4d} px, Y = {row:4d} px   |   Intensity = {val:5d} ADU")
        else:
            self.lbl_cursor.setText("Cursor: —")

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
