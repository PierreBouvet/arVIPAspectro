"""
2D Camera Frame & Calibration Diagnostic Viewer Widget (`ImageViewerWidget`)
for displaying acquired interferograms, live 2D spatial maps, Brightfield alignments,
and step-by-step calibration diagnostics using PyQtGraph.
"""

import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

from .. import plotting as pg_helpers


class ImageViewerWidget(QtWidgets.QWidget):
    """
    Interactive 2D PyQtGraph image and multi-panel diagnostic canvas with colormap/contrast
    controls, frame statistics, live cursor coordinate/value readout, and per-metric map limits.
    """

    DEFAULT_CMAP = "gray"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: Optional[np.ndarray] = None
        self._current_title: str = "No image acquired yet"
        self._in_diagnostic_view: bool = False
        self._view_mode: str = "single"  # "single", "live_map", "diagnostic", or "dual_brightfield"
        self._last_live_map_state: Optional[Dict[str, Any]] = None
        self._live_map_widgets: Optional[Dict[str, Any]] = None
        self._dual_bf_widgets: Optional[Dict[str, Any]] = None
        self._dual_ref_img: Optional[np.ndarray] = None
        self._dual_ref_title: str = "Reference"
        self._dual_live_img: Optional[np.ndarray] = None
        self._syncing_map_range: bool = False
        self._syncing_contrast: bool = False
        self._last_pg_shape: Optional[tuple] = None
        self._last_pg_cmap: Optional[str] = None
        self._last_stats_ts: float = 0.0
        self._hover_plots: List[Dict[str, Any]] = []
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
            "mean_amp": {
                "state_key": "map_mean_amp",
                "title": "Live 2D Map — Mean Amplitude a (ADU)",
                "cbar_label": "Amplitude (ADU)",
                "cmap": "inferno",
                "auto": True,
                "vmin": 50.0,
                "vmax": 5000.0,
                "decimals": 1,
                "step": 25.0,
                "bounds": (0.0, 100000.0),
                "suffix": " ADU",
            },
            "mean_baseline": {
                "state_key": "map_mean_baseline",
                "title": "Live 2D Map — Mean Baseline b (ADU)",
                "cbar_label": "Baseline (ADU)",
                "cmap": "cividis",
                "auto": True,
                "vmin": 0.0,
                "vmax": 2000.0,
                "decimals": 1,
                "step": 10.0,
                "bounds": (-10000.0, 100000.0),
                "suffix": " ADU",
            },
        }

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # 1. Top Detector Display Controls Bar
        ctrl_bar = QtWidgets.QHBoxLayout()

        ctrl_bar.addWidget(QtWidgets.QLabel("Contrast Quantiles:"))
        ctrl_bar.addWidget(QtWidgets.QLabel("Low:"))
        self.spin_contrast_low = QtWidgets.QDoubleSpinBox()
        self.spin_contrast_low.setRange(0.0, 99.9)
        self.spin_contrast_low.setDecimals(1)
        self.spin_contrast_low.setSingleStep(0.5)
        self.spin_contrast_low.setValue(1.0)
        self.spin_contrast_low.setSuffix(" %")
        self.spin_contrast_low.setKeyboardTracking(False)
        self.spin_contrast_low.valueChanged.connect(self._on_contrast_bounds_changed)
        ctrl_bar.addWidget(self.spin_contrast_low)

        ctrl_bar.addWidget(QtWidgets.QLabel("High:"))
        self.spin_contrast_high = QtWidgets.QDoubleSpinBox()
        self.spin_contrast_high.setRange(0.1, 100.0)
        self.spin_contrast_high.setDecimals(1)
        self.spin_contrast_high.setSingleStep(0.5)
        self.spin_contrast_high.setValue(99.8)
        self.spin_contrast_high.setSuffix(" %")
        self.spin_contrast_high.setKeyboardTracking(False)
        self.spin_contrast_high.valueChanged.connect(self._on_contrast_bounds_changed)
        ctrl_bar.addWidget(self.spin_contrast_high)

        ctrl_bar.addStretch()
        self.lbl_stats = QtWidgets.QLabel("No frame loaded")
        self.lbl_stats.setStyleSheet("font-family: monospace; font-size: 11px;")
        ctrl_bar.addWidget(self.lbl_stats)
        layout.addLayout(ctrl_bar)

        # 1b. Live Spatial Map Display & Colorscale Range Bar (shown only during mapping)
        self.map_bar_widget = QtWidgets.QWidget()
        map_bar = QtWidgets.QHBoxLayout(self.map_bar_widget)
        map_bar.setContentsMargins(0, 0, 0, 0)
        map_bar.addWidget(QtWidgets.QLabel("Map Metric:"))
        self.combo_map_metric = QtWidgets.QComboBox()
        self.combo_map_metric.addItem("Mean Shift ν_B (GHz)", "mean_shift")
        self.combo_map_metric.addItem("Velocity V_s (m/s)", "v_s")
        self.combo_map_metric.addItem("Mean Linewidth Γ_B (GHz)", "mean_lw")
        self.combo_map_metric.addItem("Normal Shift S(0) (GHz)", "shift_0")
        self.combo_map_metric.addItem("Mean Amplitude a (ADU)", "mean_amp")
        self.combo_map_metric.addItem("Mean Baseline b (ADU)", "mean_baseline")
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
        self.map_bar_widget.setVisible(False)
        layout.addWidget(self.map_bar_widget)
        self._sync_map_range_widgets()

        # 2. Unified PyQtGraph GraphicsLayoutWidget (handles single-frame, live 2D maps & diagnostics)
        self._pg_widget = pg.GraphicsLayoutWidget()
        self._pg_widget.setBackground("#1e1e1e")
        layout.addWidget(self._pg_widget, stretch=1)

        self._pg_plot: Optional[pg.PlotItem] = None
        self._pg_img_item: Optional[pg.ImageItem] = None
        self._pg_cbar: Optional[pg.ColorBarItem] = None
        self._build_single_plot()

        # 3. Bottom Cursor Readout Bar
        self.lbl_cursor = QtWidgets.QLabel("Cursor: —")
        self.lbl_cursor.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.lbl_cursor)

        self._pg_widget.scene().sigMouseMoved.connect(self._on_pg_mouse_move)

    # ==========================================================================
    # Shared Dark-Theme PyQtGraph Panel Helpers
    # ==========================================================================

    @staticmethod
    def _style_dark_plot(
        plot_item: pg.PlotItem,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        grid: bool = False,
    ):
        if title:
            plot_item.setTitle(title, color="#e0e0e0", size="10pt")
        for side, lbl in (("bottom", xlabel), ("left", ylabel)):
            ax = plot_item.getAxis(side)
            ax.setPen(pg.mkPen("#90a4ae"))
            ax.setTextPen(pg.mkPen("#cfd8dc"))
            if lbl:
                plot_item.setLabel(side, lbl, color="#cfd8dc")
        if grid:
            plot_item.showGrid(x=True, y=True, alpha=0.25)

    def _clear_pg_grid(self):
        self._pg_widget.clear()
        self._hover_plots = []
        self._live_map_widgets = None
        self._dual_bf_widgets = None
        self._pg_plot = None
        self._pg_img_item = None
        self._pg_cbar = None

    def set_map_bar_visible(self, visible: bool):
        if hasattr(self, "map_bar_widget"):
            self.map_bar_widget.setVisible(bool(visible))

    def _build_single_plot(self):
        self._clear_pg_grid()
        self._view_mode = "single"
        self.set_map_bar_visible(False)
        self._pg_plot = self._pg_widget.addPlot(title=self._current_title)
        self._pg_plot.invertY(True)
        self._pg_plot.setAspectLocked(True)
        self._style_dark_plot(
            self._pg_plot,
            title=self._current_title,
            xlabel="Detector X (px)",
            ylabel="Detector Y (px)",
        )
        self._pg_img_item = pg.ImageItem(axisOrder="row-major")
        self._pg_plot.addItem(self._pg_img_item)

        init_cmap = pg_helpers.get_pg_colormap(self.DEFAULT_CMAP)
        self._pg_cbar = pg.ColorBarItem(
            values=(0, 65535),
            colorMap=init_cmap,
            label="Counts (ADU)",
            interactive=True,
        )
        self._pg_cbar.axis.setTextPen(pg.mkPen("#cfd8dc"))
        self._pg_cbar.setImageItem(self._pg_img_item, insert_in=self._pg_plot)
        self._last_pg_cmap = self.DEFAULT_CMAP
        self._last_pg_shape = None

    def _enter_diagnostic_mode(self):
        self._in_diagnostic_view = True
        self._view_mode = "diagnostic"
        self.set_map_bar_visible(False)
        self._clear_pg_grid()

    def _add_dark_image_panel(
        self,
        row: int,
        col: int,
        data: np.ndarray,
        cmap_name: str,
        title: str,
        xlabel: str,
        ylabel: str,
        cbar_label: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        aspect_equal: bool = True,
        invert_y: bool = True,
        colspan: int = 1,
        extent: Optional[Tuple[float, float, float, float]] = None,
        is_rgb: bool = False,
        value_unit: str = "",
    ) -> Tuple[pg.PlotItem, pg.ImageItem]:
        """Adds a styled dark-theme image panel (scalar or RGB) with optional ColorBarItem."""
        sub = self._pg_widget.addLayout(row=row, col=col, colspan=colspan)
        p = sub.addPlot(row=0, col=0)
        p.invertY(invert_y)
        if aspect_equal:
            p.setAspectLocked(True)
        self._style_dark_plot(p, title=title, xlabel=xlabel, ylabel=ylabel)

        arr = np.asarray(data)
        h, w = arr.shape[:2]
        im = pg.ImageItem(axisOrder="row-major")

        if extent is not None:
            x0, x1, y0, y1 = [float(v) for v in extent]
            im.setRect(QtCore.QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))
        else:
            im.setRect(QtCore.QRectF(-0.5, -0.5, float(w), float(h)))

        if is_rgb:
            rgb_u8 = np.clip(np.asarray(arr, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.ubyte)
            im.setImage(rgb_u8, autoLevels=False)
            p.addItem(im)
        else:
            arr_f = np.asarray(arr, dtype=np.float64)
            finite = np.isfinite(arr_f)
            if vmin is None or vmax is None:
                if np.any(finite):
                    lo = float(np.min(arr_f[finite])) if vmin is None else float(vmin)
                    hi = float(np.max(arr_f[finite])) if vmax is None else float(vmax)
                else:
                    lo, hi = 0.0, 1.0
            else:
                lo, hi = float(vmin), float(vmax)
            if hi <= lo:
                hi = lo + 1e-6

            if not np.all(finite):
                rgba, lo, hi = pg_helpers.array_to_rgba_with_nan(
                    arr_f, cmap_name, lo, hi, nan_rgba=(30, 30, 30, 0)
                )
                im.setImage(rgba, autoLevels=False)
                p.addItem(im)
                if cbar_label is not None:
                    cb = pg.ColorBarItem(
                        values=(lo, hi),
                        colorMap=pg_helpers.get_pg_colormap(cmap_name),
                        label=cbar_label,
                        interactive=False,
                        width=14,
                    )
                    cb.axis.setTextPen(pg.mkPen("#cfd8dc"))
                    sub.addItem(cb, row=0, col=1)
            else:
                im.setImage(arr_f, autoLevels=False, levels=(lo, hi))
                p.addItem(im)
                if cbar_label is not None:
                    cb = pg.ColorBarItem(
                        values=(lo, hi),
                        colorMap=pg_helpers.get_pg_colormap(cmap_name),
                        label=cbar_label,
                        interactive=True,
                        width=14,
                    )
                    cb.axis.setTextPen(pg.mkPen("#cfd8dc"))
                    cb.setImageItem(im, insert_in=p)

        p.autoRange()
        self._hover_plots.append(
            {
                "plot": p,
                "data": None if is_rgb else arr,
                "unit": value_unit,
                "origin": "upper" if invert_y else "lower",
                "extent": extent,
            }
        )
        return p, im

    def _add_dark_scatter_panel(
        self,
        row: int,
        col: int,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        cmap_name: str,
        title: str,
        xlabel: str,
        ylabel: str,
        cbar_label: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        size: float = 5.0,
        alpha: int = 175,
        invert_y: bool = False,
        aspect_equal: bool = False,
        colspan: int = 1,
    ) -> Tuple[pg.PlotItem, pg.ScatterPlotItem]:
        sub = self._pg_widget.addLayout(row=row, col=col, colspan=colspan)
        p = sub.addPlot(row=0, col=0)
        p.invertY(invert_y)
        if aspect_equal:
            p.setAspectLocked(True)
        self._style_dark_plot(p, title=title, xlabel=xlabel, ylabel=ylabel, grid=True)

        brushes, lo, hi, cmap = pg_helpers.scatter_brushes(c, cmap_name, vmin=vmin, vmax=vmax, alpha=alpha)
        sc = pg.ScatterPlotItem(
            x=np.asarray(x, dtype=np.float64).ravel(),
            y=np.asarray(y, dtype=np.float64).ravel(),
            brush=brushes,
            pen=None,
            size=size,
        )
        p.addItem(sc)
        cb = pg.ColorBarItem(
            values=(lo, hi),
            colorMap=cmap,
            label=cbar_label,
            interactive=False,
            width=14,
        )
        cb.axis.setTextPen(pg.mkPen("#cfd8dc"))
        sub.addItem(cb, row=0, col=1)
        self._hover_plots.append({"plot": p, "data": None, "unit": "", "origin": "lower", "extent": None})
        return p, sc

    # ==========================================================================
    # Single-Frame Camera Viewer
    # ==========================================================================

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
        self._current_image = np.array(image, copy=True)
        h, w = self._current_image.shape[:2]

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

    def _on_contrast_bounds_changed(self):
        if self._syncing_contrast:
            return
        self._syncing_contrast = True
        try:
            low = float(self.spin_contrast_low.value())
            high = float(self.spin_contrast_high.value())
            if low >= high:
                if self.sender() is self.spin_contrast_low:
                    high = min(100.0, low + 0.1)
                    if high <= low:
                        low = max(0.0, high - 0.1)
                        self.spin_contrast_low.setValue(low)
                    self.spin_contrast_high.setValue(high)
                else:
                    low = max(0.0, high - 0.1)
                    if low >= high:
                        high = min(100.0, low + 0.1)
                        self.spin_contrast_high.setValue(high)
                    self.spin_contrast_low.setValue(low)
        finally:
            self._syncing_contrast = False
        self._on_display_settings_changed()

    def _compute_clim(self, img: np.ndarray) -> Tuple[float, float]:
        low_pct = float(self.spin_contrast_low.value())
        high_pct = float(self.spin_contrast_high.value())
        if high_pct <= low_pct:
            high_pct = min(100.0, low_pct + 0.1)
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

        if self._view_mode != "single" or self._pg_plot is None or self._pg_img_item is None:
            self._build_single_plot()

        cmap_name = self.DEFAULT_CMAP
        vmin, vmax = self._compute_clim(self._current_image)
        h, w = self._current_image.shape[:2]

        if cmap_name != self._last_pg_cmap:
            self._pg_cbar.setColorMap(pg_helpers.get_pg_colormap(cmap_name))
            self._last_pg_cmap = cmap_name

        self._pg_img_item.setImage(self._current_image, autoLevels=False, levels=(vmin, vmax))
        self._pg_img_item.setRect(QtCore.QRectF(-0.5, -0.5, float(w), float(h)))
        self._pg_cbar.setLevels((vmin, vmax))
        self._pg_plot.setTitle(self._current_title, color="#e0e0e0", size="10pt")

        if self._last_pg_shape != (h, w):
            self._last_pg_shape = (h, w)
            self._pg_plot.autoRange()

    def _on_display_settings_changed(self):
        if self._view_mode == "live_map" and self._last_live_map_state is not None:
            self._render_live_map_state()
        elif self._view_mode == "dual_brightfield":
            self._refresh_dual_brightfield_images()
        elif self._view_mode == "single":
            self._render_image()

    # ==========================================================================
    # Dual-Graph Mode ('Reference' | 'Current Brightfield') for Mapping Parameters
    # ==========================================================================

    def show_dual_brightfield_view(
        self,
        ref_img: Optional[np.ndarray] = None,
        ref_title: str = "Reference",
        live_img: Optional[np.ndarray] = None,
    ):
        """Splits the viewer into 'Reference' (left) and 'Current Brightfield' (right)."""
        self._last_live_map_state = None
        self._in_diagnostic_view = False
        self._clear_pg_grid()
        self._view_mode = "dual_brightfield"
        self.set_map_bar_visible(False)

        if ref_img is not None:
            self._dual_ref_img = np.asarray(ref_img)
            self._dual_ref_title = ref_title
        if live_img is not None:
            self._dual_live_img = np.asarray(live_img)

        p_ref = self._pg_widget.addPlot(row=0, col=0, title=self._dual_ref_title)
        p_ref.invertY(True)
        p_ref.setAspectLocked(True)
        self._style_dark_plot(p_ref, title=self._dual_ref_title, xlabel="X (px)", ylabel="Y (px)")
        im_ref = pg.ImageItem(axisOrder="row-major")
        p_ref.addItem(im_ref)

        p_live = self._pg_widget.addPlot(row=0, col=1, title="Current Brightfield")
        p_live.invertY(True)
        p_live.setAspectLocked(True)
        self._style_dark_plot(p_live, title="Current Brightfield", xlabel="X (px)", ylabel="Y (px)")
        im_live = pg.ImageItem(axisOrder="row-major")
        p_live.addItem(im_live)

        gray_lut = pg_helpers.get_pg_colormap(self.DEFAULT_CMAP).getLookupTable(nPts=256)
        im_ref.setLookupTable(gray_lut)
        im_live.setLookupTable(gray_lut)

        self._dual_bf_widgets = {
            "ref_plot": p_ref,
            "ref_img_item": im_ref,
            "ref_shape": None,
            "live_plot": p_live,
            "live_img_item": im_live,
            "live_shape": None,
        }
        self._hover_plots = [
            {"plot": p_ref, "data": self._dual_ref_img, "unit": "ADU", "origin": "upper", "extent": None},
            {"plot": p_live, "data": self._dual_live_img, "unit": "ADU", "origin": "upper", "extent": None},
        ]
        self._refresh_dual_brightfield_images()

    def set_dual_reference_image(self, ref_img: Optional[np.ndarray], ref_title: str = "Reference"):
        self._dual_ref_img = np.asarray(ref_img) if ref_img is not None else None
        self._dual_ref_title = ref_title
        if self._view_mode != "dual_brightfield" or self._dual_bf_widgets is None:
            self.show_dual_brightfield_view(ref_img=self._dual_ref_img, ref_title=ref_title)
            return
        self._dual_bf_widgets["ref_plot"].setTitle(ref_title, color="#e0e0e0", size="10pt")
        if len(self._hover_plots) >= 1:
            self._hover_plots[0]["data"] = self._dual_ref_img
        self._refresh_dual_brightfield_images()

    def set_dual_live_image(self, live_img: np.ndarray):
        self._dual_live_img = np.asarray(live_img)
        if self._view_mode != "dual_brightfield" or self._dual_bf_widgets is None:
            self.show_dual_brightfield_view(live_img=self._dual_live_img)
            return
        if len(self._hover_plots) >= 2:
            self._hover_plots[1]["data"] = self._dual_live_img
        self._refresh_dual_brightfield_images()

    def _refresh_dual_brightfield_images(self):
        if self._dual_bf_widgets is None:
            return
        w = self._dual_bf_widgets
        if self._dual_ref_img is not None:
            arr = self._dual_ref_img
            vmin, vmax = self._compute_clim(arr)
            h, wd = arr.shape[:2]
            w["ref_img_item"].setImage(arr, autoLevels=False, levels=(vmin, vmax))
            w["ref_img_item"].setRect(QtCore.QRectF(-0.5, -0.5, float(wd), float(h)))
            if w["ref_shape"] != (h, wd):
                w["ref_shape"] = (h, wd)
                w["ref_plot"].autoRange()
        else:
            w["ref_img_item"].clear()
            w["ref_shape"] = None

        if self._dual_live_img is not None:
            arr = self._dual_live_img
            vmin, vmax = self._compute_clim(arr)
            h, wd = arr.shape[:2]
            w["live_img_item"].setImage(arr, autoLevels=False, levels=(vmin, vmax))
            w["live_img_item"].setRect(QtCore.QRectF(-0.5, -0.5, float(wd), float(h)))
            if w["live_shape"] != (h, wd):
                w["live_shape"] = (h, wd)
                w["live_plot"].autoRange()
        else:
            w["live_img_item"].clear()
            w["live_shape"] = None

    # ==========================================================================
    # Step 5.1: Laser Reference Calibration Diagnostic Views
    # ==========================================================================

    def show_laser_calibration_diagnostics(
        self,
        calib_res: Dict[str, Any],
        view_mode: str = "fringes_geom",
    ):
        """
        Renders diagnostic plots for Step 5.1 (Laser Reference Calibration) in PyQtGraph:
          - `'denoise_seeds'`: Step 2–3 Denoised laser image + central row profile & seed peaks.
          - `'fringes_geom'`: Step 4–5 Traced & refined sub-pixel fringes + gnomonic optical axis.
          - `'anamorphic_res'`: Step 6 2D anamorphic polynomial surface residuals (MHz).
        """
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

        cmap = self.DEFAULT_CMAP
        data_sub = calib_res["data_subtracted"]
        vmin, vmax = self._compute_clim(data_sub)

        if view_mode == "denoise_seeds":
            p_img, _ = self._add_dark_image_panel(
                0, 0, data_sub, cmap,
                "Step 5.1A — Denoised Laser Interferogram & Central Seed Band",
                "Detector X (px)", "Detector Y (px)", "Net ADU",
                vmin=vmin, vmax=vmax, aspect_equal=False, value_unit="ADU",
            )
            yc = float(calib_res["y_center_laser"])
            dy = float(calib_res["dy_laser"])
            peaks = np.asarray(calib_res["peaks"], dtype=np.float64).ravel()
            dash_c = pg.mkPen("#00e5ff", width=1.1, style=QtCore.Qt.PenStyle.DashLine)
            p_img.addItem(pg.InfiniteLine(pos=yc - dy, angle=0, pen=dash_c))
            p_img.addItem(pg.InfiniteLine(pos=yc + dy, angle=0, pen=dash_c))
            p_img.plot(
                peaks, np.full_like(peaks, yc),
                pen=None, symbol="star", symbolSize=9,
                symbolBrush=pg.mkBrush("#ff5252"), symbolPen=pg.mkPen("#ff5252"),
            )

            p_prof = self._pg_widget.addPlot(row=1, col=0)
            self._style_dark_plot(
                p_prof,
                "Central Row-Averaged Profile & Detected Laser Fringes",
                "Detector X (px)",
                "Net Intensity (ADU)",
                grid=True,
            )
            p_prof.addLegend(offset=(10, 10))
            avg_row = calib_res["average_row"]
            p_prof.plot(
                np.arange(len(avg_row)), avg_row,
                pen=pg.mkPen("#29b6f6", width=1.4), name="Row average",
            )
            p_prof.plot(
                peaks, avg_row[peaks.astype(int)],
                pen=None, symbol="o", symbolSize=6,
                symbolBrush=pg.mkBrush("#ff5252"), symbolPen=None, name="Detected modes",
            )
            p_prof.setXLink(p_img)
            self._hover_plots.append({"plot": p_prof, "data": None, "unit": "", "origin": "lower", "extent": None})

        elif view_mode == "anamorphic_res":
            all_x = calib_res["all_x_laser"]
            all_y = calib_res["all_y_laser"]
            err_raw = calib_res["err_2d_mhz"]
            err_sm = calib_res["err_2d_smooth_mhz"]
            rms_raw = calib_res["rms_raw_mhz"]
            rms_sm = calib_res["rms_smooth_mhz"]

            lim = max(5.0, float(np.percentile(np.abs(err_raw), 98)))
            p1, _ = self._add_dark_scatter_panel(
                0, 0, all_x, all_y, err_raw, "coolwarm",
                f"2D Surface Residuals (Raw RMS = {rms_raw:.2f} MHz)",
                "Detector X (px)", "Detector Y (px)", "Residual (MHz)",
                vmin=-lim, vmax=lim, size=3.5, alpha=210, invert_y=True,
            )

            p2 = self._pg_widget.addPlot(row=0, col=1)
            p2.invertY(True)
            self._style_dark_plot(
                p2,
                "Anamorphic Residual vs. Detector Y",
                "Frequency Residual (MHz)",
                "Detector Y (px)",
                grid=True,
            )
            p2.addLegend(offset=(10, 10))
            p2.plot(
                err_raw, all_y, pen=None, symbol="o", symbolSize=3,
                symbolBrush=pg.mkBrush(100, 181, 246, 95), symbolPen=None,
                name=f"Raw ({rms_raw:.2f} MHz)",
            )
            p2.plot(
                err_sm, all_y, pen=None, symbol="o", symbolSize=3,
                symbolBrush=pg.mkBrush(229, 57, 53, 185), symbolPen=None,
                name=f"Smoothed ({rms_sm:.2f} MHz)",
            )
            p2.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#cfd8dc", style=QtCore.Qt.PenStyle.DashLine)))
            p2.setYLink(p1)
            self._hover_plots.append({"plot": p2, "data": None, "unit": "", "origin": "upper", "extent": None})

        else:
            ref_fringes = calib_res["refined_fringes"]
            x0_fit = float(calib_res["x0_fit"])
            y0_fit = float(calib_res["y0_fit"])
            xc_gnom = float(calib_res["xc_gnom"])

            p_img, _ = self._add_dark_image_panel(
                0, 0, data_sub, cmap,
                f"Step 5.1B — {len(ref_fringes)} Refined Fringes | "
                f"Optical Axis ({x0_fit:.1f}, {y0_fit:.1f}) px | Gnomonic Xc = {xc_gnom:.1f} px",
                "Detector X (px)", "Detector Y (px)", "Net ADU",
                vmin=vmin, vmax=vmax, aspect_equal=True, value_unit="ADU",
            )
            p_img.addLegend(offset=(10, 10))
            fringe_pen = pg.mkPen("#00e5ff", width=1.2)
            for k, (fx, fy) in enumerate(ref_fringes):
                p_img.plot(fx, fy, pen=fringe_pen, name="Refined fringes (k)" if k == 0 else None)

            p_img.addItem(
                pg.InfiniteLine(
                    pos=y0_fit,
                    angle=0,
                    pen=pg.mkPen("#ffeb3b", width=1.3, style=QtCore.Qt.PenStyle.DashLine),
                )
            )
            p_img.plot(
                [x0_fit], [y0_fit], pen=None, symbol="+", symbolSize=14,
                symbolPen=pg.mkPen("#ff5252", width=2.2),
                name=f"Optical axis ({x0_fit:.1f}, {y0_fit:.1f}) px",
            )

        self.lbl_stats.setText(
            f"Laser Calib: {len(calib_res['refined_fringes'])} modes  |  "
            f"(x₀, y₀)=({calib_res['x0_fit']:.1f}, {calib_res['y0_fit']:.1f}) px  |  "
            f"RMS: {calib_res['rms_raw_mhz']:.2f} MHz raw, {calib_res['rms_smooth_mhz']:.2f} MHz smooth"
        )

    # ==========================================================================
    # Step 5.2: Rubidium Absorption & 2D Frequency Map Diagnostic Views
    # ==========================================================================

    def show_rb_calibration_diagnostics(
        self,
        rb_res: Dict[str, Any],
        view_mode: str = "fsr_fit",
    ):
        """
        Renders diagnostic plots for Step 5.2 (Rubidium Calibration & 2D Frequency Map) in PyQtGraph:
          - `'rb_denoise_radial'`: Step 8–9 Denoised Rb image + radial order profile.
          - `'fsr_fit'`: Step 10 Folded multi-order Rb spectrum & hyperfine FSR fit.
          - `'freq_map'`: Step 10B Full 2D calibrated frequency map & intra-order detuning map.
        """
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

        cmap = self.DEFAULT_CMAP
        fsr_cal = float(rb_res["FSR_cal"])
        dnu = float(rb_res["delta_nu_opt"])

        if view_mode == "rb_denoise_radial":
            data_abs = rb_res["data_absorption"]
            vmin, vmax = self._compute_clim(data_abs)
            self._add_dark_image_panel(
                0, 0, data_abs, cmap,
                "Step 5.2A — Denoised Rubidium Absorption Interferogram",
                "Detector X (px)", "Detector Y (px)", "Net ADU",
                vmin=vmin, vmax=vmax, aspect_equal=False, value_unit="ADU",
            )

            p_rad = self._pg_widget.addPlot(row=1, col=0)
            self._style_dark_plot(
                p_rad,
                "Radially Averaged Rb Transmission vs. Continuous Interference Order φ(r)",
                "Interference Order φ",
                "Radial Mean Intensity (ADU)",
                grid=True,
            )
            p_rad.plot(
                rb_res["phi_order_r"], rb_res["avg_spectrum_radius"],
                pen=pg.mkPen("#ce93d8", width=1.5),
            )
            self._hover_plots.append({"plot": p_rad, "data": None, "unit": "", "origin": "lower", "extent": None})

        elif view_mode == "freq_map":
            freq_map = rb_res["freq_map"]
            p1, _ = self._add_dark_image_panel(
                0, 0, freq_map, "viridis",
                f"Step 5.2C — 2D Frequency Map (FSR = {fsr_cal:.4f} GHz)",
                "Detector X (px)", "Detector Y (px)", "Frequency (GHz)",
                aspect_equal=True, value_unit="GHz",
            )
            intra_ghz = ((freq_map + 0.5 * fsr_cal) % fsr_cal) - 0.5 * fsr_cal
            p2, _ = self._add_dark_image_panel(
                0, 1, intra_ghz, "coolwarm",
                f"Intra-Order Detuning (Elastic Zero ξ = {rb_res['xi_elastic']:.4f})",
                "Detector X (px)", "Detector Y (px)", "Δν from Elastic Order (GHz)",
                vmin=-0.5 * fsr_cal, vmax=+0.5 * fsr_cal,
                aspect_equal=True, value_unit="GHz",
            )
            p2.setXLink(p1)
            p2.setYLink(p1)

        else:
            p_fold = self._pg_widget.addPlot(row=0, col=0)
            self._style_dark_plot(
                p_fold,
                f"Step 5.2B — Coherently Folded Rb Orders (Calibrated FSR = {fsr_cal:.4f} GHz)",
                "Relative Frequency within Order (GHz)",
                "Intensity (ADU)",
                grid=True,
            )
            p_fold.addLegend(offset=(10, 10))
            f_grid = rb_res["f_grid"]
            order_spectra = rb_res["order_spectra"]
            orders_used = rb_res["orders_used"]
            for idx_o, _ in enumerate(orders_used):
                lbl = f"Orders k={orders_used[0]}..{orders_used[-1]}" if idx_o == 0 else None
                p_fold.plot(
                    f_grid,
                    order_spectra[idx_o],
                    pen=pg.mkPen(pg.intColor(idx_o, max(len(orders_used), 1), alpha=120), width=1.0),
                    name=lbl,
                )
            p_fold.plot(
                rb_res["f_valid"], rb_res["sig_valid"],
                pen=pg.mkPen("#4fc3f7", width=1.8), name="Order-averaged spectrum",
            )
            p_fold.plot(
                rb_res["f_valid"], rb_res["baseline_env"],
                pen=pg.mkPen("#ff5252", width=1.4, style=QtCore.Qt.PenStyle.DashLine),
                name="Envelope baseline",
            )

            p_fit = self._pg_widget.addPlot(row=1, col=0)
            self._style_dark_plot(
                p_fit,
                f"Hyperfine Cross-Correlation Fit: Δν = {dnu:+.3f} GHz  |  FSR = {fsr_cal:.4f} GHz",
                "Relative Frequency within Order (GHz)",
                "Normalized Absorption",
                grid=True,
            )
            p_fit.addLegend(offset=(10, 10))
            f_fit = rb_res["f_fit_rb"]
            meas_abs = rb_res["meas_abs_fit"]
            theo_abs = rb_res["theo_fit_rb"]
            meas_norm = meas_abs / max(float(np.max(meas_abs)), 1e-9)
            theo_norm = theo_abs / max(float(np.max(theo_abs)), 1e-9)
            p_fit.plot(
                f_fit, meas_norm, pen=pg.mkPen("#26a69a", width=1.6),
                name="Measured Rb absorption (norm.)",
            )
            p_fit.plot(
                f_fit, theo_norm,
                pen=pg.mkPen("#ff4081", width=1.6, style=QtCore.Qt.PenStyle.DashLine),
                name=f"Theoretical Rb D2 (T={rb_res['T_cell']:.0f}°C, Δν={dnu:+.3f} GHz)",
            )
            self._hover_plots.append({"plot": p_fold, "data": None, "unit": "", "origin": "lower", "extent": None})
            self._hover_plots.append({"plot": p_fit, "data": None, "unit": "", "origin": "lower", "extent": None})

        self.lbl_stats.setText(
            f"Rb Calib: FSR = {fsr_cal:.4f} GHz  |  "
            f"Δν = {dnu:+.3f} GHz  |  "
            f"ξ_elastic = {rb_res['xi_elastic']:.4f}  |  "
            f"Freq Map: {rb_res['freq_map'].shape[1]}×{rb_res['freq_map'].shape[0]} px"
        )

    # ==========================================================================
    # Step 5.3: Sample In-Situ Calibration, DHO Fit & q-Vector Diagnostic Views
    # ==========================================================================

    def show_sample_treatment_diagnostics(
        self,
        sample_res: Dict[str, Any],
        view_mode: str = "dho_maps",
    ):
        """
        Renders diagnostic plots for Step 5.3 (Sample DHO Fitting & Reference Geometry) in PyQtGraph:
          - `'insitu_freq'`: Step 11 In-situ frequency self-calibration & Brillouin windows.
          - `'dho_maps'`: Step 12–13 2D Fitted Brillouin Shift & Linewidth maps.
          - `'q_dispersion'`: Step 14 2D Coaxial Ellipse Geometry & Acoustic Dispersion V_s.
        """
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

        dho = sample_res["dho_results"]
        q_res = sample_res.get("q_results")

        if view_mode == "insitu_freq":
            p_prof = self._pg_widget.addPlot(row=0, col=0, colspan=2)
            self._style_dark_plot(
                p_prof,
                f"Step 5.3A — Central Spectrum & Brillouin Search Windows (Coarse df = {sample_res['df_coarse']*1e3:+.1f} MHz)",
                "Calibrated Frequency (GHz)",
                "Intensity (ADU)",
                grid=True,
            )
            p_prof.addLegend(offset=(10, 10))
            p_prof.plot(
                sample_res["freq_avg_data_sample"],
                sample_res["avg_data_sample"],
                pen=pg.mkPen("#42a5f5", width=1.4),
                name="Central row-averaged spectrum",
            )
            for f_as, f_s in sample_res["bls_peaks"]:
                p_prof.addItem(
                    pg.InfiniteLine(
                        pos=float(f_as), angle=90,
                        pen=pg.mkPen("#ef5350", width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                    )
                )
                p_prof.addItem(
                    pg.InfiniteLine(
                        pos=float(f_s), angle=90,
                        pen=pg.mkPen("#66bb6a", width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                    )
                )
            self._hover_plots.append({"plot": p_prof, "data": None, "unit": "", "origin": "lower", "extent": None})

            dz_mhz = sample_res["dz_map_2d"] * 1e3
            self._add_dark_image_panel(
                1, 0, dz_mhz, "coolwarm",
                f"2D Zero-Offset Map (RMS={np.std(dz_mhz):.1f} MHz)",
                "Detector X (px)", "Detector Y (px)", "MHz",
                aspect_equal=False, value_unit="MHz",
            )
            di_mhz = sample_res["dinst_map_2d"] * 1e3
            self._add_dark_image_panel(
                1, 1, di_mhz, "coolwarm",
                f"Higher-Order Dispersion Residual (RMS={np.std(di_mhz):.1f} MHz)",
                "Detector X (px)", "Detector Y (px)", "MHz",
                aspect_equal=False, value_unit="MHz",
            )

        elif view_mode == "q_dispersion" and q_res is not None:
            x_f = q_res["x_coords"]
            y_f = q_res["y_coords"]
            s_f = q_res["shift"]
            opt = q_res["opt_params"]

            p1, _ = self._add_dark_scatter_panel(
                0, 0, x_f, y_f, s_f, "viridis",
                f"Step 5.3C — Coaxial Ellipses: Center=({opt['xc']:.1f}, {opt['yc']:.1f}), γ={opt['gamma']:.3f}",
                "Detector X (px)", "Detector Y (px)", "Shift ν_B (GHz)",
                size=6.5, alpha=190, invert_y=True,
            )
            theta_t = np.linspace(0.0, 2.0 * np.pi, 200)
            ct = np.cos(opt["phi_rad"])
            st = np.sin(opt["phi_rad"])
            dash_r = pg.mkPen("#ff5252", width=1.1, style=QtCore.Qt.PenStyle.DashLine)
            for r_iso in (300.0, 600.0, 900.0, 1200.0):
                xp = r_iso * np.cos(theta_t)
                yp = opt["gamma"] * r_iso * np.sin(theta_t)
                xe = opt["xc"] + xp * ct - yp * st
                ye = opt["yc"] + xp * st + yp * ct
                p1.plot(xe, ye, pen=dash_r)
            p1.plot(
                [opt["xc"]], [opt["yc"]],
                pen=None, symbol="+", symbolSize=13,
                symbolPen=pg.mkPen("#ff5252", width=2.2),
            )

            p2 = self._pg_widget.addPlot(row=0, col=1)
            vs = q_res["V_s"]
            vs_err = q_res.get("V_s_err") or 0.0
            self._style_dark_plot(
                p2,
                f"Acoustic Dispersion: V_s = {vs:.1f} ± {vs_err:.1f} m/s",
                "Scattering Angle θ (°)",
                "Brillouin Shift ν_B (GHz)",
                grid=True,
            )
            p2.addLegend(offset=(10, 10))
            theta_deg = q_res["theta_deg"]
            p2.plot(
                theta_deg, s_f, pen=None, symbol="o", symbolSize=5,
                symbolBrush=pg.mkBrush(66, 165, 245, 115), symbolPen=None,
                name="Measured DHO Shift",
            )
            th_grid = np.linspace(0.0, max(float(np.max(theta_deg)), 5.0), 200)
            s_curve = q_res["Shift_0"] * np.cos(0.5 * np.deg2rad(th_grid))
            p2.plot(
                th_grid, s_curve, pen=pg.mkPen("#ff5252", width=2.0),
                name=f"Fit: S(0)={q_res['Shift_0']:.3f} GHz, M_θ={q_res['M_theta']:.0f} px",
            )
            self._hover_plots.append({"plot": p2, "data": None, "unit": "", "origin": "lower", "extent": None})

        else:
            sa = dho["shift_clipped_averaged"]
            la = dho["linewidth_clipped_averaged"]
            xa = dho["x_coords_averaged"]
            ya = dho["y_coords_averaged"]
            valid = np.isfinite(sa) & np.isfinite(la) & np.isfinite(xa) & np.isfinite(ya)

            if np.any(valid):
                p1, _ = self._add_dark_scatter_panel(
                    0, 0, xa[valid], ya[valid], sa[valid], "viridis",
                    f"Step 5.3B — Order-Averaged Brillouin Shift ({dho['bin_val_h']}×{dho['bin_val_w']} bin)",
                    "Detector X (px)", "Detector Y (px)", "Shift ν_B (GHz)",
                    size=6.5, alpha=190, invert_y=True,
                )
                p2, _ = self._add_dark_scatter_panel(
                    0, 1, xa[valid], ya[valid], la[valid], "magma",
                    f"Brillouin Linewidth (S/AS Splitting RMS = {dho['stokes_antistokes_rms_mhz']:.1f} MHz)",
                    "Detector X (px)", "Detector Y (px)", "Linewidth Γ_B (GHz)",
                    size=6.5, alpha=190, invert_y=True,
                )
                p2.setXLink(p1)
                p2.setYLink(p1)
            else:
                p1 = self._pg_widget.addPlot(row=0, col=0)
                p1.invertY(True)
                self._style_dark_plot(p1, "No valid fitted orders", "Detector X (px)", "Detector Y (px)", grid=True)
                p2 = self._pg_widget.addPlot(row=0, col=1)
                p2.invertY(True)
                self._style_dark_plot(p2, "No valid linewidth fits", "Detector X (px)", "Detector Y (px)", grid=True)

        vs_str = (
            f"  |  V_s = {q_res['V_s']:.1f} ± {(q_res.get('V_s_err') or 0.0):.1f} m/s"
            if q_res is not None and "V_s" in q_res
            else ""
        )
        amp_str = (
            f"  |  Mean A = {float(np.nanmean(dho['amplitude_averaged'])):.1f} ADU"
            if "amplitude_averaged" in dho and np.any(np.isfinite(dho["amplitude_averaged"]))
            else ""
        )
        base_str = (
            f"  |  Mean b = {float(np.nanmean(dho['baseline_averaged'])):.1f} ADU"
            if "baseline_averaged" in dho and np.any(np.isfinite(dho["baseline_averaged"]))
            else ""
        )
        self.lbl_stats.setText(
            f"DHO Fit ({dho['bin_val_h']}×{dho['bin_val_w']}): {dho['n_fringes']} orders  |  "
            f"Mean Shift = {float(np.nanmean(dho['shift_clipped_averaged'])):.3f} GHz  |  "
            f"Mean LW = {float(np.nanmean(dho['linewidth_clipped_averaged'])):.3f} GHz{amp_str}{base_str}{vs_str}"
        )

    # ==========================================================================
    # Step 5.4: Live 2D Spatial Map Heatmap + Interferogram View (Dual PyQtGraph)
    # ==========================================================================

    def _build_live_map_layout(self):
        self._clear_pg_grid()
        self._view_mode = "live_map"
        self._in_diagnostic_view = True

        # Left panel: Detector frame
        sub_left = self._pg_widget.addLayout(row=0, col=0)
        p_img = sub_left.addPlot(row=0, col=0)
        p_img.invertY(True)
        p_img.setAspectLocked(True)
        self._style_dark_plot(p_img, "Detector Frame", "Detector X (px)", "Detector Y (px)")
        im_frame = pg.ImageItem(axisOrder="row-major")
        p_img.addItem(im_frame)
        cb_frame = pg.ColorBarItem(
            values=(0, 65535),
            colorMap=pg_helpers.get_pg_colormap(self.DEFAULT_CMAP),
            label="Counts (ADU)",
            interactive=True,
            width=14,
        )
        cb_frame.axis.setTextPen(pg.mkPen("#cfd8dc"))
        cb_frame.setImageItem(im_frame, insert_in=p_img)
        txt_waiting = pg.TextItem(
            text="Map scan started\nAcquiring first point...",
            color="#cfd8dc",
            anchor=(0.5, 0.5),
            fill=pg.mkBrush(30, 30, 30, 200),
            border=pg.mkPen("#455a64"),
        )
        p_img.addItem(txt_waiting)

        # Right panel: Spatial map heatmap (proportional µm aspect)
        sub_right = self._pg_widget.addLayout(row=0, col=1)
        p_map = sub_right.addPlot(row=0, col=0)
        p_map.invertY(False)
        p_map.setAspectLocked(True)
        self._style_dark_plot(p_map, "Live 2D Map", "Stage X (µm)", "Stage Y (µm)")
        im_map = pg.ImageItem(axisOrder="row-major")
        p_map.addItem(im_map)
        crosshair = p_map.plot(
            [], [], pen=None, symbol="+", symbolSize=13,
            symbolPen=pg.mkPen("#ff5252", width=2.2),
        )
        cb_map = pg.ColorBarItem(
            values=(0.0, 1.0),
            colorMap=pg_helpers.get_pg_colormap("viridis"),
            label="Value",
            interactive=False,
            width=14,
        )
        cb_map.axis.setTextPen(pg.mkPen("#cfd8dc"))
        sub_right.addItem(cb_map, row=0, col=1)

        self._live_map_widgets = {
            "p_img": p_img,
            "im_frame": im_frame,
            "cb_frame": cb_frame,
            "txt_waiting": txt_waiting,
            "last_frame_shape": None,
            "last_frame_cmap": self.DEFAULT_CMAP,
            "p_map": p_map,
            "im_map": im_map,
            "cb_map": cb_map,
            "crosshair": crosshair,
            "last_map_rect": None,
            "last_map_metric": None,
        }
        self._hover_plots = [
            {"plot": p_img, "data": None, "unit": "ADU", "origin": "upper", "extent": None},
            {"plot": p_map, "data": None, "unit": "", "origin": "lower", "extent": None},
        ]

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
        map_mean_amp: Optional[np.ndarray] = None,
        map_mean_baseline: Optional[np.ndarray] = None,
    ):
        """
        Renders a side-by-side PyQtGraph live view during spatial mapping:
          - Left panel: Latest acquired 2D camera interferogram at `(iy, ix)`.
          - Right panel: Live 2D spatial heatmap `(X_um, Y_um)` of the selected Brillouin metric
            with proportional physical step aspect ratio (`setAspectLocked(True)`).
        """
        self.set_map_bar_visible(True)
        shift_arr = np.array(map_mean_shift, dtype=np.float64, copy=True)
        amp_arr = (
            np.full_like(shift_arr, np.nan)
            if map_mean_amp is None
            else np.array(map_mean_amp, dtype=np.float64, copy=True)
        )
        base_arr = (
            np.full_like(shift_arr, np.nan)
            if map_mean_baseline is None
            else np.array(map_mean_baseline, dtype=np.float64, copy=True)
        )
        self._last_live_map_state = {
            "frame": None if frame is None else np.asarray(frame),
            "x_coords_um": np.asarray(x_coords_um, dtype=np.float64),
            "y_coords_um": np.asarray(y_coords_um, dtype=np.float64),
            "map_mean_shift": shift_arr,
            "map_mean_lw": np.array(map_mean_lw, dtype=np.float64, copy=True),
            "map_shift_0": np.array(map_shift_0, dtype=np.float64, copy=True),
            "map_vs": np.array(map_vs, dtype=np.float64, copy=True),
            "map_mean_amp": amp_arr,
            "map_mean_baseline": base_arr,
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
        if self._view_mode == "live_map" and self._last_live_map_state is not None:
            self._render_live_map_state()

    def _on_map_auto_toggled(self, checked: bool):
        if self._syncing_map_range:
            return
        metric_key = self._current_metric_key()
        cfg = self._map_range_settings[metric_key]
        cfg["auto"] = bool(checked)

        if not checked and self._last_live_map_state is not None:
            minmax = self._compute_finite_minmax(self._last_live_map_state[cfg["state_key"]])
            if minmax is not None:
                d_min, d_max = minmax
                eps = 1.0 if metric_key in ("v_s", "mean_amp", "mean_baseline") else 0.001
                cfg["vmin"] = d_min
                cfg["vmax"] = d_max if d_max > d_min else d_min + eps

        self._sync_map_range_widgets()
        if self._view_mode == "live_map" and self._last_live_map_state is not None:
            self._render_live_map_state()

    def _on_map_bounds_changed(self):
        if self._syncing_map_range:
            return
        metric_key = self._current_metric_key()
        cfg = self._map_range_settings[metric_key]
        cfg["vmin"] = float(self.spin_map_vmin.value())
        cfg["vmax"] = float(self.spin_map_vmax.value())
        if not cfg["auto"] and self._view_mode == "live_map" and self._last_live_map_state is not None:
            self._render_live_map_state()

    def _render_live_map_state(self):
        st = self._last_live_map_state
        if st is None:
            return

        if self._view_mode != "live_map" or self._live_map_widgets is None:
            self._build_live_map_layout()

        w_map = self._live_map_widgets
        p_img = w_map["p_img"]
        im_frame = w_map["im_frame"]
        cb_frame = w_map["cb_frame"]
        txt_waiting = w_map["txt_waiting"]
        p_map = w_map["p_map"]
        im_map = w_map["im_map"]
        cb_map = w_map["cb_map"]
        crosshair = w_map["crosshair"]

        frame = st["frame"]
        xs = st["x_coords_um"]
        ys = st["y_coords_um"]
        iy, ix = st["iy"], st["ix"]
        pos_um = st["pos_um"] or {}
        point_fit = st["point_fit"]

        x_val = pos_um.get("X")
        if x_val is None:
            x_val = float(xs[min(ix, len(xs) - 1)])
        y_val = pos_um.get("Y")
        if y_val is None:
            y_val = float(ys[min(iy, len(ys) - 1)])

        if frame is not None:
            cmap_img = self.DEFAULT_CMAP
            vmin, vmax = self._compute_clim(frame)
            fh, fw = frame.shape[:2]
            if cmap_img != w_map["last_frame_cmap"]:
                cb_frame.setColorMap(pg_helpers.get_pg_colormap(cmap_img))
                w_map["last_frame_cmap"] = cmap_img
            im_frame.setVisible(True)
            im_frame.setImage(frame, autoLevels=False, levels=(vmin, vmax))
            im_frame.setRect(QtCore.QRectF(-0.5, -0.5, float(fw), float(fh)))
            cb_frame.setLevels((vmin, vmax))
            txt_waiting.setVisible(False)
            p_img.setTitle(
                f"Point [row={iy}, col={ix}] @ ({x_val:.1f}, {y_val:.1f}) µm",
                color="#e0e0e0",
                size="10pt",
            )
            if w_map["last_frame_shape"] != (fh, fw):
                w_map["last_frame_shape"] = (fh, fw)
                p_img.autoRange()
            self._hover_plots[0]["data"] = frame
        else:
            im_frame.setVisible(False)
            txt_waiting.setText(
                f"Map scan started\nAcquiring [row={iy}, col={ix}] @ ({x_val:.1f}, {y_val:.1f}) µm..."
            )
            txt_waiting.setPos(0.5, 0.5)
            txt_waiting.setVisible(True)
            p_img.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0.0)
            p_img.setTitle("Detector Frame (waiting for point 1)", color="#e0e0e0", size="10pt")
            self._hover_plots[0]["data"] = None
            self.lbl_stats.setText("Map initialized — acquiring first point...")

        metric_key = self._current_metric_key()
        cfg = self._map_range_settings[metric_key]
        data_2d = st[cfg["state_key"]]
        cmap_m = cfg["cmap"]

        live_bounds = self._compute_finite_minmax(data_2d)
        if cfg["auto"]:
            if live_bounds is not None:
                self._sync_map_range_widgets(live_bounds=live_bounds)
                d_min, d_max = live_bounds
                map_vmin, map_vmax = (
                    (d_min, d_max)
                    if d_max > d_min
                    else (d_min, d_min + (1.0 if metric_key in ("v_s", "mean_amp", "mean_baseline") else 1e-3))
                )
            else:
                map_vmin, map_vmax = cfg["vmin"], cfg["vmax"]
        else:
            map_vmin = float(cfg["vmin"])
            map_vmax = float(cfg["vmax"])
            if map_vmax <= map_vmin:
                eps = 1.0 if metric_key in ("v_s", "mean_amp", "mean_baseline") else 1e-3
                map_vmax = map_vmin + eps

        # Orientation handling if user acquired reverse X or Y grid directions
        disp_2d = data_2d
        if len(xs) > 1 and xs[1] < xs[0]:
            disp_2d = np.fliplr(disp_2d)
        if len(ys) > 1 and ys[1] < ys[0]:
            disp_2d = np.flipud(disp_2d)

        rgba_map, lo_m, hi_m = pg_helpers.array_to_rgba_with_nan(
            disp_2d, cmap_m, vmin=map_vmin, vmax=map_vmax, nan_rgba=(30, 30, 30, 0)
        )

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

        map_rect = (x_min, x_max, y_min, y_max)
        im_map.setImage(rgba_map, autoLevels=False)
        im_map.setRect(QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
        if w_map["last_map_metric"] != metric_key:
            cb_map.setColorMap(pg_helpers.get_pg_colormap(cmap_m))
            cb_map.axis.setLabel(cfg["cbar_label"])
            p_map.setTitle(cfg["title"], color="#e0e0e0", size="10pt")
            w_map["last_map_metric"] = metric_key
        cb_map.setLevels((lo_m, hi_m))

        crosshair.setData([x_val], [y_val])
        if w_map["last_map_rect"] != map_rect:
            w_map["last_map_rect"] = map_rect
            p_map.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0.02)

        self._hover_plots[1]["data"] = disp_2d
        self._hover_plots[1]["extent"] = map_rect
        self._hover_plots[1]["unit"] = cfg["suffix"].strip()

        if point_fit is not None:
            amp_txt = (
                f"  |  A = {point_fit['mean_amp_adu']:.1f} ADU"
                if "mean_amp_adu" in point_fit and np.isfinite(point_fit["mean_amp_adu"])
                else ""
            )
            base_txt = (
                f"  |  b = {point_fit['mean_baseline_adu']:.1f} ADU"
                if "mean_baseline_adu" in point_fit and np.isfinite(point_fit["mean_baseline_adu"])
                else ""
            )
            self.lbl_stats.setText(
                f"Map [{iy},{ix}]: ν_B = {point_fit['mean_shift_ghz']:.3f} GHz  |  "
                f"Γ_B = {point_fit['mean_lw_ghz']:.3f} GHz  |  "
                f"S(0) = {point_fit['Shift_0_ghz']:.3f} GHz  |  "
                f"V_s = {point_fit['V_s_ms']:.1f} m/s{amp_txt}{base_txt}"
            )

    # ==========================================================================
    # Brightfield Alignment & Stage Plate Calibration Views (PyQtGraph)
    # ==========================================================================

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

        # Magenta (R=ref, B=ref) vs Green (G=mov): aligned overlap -> neutral gray/white
        rgb_before = np.stack([ref_n, mov_n, ref_n], axis=-1)
        rgb_after = np.stack([ref_n, mov_aligned, ref_n], axis=-1)

        p1, _ = self._add_dark_image_panel(
            0, 0, rgb_before, "gray",
            f"Before Alignment: {ref_label} (Mag) vs {cur_label} (Grn) | Shift = ({dcol_px:+.2f}, {drow_px:+.2f}) px",
            "X (px)", "Y (px)", is_rgb=True,
        )
        h, w = ref_n.shape[:2]
        cx, cy = 0.5 * w, 0.5 * h
        if abs(dcol_px) > 0.5 or abs(drow_px) > 0.5:
            pg_helpers.add_vector_arrow(p1, cx, cy, cx + dcol_px, cy + drow_px, "#ffeb3b", width=2.2, head_len=14.0)

        p2, _ = self._add_dark_image_panel(
            0, 1, rgb_after, "gray",
            f"After Stage Correction: ΔX = {dx_um:+.2f} µm, ΔY = {dy_um:+.2f} µm (Aligned Overlay)",
            "X (px)", "Y (px)", is_rgb=True,
        )
        p2.setXLink(p1)
        p2.setYLink(p1)

        self.lbl_stats.setText(
            f"BF Align [{cur_label} → {ref_label}]: Δpx=({dcol_px:+.2f}, {drow_px:+.2f}) | "
            f"Stage Δ=({dx_um:+.2f}, {dy_um:+.2f}) µm"
        )

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
        """Renders the 3 brightfield jog images in a 2x2 PyQtGraph grid."""
        self._last_live_map_state = None
        self._enter_diagnostic_mode()

        cmap = self.DEFAULT_CMAP
        specs = [
            (0, 0, img0, f"1. Base ({x0:.1f}, {y0:.1f}) µm"),
            (0, 1, img_x, f"2. +X ({x0 + dx_um:.1f}, {y0:.1f}) µm [ΔX={dx_um:+.1f}]"),
            (1, 0, img_y, f"3. +Y ({x0:.1f}, {y0 + dy_um:.1f}) µm [ΔY={dy_um:+.1f}]"),
        ]
        for row, col, arr, title in specs:
            vmin, vmax = np.percentile(arr, [1.0, 99.5])
            if vmax <= vmin:
                vmin, vmax = float(np.min(arr)), float(np.max(arr)) + 1.0
            self._add_dark_image_panel(
                row, col, arr, cmap, title, "X (px)", "Y (px)",
                cbar_label=None, vmin=float(vmin), vmax=float(vmax), value_unit="ADU",
            )

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
        self._render_plate_jog_panels(img0, img_x, img_y, x0, y0, dx_um, dy_um)
        p4 = self._pg_widget.addPlot(row=1, col=1)
        self._style_dark_plot(p4, "4. Correlation Result (Pending)", "", "")
        p4.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0.0)
        p4.hideAxis("bottom")
        p4.hideAxis("left")
        txt = pg.TextItem(
            text="3 jog frames captured.\nClick 'Continue' to compute\nautocorrelation & cross-correlation.",
            color="#cfd8dc",
            anchor=(0.5, 0.5),
            fill=pg.mkBrush(30, 30, 30, 220),
            border=pg.mkPen("#455a64"),
        )
        txt.setPos(0.5, 0.5)
        p4.addItem(txt)
        self.lbl_stats.setText(
            f"Plate Calib Step 1: 3 frames captured at Base ({x0:.1f}, {y0:.1f}), +{dx_um:.1f} X, +{dy_um:.1f} Y µm"
        )

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
        Step 2 diagnostic view: shows the 3 jog images plus panel 4 as an RGB composite of
        the correlation maps centered at (0, 0): Red=CC(+X), Green=CC(+Y), Blue=AutoCorr(Base).
        """
        self._render_plate_jog_panels(img0, img_x, img_y, x0, y0, dx_um, dy_um)

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
        extent = (-half_span - 0.5, half_span + 0.5, -half_span - 0.5, half_span + 0.5)
        p4, _ = self._add_dark_image_panel(
            1, 1, rgb_corr, "gray",
            f"4. RGB Correlation (R=+X, G=+Y, B=Auto) | +X=({dc_x:+.1f},{dr_x:+.1f}) [{pk_x:.2f}] | +Y=({dc_y:+.1f},{dr_y:+.1f}) [{pk_y:.2f}]",
            "Shift ΔX (px)", "Shift ΔY (px)",
            extent=extent, is_rgb=True, invert_y=True,
        )
        grid_pen = pg.mkPen(255, 255, 255, 110, width=0.8, style=QtCore.Qt.PenStyle.DotLine)
        p4.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=grid_pen))
        p4.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=grid_pen))
        p4.plot([0.0], [0.0], pen=None, symbol="o", symbolSize=6, symbolBrush=None, symbolPen=pg.mkPen("w", width=1.2))
        pg_helpers.add_vector_arrow(p4, 0.0, 0.0, dc_x, dr_x, "#ff5252", width=2.0, head_len=12.0)
        pg_helpers.add_vector_arrow(p4, 0.0, 0.0, dc_y, dr_y, "#69f0ae", width=2.0, head_len=12.0)

        self.lbl_stats.setText(
            f"Plate Calib Step 2: +X=({dc_x:+.2f}, {dr_x:+.2f}) px [pk={pk_x:.3f}] | "
            f"+Y=({dc_y:+.2f}, {dr_y:+.2f}) px [pk={pk_y:.3f}]"
        )

    # ==========================================================================
    # Unified PyQtGraph Hover Readout
    # ==========================================================================

    def _on_pg_mouse_move(self, scene_pos):
        if self._view_mode == "single":
            if self._current_image is None or self._pg_plot is None:
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
            return

        for entry in self._hover_plots:
            p = entry["plot"]
            if p.sceneBoundingRect().contains(scene_pos):
                pt = p.vb.mapSceneToView(scene_pos)
                x_v, y_v = float(pt.x()), float(pt.y())
                arr = entry["data"]
                unit = entry["unit"]
                ext = entry["extent"]
                if arr is not None:
                    h, w = arr.shape[:2]
                    if ext is not None:
                        x0, x1, y0, y1 = ext
                        span_x = max(x1 - x0, 1e-12)
                        span_y = max(y1 - y0, 1e-12)
                        ix = int(np.floor((x_v - x0) / span_x * w))
                        iy = int(np.floor((y_v - y0) / span_y * h))
                        if 0 <= iy < h and 0 <= ix < w:
                            val = float(arr[iy, ix])
                            val_txt = f"  |  Value = {val:.3f} {unit}".rstrip() if np.isfinite(val) else "  |  Value = NaN"
                            self.lbl_cursor.setText(f"Cursor: X = {x_v:.2f}, Y = {y_v:.2f}{val_txt}")
                            return
                    else:
                        col = int(round(x_v))
                        row = int(round(y_v))
                        if 0 <= row < h and 0 <= col < w:
                            val = float(arr[row, col])
                            if np.isfinite(val):
                                val_txt = f"{int(val):5d}" if unit == "ADU" and abs(val - round(val)) < 1e-6 else f"{val:.3f}"
                                self.lbl_cursor.setText(
                                    f"Cursor: X = {col:4d} px, Y = {row:4d} px   |   {val_txt} {unit}".rstrip()
                                )
                                return
                self.lbl_cursor.setText(f"Cursor: X = {x_v:.2f}, Y = {y_v:.2f}")
                return

        self.lbl_cursor.setText("Cursor: —")
