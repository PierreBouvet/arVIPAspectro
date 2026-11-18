"""
Visualization helpers and calibration diagnostic plots built entirely on PyQtGraph.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters


_PG_CMAP_CACHE: Dict[str, pg.ColorMap] = {}


def get_pg_colormap(name: str) -> pg.ColorMap:
    """
    Returns a cached native `pg.ColorMap` without invoking Matplotlib.
    Supports PyQtGraph built-ins (`viridis`, `inferno`, `magma`, `plasma`, `turbo`,
    `cividis`) as well as custom native stops for `gray`, `coolwarm`, and `RdBu_r`.
    """
    key = (name or "viridis").strip()
    if key in _PG_CMAP_CACHE:
        return _PG_CMAP_CACHE[key]

    low = key.lower()
    if low == "gray":
        cmap = pg.ColorMap(
            pos=np.array([0.0, 1.0]),
            color=np.array([[0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte),
        )
    elif low == "coolwarm":
        cmap = pg.ColorMap(
            pos=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            color=np.array(
                [
                    [59, 76, 192, 255],
                    [141, 176, 254, 255],
                    [221, 221, 221, 255],
                    [244, 154, 123, 255],
                    [180, 4, 38, 255],
                ],
                dtype=np.ubyte,
            ),
        )
    elif low == "rdbu_r":
        cmap = pg.ColorMap(
            pos=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            color=np.array(
                [
                    [5, 48, 97, 255],
                    [67, 147, 195, 255],
                    [247, 247, 247, 255],
                    [214, 96, 77, 255],
                    [103, 0, 31, 255],
                ],
                dtype=np.ubyte,
            ),
        )
    else:
        cmap = pg.colormap.get(key)
        if cmap is None:
            cmap = pg.colormap.get("viridis")

    _PG_CMAP_CACHE[key] = cmap
    return cmap


def array_to_rgba_with_nan(
    data_2d: np.ndarray,
    cmap_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nan_rgba: Tuple[int, int, int, int] = (30, 30, 30, 0),
) -> Tuple[np.ndarray, float, float]:
    """
    Maps a 2D scalar array (potentially containing NaNs) to an `(H, W, 4)` uint8 RGBA array.
    Non-finite pixels receive `nan_rgba` (transparent by default) instead of saturating endpoints.
    """
    arr = np.asarray(data_2d, dtype=np.float64)
    finite = np.isfinite(arr)
    if vmin is None or vmax is None:
        if np.any(finite):
            d_min = float(np.min(arr[finite])) if vmin is None else float(vmin)
            d_max = float(np.max(arr[finite])) if vmax is None else float(vmax)
        else:
            d_min, d_max = 0.0, 1.0
    else:
        d_min, d_max = float(vmin), float(vmax)

    if d_max <= d_min:
        d_max = d_min + 1e-6

    norm = np.zeros_like(arr, dtype=np.float64)
    norm[finite] = np.clip((arr[finite] - d_min) / (d_max - d_min), 0.0, 1.0)
    cmap = get_pg_colormap(cmap_name)
    rgba = cmap.map(norm, mode="byte")
    if not np.all(finite):
        rgba[~finite] = np.asarray(nan_rgba, dtype=np.ubyte)
    return rgba, d_min, d_max


def scatter_brushes(
    values: np.ndarray,
    cmap_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: int = 160,
) -> Tuple[list, float, float, pg.ColorMap]:
    """Builds per-point QBrush colors from a scalar array via a native PyQtGraph ColorMap."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(vals)
    if vmin is None or vmax is None:
        if np.any(finite):
            lo = float(np.min(vals[finite])) if vmin is None else float(vmin)
            hi = float(np.max(vals[finite])) if vmax is None else float(vmax)
        else:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = float(vmin), float(vmax)

    if hi <= lo:
        hi = lo + 1e-6

    norm = np.zeros_like(vals, dtype=np.float64)
    norm[finite] = np.clip((vals[finite] - lo) / (hi - lo), 0.0, 1.0)
    cmap = get_pg_colormap(cmap_name)
    rgba = cmap.map(norm, mode="byte")
    brushes = [
        pg.mkBrush(int(r), int(g), int(b), alpha if ok else 0)
        for (r, g, b, _), ok in zip(rgba, finite)
    ]
    return brushes, lo, hi, cmap


def add_vector_arrow(
    plot_item: pg.PlotItem,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: Any,
    width: float = 2.0,
    head_len: float = 12.0,
) -> Tuple[pg.PlotDataItem, pg.ArrowItem]:
    """
    Draws a clean shaft + arrowhead from `(x0, y0)` to `(x1, y1)` in `plot_item` data coords,
    handling both inverted (`invertY(True)`) and non-inverted Y axes accurately.
    """
    shaft = pg.PlotDataItem([x0, x1], [y0, y1], pen=pg.mkPen(color, width=width))
    plot_item.addItem(shaft)
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    dy_screen = dy if plot_item.vb.yInverted() else -dy
    angle_deg = float(np.degrees(np.arctan2(-dy_screen, -dx)))
    head = pg.ArrowItem(
        pos=(x1, y1),
        angle=angle_deg,
        headLen=head_len,
        tipAngle=30,
        baseAngle=20,
        brush=pg.mkBrush(color),
        pen=None,
    )
    plot_item.addItem(head)
    return shaft, head


def attach_top_theta_axis(
    plot_item: pg.PlotItem,
    theta_deg: np.ndarray,
    n_medium: float,
    lambda_vacuum: float,
    label_color: Any = "#455a64",
):
    """Adds scattering-angle ticks along the top axis matching `q = 2*n*k0*cos(theta/2)`."""
    k0 = 2.0 * np.pi / float(lambda_vacuum)
    th_ticks = np.linspace(float(np.min(theta_deg)), float(np.max(theta_deg)), 5)
    q_ticks = 2.0 * float(n_medium) * k0 * np.cos(np.deg2rad(th_ticks) / 2.0) * 1e-6
    plot_item.showAxis("top")
    ax_top = plot_item.getAxis("top")
    ax_top.setLabel("Scattering Angle θ (°)", color=label_color)
    ax_top.setPen(pg.mkPen(label_color))
    ax_top.setTextPen(pg.mkPen(label_color))
    ax_top.setTicks([[(float(qv), f"{tv:.1f}°") for qv, tv in zip(q_ticks, th_ticks)]])


def style_plot_light(
    plot_item: pg.PlotItem,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
):
    """Configures readable dark axes/text for white-background exported figures."""
    if title:
        plot_item.setTitle(title, color="#111111", size="10pt")
    for side, text in (("bottom", xlabel), ("left", ylabel)):
        ax = plot_item.getAxis(side)
        ax.setPen(pg.mkPen("#333333"))
        ax.setTextPen(pg.mkPen("#222222"))
        if text:
            plot_item.setLabel(side, text, color="#111111")
    if grid:
        plot_item.showGrid(x=True, y=True, alpha=0.25)


def add_light_image_panel(
    layout_parent: pg.GraphicsLayout,
    row: int,
    col: int,
    data: np.ndarray,
    cmap_name: str = "gray",
    title: str = "",
    xlabel: str = "Pixel x",
    ylabel: str = "Pixel y",
    cbar_label: str = "Counts",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    aspect_equal: bool = True,
    invert_y: bool = True,
    colspan: int = 1,
) -> Tuple[pg.PlotItem, pg.ImageItem, pg.ColorBarItem]:
    """Creates a white-background image PlotItem with pixel-centered bounds and a ColorBarItem."""
    arr = np.asarray(data, dtype=np.float64)
    sub = layout_parent.addLayout(row=row, col=col, colspan=colspan)
    p = sub.addPlot(row=0, col=0)
    p.invertY(invert_y)
    if aspect_equal:
        p.setAspectLocked(True)
    style_plot_light(p, title=title, xlabel=xlabel, ylabel=ylabel, grid=False)

    finite = np.isfinite(arr)
    if vmin is None or vmax is None:
        if np.any(finite):
            lo = float(np.min(arr[finite])) if vmin is None else float(vmin)
            hi = float(np.max(arr[finite])) if vmax is None else float(vmax)
        else:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = float(vmin), float(vmax)
    if hi <= lo:
        hi = lo + 1e-6

    im = pg.ImageItem(axisOrder="row-major")
    h, w = arr.shape[:2]
    im.setRect(QtCore.QRectF(-0.5, -0.5, float(w), float(h)))
    if not np.all(finite):
        rgba, lo, hi = array_to_rgba_with_nan(arr, cmap_name, lo, hi, nan_rgba=(240, 240, 240, 0))
        im.setImage(rgba, autoLevels=False)
        p.addItem(im)
        cb = pg.ColorBarItem(
            values=(lo, hi),
            colorMap=get_pg_colormap(cmap_name),
            label=cbar_label,
            interactive=False,
            width=14,
        )
        cb.axis.setTextPen(pg.mkPen("#222222"))
        sub.addItem(cb, row=0, col=1)
    else:
        im.setImage(arr, autoLevels=False, levels=(lo, hi))
        p.addItem(im)
        cb = pg.ColorBarItem(
            values=(lo, hi),
            colorMap=get_pg_colormap(cmap_name),
            label=cbar_label,
            interactive=True,
            width=14,
        )
        cb.axis.setTextPen(pg.mkPen("#222222"))
        cb.setImageItem(im, insert_in=p)
    p.autoRange()
    return p, im, cb


def add_light_scatter_with_cbar(
    layout_parent: pg.GraphicsLayout,
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
    alpha: int = 160,
    invert_y: bool = False,
    aspect_equal: bool = False,
    colspan: int = 1,
) -> Tuple[pg.PlotItem, pg.ScatterPlotItem]:
    """Creates a white-background scatter PlotItem inside a sub-layout with its own ColorBarItem."""
    sub = layout_parent.addLayout(row=row, col=col, colspan=colspan)
    p = sub.addPlot(row=0, col=0)
    p.invertY(invert_y)
    if aspect_equal:
        p.setAspectLocked(True)
    style_plot_light(p, title=title, xlabel=xlabel, ylabel=ylabel)

    brushes, lo, hi, cmap = scatter_brushes(c, cmap_name, vmin=vmin, vmax=vmax, alpha=alpha)
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
    cb.axis.setTextPen(pg.mkPen("#222222"))
    sub.addItem(cb, row=0, col=1)
    return p, sc


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _create_export_widget(width: int = 1400, height: int = 650, title: str = "") -> pg.GraphicsLayoutWidget:
    _ensure_qapp()
    win = pg.GraphicsLayoutWidget()
    win.setBackground("w")
    win.resize(width, height)
    win.ci.resize(width, height)
    if title:
        win.setWindowTitle(title)
    return win


def _finalize_pg_figure(
    win: pg.GraphicsLayoutWidget,
    out_dir: Optional[str],
    filename: str,
    show_plots: bool = False,
):
    app = _ensure_qapp()
    win.ci.layout.activate()
    app.processEvents()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, filename)
        exporter = pyqtgraph.exporters.ImageExporter(win.scene())
        exporter.parameters()["width"] = max(int(win.width()), 800)
        exporter.export(save_path)

    if show_plots:
        win.show()
        app.exec()
    win.close()


# ==============================================================================
# Step 5.1 – 5.4 Diagnostic Plot Functions (PyQtGraph)
# ==============================================================================


def plot_denoise_laser(
    data_laser_reference,
    baseline_2d,
    data_subtracted,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots the raw laser reference, morphological baseline, and background-subtracted fringes.
    """
    win = _create_export_widget(1600, 520, "Laser Reference Denoising")
    add_light_image_panel(
        win, 0, 0, data_laser_reference, "gray",
        "(a) Raw Monochromatic Laser Reference", "Pixel x", "Pixel y", "Counts",
        vmin=float(np.min(data_laser_reference)), vmax=float(np.percentile(data_laser_reference, 95)),
    )
    add_light_image_panel(
        win, 0, 1, baseline_2d, "gray",
        "(b) Morphological Background Baseline", "Pixel x", "Pixel y", "Counts",
        vmin=float(np.min(baseline_2d)), vmax=float(np.percentile(baseline_2d, 95)),
    )
    clipped = np.clip(data_subtracted, 0, np.percentile(data_subtracted, 95))
    add_light_image_panel(
        win, 0, 2, clipped, "gray",
        "(c) Background-Subtracted Laser Fringes", "Pixel x", "Pixel y", "Net Fringe Counts",
    )
    _finalize_pg_figure(win, out_dir, f"{prefix}2 - Cleaning laser lines.png", show_plots)


def plot_laser_fringes(
    average_data,
    peaks,
    data_subtracted,
    points,
    y_center_laser,
    dy_laser,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots detected fringe seeds along the center row-average and full-height traced fringes.
    """
    win = _create_export_widget(1500, 560, "Laser Fringe Identification")
    p0 = win.addPlot(row=0, col=0)
    style_plot_light(
        p0,
        f"Detected Laser Fringe Seeds ({len(peaks)} Orders)",
        "Pixel x",
        "Net Intensity (counts)",
    )
    p0.addLegend(offset=(10, 10))
    p0.plot(
        np.arange(len(average_data)),
        average_data,
        pen=pg.mkPen("#1f77b4", width=1.4),
        name=f"Row avg (y={y_center_laser}±{dy_laser})",
    )
    for peak in peaks:
        p0.addItem(
            pg.InfiniteLine(
                pos=float(peak[0]),
                angle=90,
                pen=pg.mkPen("#d62728", width=1.0, style=QtCore.Qt.PenStyle.DashLine),
            )
        )

    p1, _, _ = add_light_image_panel(
        win, 0, 1, data_subtracted, "gray",
        f"Full-Height Traced Laser Fringes (All {len(points)} Orders Recovered)",
        "Pixel x", "Pixel y", "Net ADU",
        vmin=float(np.percentile(average_data, 20)),
        vmax=float(np.percentile(average_data, 90)),
    )
    for fringe in points:
        p1.plot(
            np.asarray(fringe[0], dtype=np.float64),
            np.asarray(fringe[1], dtype=np.float64),
            pen=None,
            symbol="+",
            symbolSize=4,
            symbolPen=pg.mkPen("#e53935"),
        )

    _finalize_pg_figure(win, out_dir, f"{prefix}3 - Laser fringe identification.png", show_plots)


def plot_fringe_refinement(
    all_x_laser,
    all_x_laser_smooth,
    all_y_laser,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots the raw sub-pixel centroid coordinates versus polynomial-smoothed coordinates.
    """
    win = _create_export_widget(900, 760, "Laser Fringe Refinement")
    p = win.addPlot(row=0, col=0)
    p.invertY(True)
    style_plot_light(p, "Laser Fringe Sub-Pixel Refinement & Smoothing", "Pixel x", "Pixel y")
    p.addLegend(offset=(10, 10))
    p.plot(
        np.asarray(all_x_laser, dtype=np.float64),
        np.asarray(all_y_laser, dtype=np.float64),
        pen=None,
        symbol="+",
        symbolSize=5,
        symbolPen=pg.mkPen(30, 100, 220, 150),
        name="Sub-pixel centroided",
    )
    p.plot(
        np.asarray(all_x_laser_smooth, dtype=np.float64),
        np.asarray(all_y_laser, dtype=np.float64),
        pen=None,
        symbol="o",
        symbolSize=3,
        symbolBrush=pg.mkBrush(220, 40, 40, 170),
        symbolPen=None,
        name="Polynomial-smoothed",
    )
    _finalize_pg_figure(win, out_dir, f"{prefix}4 - Laser fringe refinement.png", show_plots)


def plot_concentric_fringes_and_geometry(
    data_subtracted,
    refined_fringes,
    x0_fit,
    y0_fit,
    xc_gnom,
    yc_gnom,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots refined fringes overlaid with the fitted optical axis (x0, y0) and gnomonic ring center (xc, yc).
    """
    win = _create_export_widget(960, 820, "Gnomonic Projection & Optical Axis")
    p, _, _ = add_light_image_panel(
        win, 0, 0, data_subtracted, "gray",
        "Gnomonic Projection & Optical Axis Identification",
        "Pixel x", "Pixel y", "Net ADU",
        vmin=float(np.percentile(data_subtracted, 20)),
        vmax=float(np.percentile(data_subtracted, 90)),
    )
    p.addLegend(offset=(10, 10))
    dash_pen = pg.mkPen("#e53935", width=1.1, style=QtCore.Qt.PenStyle.DashLine)
    for fringe in refined_fringes:
        sort_idx = np.argsort(fringe[1])
        p.plot(fringe[0][sort_idx], fringe[1][sort_idx], pen=dash_pen)
    p.plot(
        [x0_fit], [y0_fit], pen=None, symbol="x", symbolSize=13,
        symbolPen=pg.mkPen("#222222", width=2),
        name=f"Optical axis ({x0_fit:.1f}, {y0_fit:.1f})",
    )
    p.plot(
        [xc_gnom], [yc_gnom], pen=None, symbol="+", symbolSize=13,
        symbolPen=pg.mkPen("#00acc1", width=2),
        name=f"Gnomonic center ({xc_gnom:.1f}, {yc_gnom:.1f})",
    )
    _finalize_pg_figure(win, out_dir, f"{prefix}5 - Identification gnomonic projection.png", show_plots)


def plot_anamorphic_calibration(
    q_orders,
    q_pred_2d,
    all_k_laser,
    all_x_laser,
    all_y_laser,
    err_2d_mhz,
    err_2d_smooth_mhz,
    fsr_vipa,
    nb_laser_modes,
    out_dir,
    prefix="",
    show_plots=False,
    lower_order_study=False,
    q_pred_lin=None,
    q_pred_quad=None,
):
    """
    Plots inter-order frequency residuals and 2D spatial residual map after anamorphic calibration.
    """
    win = _create_export_widget(1500, 560, "Dispersion Calibration Residuals")
    p0 = win.addPlot(row=0, col=0)
    style_plot_light(
        p0,
        "Inter-Order Frequency Residual vs Order q",
        "Fringe Order q",
        "Mean Frequency Residual (MHz)",
    )
    p0.addLegend(offset=(10, 10))
    order_means_2d = [
        (np.mean(q_pred_2d[all_k_laser == k]) - k) * fsr_vipa * 1e3
        if np.any(all_k_laser == k)
        else 0.0
        for k in range(nb_laser_modes)
    ]
    p0.plot(
        q_orders,
        order_means_2d,
        pen=pg.mkPen("#2ca02c", width=2.0),
        symbol="d",
        symbolBrush="#2ca02c",
        symbolSize=7,
        name="2D Anamorphic φ_order(g, y)",
    )
    p0.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen("#555555", style=QtCore.Qt.PenStyle.DotLine)))

    if lower_order_study and q_pred_lin is not None and q_pred_quad is not None:
        order_means_lin = [
            (np.mean(q_pred_lin[all_k_laser == k]) - k) * fsr_vipa * 1e3
            if np.any(all_k_laser == k)
            else 0.0
            for k in range(nb_laser_modes)
        ]
        order_means_quad = [
            (np.mean(q_pred_quad[all_k_laser == k]) - k) * fsr_vipa * 1e3
            if np.any(all_k_laser == k)
            else 0.0
            for k in range(nb_laser_modes)
        ]
        p0.plot(
            q_orders, order_means_lin,
            pen=pg.mkPen("#d62728", width=1.6, style=QtCore.Qt.PenStyle.DashLine),
            symbol="o", symbolBrush="#d62728", symbolSize=6, name="1D Linear g → q",
        )
        p0.plot(
            q_orders, order_means_quad,
            pen=pg.mkPen("#ff7f0e", width=1.6, style=QtCore.Qt.PenStyle.DotLine),
            symbol="s", symbolBrush="#ff7f0e", symbolSize=6, name="1D Quadratic g → q",
        )

    rms_raw = float(np.sqrt(np.mean(err_2d_mhz**2)))
    rms_smooth = float(np.sqrt(np.mean(err_2d_smooth_mhz**2)))
    add_light_scatter_with_cbar(
        win,
        row=0,
        col=1,
        x=all_x_laser,
        y=all_y_laser,
        c=err_2d_mhz,
        cmap_name="RdBu_r",
        title=f"2D Residuals (RMS: {rms_raw:.1f} MHz raw, {rms_smooth:.1f} MHz smooth)",
        xlabel="Pixel x",
        ylabel="Pixel y",
        cbar_label="Residual (MHz)",
        vmin=-60.0,
        vmax=60.0,
        size=3.0,
        alpha=200,
        invert_y=True,
        aspect_equal=True,
    )
    _finalize_pg_figure(win, out_dir, f"{prefix}6 - Dispersion calibration residuals.png", show_plots)


def plot_rb_theoretical_spectrum(
    det_table_MHz,
    trans_table,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots the theoretical natural Rubidium D2 transmission spectrum.
    """
    win = _create_export_widget(960, 480, "Theoretical Rb D2 Spectrum")
    p = win.addPlot(row=0, col=0)
    style_plot_light(
        p,
        "Theoretical Transmission Spectrum of Rubidium D2 Line",
        "Detuning from Center of Gravity (GHz)",
        "Transmittivity",
    )
    p.plot(det_table_MHz * 1e-3, trans_table, pen=pg.mkPen("#d62728", width=1.6))
    _finalize_pg_figure(win, out_dir, f"{prefix}7 - Absorption profile of the Rubidium cell.png", show_plots)


def plot_denoise_rb(
    data_raw,
    baseline_absorption,
    data_absorption,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots the raw Rubidium absorption image, morphological baseline, and background-subtracted absorption.
    """
    win = _create_export_widget(1600, 520, "Rubidium Absorption Denoising")
    add_light_image_panel(
        win, 0, 0, data_raw, "gray",
        "(a) Raw Rubidium Absorption", "Pixel x", "Pixel y", "Counts",
        vmin=float(np.min(data_raw)), vmax=float(np.percentile(data_raw, 95)),
    )
    add_light_image_panel(
        win, 0, 1, baseline_absorption, "gray",
        "(b) Morphological Background Baseline", "Pixel x", "Pixel y", "Counts",
        vmin=float(np.min(baseline_absorption)), vmax=float(np.percentile(baseline_absorption, 95)),
    )
    clipped = np.clip(data_absorption, 0, np.percentile(data_absorption, 95))
    add_light_image_panel(
        win, 0, 2, clipped, "gray",
        "(c) Background-Subtracted Absorption Image", "Pixel x", "Pixel y", "Counts",
    )
    _finalize_pg_figure(win, out_dir, f"{prefix}8 - Cleaning Rubidium absorption figure.png", show_plots)


def plot_rb_experimental_profile(
    phi_order_r,
    avg_spectrum_radius,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots the experimentally extracted radial absorption profile vs interference orders.
    """
    win = _create_export_widget(960, 480, "Experimental Rb Radial Profile")
    p = win.addPlot(row=0, col=0)
    style_plot_light(
        p,
        "Experimental Rubidium Absorption Profile vs Interference Order",
        "Interference Order φ_order",
        "Intensity (counts)",
    )
    p.plot(phi_order_r, avg_spectrum_radius, pen=pg.mkPen("#1f77b4", width=1.4))
    _finalize_pg_figure(
        win, out_dir, f"{prefix}9 - Absorption profile versus orders of interference.png", show_plots
    )


def plot_rb_calibration(
    f_grid,
    orders_used,
    order_spectra,
    f_valid,
    sig_valid,
    baseline_env,
    f_fit_min,
    f_fit_max,
    delta_nu_opt,
    delta_nu_fine,
    fsr_cal,
    trans_fast,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots the coherently folded multi-order spectra and the hyperfine theoretical alignment.
    """
    win = _create_export_widget(1600, 560, "Rubidium FSR Calibration")
    p0 = win.addPlot(row=0, col=0)
    style_plot_light(
        p0,
        "Coherently Folded Rubidium Spectra Across Orders",
        "Δν from Rb D2 Center (GHz)",
        "Intensity (counts)",
    )
    p0.addLegend(offset=(10, 10))
    for idx_o, (k_ord, sp) in enumerate(zip(orders_used, order_spectra)):
        p0.plot(
            f_grid,
            sp,
            pen=pg.mkPen(pg.intColor(idx_o, max(len(orders_used), 1), alpha=130), width=1.0),
            name=f"Order {k_ord}" if idx_o < 6 else None,
        )
    p0.plot(f_valid, sig_valid, pen=pg.mkPen("#111111", width=2.0), name="Order average")
    reg0 = pg.LinearRegionItem(
        values=(float(f_fit_min), float(f_fit_max)),
        brush=pg.mkBrush(44, 160, 44, 35),
        pen=pg.mkPen(44, 160, 44, 120),
        movable=False,
    )
    p0.addItem(reg0)

    p1 = win.addPlot(row=0, col=1)
    style_plot_light(
        p1,
        f"Rubidium D2 Hyperfine Alignment (FSR={fsr_cal:.4f} GHz)",
        "Δν from Rb D2 Center (GHz)",
        "Measured Intensity (counts)",
    )
    p1.addLegend(offset=(10, 10))
    p1.plot(f_valid, sig_valid, pen=pg.mkPen("#1f77b4", width=1.6), name="Order-averaged spectrum")
    p1.plot(
        f_valid,
        baseline_env,
        pen=pg.mkPen("#777777", width=1.2, style=QtCore.Qt.PenStyle.DotLine),
        name="Illumination envelope",
    )
    reg1 = pg.LinearRegionItem(
        values=(float(f_fit_min), float(f_fit_max)),
        brush=pg.mkBrush(44, 160, 44, 35),
        pen=pg.mkPen(44, 160, 44, 120),
        movable=False,
    )
    p1.addItem(reg1)

    # Secondary right axis for normalized theoretical Rb absorption
    p1.showAxis("right")
    p1.getAxis("right").setLabel("Rubidium Absorption (1 - T_Rb)", color="#d62728")
    p1.getAxis("right").setPen(pg.mkPen("#d62728"))
    p1.getAxis("right").setTextPen(pg.mkPen("#d62728"))
    vb2 = pg.ViewBox()
    p1.scene().addItem(vb2)
    p1.getAxis("right").linkToView(vb2)
    vb2.setXLink(p1)
    vb2.setYRange(-0.05, 1.15)

    a_theo = 1.0 - trans_fast((f_valid - delta_nu_fine) * 1e3)
    theo_curve = pg.PlotDataItem(
        f_valid,
        a_theo,
        pen=pg.mkPen("#d62728", width=1.8),
    )
    vb2.addItem(theo_curve)

    def _sync_vb():
        vb2.setGeometry(p1.vb.sceneBoundingRect())
        vb2.linkedViewChanged(p1.vb, vb2.XAxis)

    p1.vb.sigResized.connect(_sync_vb)
    _sync_vb()

    _finalize_pg_figure(
        win, out_dir, f"{prefix}10 - Calibration VIPA Free Spectral Range on Rb absorption.png", show_plots
    )


def plot_frequency_map(freq_map, out_dir, prefix="", show_plots=False):
    """
    Plots the calibrated 2D frequency map zero-referenced to the elastic line.
    """
    win = _create_export_widget(960, 820, "2D VIPA Frequency Map")
    add_light_image_panel(
        win, 0, 0, freq_map, "viridis",
        "Calibrated 2D VIPA Frequency Map (GHz)",
        "Pixel x", "Pixel y", "Frequency Relative to Elastic Line (GHz)",
    )
    _finalize_pg_figure(win, out_dir, f"{prefix}10B - Full 2D Frequency Map.png", show_plots)


def plot_sample_baseline_and_windows(
    data_sample,
    freq_avg_data_sample,
    avg_data_sample,
    bls_peaks,
    dinst_disp_map,
    path_sample,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Plots sample interferogram, central row-averaged spectrum with Anti-Stokes/Stokes windows,
    and 2D higher-order dispersion distortion correction.
    """
    win = _create_export_widget(1800, 520, "Sample Baseline and Brillouin Windows")
    sample_name = os.path.basename(path_sample)

    add_light_image_panel(
        win, 0, 0, np.clip(data_sample, 0, np.percentile(data_sample, 98)), "inferno",
        f"Sample Interferogram ({sample_name})", "Pixel x", "Pixel y", "Counts",
        aspect_equal=False,
    )

    p1 = win.addPlot(row=0, col=1)
    style_plot_light(
        p1,
        "Calibrated Central Spectrum & Brillouin Windows",
        "Calibrated Frequency (GHz)",
        "Intensity (counts)",
    )
    p1.addLegend(offset=(10, 10))
    p1.plot(freq_avg_data_sample, avg_data_sample, pen=pg.mkPen("#1f77b4", width=1.4), name="Central profile")
    for p in bls_peaks[:3]:
        p1.addItem(
            pg.InfiniteLine(
                pos=float(p[0]),
                angle=90,
                pen=pg.mkPen("#2ca02c", width=1.1, style=QtCore.Qt.PenStyle.DashLine),
            )
        )
        p1.addItem(
            pg.InfiniteLine(
                pos=float(p[1]),
                angle=90,
                pen=pg.mkPen("#d62728", width=1.1, style=QtCore.Qt.PenStyle.DashLine),
            )
        )
    p1.setXRange(15, 110)

    add_light_image_panel(
        win, 0, 2, dinst_disp_map * 1e3, "RdBu_r",
        "2D Higher-Order Dispersion Correction Δν_inst(x, y) (MHz)",
        "Pixel x", "Pixel y", "Dispersion Distortion Removed (MHz)",
        vmin=-150.0, vmax=150.0, aspect_equal=False,
    )
    _finalize_pg_figure(win, out_dir, f"{prefix}11 - {sample_name} baseline and windows.png", show_plots)


def plot_dho_overview(
    shift_clipped,
    shift_clipped_averaged,
    linewidth_clipped,
    linewidth_clipped_averaged,
    x_coords,
    y_coords,
    x_coords_averaged,
    y_coords_averaged,
    path_sample,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    2x2 overview plot of extracted Brillouin shift and linewidth across the detector.
    """
    win = _create_export_widget(1500, 1150, "DHO Shift and Linewidth Overview")
    sample_name = os.path.basename(path_sample)

    add_light_scatter_with_cbar(
        win, 0, 0,
        x_coords.ravel(), y_coords.ravel(), shift_clipped.ravel(),
        "plasma", f"Brillouin Shift Δν_B on Detector ({sample_name})",
        "Pixel x", "Pixel y", "Shift (GHz)",
        size=4.0, alpha=170, invert_y=True,
    )

    p01 = win.addPlot(row=0, col=1)
    style_plot_light(p01, "Order-Averaged Brillouin Shift vs Detector Height y", "Pixel y", "Shift (GHz)")
    n_orders = shift_clipped_averaged.shape[1]
    if n_orders <= 10:
        p01.addLegend(offset=(10, 10))
    for k in range(n_orders):
        valid = np.isfinite(shift_clipped_averaged[:, k])
        if np.any(valid):
            p01.plot(
                y_coords_averaged[valid, k],
                shift_clipped_averaged[valid, k],
                pen=pg.mkPen(pg.intColor(k, max(n_orders, 1)), width=1.3),
                name=f"Order {k + 1}" if n_orders <= 10 else None,
            )

    add_light_scatter_with_cbar(
        win, 1, 0,
        x_coords.ravel(), y_coords.ravel(), linewidth_clipped.ravel(),
        "viridis", f"Brillouin Linewidth Γ_B on Detector ({sample_name})",
        "Pixel x", "Pixel y", "Linewidth (GHz)",
        size=4.0, alpha=170, invert_y=True,
    )

    p11 = win.addPlot(row=1, col=1)
    style_plot_light(p11, "Order-Averaged Brillouin Linewidth vs Detector Height y", "Pixel y", "Linewidth (GHz)")
    if n_orders <= 10:
        p11.addLegend(offset=(10, 10))
    for k in range(n_orders):
        valid = np.isfinite(linewidth_clipped_averaged[:, k])
        if np.any(valid):
            p11.plot(
                y_coords_averaged[valid, k],
                linewidth_clipped_averaged[valid, k],
                pen=pg.mkPen(pg.intColor(k, max(n_orders, 1)), width=1.3),
                name=f"Order {k + 1}" if n_orders <= 10 else None,
            )

    _finalize_pg_figure(
        win, out_dir, f"{prefix}13 - {sample_name} DHO shift and linewidth overview.png", show_plots
    )


def plot_dho_fringe_profiles(
    shift_clipped,
    linewidth_clipped,
    shift_clipped_averaged,
    linewidth_clipped_averaged,
    y_coords_averaged,
    path_sample,
    out_dir,
    prefix="",
    show_plots=False,
):
    """
    Per-fringe comparison plot: Anti-Stokes vs Stokes vs Order-Averaged.
    """
    sample_name = os.path.basename(path_sample)
    n_orders = shift_clipped_averaged.shape[1]
    n_cols = min(5, n_orders)
    n_rows = int(np.ceil(n_orders / n_cols))

    win = _create_export_widget(380 * n_cols, 340 * n_rows, "Per-Fringe Breakdown")
    for k in range(n_orders):
        r, c = divmod(k, n_cols)
        p = win.addPlot(row=r, col=c)
        style_plot_light(p, f"Order {k + 1}", "Pixel y", "Shift (GHz)")
        if k == 0:
            p.addLegend(offset=(8, 8))
        y_k = y_coords_averaged[:, k]
        s_as = shift_clipped[:, 2 * k]
        s_st = shift_clipped[:, 2 * k + 1]
        s_avg = shift_clipped_averaged[:, k]

        m_as = np.isfinite(s_as) & np.isfinite(y_k)
        m_st = np.isfinite(s_st) & np.isfinite(y_k)
        m_av = np.isfinite(s_avg) & np.isfinite(y_k)
        if np.any(m_as):
            p.plot(
                y_k[m_as], s_as[m_as], pen=None, symbol="o", symbolSize=3,
                symbolBrush=pg.mkBrush(44, 160, 44, 140), symbolPen=None,
                name="Anti-Stokes" if k == 0 else None,
            )
        if np.any(m_st):
            p.plot(
                y_k[m_st], s_st[m_st], pen=None, symbol="o", symbolSize=3,
                symbolBrush=pg.mkBrush(214, 39, 40, 140), symbolPen=None,
                name="Stokes" if k == 0 else None,
            )
        if np.any(m_av):
            p.plot(
                y_k[m_av], s_avg[m_av],
                pen=pg.mkPen("#1f77b4", width=1.4),
                name="Order Avg" if k == 0 else None,
            )

    _finalize_pg_figure(win, out_dir, f"{prefix}13B - {sample_name} per-fringe breakdown.png", show_plots)


def plot_q_vector_dispersion(
    q_vector,
    shift,
    linewidth,
    theta_deg,
    shift_err=None,
    linewidth_err=None,
    Shift_0=None,
    M_theta=None,
    n_medium=None,
    lambda_vacuum=None,
    V_s=None,
    V_s_err=None,
    out_dir=None,
    prefix="",
    show_plots=False,
):
    """
    Displays Shift(q_vector) and Linewidth(q_vector) with dual axes (q in rad/µm and theta in degrees).
    """
    q_um = q_vector * 1e-6
    win = _create_export_widget(1500, 620, "Angle-Resolved BLS Acoustic Dispersion")

    row_offset = 0
    if lambda_vacuum is not None and n_medium is not None:
        win.addLabel(
            f"Angle-Resolved BLS Acoustic Dispersion — λ={lambda_vacuum*1e9:.2f} nm, n={n_medium:.2f}",
            row=0, col=0, colspan=2, color="#111111", size="12pt", bold=True,
        )
        row_offset = 1

    # Panel 1: Brillouin Shift(q_vector)
    p1, _ = add_light_scatter_with_cbar(
        win, row_offset, 0,
        q_um, shift, theta_deg,
        "viridis", "Brillouin Frequency Shift Shift(q)",
        "Acoustic Wavevector q (rad/µm)", "Brillouin Shift (GHz)", "Scattering Angle θ (°)",
        size=6.0, alpha=145,
    )
    p1.addLegend(offset=(10, 10))
    if lambda_vacuum is not None and n_medium is not None:
        attach_top_theta_axis(p1, theta_deg, n_medium, lambda_vacuum)

    if V_s is not None:
        q_model_um = np.linspace(np.min(q_um), np.max(q_um), 200)
        q_model_si = q_model_um * 1e6
        shift_model_ghz = (V_s * q_model_si / (2.0 * np.pi)) * 1e-9

        if V_s_err is not None and np.isfinite(V_s_err):
            c_lo = pg.PlotDataItem(
                q_model_um, ((V_s - V_s_err) * q_model_si / (2.0 * np.pi)) * 1e-9,
                pen=pg.mkPen(220, 40, 40, 80),
            )
            c_hi = pg.PlotDataItem(
                q_model_um, ((V_s + V_s_err) * q_model_si / (2.0 * np.pi)) * 1e-9,
                pen=pg.mkPen(220, 40, 40, 80),
            )
            p1.addItem(c_lo)
            p1.addItem(c_hi)
            p1.addItem(pg.FillBetweenItem(c_lo, c_hi, brush=pg.mkBrush(220, 40, 40, 45)))
            line_lbl = f"Linear Dispersion Ω = V_s q (V_s = {V_s:.1f} ± {V_s_err:.1f} m/s)"
        else:
            line_lbl = f"Linear Dispersion Ω = V_s q (V_s = {V_s:.1f} m/s)"

        p1.plot(q_model_um, shift_model_ghz, pen=pg.mkPen("#d62728", width=2.2), name=line_lbl)

    # Panel 2: Brillouin Linewidth(q_vector) + Weighted Binned Mean ± 1 SE
    p2, _ = add_light_scatter_with_cbar(
        win, row_offset, 1,
        q_um, linewidth, theta_deg,
        "magma", "Brillouin Linewidth Linewidth(q)",
        "Acoustic Wavevector q (rad/µm)", "Brillouin Linewidth FWHM (GHz)", "Scattering Angle θ (°)",
        size=6.0, alpha=145,
    )
    p2.addLegend(offset=(10, 10))
    if lambda_vacuum is not None and n_medium is not None:
        attach_top_theta_axis(p2, theta_deg, n_medium, lambda_vacuum)

    nb_bins = 25
    bin_edges = np.linspace(np.min(q_um), np.max(q_um), nb_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_lw_mean = np.full(nb_bins, np.nan)
    bin_lw_err = np.full(nb_bins, np.nan)
    w_lw = 1.0 / linewidth_err if linewidth_err is not None else np.ones_like(linewidth)
    for b in range(nb_bins):
        in_b = (q_um >= bin_edges[b]) & (q_um < bin_edges[b + 1])
        if np.sum(in_b) > 3:
            wb = w_lw[in_b]
            lwb = linewidth[in_b]
            mean_wb = np.sum(wb * lwb) / np.sum(wb)
            bin_lw_mean[b] = mean_wb
            v1 = np.sum(wb)
            v2 = np.sum(wb**2)
            if v1**2 > v2:
                var_wb = (v1 / (v1**2 - v2)) * np.sum(wb * (lwb - mean_wb) ** 2)
                bin_lw_err[b] = np.sqrt(max(0.0, var_wb) / np.sum(in_b))
            else:
                bin_lw_err[b] = np.std(lwb) / np.sqrt(np.sum(in_b))

    valid_bins = np.isfinite(bin_lw_mean)
    if np.any(valid_bins):
        bx = bin_centers[valid_bins]
        by = bin_lw_mean[valid_bins]
        be = bin_lw_err[valid_bins]
        p2.addItem(pg.ErrorBarItem(x=bx, y=by, top=be, bottom=be, beam=0.04, pen=pg.mkPen("#dc143c", width=1.4)))
        p2.plot(
            bx, by, pen=pg.mkPen("#dc143c", width=1.8),
            symbol="o", symbolSize=5, symbolBrush="#dc143c",
            name="Error-Weighted Mean ± 1 SE",
        )

    _finalize_pg_figure(win, out_dir, f"14 - {prefix} - Shift and linewidth vs q_vector.png", show_plots)


def plot_complete_diagnostic_suite(
    x,
    y,
    shift,
    linewidth,
    shift_err,
    linewidth_err,
    x_sph,
    y_sph,
    r_sph,
    q_vector,
    theta_deg,
    opt_params,
    S0_fit,
    M_theta_fit,
    S0_err=None,
    M_theta_err=None,
    n_medium=None,
    lambda_vacuum=None,
    V_s=None,
    V_s_err=None,
    out_dir=None,
    prefix="",
    show_plots=False,
):
    """
    Generates a 6-panel comprehensive diagnostic figure using PyQtGraph:
        1. 2D detector coordinates (x, y) with fitted concentric ellipses & axes.
        2. Spherically symmetric coordinates (x_sph, y_sph) confirming circularity.
        3. Shift(r_sph) vs theoretical M_theta kinematic fit.
        4. Linewidth(r_sph) vs r_sph.
        5. Shift(q_vector) dispersion.
        6. Linewidth(q_vector) broadening.
    """
    win = _create_export_widget(1800, 1060, "2D Elliptical Geometry & Acoustic Wavevector Diagnostics")
    win.addLabel(
        "Complete 2D Elliptical Geometry & Acoustic Wavevector Diagnostics",
        row=0, col=0, colspan=3, color="#111111", size="13pt", bold=True,
    )

    xc = opt_params["xc"]
    yc = opt_params["yc"]
    phi = opt_params["phi_rad"]
    gamma = opt_params["gamma"]
    th_dense = np.linspace(0, 2 * np.pi, 200)

    # Panel 1: Detector Space with Fitted Concentric Ellipses
    p1, _ = add_light_scatter_with_cbar(
        win, 1, 0, x, y, shift, "coolwarm",
        "1. Detector Coordinates & Ellipses", "Detector x (px)", "Detector y (px)", "Shift (GHz)",
        size=6.0, alpha=145, invert_y=True, aspect_equal=True,
    )
    p1.addLegend(offset=(10, 10))
    dash_pen = pg.mkPen("#333333", width=1.2, style=QtCore.Qt.PenStyle.DashLine)
    for r_level in (400, 800, 1200, 1600):
        x_ell = r_level * np.cos(th_dense)
        y_ell = (gamma * r_level) * np.sin(th_dense)
        x_rot = xc + x_ell * np.cos(phi) - y_ell * np.sin(phi)
        y_rot = yc + x_ell * np.sin(phi) + y_ell * np.cos(phi)
        p1.plot(x_rot, y_rot, pen=dash_pen)
    l_ax = 1000.0
    p1.plot(
        [xc - l_ax * np.cos(phi), xc + l_ax * np.cos(phi)],
        [yc - l_ax * np.sin(phi), yc + l_ax * np.sin(phi)],
        pen=pg.mkPen("#1565c0", width=1.8), name=f"Principal Axis 1 (θ={np.degrees(phi):.1f}°)",
    )
    p1.plot(
        [xc + l_ax * np.sin(phi), xc - l_ax * np.sin(phi)],
        [yc - l_ax * np.cos(phi), yc + l_ax * np.cos(phi)],
        pen=pg.mkPen("#1565c0", width=1.4, style=QtCore.Qt.PenStyle.DotLine), name="Principal Axis 2",
    )
    p1.plot(
        [xc], [yc], pen=None, symbol="+", symbolSize=13,
        symbolPen=pg.mkPen("#d32f2f", width=2.2), name=f"Center ({xc:.0f}, {yc:.0f})",
    )

    # Panel 2: Spherical Symmetry
    p2, _ = add_light_scatter_with_cbar(
        win, 1, 1, x_sph, y_sph, shift, "coolwarm",
        f"2. Transformed Coordinates (γ={gamma:.3f})", "x_sph (px)", "y_sph (px)", "Shift (GHz)",
        size=6.0, alpha=145, aspect_equal=True,
    )
    for r_level in (400, 800, 1200, 1600):
        p2.plot(r_level * np.cos(th_dense), r_level * np.sin(th_dense), pen=dash_pen)
    p2.plot([0.0], [0.0], pen=None, symbol="+", symbolSize=13, symbolPen=pg.mkPen("#d32f2f", width=2.2))

    # Panel 3: Shift(r_sph) & Kinematic Fit
    p3 = win.addPlot(row=1, col=2)
    style_plot_light(p3, "3. Shift(r_sph) & Kinematic Fit", "Spherical Radius r_sph (px)", "Shift (GHz)")
    p3.addLegend(offset=(10, 10))
    p3.plot(
        r_sph, shift, pen=None, symbol="o", symbolSize=5,
        symbolBrush=pg.mkBrush(65, 105, 225, 120), symbolPen=None, name="Experimental Shift",
    )
    r_arr = np.linspace(0, float(np.max(r_sph)), 200)
    u_arr = np.clip(r_arr / max(M_theta_fit, 1e-6), -0.999999, 0.999999)
    s_model = S0_fit * np.cos(np.arcsin(u_arr) / 2.0)
    p3.plot(
        r_arr, s_model, pen=pg.mkPen("#d62728", width=2.2),
        name=f"Kinematic Fit (S0={S0_fit:.3f} GHz, M_θ={M_theta_fit:.0f} px)",
    )
    s0_t = f"S0 = {S0_fit:.4f} ± {S0_err:.4f} GHz" if (S0_err is not None and np.isfinite(S0_err)) else f"S0 = {S0_fit:.4f} GHz"
    m_t = f"M_θ = {M_theta_fit:.1f} ± {M_theta_err:.1f} px" if (M_theta_err is not None and np.isfinite(M_theta_err)) else f"M_θ = {M_theta_fit:.1f} px"
    box3 = pg.TextItem(
        text=f"Kinematic Model:\nShift(r) = S0·cos(0.5·arcsin(r/M_θ))\n{s0_t}\n{m_t}",
        color="#111111", border=pg.mkPen("#cccccc"), fill=pg.mkBrush(255, 255, 255, 235), anchor=(1, 0),
    )
    box3.setPos(float(np.max(r_arr)), float(np.nanmax(shift)))
    p3.addItem(box3)

    # Panel 4: Linewidth(r_sph)
    p4 = win.addPlot(row=2, col=0)
    style_plot_light(p4, "4. Linewidth(r_sph)", "Spherical Radius r_sph (px)", "Linewidth (GHz)")
    p4.plot(
        r_sph, linewidth, pen=None, symbol="o", symbolSize=5,
        symbolBrush=pg.mkBrush(46, 139, 87, 120), symbolPen=None,
    )

    # Panel 5: Shift(q) Acoustic Dispersion
    q_um = q_vector * 1e-6
    p5, _ = add_light_scatter_with_cbar(
        win, 2, 1, q_um, shift, theta_deg, "viridis",
        "5. Shift(q) Acoustic Dispersion", "Acoustic Wavevector q (rad/µm)", "Shift (GHz)", "Angle θ (°)",
        size=5.0, alpha=145,
    )
    if V_s is not None:
        p5.addLegend(offset=(10, 10))
        q_lin_um = np.linspace(np.min(q_um), np.max(q_um), 200)
        q_lin_si = q_lin_um * 1e6
        shift_model = (V_s * q_lin_si / (2.0 * np.pi)) * 1e-9
        if V_s_err is not None and np.isfinite(V_s_err):
            c_lo = pg.PlotDataItem(q_lin_um, ((V_s - V_s_err) * q_lin_si / (2.0 * np.pi)) * 1e-9, pen=pg.mkPen(220, 40, 40, 80))
            c_hi = pg.PlotDataItem(q_lin_um, ((V_s + V_s_err) * q_lin_si / (2.0 * np.pi)) * 1e-9, pen=pg.mkPen(220, 40, 40, 80))
            p5.addItem(c_lo)
            p5.addItem(c_hi)
            p5.addItem(pg.FillBetweenItem(c_lo, c_hi, brush=pg.mkBrush(220, 40, 40, 45)))
            vs_txt = f"V_s = {V_s:.1f} ± {V_s_err:.1f} m/s"
        else:
            vs_txt = f"V_s = {V_s:.1f} m/s"
        p5.plot(q_lin_um, shift_model, pen=pg.mkPen("#d62728", width=2.2), name=f"Linear Fit: {vs_txt}")
        box5 = pg.TextItem(
            text=f"Speed of Sound:\nShift(q) = (V_s / 2π)·q\n{vs_txt}",
            color="#111111", border=pg.mkPen("#cccccc"), fill=pg.mkBrush(255, 255, 255, 235), anchor=(0, 0),
        )
        box5.setPos(float(np.min(q_um)), float(np.nanmax(shift)))
        p5.addItem(box5)

    # Panel 6: Linewidth(q)
    add_light_scatter_with_cbar(
        win, 2, 2, q_um, linewidth, theta_deg, "magma",
        "6. Linewidth(q) Broadening", "q (rad/µm)", "Linewidth (GHz)", "Angle θ (°)",
        size=5.0, alpha=145,
    )

    _finalize_pg_figure(
        win, out_dir, f"14B - {prefix} - Complete elliptical qvector diagnostics.png", show_plots
    )
