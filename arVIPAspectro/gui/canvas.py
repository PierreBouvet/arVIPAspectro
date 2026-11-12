"""
canvas.py: Interactive Matplotlib Canvas and Visualizer Widget for arVIPAspectro GUI.
"""

import inspect
from typing import Callable, Dict, Optional, Any
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6 import QtWidgets, QtCore, QtGui

from .theme import get_mpl_theme, get_system_theme, MplTheme, apply_axis_theme


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib FigureCanvas integrated into Qt with high-DPI awareness."""

    def __init__(self, parent=None, width=8, height=6, dpi=100, bg_color="#1e1e1e"):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.fig.patch.set_facecolor(bg_color)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.updateGeometry()

    def set_bg_color(self, bg_color: str):
        self.fig.patch.set_facecolor(bg_color)
        self.draw_idle()


class PlotVisualizer(QtWidgets.QWidget):
    """
    Two-frame left visualizer widget containing:
      - Top control bar: View selector dropdown, colormap dropdown, cursor readout.
      - Matplotlib NavigationToolbar.
      - MplCanvas.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._views: Dict[str, Any] = {}
        self._current_view_key: Optional[str] = None
        self._current_2d_image: Optional[np.ndarray] = None
        self._current_im_artist = None
        self.theme_name: str = get_system_theme()

        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 1. Top Control Bar
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(8)

        # View Selector
        self.lbl_view = QtWidgets.QLabel("View:")
        self.lbl_view.setStyleSheet("font-weight: bold;")
        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.setMinimumWidth(220)
        self.combo_view.currentIndexChanged.connect(self._on_view_changed)

        # Colormap Selector
        self.lbl_cmap = QtWidgets.QLabel("Colormap:")
        self.combo_cmap = QtWidgets.QComboBox()
        for cmap in ["inferno", "viridis", "plasma", "magma", "cividis", "turbo", "coolwarm", "gray"]:
            self.combo_cmap.addItem(cmap)
        self.combo_cmap.setCurrentText("inferno")
        self.combo_cmap.currentIndexChanged.connect(self._on_cmap_changed)

        # Value readout label
        self.lbl_cursor = QtWidgets.QLabel("x: -- | y: -- | val: --")
        self.lbl_cursor.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        top_bar.addWidget(self.lbl_view)
        top_bar.addWidget(self.combo_view)
        top_bar.addWidget(self.lbl_cmap)
        top_bar.addWidget(self.combo_cmap)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_cursor)

        layout.addLayout(top_bar)

        # 2. Matplotlib Canvas
        mpl_theme = get_mpl_theme(self.theme_name)
        self.canvas = MplCanvas(self, bg_color=mpl_theme.fig_bg)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # 3. Matplotlib Navigation Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._apply_theme_styling()
        # Initial placeholder display
        self.show_placeholder("No dataset or calibration step selected.\nOpen an HDF5 file or run a step from the right panel.")

    # --------------------------------------------------------------------------
    # Theme Support
    # --------------------------------------------------------------------------

    def set_theme(self, theme_name: str):
        """Switches visualizer between 'light' and 'dark' themes."""
        self.theme_name = theme_name
        mpl_theme = get_mpl_theme(self.theme_name)
        self.canvas.fig.patch.set_facecolor(mpl_theme.fig_bg)
        self._apply_theme_styling()

        if self._current_view_key:
            self._render_view(self._current_view_key)
        else:
            self.show_placeholder("No dataset or calibration step selected.\nOpen an HDF5 file or run a step from the right panel.")

    def _apply_theme_styling(self):
        mpl_theme = get_mpl_theme(self.theme_name)
        if self.theme_name == "dark":
            self.lbl_view.setStyleSheet("font-weight: bold; color: #dcdcdc;")
            self.lbl_cmap.setStyleSheet("color: #aaaaaa;")
            self.lbl_cursor.setStyleSheet(f"color: {mpl_theme.cursor_text}; font-family: monospace; font-size: 11px;")
            self.toolbar.setStyleSheet("background: transparent; color: #dcdcdc; border: none;")
        else:
            self.lbl_view.setStyleSheet("font-weight: bold; color: #1f2937;")
            self.lbl_cmap.setStyleSheet("color: #4b5563;")
            self.lbl_cursor.setStyleSheet(f"color: {mpl_theme.cursor_text}; font-family: monospace; font-size: 11px;")
            self.toolbar.setStyleSheet("background: transparent; color: #1f2937; border: none;")

    # --------------------------------------------------------------------------
    # View Management
    # --------------------------------------------------------------------------

    def set_views(self, views_dict: Dict[str, Any], default_key: Optional[str] = None):
        """
        Populates the view selector dropdown with named view render callables.
        Each callable takes the Matplotlib `Figure` object (and optional `MplTheme`).
        """
        self._views = views_dict
        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        for name in self._views.keys():
            self.combo_view.addItem(name)

        if default_key and default_key in self._views:
            idx = self.combo_view.findText(default_key)
            if idx >= 0:
                self.combo_view.setCurrentIndex(idx)
        elif self.combo_view.count() > 0:
            self.combo_view.setCurrentIndex(0)

        self.combo_view.blockSignals(False)

        current_text = self.combo_view.currentText()
        if current_text:
            self._render_view(current_text)

    def _on_view_changed(self, index: int):
        if index < 0:
            return
        key = self.combo_view.currentText()
        self._render_view(key)

    def _render_view(self, key: str):
        if key not in self._views:
            return
        self._current_view_key = key
        self._current_2d_image = None
        self._current_im_artist = None

        mpl_theme = get_mpl_theme(self.theme_name)
        self.canvas.fig.clf()
        self.canvas.fig.patch.set_facecolor(mpl_theme.fig_bg)

        try:
            render_func = self._views[key]
            # Inspect whether render_func accepts theme
            sig = inspect.signature(render_func)
            if len(sig.parameters) >= 2 or "theme" in sig.parameters:
                render_func(self.canvas.fig, mpl_theme)
            else:
                render_func(self.canvas.fig)
        except Exception as e:
            self.canvas.fig.clf()
            ax = self.canvas.fig.add_subplot(111)
            ax.set_facecolor(mpl_theme.ax_bg)
            ax.text(
                0.5, 0.5, f"Error rendering view '{key}':\n{e}",
                ha="center", va="center", color="#ef4444", fontsize=11, wrap=True
            )
            ax.axis("off")

        self.canvas.draw_idle()

    def _on_cmap_changed(self, index: int):
        cmap_name = self.combo_cmap.currentText()
        if self._current_im_artist is not None:
            self._current_im_artist.set_cmap(cmap_name)
            self.canvas.draw_idle()
        elif self._current_view_key:
            self._render_view(self._current_view_key)

    # --------------------------------------------------------------------------
    # Specialized Quick Renderers
    # --------------------------------------------------------------------------

    def show_placeholder(self, message: str):
        """Displays a clean placeholder message on the canvas."""
        mpl_theme = get_mpl_theme(self.theme_name)
        self._views.clear()
        self.combo_view.blockSignals(True)
        self.combo_view.clear()
        self.combo_view.blockSignals(False)

        self.canvas.fig.clf()
        self.canvas.fig.patch.set_facecolor(mpl_theme.fig_bg)
        ax = self.canvas.fig.add_subplot(111)
        ax.set_facecolor(mpl_theme.ax_bg)
        ax.text(
            0.5, 0.5, message,
            ha="center", va="center", color=mpl_theme.text_tick, fontsize=12, style="italic", wrap=True
        )
        ax.axis("off")
        self.canvas.draw_idle()

    def show_2d_image(
        self,
        data: np.ndarray,
        title: str = "",
        xlabel: str = "X (pixels)",
        ylabel: str = "Y (pixels)",
        extent=None,
        vmin=None,
        vmax=None,
    ):
        """Quickly displays a 2D image matrix with a colorbar and interactive hover."""
        self._current_2d_image = data

        def _render(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(self.theme_name)
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax = fig.add_subplot(111)
            apply_axis_theme(ax, t, title=title, xlabel=xlabel, ylabel=ylabel)

            cmap = self.combo_cmap.currentText()
            p1, p99 = np.nanpercentile(data, (1, 99.5)) if vmin is None else (vmin, vmax)
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                extent=extent,
                vmin=p1,
                vmax=p99,
                interpolation="nearest",
            )
            self._current_im_artist = im

            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.ax.tick_params(colors=t.text_tick)
            cbar.outline.set_edgecolor(t.spine)

        self.set_views({title or "2D Image": _render})

    def show_1d_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_err: Optional[np.ndarray] = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        label: str = "Data",
    ):
        """Quickly displays a 1D curve with optional error bars."""
        def _render(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(self.theme_name)
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax = fig.add_subplot(111)
            apply_axis_theme(ax, t, title=title, xlabel=xlabel, ylabel=ylabel)

            curve_color = "#00bcd4" if self.theme_name == "dark" else "#0288d1"
            if y_err is not None:
                ax.errorbar(
                    x, y, yerr=y_err, fmt="o", color=curve_color, ecolor=curve_color,
                    elinewidth=1.0, capsize=2, markersize=3, label=label
                )
            else:
                ax.plot(x, y, "-", color=curve_color, linewidth=1.5, label=label)

            ax.grid(True, linestyle="--", alpha=t.grid_alpha, color=t.grid)
            if label:
                legend = ax.legend(loc="best", facecolor=t.legend_bg, edgecolor=t.legend_edge)
                for text in legend.get_texts():
                    text.set_color(t.text_title)

        self.set_views({title or "1D Spectrum": _render})

    # --------------------------------------------------------------------------
    # Interactive Callbacks
    # --------------------------------------------------------------------------

    def _on_mouse_move(self, event):
        """Displays coordinate and pixel intensity value when moving mouse over axes."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self.lbl_cursor.setText("x: -- | y: -- | val: --")
            return

        x = event.xdata
        y = event.ydata

        if self._current_2d_image is not None:
            r = int(round(y))
            c = int(round(x))
            rows, cols = self._current_2d_image.shape[:2]
            if 0 <= r < rows and 0 <= c < cols:
                val = self._current_2d_image[r, c]
                self.lbl_cursor.setText(f"x: {c:4d} | y: {r:4d} | val: {val:.2f}")
                return

        self.lbl_cursor.setText(f"x: {x:.2f} | y: {y:.2f}")
