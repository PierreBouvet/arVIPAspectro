"""
2D Camera Frame Viewer Widget (`ImageViewerWidget`) for displaying acquired
Hamamatsu Orca Flash 4.0 v3 interferograms in real time.
"""

from typing import Optional, Dict
import numpy as np
from PySide6 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class ImageViewerWidget(QtWidgets.QWidget):
    """
    Interactive 2D image canvas with colormap/contrast controls, frame statistics,
    and live cursor pixel/intensity readout.
    """

    COLORMAPS = ["inferno", "viridis", "magma", "plasma", "gray", "turbo"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: Optional[np.ndarray] = None
        self._current_title: str = "No image acquired yet"
        self._im_artist = None
        self._cbar = None

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
    ):
        """
        Displays a newly acquired 2D image and updates statistics and stage coordinates in the title.
        """
        self._current_image = np.asarray(image)
        h, w = self._current_image.shape[:2]

        # Build title string
        parts = [f"Frame {w}×{h} px"]
        if stage_pos_um is not None and stage_pos_um.get("X") is not None:
            pos_str = f"X={stage_pos_um['X']:.1f} µm, Y={stage_pos_um['Y']:.1f} µm"
            if stage_pos_um.get("Z") is not None:
                pos_str += f", Z={stage_pos_um['Z']:.1f} µm"
            parts.append(f"Stage: ({pos_str})")
        if exposure_s is not None:
            parts.append(f"Exp: {exposure_s * 1e3:.1f} ms")
        self._current_title = "  |  ".join(parts)

        # Compute statistics
        vmin_raw = int(np.min(self._current_image))
        vmax_raw = int(np.max(self._current_image))
        vmean = float(np.mean(self._current_image))
        n_sat = int(np.sum(self._current_image >= 65000))
        self.lbl_stats.setText(
            f"Min: {vmin_raw}  |  Max: {vmax_raw}  |  Mean: {vmean:.1f}  |  Sat(≥65k): {n_sat} px"
        )

        self._render_image()

    def _compute_clim(self, img: np.ndarray):
        mode = self.combo_contrast.currentData()
        if mode == "uint16":
            return 0.0, 65535.0
        if mode == "minmax":
            vmin, vmax = float(np.min(img)), float(np.max(img))
            return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)
        # Default: fast strided percentile for large 2048x2048 frames
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

        if self._im_artist is not None and self._im_artist.get_array().shape == (h, w):
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
        self._render_image()

    def _on_mouse_move(self, event):
        if self._current_image is None or event.inaxes != self.ax:
            self.lbl_cursor.setText("Cursor: —")
            return
        if event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        h, w = self._current_image.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            val = int(self._current_image[row, col])
            self.lbl_cursor.setText(f"Cursor: X = {col:4d} px, Y = {row:4d} px   |   Intensity = {val:5d} ADU")
        else:
            self.lbl_cursor.setText("Cursor: —")
