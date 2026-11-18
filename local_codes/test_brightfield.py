"""
Live brightfield viewer for the Teledyne FLIR CM3-U3-31S4M-CS camera.

Automatically detects the connected Spinnaker camera, starts continuous
acquisition, and opens a PySide6 window displaying the live feed.

Usage:
    python local_codes/test_brightfield.py
    python local_codes/test_brightfield.py --simulate
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Ensure project root is importable when executed directly as `python local_codes/test_brightfield.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arVIPAspectro.hardware import (
    TeledyneBrightfieldCamera,
    TeledyneCameraError,
    diagnose_teledyne_backend,
    find_chameleon3_camera,
    list_teledyne_cameras,
)


def _to_uint8_display(frame: np.ndarray, auto_contrast: bool) -> np.ndarray:
    """Converts a 2D mono frame (uint8/uint16) to contiguous uint8 for QImage."""
    if frame.ndim == 3:
        frame = frame[..., 0]
    if not auto_contrast and frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)

    arr = frame.astype(np.float32, copy=False)
    if auto_contrast:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    else:
        lo = 0.0
        hi = 65535.0 if np.issubdtype(frame.dtype, np.uint16) or arr.max() > 255.0 else 255.0

    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(scaled)


class BrightfieldLiveWindow(QMainWindow):
    """Simple PySide6 live display window for TeledyneBrightfieldCamera."""

    def __init__(self, cam: TeledyneBrightfieldCamera, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cam = cam
        self._paused = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_pixmap: Optional[QPixmap] = None
        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0
        self._fps = 0.0

        status = self.cam.get_status()
        sim_tag = " [SIMULATED]" if status["simulate"] else ""
        self.setWindowTitle(
            f"Brightfield Live — {status['model']} (S/N: {status['serial_number']}){sim_tag}"
        )
        self.resize(1024, 800)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Top control bar
        ctrl_layout = QHBoxLayout()
        root_layout.addLayout(ctrl_layout)

        ctrl_layout.addWidget(QLabel("Exposure (ms):"))
        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setRange(0.02, 5000.0)
        self.exp_spin.setDecimals(2)
        self.exp_spin.setSingleStep(5.0)
        cur_exp_ms = (status["exposure_s"] or 0.020) * 1e3
        self.exp_spin.setValue(cur_exp_ms)
        self.exp_spin.valueChanged.connect(self._on_exposure_changed)
        ctrl_layout.addWidget(self.exp_spin)

        ctrl_layout.addWidget(QLabel("Gain (dB):"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 24.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(1.0)
        self.gain_spin.setValue(float(status["gain_db"] or 0.0))
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        ctrl_layout.addWidget(self.gain_spin)

        self.auto_contrast_chk = QCheckBox("Auto Contrast")
        self.auto_contrast_chk.setChecked(True)
        ctrl_layout.addWidget(self.auto_contrast_chk)

        ctrl_layout.addStretch(1)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        ctrl_layout.addWidget(self.pause_btn)

        self.save_btn = QPushButton("Save Frame...")
        self.save_btn.clicked.connect(self._save_frame)
        ctrl_layout.addWidget(self.save_btn)

        # Live image viewport
        self.image_label = QLabel("Waiting for frames...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #111; color: #bbb;")
        root_layout.addWidget(self.image_label, stretch=1)

        # Poll timer (~60 Hz UI refresh; reads newest frame from Spinnaker buffer)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_frame)
        self._timer.start(16)

    def _on_exposure_changed(self, value_ms: float):
        try:
            self.cam.set_exposure(value_ms * 1e-3)
        except Exception as exc:
            self.statusBar().showMessage(f"Exposure error: {exc}", 3000)

    def _on_gain_changed(self, value_db: float):
        try:
            self.cam.set_gain(value_db)
        except Exception as exc:
            self.statusBar().showMessage(f"Gain error: {exc}", 3000)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("Resume" if self._paused else "Pause")

    def _save_frame(self):
        if self._last_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Brightfield Frame",
            "brightfield_frame.png",
            "PNG Image (*.png);;TIFF Image (*.tif *.tiff);;NumPy Array (*.npy)",
        )
        if not path:
            return
        if path.lower().endswith(".npy"):
            np.save(path, self._last_frame)
        else:
            img8 = _to_uint8_display(self._last_frame, self.auto_contrast_chk.isChecked())
            h, w = img8.shape
            qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
            qimg.save(path)
        self.statusBar().showMessage(f"Saved frame to {path}", 4000)

    def _poll_frame(self):
        if self._paused:
            return
        try:
            frame = self.cam.read_live_frame(timeout_s=0.05)
        except Exception as exc:
            self.statusBar().showMessage(f"Acquisition error: {exc}", 3000)
            return

        if frame is None:
            return

        self._last_frame = frame
        self._fps_frames += 1
        now = time.perf_counter()
        dt = now - self._fps_t0
        if dt >= 0.5:
            self._fps = self._fps_frames / dt
            self._fps_frames = 0
            self._fps_t0 = now

        img8 = _to_uint8_display(frame, self.auto_contrast_chk.isChecked())
        h, w = img8.shape
        qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
        self._last_pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap_view()

        self.statusBar().showMessage(
            f"{w}x{h} px ({frame.dtype}) | min={int(frame.min())}, max={int(frame.max())} | {self._fps:.1f} fps"
        )

    def _update_pixmap_view(self):
        if self._last_pixmap is None:
            return
        scaled = self._last_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap_view()

    def closeEvent(self, event):
        self._timer.stop()
        try:
            self.cam.stop_live()
        finally:
            self.cam.disconnect()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Live brightfield view for Teledyne CM3-U3-31S4M-CS.")
    parser.add_argument("--simulate", action="store_true", help="Run with built-in camera simulator.")
    parser.add_argument("--exposure-ms", type=float, default=20.0, help="Initial exposure in ms.")
    parser.add_argument("--gain-db", type=float, default=0.0, help="Initial analog gain in dB.")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    cam = TeledyneBrightfieldCamera(logger=print)
    try:
        if args.simulate:
            cam.connect(simulate=True)
        else:
            diag = diagnose_teledyne_backend()
            print("Teledyne backend diagnostics:", diag)
            detected = list_teledyne_cameras()
            print(f"Detected Teledyne Spinnaker cameras: {detected}")
            idx = find_chameleon3_camera("CM3-U3-31S4M")
            if idx is None:
                raise TeledyneCameraError(
                    "No Teledyne / Point Grey camera found.\n"
                    f"PyCapture2 available: {diag.get('pycapture2_available')} "
                    f"(error: {diag.get('pycapture2_error')})\n"
                    f"Spinnaker C DLL dir: {diag.get('spinnaker_c_dll_dir')} "
                    f"(error: {diag.get('spinnaker_error')})\n\n"
                    "Make sure FlyCapture2 / SpinView is closed and the camera is connected via USB 3.0 "
                    "(or pass --simulate to test offline)."
                )
            cam.connect(camera_idx=idx)

        cam.set_exposure(args.exposure_ms * 1e-3)
        cam.set_gain(args.gain_db)
        cam.start_live()
    except Exception as exc:
        QMessageBox.critical(None, "Brightfield Camera Error", str(exc))
        cam.disconnect()
        sys.exit(1)

    win = BrightfieldLiveWindow(cam)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
