"""
Step 4 GUI Panel & Worker (`arVIPAspectro/gui/pipeline_panel.py`):
Guided acquisition workflow for:
  1. Opening / creating the HDF5_BLS container (`Measures.h5`)
  2. Prompting & acquiring (or loading) `laser_reference`, `Rb_absorption`,
     and optional `reference_sample` into `Brillouin/Calibration/<role>/image`
  3. Configuring the 2D spatial map (`Nx, Ny, ΔX, ΔY` or Corner A / Corner B,
     point exposure, frame averaging, Snake vs. Raster, optional 4D `Raw_data` saving)
  4. Executing the automated 2D stage + camera map scan with Pause / Resume / Abort.
"""

import os
import time
import queue
import threading
from typing import Optional, Dict, Any, Callable, Tuple
import numpy as np
from PySide6 import QtWidgets, QtCore

from ..hardware import MS2000Stage, OrcaFlashCamera
from ..acquisition_pipeline import AcquisitionSession, MapGridConfig
from .. import config


def _format_bytes(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            return f"{val:.2f} {unit}" if unit in ("MB", "GB", "TB") else f"{val:.0f} {unit}"
        val /= 1024.0
    return f"{val:.2f} TB"


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


class CalibrationCaptureWorker(QtCore.QThread):
    """
    Background worker that applies the requested exposure & frame averaging,
    snaps a calibration image (`laser_reference`, `Rb_absorption`, or `reference_sample`),
    and saves it into `Brillouin/Calibration/<role>/image` in the HDF5_BLS session.
    """

    sig_log = QtCore.Signal(str)
    sig_captured = QtCore.Signal(str, object, object, float, int, int)  # (role, frame, pos, exp_s, n_avg, binning)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        role: str,
        exposure_ms: float,
        n_avg: int,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.camera = camera
        self.stage = stage
        self.role = role
        self.exposure_ms = exposure_ms
        self.n_avg = n_avg

    def run(self):
        try:
            exp_s = self.camera.set_exposure(float(self.exposure_ms) / 1000.0)
            n_avg = self.camera.set_frame_averaging(int(self.n_avg))
            roi = self.camera.get_roi()
            binning = int(roi[4])

            pos = None
            if self.stage.is_connected:
                try:
                    pos = self.stage.get_position()
                except Exception:
                    pos = None

            sim_type_map = {
                AcquisitionSession.ROLE_LASER: "laser",
                AcquisitionSession.ROLE_RB: "rb",
                AcquisitionSession.ROLE_REF_SAMPLE: "sample",
            }
            sim_type = sim_type_map.get(self.role, "sample")

            self.sig_log.emit(
                f"Acquiring '{self.role}' (Exp={exp_s*1e3:.2f} ms, Avg={n_avg})..."
            )
            frame = self.camera.snap_frame(
                n_frames=n_avg,
                stage_pos_um=pos,
                sim_spectrum_type=sim_type,
            )

            self.session.save_calibration_image(
                role=self.role,
                image=frame,
                exposure_s=exp_s,
                n_avg=n_avg,
                roi=roi,
                stage_pos_um=pos,
            )
            self.sig_log.emit(
                f"Saved '{self.role}' to HDF5: Brillouin/Calibration/{self.role}/image "
                f"({frame.shape[1]}x{frame.shape[0]} px)."
            )
            self.sig_captured.emit(self.role, frame, pos, exp_s, n_avg, binning)
        except Exception as exc:
            self.sig_error.emit(f"Calibration Capture Error ({self.role})", str(exc))


class LaserCalibrationWorker(QtCore.QThread):
    """
    Background worker for Step 5.1: Executes `session.run_laser_calibration(...)`
    (Steps 2–6 of `arVIPAspectro.laser_calib`: denoising, fringe tracing, sub-pixel
    refinement, 2D gnomonic VIPA geometry fit, and 2D anamorphic order surface fit).
    """

    sig_log = QtCore.Signal(str)
    sig_calibrated = QtCore.Signal(object)  # calib_res dict
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.params = params

    def run(self):
        try:
            t0 = time.time()
            self.sig_log.emit("Step 5.1 — Running Laser Reference Calibration (Steps 2–6)...")
            res = self.session.run_laser_calibration(
                scale_laser=int(self.params.get("scale_laser", config.SCALE_LASER)),
                ksize_se=int(self.params.get("ksize_se", config.KSIZE_STRUCTURING_ELEMENTS_LASER)),
                ksize_blur=int(self.params.get("ksize_blur", config.KSIZE_BLUR_LASER)),
                y_center_laser=int(self.params.get("y_center_laser", config.Y_CENTER_LASER)),
                dy_laser=int(self.params.get("dy_laser", config.DY_LASER)),
                nb_laser_modes=int(self.params.get("nb_laser_modes", config.NB_LASER_MODES)),
                distance_peaks=int(self.params.get("distance_peaks", config.DISTANCE_PEAKS_LASER)),
                poly_deg=int(self.params.get("poly_deg", 4)),
            )
            dt = time.time() - t0
            self.sig_log.emit(
                f"Step 5.1 Completed in {dt:.2f} s: {len(res['refined_fringes'])} modes | "
                f"Optical axis (x0={res['x0_fit']:.2f}, y0={res['y0_fit']:.2f}) px | "
                f"Gnomonic Xc={res['xc_gnom']:.2f} px | "
                f"Anamorphic RMS: {res['rms_raw_mhz']:.2f} MHz raw, {res['rms_smooth_mhz']:.2f} MHz smooth."
            )
            self.sig_calibrated.emit(res)
        except Exception as exc:
            self.sig_error.emit("Step 5.1 Laser Calibration Error", str(exc))


class RbCalibrationWorker(QtCore.QThread):
    """
    Background worker for Step 5.2: Executes `session.run_rb_calibration(...)`
    (Steps 7–10B of `arVIPAspectro.Rb_calib`: theoretical Rb D2 table, denoising,
    radial profile extraction, VIPA FSR & hyperfine offset calibration, and 2D Frequency Map).
    """

    sig_log = QtCore.Signal(str)
    sig_calibrated = QtCore.Signal(object)  # rb_res dict
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.params = params

    def run(self):
        try:
            t0 = time.time()
            self.sig_log.emit("Step 5.2 — Running Rubidium & 2D Frequency Map Calibration (Steps 7–10B)...")
            res = self.session.run_rb_calibration(
                t_cell_c=float(self.params.get("t_cell_c", 70.0)),
                l_cell_m=float(self.params.get("l_cell_m", 0.07)),
                scale_rb=int(self.params.get("scale_rb", 9)),
                delta_nu_laser_offset_ghz=float(self.params.get("delta_nu_laser_offset_ghz", 0.0)),
                fit_window_ghz=float(self.params.get("fit_window_ghz", 10.0)),
                rb_lock_offset_ghz=float(self.params.get("rb_lock_offset_ghz", 1.323)),
            )
            dt = time.time() - t0
            self.sig_log.emit(
                f"Step 5.2 Completed in {dt:.2f} s: Calibrated VIPA FSR = {res['FSR_cal']:.4f} GHz | "
                f"Hyperfine offset Δν = {res['delta_nu_opt']:+.3f} GHz | "
                f"Elastic zero ξ = {res['xi_elastic']:.4f} | "
                f"Saved 2D Frequency Map ({res['freq_map'].shape[1]}x{res['freq_map'].shape[0]} px)."
            )
            self.sig_calibrated.emit(res)
        except Exception as exc:
            self.sig_error.emit("Step 5.2 Rubidium Calibration Error", str(exc))


class SampleTreatmentWorker(QtCore.QThread):
    """
    Background worker for Step 5.3: Executes `session.run_sample_treatment_and_q_calibration(...)`
    (Step 11 in-situ 2D frequency map calibration, Step 12 spatial binning & vectorized DHO fitting,
    and Step 14 2D concentric ellipse & q-vector / speed-of-sound V_s extraction).
    """

    sig_log = QtCore.Signal(str)
    sig_fitted = QtCore.Signal(object)  # sample_res dict
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.params = params

    def run(self):
        try:
            t0 = time.time()
            self.sig_log.emit(
                "Step 5.3 — Running Sample In-Situ Calibration, Vectorized DHO Fit & q-Vector (Steps 11–14)..."
            )
            res = self.session.run_sample_treatment_and_q_calibration(
                bin_val_h=int(self.params.get("bin_val_h", 4)),
                bin_val_w=int(self.params.get("bin_val_w", 2)),
                expected_shift_ghz=float(self.params.get("expected_shift_ghz", 5.1)),
                shift_tol_ghz=float(self.params.get("shift_tol_ghz", 2.0)),
                expected_lw_ghz=float(self.params.get("expected_lw_ghz", 0.8)),
                lw_bounds_ghz=tuple(self.params.get("lw_bounds_ghz", (0.4, 1.3))),
                run_q_calibration=bool(self.params.get("run_q_calibration", True)),
                refractive_index=float(self.params.get("refractive_index", 1.33)),
                shift_filter_bounds=tuple(self.params.get("shift_filter_bounds", (4.8, 5.3))),
                lw_filter_bounds=tuple(self.params.get("lw_bounds_ghz", (0.4, 1.3))),
            )
            dt = time.time() - t0
            dho = res["dho_results"]
            q_res = res.get("q_results")
            vs_msg = (
                f" | S(0)={q_res['Shift_0']:.4f} GHz, M_θ={q_res['M_theta']:.1f} px, V_s={q_res['V_s']:.2f} m/s"
                if q_res is not None
                else ""
            )
            self.sig_log.emit(
                f"Step 5.3 Completed in {dt:.2f} s: {dho['n_fringes']} orders fitted ({dho['bin_val_h']}x{dho['bin_val_w']} bin) | "
                f"S/AS RMS={dho['stokes_antistokes_rms_mhz']:.2f} MHz{vs_msg}."
            )
            self.sig_fitted.emit(res)
        except Exception as exc:
            self.sig_error.emit("Step 5.3 Sample Fitting Error", str(exc))


class MapScanWorker(QtCore.QThread):
    """
    Asynchronous Producer-Consumer worker that executes the 2D spatial map scan:
      - Producer (Stage + Camera thread): Moves the ASI MS-2000 to `(target_x_um, target_y_um)`,
        waits `settling_ms`, acquires an Orca Flash 4.0 v3 frame, and enqueues it.
      - Consumer (Fitting + HDF5 thread): Dequeues each frame `(iy, ix)`, performs on-the-fly
        DHO fitting (`treat_single_map_point`), persists treated parameters (and optional raw frame)
        to `Brillouin/<Map_name>` in the HDF5 file, and emits live 2D heatmap updates.
    """

    sig_log = QtCore.Signal(str)
    sig_progress = QtCore.Signal(int, int, int, int, float, float)  # (step_idx, total, iy, ix, x_um, y_um)
    sig_stage_updated = QtCore.Signal()
    sig_point_acquired = QtCore.Signal(object, object, float, int, int)  # (frame, pos, exp_s, n_avg, binning)
    sig_live_map_updated = QtCore.Signal(
        object, object, object, object, object, object, object, int, int, object, object
    )  # (frame, x_um_1d, y_um_1d, map_shift, map_lw, map_s0, map_vs, iy, ix, pos, point_fit)
    sig_finished_map = QtCore.Signal(str, int)                           # (map_group_path, points_completed)
    sig_error = QtCore.Signal(str, str)

    def __init__(
        self,
        session: AcquisitionSession,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        map_name: str,
        grid: MapGridConfig,
        exposure_ms: float,
        n_avg: int,
        settling_ms: float,
        save_raw_data: bool,
        on_the_fly_treatment: bool = True,
        fit_params: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.camera = camera
        self.stage = stage
        self.map_name = map_name
        self.grid = grid
        self.exposure_ms = exposure_ms
        self.n_avg = n_avg
        self.settling_ms = settling_ms
        self.save_raw_data = save_raw_data
        self.on_the_fly_treatment = bool(on_the_fly_treatment)
        self.fit_params = fit_params or {}

        self._pause_requested = False
        self._abort_requested = False

    def request_pause(self, paused: bool):
        self._pause_requested = bool(paused)

    def request_abort(self):
        self._abort_requested = True
        self._pause_requested = False

    def run(self):
        try:
            exp_s = self.camera.set_exposure(float(self.exposure_ms) / 1000.0)
            n_avg = self.camera.set_frame_averaging(int(self.n_avg))
            roi = self.camera.get_roi()
            binning = int(roi[4])
            out_w = (roi[1] - roi[0]) // max(1, binning)
            out_h = (roi[3] - roi[2]) // max(1, binning)
            frame_shape = (out_h, out_w)

            bin_h = int(self.fit_params.get("bin_val_h", 4))
            bin_w = int(self.fit_params.get("bin_val_w", 2))
            n_idx = float(self.fit_params.get("refractive_index", 1.33))

            # Initialize HDF5 Measure group and pre-allocate datasets (including 4D Treated group)
            map_group_path, treated_group_path, n_orders = self.session.initialize_map_group(
                map_name=self.map_name,
                grid=self.grid,
                frame_shape=frame_shape,
                exposure_s=exp_s,
                n_avg=n_avg,
                roi=roi,
                save_raw_data=self.save_raw_data,
                on_the_fly_treatment=self.on_the_fly_treatment,
                bin_val_h=bin_h,
                bin_val_w=bin_w,
                refractive_index=n_idx,
                overwrite=True,
            )

            trajectory = self.grid.build_trajectory()
            total_pts = len(trajectory)
            settling_s = max(0.0, float(self.settling_ms) / 1000.0)

            ny, nx = max(1, int(self.grid.ny)), max(1, int(self.grid.nx))
            x_coords_1d = self.grid.x_coords_um
            y_coords_1d = self.grid.y_coords_um
            map_mean_shift = np.full((ny, nx), np.nan, dtype=np.float64)
            map_mean_lw = np.full((ny, nx), np.nan, dtype=np.float64)
            map_shift_0 = np.full((ny, nx), np.nan, dtype=np.float64)
            map_vs = np.full((ny, nx), np.nan, dtype=np.float64)

            self.sig_log.emit(
                f"Starting Pipelined Map '{map_group_path}': {nx}x{ny} = {total_pts} point(s), "
                f"pattern={self.grid.scan_pattern}, Exp={exp_s*1e3:.2f} ms x {n_avg}, "
                f"On-the-Fly Fit={treated_group_path is not None} ({bin_h}x{bin_w} bin, {n_orders} orders), "
                f"Save Raw={self.save_raw_data}."
            )

            work_queue: "queue.Queue[Optional[Tuple]]" = queue.Queue(maxsize=8)
            completed_box = [0]
            consumer_error: list = []

            def _consumer_loop():
                while True:
                    item = work_queue.get()
                    if item is None:
                        work_queue.task_done()
                        break
                    try:
                        c_idx, c_iy, c_ix, c_pos, c_frame, c_tx, c_ty = item
                        point_fit = None
                        if treated_group_path is not None:
                            point_fit = self.session.treat_single_map_point(
                                frame=c_frame,
                                bin_val_h=bin_h,
                                bin_val_w=bin_w,
                                n_orders=n_orders,
                                expected_shift_ghz=float(self.fit_params.get("expected_shift_ghz", 5.1)),
                                shift_tol_ghz=float(self.fit_params.get("shift_tol_ghz", 2.0)),
                                expected_lw_ghz=float(self.fit_params.get("expected_lw_ghz", 0.8)),
                                lw_bounds_ghz=tuple(self.fit_params.get("lw_bounds_ghz", (0.4, 1.3))),
                                refractive_index=n_idx,
                                fast_drift_only=bool(self.fit_params.get("fast_drift_only", True)),
                            )
                            map_mean_shift[c_iy, c_ix] = point_fit["mean_shift_ghz"]
                            map_mean_lw[c_iy, c_ix] = point_fit["mean_lw_ghz"]
                            map_shift_0[c_iy, c_ix] = point_fit["Shift_0_ghz"]
                            map_vs[c_iy, c_ix] = point_fit["V_s_ms"]

                        self.session.write_map_point(
                            map_group_path=map_group_path,
                            iy=c_iy,
                            ix=c_ix,
                            x_meas_um=c_pos.get("X"),
                            y_meas_um=c_pos.get("Y"),
                            frame=c_frame,
                            save_raw_data=self.save_raw_data,
                            treated_group_path=treated_group_path,
                            point_fit=point_fit,
                        )
                        completed_box[0] += 1

                        if treated_group_path is not None:
                            self.sig_live_map_updated.emit(
                                c_frame,
                                x_coords_1d,
                                y_coords_1d,
                                map_mean_shift,
                                map_mean_lw,
                                map_shift_0,
                                map_vs,
                                c_iy,
                                c_ix,
                                c_pos,
                                point_fit,
                            )
                        else:
                            self.sig_point_acquired.emit(c_frame, c_pos, exp_s, n_avg, binning)

                        self.sig_progress.emit(
                            c_idx,
                            total_pts,
                            c_iy,
                            c_ix,
                            float(c_pos.get("X") or c_tx),
                            float(c_pos.get("Y") or c_ty),
                        )
                    except Exception as c_exc:
                        consumer_error.append(c_exc)
                    finally:
                        work_queue.task_done()

            consumer_thread = threading.Thread(target=_consumer_loop, daemon=True)
            consumer_thread.start()

            # Producer Loop: Move Stage + Expose Camera
            for idx, (iy, ix, target_y_um, target_x_um) in enumerate(trajectory, start=1):
                while self._pause_requested and not self._abort_requested:
                    time.sleep(0.05)

                if self._abort_requested or consumer_error:
                    if self._abort_requested:
                        self.sig_log.emit(f"Map scan aborted by user at point {idx}/{total_pts}.")
                    break

                # 1. Move stage to (target_x_um, target_y_um)
                pos = self.stage.move_to(x_um=target_x_um, y_um=target_y_um, wait=True)
                self.sig_stage_updated.emit()

                if settling_s > 0:
                    time.sleep(settling_s)

                if self._abort_requested or consumer_error:
                    if self._abort_requested:
                        self.sig_log.emit(f"Map scan aborted by user at point {idx}/{total_pts}.")
                    break

                # 2. Acquire camera frame
                frame = self.camera.snap_frame(
                    n_frames=n_avg,
                    stage_pos_um=pos,
                    sim_spectrum_type="sample",
                )

                # 3. Enqueue for asynchronous Consumer fitting & HDF5 disk flush
                work_queue.put((idx, iy, ix, pos, frame, target_x_um, target_y_um))

            # Signal Consumer to finish remaining queued frames
            work_queue.put(None)
            consumer_thread.join()

            if consumer_error:
                raise consumer_error[0]

            self.sig_finished_map.emit(map_group_path, completed_box[0])

        except Exception as exc:
            self.sig_error.emit("Map Acquisition Error", str(exc))


class PipelineAcquisitionWidget(QtWidgets.QWidget):
    """
    Step 4 UI Widget:
      - 4A: HDF5_BLS File & Calibration Spectra Acquisition (Laser Reference, Rb Absorption, Ref Sample)
      - 4B: 2D Spatial Map Definition (Grid / Corners, Exposure, Optional Raw Data) & Scan Execution
    """

    sig_log = QtCore.Signal(str)
    sig_display_frame = QtCore.Signal(object, object, float, int, int)  # (frame, pos, exp_s, n_avg, binning)
    sig_show_laser_diagnostics = QtCore.Signal(object, str)             # (calib_res, view_mode)
    sig_show_rb_diagnostics = QtCore.Signal(object, str)                # (rb_res, view_mode)
    sig_show_sample_diagnostics = QtCore.Signal(object, str)            # (sample_res, view_mode)
    sig_update_live_map = QtCore.Signal(
        object, object, object, object, object, object, object, int, int, object, object
    )
    sig_stage_updated = QtCore.Signal()
    sig_camera_updated = QtCore.Signal()
    sig_lock_geometry = QtCore.Signal(bool)

    def __init__(
        self,
        camera: OrcaFlashCamera,
        stage: MS2000Stage,
        stop_live_callback: Callable[[], None],
        parent=None,
    ):
        super().__init__(parent)
        self.camera = camera
        self.stage = stage
        self._stop_live_cb = stop_live_callback
        self.session = AcquisitionSession()

        self._cal_worker: Optional[CalibrationCaptureWorker] = None
        self._laser_calib_worker: Optional[LaserCalibrationWorker] = None
        self._rb_calib_worker: Optional[RbCalibrationWorker] = None
        self._sample_fit_worker: Optional[SampleTreatmentWorker] = None
        self._map_worker: Optional[MapScanWorker] = None
        self._map_paused: bool = False

        # Taught corners (X, Y) in µm
        self._corner_a: Optional[Tuple[float, float]] = None
        self._corner_b: Optional[Tuple[float, float]] = None

        self._build_ui()
        self._update_calibration_badges()
        self._update_map_summary()

    # ==========================================================================
    # UI Construction
    # ==========================================================================

    def _build_ui(self):
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 6, 2)
        layout.setSpacing(8)

        layout.addWidget(self._build_h5_session_card())
        layout.addWidget(self._build_calibration_prompt_card())
        layout.addWidget(self._build_step5_1_laser_calib_card())
        layout.addWidget(self._build_step5_2_rb_calib_card())
        layout.addWidget(self._build_step5_3_sample_fit_card())
        layout.addWidget(self._build_map_configuration_card())
        layout.addStretch()

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

    def _build_h5_session_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("4A. HDF5_BLS Session Container")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        self.txt_h5_path = QtWidgets.QLineEdit()
        default_path = os.path.abspath("Measures.h5")
        self.txt_h5_path.setText(default_path)
        self.txt_h5_path.setPlaceholderText("Path to output/existing .h5 file...")

        self.btn_browse_h5 = QtWidgets.QPushButton("Browse / New...")
        self.btn_browse_h5.clicked.connect(self._on_browse_h5)

        self.btn_open_h5 = QtWidgets.QPushButton("Open / Create HDF5")
        self.btn_open_h5.setStyleSheet("font-weight: bold;")
        self.btn_open_h5.clicked.connect(self._on_open_or_create_h5)

        row.addWidget(self.txt_h5_path, stretch=1)
        row.addWidget(self.btn_browse_h5)
        row.addWidget(self.btn_open_h5)
        layout.addLayout(row)

        self.lbl_h5_status = QtWidgets.QLabel("No HDF5 file opened yet. Click 'Open / Create HDF5' to begin.")
        self.lbl_h5_status.setStyleSheet("font-size: 11px; color: #b0bec5;")
        layout.addWidget(self.lbl_h5_status)
        return box

    def _build_calibration_prompt_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(
            "4B. Step-by-Step Calibration Spectra (Brillouin/Calibration/<role>/image)"
        )
        grid = QtWidgets.QGridLayout(box)
        grid.setSpacing(6)

        grid.addWidget(QtWidgets.QLabel("<b>Spectrum Role</b>"), 0, 0)
        grid.addWidget(QtWidgets.QLabel("<b>Exposure</b>"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("<b>Avg</b>"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("<b>Acquire</b>"), 0, 3)
        grid.addWidget(QtWidgets.QLabel("<b>Preview</b>"), 0, 4)
        grid.addWidget(QtWidgets.QLabel("<b>Status</b>"), 0, 5)

        # Row 1: Laser Reference
        grid.addWidget(QtWidgets.QLabel("1. Laser Reference:"), 1, 0)
        self.spin_laser_exp = QtWidgets.QDoubleSpinBox()
        self.spin_laser_exp.setDecimals(2)
        self.spin_laser_exp.setRange(0.02, 30000.0)
        self.spin_laser_exp.setValue(20.0)
        self.spin_laser_exp.setSuffix(" ms")
        grid.addWidget(self.spin_laser_exp, 1, 1)

        self.spin_laser_avg = QtWidgets.QSpinBox()
        self.spin_laser_avg.setRange(1, 500)
        self.spin_laser_avg.setValue(1)
        grid.addWidget(self.spin_laser_avg, 1, 2)

        self.btn_acq_laser = QtWidgets.QPushButton("📷 Acquire Laser")
        self.btn_acq_laser.clicked.connect(
            lambda: self._on_acquire_calibration(
                AcquisitionSession.ROLE_LASER,
                self.spin_laser_exp.value(),
                self.spin_laser_avg.value(),
            )
        )
        grid.addWidget(self.btn_acq_laser, 1, 3)

        self.btn_view_laser = QtWidgets.QPushButton("👁 View")
        self.btn_view_laser.clicked.connect(lambda: self._on_view_calibration(AcquisitionSession.ROLE_LASER))
        grid.addWidget(self.btn_view_laser, 1, 4)

        self.lbl_status_laser = QtWidgets.QLabel("MISSING")
        self.lbl_status_laser.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status_laser.setMinimumWidth(95)
        grid.addWidget(self.lbl_status_laser, 1, 5)

        # Row 2: Rubidium Absorption
        grid.addWidget(QtWidgets.QLabel("2. Rb Absorption:"), 2, 0)
        self.spin_rb_exp = QtWidgets.QDoubleSpinBox()
        self.spin_rb_exp.setDecimals(2)
        self.spin_rb_exp.setRange(0.02, 30000.0)
        self.spin_rb_exp.setValue(50.0)
        self.spin_rb_exp.setSuffix(" ms")
        grid.addWidget(self.spin_rb_exp, 2, 1)

        self.spin_rb_avg = QtWidgets.QSpinBox()
        self.spin_rb_avg.setRange(1, 500)
        self.spin_rb_avg.setValue(1)
        grid.addWidget(self.spin_rb_avg, 2, 2)

        self.btn_acq_rb = QtWidgets.QPushButton("📷 Acquire Rb")
        self.btn_acq_rb.clicked.connect(
            lambda: self._on_acquire_calibration(
                AcquisitionSession.ROLE_RB,
                self.spin_rb_exp.value(),
                self.spin_rb_avg.value(),
            )
        )
        grid.addWidget(self.btn_acq_rb, 2, 3)

        self.btn_view_rb = QtWidgets.QPushButton("👁 View")
        self.btn_view_rb.clicked.connect(lambda: self._on_view_calibration(AcquisitionSession.ROLE_RB))
        grid.addWidget(self.btn_view_rb, 2, 4)

        self.lbl_status_rb = QtWidgets.QLabel("MISSING")
        self.lbl_status_rb.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status_rb.setMinimumWidth(95)
        grid.addWidget(self.lbl_status_rb, 2, 5)

        # Row 3: Reference Sample (Optional, e.g. Water / Polystyrene)
        grid.addWidget(QtWidgets.QLabel("3. Ref Sample (opt):"), 3, 0)
        self.spin_ref_exp = QtWidgets.QDoubleSpinBox()
        self.spin_ref_exp.setDecimals(2)
        self.spin_ref_exp.setRange(0.02, 30000.0)
        self.spin_ref_exp.setValue(500.0)
        self.spin_ref_exp.setSuffix(" ms")
        grid.addWidget(self.spin_ref_exp, 3, 1)

        self.spin_ref_avg = QtWidgets.QSpinBox()
        self.spin_ref_avg.setRange(1, 500)
        self.spin_ref_avg.setValue(1)
        grid.addWidget(self.spin_ref_avg, 3, 2)

        self.btn_acq_ref = QtWidgets.QPushButton("📷 Acquire Ref")
        self.btn_acq_ref.clicked.connect(
            lambda: self._on_acquire_calibration(
                AcquisitionSession.ROLE_REF_SAMPLE,
                self.spin_ref_exp.value(),
                self.spin_ref_avg.value(),
            )
        )
        grid.addWidget(self.btn_acq_ref, 3, 3)

        self.btn_view_ref = QtWidgets.QPushButton("👁 View")
        self.btn_view_ref.clicked.connect(lambda: self._on_view_calibration(AcquisitionSession.ROLE_REF_SAMPLE))
        grid.addWidget(self.btn_view_ref, 3, 4)

        self.lbl_status_ref = QtWidgets.QLabel("OPTIONAL")
        self.lbl_status_ref.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status_ref.setMinimumWidth(95)
        grid.addWidget(self.lbl_status_ref, 3, 5)

        return box

    def _build_step5_1_laser_calib_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(
            "Step 5.1 — Laser Reference Calibration (Steps 2–6: Fringes, VIPA Geometry & Anamorphic Surface)"
        )
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        # Row 1: Auto-calibrate toggle + Status badge
        top_row = QtWidgets.QHBoxLayout()
        self.chk_auto_calib_laser = QtWidgets.QCheckBox(
            "Auto-run Laser Calibration after acquiring Laser Reference"
        )
        self.chk_auto_calib_laser.setChecked(True)
        top_row.addWidget(self.chk_auto_calib_laser)
        top_row.addStretch()

        self.lbl_laser_calib_status = QtWidgets.QLabel("NOT CALIBRATED")
        self.lbl_laser_calib_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_laser_calib_status.setMinimumWidth(140)
        top_row.addWidget(self.lbl_laser_calib_status)
        layout.addLayout(top_row)

        # Row 2: Algorithm parameters
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("Modes:"))
        self.spin_lc_modes = QtWidgets.QSpinBox()
        self.spin_lc_modes.setRange(2, 30)
        self.spin_lc_modes.setValue(int(config.NB_LASER_MODES))
        param_row.addWidget(self.spin_lc_modes)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("Sym Y₀:"))
        self.spin_lc_ycenter = QtWidgets.QSpinBox()
        self.spin_lc_ycenter.setRange(10, 4096)
        self.spin_lc_ycenter.setValue(int(config.Y_CENTER_LASER))
        self.spin_lc_ycenter.setSuffix(" px")
        param_row.addWidget(self.spin_lc_ycenter)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("Min Peak Dist:"))
        self.spin_lc_dist = QtWidgets.QSpinBox()
        self.spin_lc_dist.setRange(10, 1000)
        self.spin_lc_dist.setValue(int(config.DISTANCE_PEAKS_LASER))
        self.spin_lc_dist.setSuffix(" px")
        param_row.addWidget(self.spin_lc_dist)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("Morph Scale:"))
        self.spin_lc_scale = QtWidgets.QSpinBox()
        self.spin_lc_scale.setRange(2, 32)
        self.spin_lc_scale.setValue(int(config.SCALE_LASER))
        param_row.addWidget(self.spin_lc_scale)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("Poly Deg:"))
        self.spin_lc_deg = QtWidgets.QSpinBox()
        self.spin_lc_deg.setRange(2, 6)
        self.spin_lc_deg.setValue(4)
        param_row.addWidget(self.spin_lc_deg)
        param_row.addStretch()
        layout.addLayout(param_row)

        # Row 3: Execute Calibration button + Diagnostic Plot Viewers
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run_laser_calib = QtWidgets.QPushButton("⚙ Calibrate Laser Reference (Steps 2–6)")
        self.btn_run_laser_calib.setStyleSheet("font-weight: bold; padding: 4px 10px;")
        self.btn_run_laser_calib.clicked.connect(self._on_run_laser_calibration)
        btn_row.addWidget(self.btn_run_laser_calib)

        self.btn_diag_seeds = QtWidgets.QPushButton("📊 5.1A: Denoise & Seeds")
        self.btn_diag_seeds.clicked.connect(lambda: self._on_show_laser_diag("denoise_seeds"))
        btn_row.addWidget(self.btn_diag_seeds)

        self.btn_diag_geom = QtWidgets.QPushButton("📊 5.1B: Fringes & Geometry")
        self.btn_diag_geom.clicked.connect(lambda: self._on_show_laser_diag("fringes_geom"))
        btn_row.addWidget(self.btn_diag_geom)

        self.btn_diag_res = QtWidgets.QPushButton("📊 5.1C: Surface Residuals")
        self.btn_diag_res.clicked.connect(lambda: self._on_show_laser_diag("anamorphic_res"))
        btn_row.addWidget(self.btn_diag_res)
        layout.addLayout(btn_row)

        # Row 4: Calibrated Geometry & Surface RMS Summary Banner
        self.lbl_laser_calib_summary = QtWidgets.QLabel(
            "Acquire or load a Laser Reference image, then click 'Calibrate Laser Reference'."
        )
        self.lbl_laser_calib_summary.setStyleSheet(
            "background-color: #1e293b; color: #e2e8f0; font-family: monospace; "
            "font-size: 11px; padding: 5px; border-radius: 4px;"
        )
        self.lbl_laser_calib_summary.setWordWrap(True)
        layout.addWidget(self.lbl_laser_calib_summary)

        return box

    def _build_step5_2_rb_calib_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(
            "Step 5.2 — Rubidium Absorption & 2D Frequency Map Calibration (Steps 7–10B)"
        )
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        # Row 1: Auto-calibrate toggle + Status badge
        top_row = QtWidgets.QHBoxLayout()
        self.chk_auto_calib_rb = QtWidgets.QCheckBox(
            "Auto-run Rb Calibration after acquiring Rb Absorption"
        )
        self.chk_auto_calib_rb.setChecked(True)
        top_row.addWidget(self.chk_auto_calib_rb)
        top_row.addStretch()

        self.lbl_rb_calib_status = QtWidgets.QLabel("NOT CALIBRATED")
        self.lbl_rb_calib_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_rb_calib_status.setMinimumWidth(140)
        top_row.addWidget(self.lbl_rb_calib_status)
        layout.addLayout(top_row)

        # Row 2: Algorithm parameters
        param_row = QtWidgets.QHBoxLayout()
        param_row.addWidget(QtWidgets.QLabel("T_cell:"))
        self.spin_rb_tcell = QtWidgets.QDoubleSpinBox()
        self.spin_rb_tcell.setDecimals(1)
        self.spin_rb_tcell.setRange(20.0, 150.0)
        self.spin_rb_tcell.setValue(70.0)
        self.spin_rb_tcell.setSuffix(" °C")
        param_row.addWidget(self.spin_rb_tcell)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("L_cell:"))
        self.spin_rb_lcell = QtWidgets.QDoubleSpinBox()
        self.spin_rb_lcell.setDecimals(3)
        self.spin_rb_lcell.setRange(0.005, 0.500)
        self.spin_rb_lcell.setValue(0.070)
        self.spin_rb_lcell.setSuffix(" m")
        param_row.addWidget(self.spin_rb_lcell)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("85Rb Lock:"))
        self.spin_rb_lock = QtWidgets.QDoubleSpinBox()
        self.spin_rb_lock.setDecimals(3)
        self.spin_rb_lock.setRange(-15.0, 15.0)
        self.spin_rb_lock.setValue(1.323)
        self.spin_rb_lock.setSuffix(" GHz")
        param_row.addWidget(self.spin_rb_lock)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("Fit Win:"))
        self.spin_rb_fitwin = QtWidgets.QDoubleSpinBox()
        self.spin_rb_fitwin.setDecimals(1)
        self.spin_rb_fitwin.setRange(2.0, 30.0)
        self.spin_rb_fitwin.setValue(10.0)
        self.spin_rb_fitwin.setSuffix(" GHz")
        param_row.addWidget(self.spin_rb_fitwin)

        param_row.addSpacing(6)
        param_row.addWidget(QtWidgets.QLabel("Morph Scale:"))
        self.spin_rb_scale = QtWidgets.QSpinBox()
        self.spin_rb_scale.setRange(2, 32)
        self.spin_rb_scale.setValue(9)
        param_row.addWidget(self.spin_rb_scale)
        param_row.addStretch()
        layout.addLayout(param_row)

        # Row 3: Execute Calibration button + Diagnostic Plot Viewers
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run_rb_calib = QtWidgets.QPushButton("⚙ Calibrate Rb & Frequency Map (Steps 7–10B)")
        self.btn_run_rb_calib.setStyleSheet("font-weight: bold; padding: 4px 10px;")
        self.btn_run_rb_calib.clicked.connect(self._on_run_rb_calibration)
        btn_row.addWidget(self.btn_run_rb_calib)

        self.btn_diag_rb_rad = QtWidgets.QPushButton("📊 5.2A: Denoise & Radial")
        self.btn_diag_rb_rad.clicked.connect(lambda: self._on_show_rb_diag("rb_denoise_radial"))
        btn_row.addWidget(self.btn_diag_rb_rad)

        self.btn_diag_rb_fsr = QtWidgets.QPushButton("📊 5.2B: Folded FSR Fit")
        self.btn_diag_rb_fsr.clicked.connect(lambda: self._on_show_rb_diag("fsr_fit"))
        btn_row.addWidget(self.btn_diag_rb_fsr)

        self.btn_diag_rb_fmap = QtWidgets.QPushButton("📊 5.2C: 2D Frequency Map")
        self.btn_diag_rb_fmap.clicked.connect(lambda: self._on_show_rb_diag("freq_map"))
        btn_row.addWidget(self.btn_diag_rb_fmap)
        layout.addLayout(btn_row)

        # Row 4: Calibrated FSR & 2D Frequency Map Summary Banner
        self.lbl_rb_calib_summary = QtWidgets.QLabel(
            "Complete Step 5.1 (Laser Calibration) and acquire/load an Rb Absorption image, then click 'Calibrate Rb & Frequency Map'."
        )
        self.lbl_rb_calib_summary.setStyleSheet(
            "background-color: #1e293b; color: #e2e8f0; font-family: monospace; "
            "font-size: 11px; padding: 5px; border-radius: 4px;"
        )
        self.lbl_rb_calib_summary.setWordWrap(True)
        layout.addWidget(self.lbl_rb_calib_summary)

        return box

    def _build_step5_3_sample_fit_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(
            "Step 5.3 — Sample DHO Fitting & Reference Geometry Calibration (Steps 11–14)"
        )
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        # Row 1: Auto-fit toggle + Presets + Status badge
        top_row = QtWidgets.QHBoxLayout()
        self.chk_auto_fit_ref = QtWidgets.QCheckBox("Auto-fit after acquiring Ref Sample")
        self.chk_auto_fit_ref.setChecked(True)
        top_row.addWidget(self.chk_auto_fit_ref)

        btn_water = QtWidgets.QPushButton("Water (5.1 GHz, n=1.33)")
        btn_water.clicked.connect(lambda: self._apply_sample_preset(5.1, 2.0, 0.8, 0.4, 1.3, 1.33, 4.8, 5.3))
        top_row.addWidget(btn_water)

        btn_plastic = QtWidgets.QPushButton("Plastic (9.7 GHz, n=1.59)")
        btn_plastic.clicked.connect(lambda: self._apply_sample_preset(9.7, 2.0, 0.8, 0.4, 1.3, 1.59, 9.3, 9.9))
        top_row.addWidget(btn_plastic)

        top_row.addStretch()
        self.lbl_sample_fit_status = QtWidgets.QLabel("NOT FITTED")
        self.lbl_sample_fit_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_sample_fit_status.setMinimumWidth(140)
        top_row.addWidget(self.lbl_sample_fit_status)
        layout.addLayout(top_row)

        # Row 2: Spatial Binning & DHO Fitting Parameters
        dho_row = QtWidgets.QHBoxLayout()
        dho_row.addWidget(QtWidgets.QLabel("Bin H×W:"))
        self.spin_bin_h = QtWidgets.QSpinBox()
        self.spin_bin_h.setRange(1, 32)
        self.spin_bin_h.setValue(4)
        dho_row.addWidget(self.spin_bin_h)

        self.spin_bin_w = QtWidgets.QSpinBox()
        self.spin_bin_w.setRange(1, 32)
        self.spin_bin_w.setValue(2)
        dho_row.addWidget(self.spin_bin_w)

        dho_row.addSpacing(6)
        dho_row.addWidget(QtWidgets.QLabel("Shift ν_B:"))
        self.spin_dho_shift = QtWidgets.QDoubleSpinBox()
        self.spin_dho_shift.setDecimals(2)
        self.spin_dho_shift.setRange(0.5, 25.0)
        self.spin_dho_shift.setValue(5.10)
        self.spin_dho_shift.setSuffix(" GHz")
        dho_row.addWidget(self.spin_dho_shift)

        dho_row.addWidget(QtWidgets.QLabel("±Tol:"))
        self.spin_dho_tol = QtWidgets.QDoubleSpinBox()
        self.spin_dho_tol.setDecimals(2)
        self.spin_dho_tol.setRange(0.2, 10.0)
        self.spin_dho_tol.setValue(2.00)
        self.spin_dho_tol.setSuffix(" GHz")
        dho_row.addWidget(self.spin_dho_tol)

        dho_row.addSpacing(6)
        dho_row.addWidget(QtWidgets.QLabel("LW Γ_B:"))
        self.spin_dho_lw = QtWidgets.QDoubleSpinBox()
        self.spin_dho_lw.setDecimals(2)
        self.spin_dho_lw.setRange(0.05, 5.0)
        self.spin_dho_lw.setValue(0.80)
        self.spin_dho_lw.setSuffix(" GHz")
        dho_row.addWidget(self.spin_dho_lw)

        dho_row.addWidget(QtWidgets.QLabel("[min..max]:"))
        self.spin_dho_lw_min = QtWidgets.QDoubleSpinBox()
        self.spin_dho_lw_min.setDecimals(2)
        self.spin_dho_lw_min.setRange(0.05, 5.0)
        self.spin_dho_lw_min.setValue(0.40)
        dho_row.addWidget(self.spin_dho_lw_min)

        self.spin_dho_lw_max = QtWidgets.QDoubleSpinBox()
        self.spin_dho_lw_max.setDecimals(2)
        self.spin_dho_lw_max.setRange(0.10, 10.0)
        self.spin_dho_lw_max.setValue(1.30)
        dho_row.addWidget(self.spin_dho_lw_max)
        dho_row.addStretch()
        layout.addLayout(dho_row)

        # Row 3: Step 14 Ellipse & Wavevector Parameters
        q_row = QtWidgets.QHBoxLayout()
        self.chk_run_q_calib = QtWidgets.QCheckBox("Step 14: Fit 2D Coaxial Ellipse & q-Vector (M_θ, V_s)")
        self.chk_run_q_calib.setChecked(True)
        q_row.addWidget(self.chk_run_q_calib)

        q_row.addSpacing(8)
        q_row.addWidget(QtWidgets.QLabel("Index n:"))
        self.spin_q_n = QtWidgets.QDoubleSpinBox()
        self.spin_q_n.setDecimals(3)
        self.spin_q_n.setRange(1.000, 3.000)
        self.spin_q_n.setValue(1.330)
        q_row.addWidget(self.spin_q_n)

        q_row.addSpacing(8)
        q_row.addWidget(QtWidgets.QLabel("Shift Filter [min..max]:"))
        self.spin_q_smin = QtWidgets.QDoubleSpinBox()
        self.spin_q_smin.setDecimals(2)
        self.spin_q_smin.setRange(0.5, 25.0)
        self.spin_q_smin.setValue(4.80)
        self.spin_q_smin.setSuffix(" GHz")
        q_row.addWidget(self.spin_q_smin)

        self.spin_q_smax = QtWidgets.QDoubleSpinBox()
        self.spin_q_smax.setDecimals(2)
        self.spin_q_smax.setRange(0.5, 25.0)
        self.spin_q_smax.setValue(5.30)
        self.spin_q_smax.setSuffix(" GHz")
        q_row.addWidget(self.spin_q_smax)
        q_row.addStretch()
        layout.addLayout(q_row)

        # Row 4: Execute Fit button + 3 Diagnostic Plot Viewers
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run_sample_fit = QtWidgets.QPushButton("⚙ Fit Ref / Test Sample (Steps 11–14)")
        self.btn_run_sample_fit.setStyleSheet("font-weight: bold; padding: 4px 10px;")
        self.btn_run_sample_fit.clicked.connect(self._on_run_sample_fit)
        btn_row.addWidget(self.btn_run_sample_fit)

        self.btn_diag_s_freq = QtWidgets.QPushButton("📊 5.3A: In-Situ Freq & Windows")
        self.btn_diag_s_freq.clicked.connect(lambda: self._on_show_sample_diag("insitu_freq"))
        btn_row.addWidget(self.btn_diag_s_freq)

        self.btn_diag_s_dho = QtWidgets.QPushButton("📊 5.3B: 2D DHO Shift & LW")
        self.btn_diag_s_dho.clicked.connect(lambda: self._on_show_sample_diag("dho_maps"))
        btn_row.addWidget(self.btn_diag_s_dho)

        self.btn_diag_s_q = QtWidgets.QPushButton("📊 5.3C: Ellipse & Dispersion (Vs)")
        self.btn_diag_s_q.clicked.connect(lambda: self._on_show_sample_diag("q_dispersion"))
        btn_row.addWidget(self.btn_diag_s_q)
        layout.addLayout(btn_row)

        # Row 5: Fitted Summary Banner
        self.lbl_sample_fit_summary = QtWidgets.QLabel(
            "Complete Step 5.2 (Rb Calibration), acquire a Reference Sample (Row 3 in 4B), then click 'Fit Ref / Test Sample'."
        )
        self.lbl_sample_fit_summary.setStyleSheet(
            "background-color: #1e293b; color: #e2e8f0; font-family: monospace; "
            "font-size: 11px; padding: 5px; border-radius: 4px;"
        )
        self.lbl_sample_fit_summary.setWordWrap(True)
        layout.addWidget(self.lbl_sample_fit_summary)

        return box

    def _build_map_configuration_card(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("4C. Spatial Map Configuration & Scan Execution (Brillouin/<Map_name>)")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(8)

        # 1. Map Name + Point Exposure + Averaging
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Map Name:"))
        self.txt_map_name = QtWidgets.QLineEdit("Map_01")
        self.txt_map_name.setMaximumWidth(130)
        top_row.addWidget(self.txt_map_name)

        top_row.addSpacing(8)
        top_row.addWidget(QtWidgets.QLabel("Point Exposure:"))
        self.spin_map_exp = QtWidgets.QDoubleSpinBox()
        self.spin_map_exp.setDecimals(2)
        self.spin_map_exp.setRange(0.02, 30000.0)
        self.spin_map_exp.setValue(100.0)
        self.spin_map_exp.setSuffix(" ms")
        self.spin_map_exp.valueChanged.connect(self._update_map_summary)
        top_row.addWidget(self.spin_map_exp)

        top_row.addSpacing(8)
        top_row.addWidget(QtWidgets.QLabel("Avg:"))
        self.spin_map_avg = QtWidgets.QSpinBox()
        self.spin_map_avg.setRange(1, 500)
        self.spin_map_avg.setValue(1)
        self.spin_map_avg.setSuffix(" frame(s)")
        self.spin_map_avg.valueChanged.connect(self._update_map_summary)
        top_row.addWidget(self.spin_map_avg)
        top_row.addStretch()
        layout.addLayout(top_row)

        # 2. Grid Coordinates & Step Sizes
        grid_form = QtWidgets.QGridLayout()
        grid_form.setSpacing(6)

        grid_form.addWidget(QtWidgets.QLabel("Start X₀:"), 0, 0)
        self.spin_x_start = QtWidgets.QDoubleSpinBox()
        self.spin_x_start.setDecimals(2)
        self.spin_x_start.setRange(-100000.0, 100000.0)
        self.spin_x_start.setValue(0.0)
        self.spin_x_start.setSuffix(" µm")
        self.spin_x_start.valueChanged.connect(self._update_map_summary)
        grid_form.addWidget(self.spin_x_start, 0, 1)

        grid_form.addWidget(QtWidgets.QLabel("Step ΔX:"), 0, 2)
        self.spin_dx = QtWidgets.QDoubleSpinBox()
        self.spin_dx.setDecimals(2)
        self.spin_dx.setRange(-50000.0, 50000.0)
        self.spin_dx.setValue(25.0)
        self.spin_dx.setSuffix(" µm")
        self.spin_dx.valueChanged.connect(self._update_map_summary)
        grid_form.addWidget(self.spin_dx, 0, 3)

        grid_form.addWidget(QtWidgets.QLabel("Points Nx:"), 0, 4)
        self.spin_nx = QtWidgets.QSpinBox()
        self.spin_nx.setRange(1, 5000)
        self.spin_nx.setValue(4)
        self.spin_nx.valueChanged.connect(self._update_map_summary)
        grid_form.addWidget(self.spin_nx, 0, 5)

        grid_form.addWidget(QtWidgets.QLabel("Start Y₀:"), 1, 0)
        self.spin_y_start = QtWidgets.QDoubleSpinBox()
        self.spin_y_start.setDecimals(2)
        self.spin_y_start.setRange(-100000.0, 100000.0)
        self.spin_y_start.setValue(0.0)
        self.spin_y_start.setSuffix(" µm")
        self.spin_y_start.valueChanged.connect(self._update_map_summary)
        grid_form.addWidget(self.spin_y_start, 1, 1)

        grid_form.addWidget(QtWidgets.QLabel("Step ΔY:"), 1, 2)
        self.spin_dy = QtWidgets.QDoubleSpinBox()
        self.spin_dy.setDecimals(2)
        self.spin_dy.setRange(-50000.0, 50000.0)
        self.spin_dy.setValue(25.0)
        self.spin_dy.setSuffix(" µm")
        self.spin_dy.valueChanged.connect(self._update_map_summary)
        grid_form.addWidget(self.spin_dy, 1, 3)

        grid_form.addWidget(QtWidgets.QLabel("Points Ny:"), 1, 4)
        self.spin_ny = QtWidgets.QSpinBox()
        self.spin_ny.setRange(1, 5000)
        self.spin_ny.setValue(4)
        self.spin_ny.valueChanged.connect(self._update_map_summary)
        grid_form.addWidget(self.spin_ny, 1, 5)

        layout.addLayout(grid_form)

        # 3. Stage Teaching Helpers (Set Start / Teach Corner A & Corner B)
        teach_row = QtWidgets.QHBoxLayout()
        self.btn_set_start_current = QtWidgets.QPushButton("Set (X₀, Y₀) from Current Stage")
        self.btn_set_start_current.clicked.connect(self._on_set_start_from_stage)
        teach_row.addWidget(self.btn_set_start_current)

        self.btn_teach_a = QtWidgets.QPushButton("Teach Corner A")
        self.btn_teach_a.clicked.connect(self._on_teach_corner_a)
        teach_row.addWidget(self.btn_teach_a)

        self.btn_teach_b = QtWidgets.QPushButton("Teach Corner B -> Compute Grid")
        self.btn_teach_b.clicked.connect(self._on_teach_corner_b)
        teach_row.addWidget(self.btn_teach_b)
        teach_row.addStretch()
        layout.addLayout(teach_row)

        # 4. Scan Pattern, Settling Delay & Optional Raw Data Saving
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.addWidget(QtWidgets.QLabel("Pattern:"))
        self.combo_pattern = QtWidgets.QComboBox()
        self.combo_pattern.addItem("Snake / Serpentine (Alternating X)", "snake")
        self.combo_pattern.addItem("Raster (Unidirectional X)", "raster")
        self.combo_pattern.currentIndexChanged.connect(self._update_map_summary)
        opt_row.addWidget(self.combo_pattern)

        opt_row.addSpacing(8)
        opt_row.addWidget(QtWidgets.QLabel("Settling:"))
        self.spin_map_settling = QtWidgets.QSpinBox()
        self.spin_map_settling.setRange(0, 5000)
        self.spin_map_settling.setValue(50)
        self.spin_map_settling.setSuffix(" ms")
        self.spin_map_settling.valueChanged.connect(self._update_map_summary)
        opt_row.addWidget(self.spin_map_settling)

        opt_row.addSpacing(12)
        self.chk_save_raw = QtWidgets.QCheckBox("Save 4D Raw Data Dataset (Ny×Nx×H×W) in HDF5")
        self.chk_save_raw.setChecked(False)
        self.chk_save_raw.toggled.connect(self._update_map_summary)
        opt_row.addWidget(self.chk_save_raw)
        opt_row.addStretch()
        layout.addLayout(opt_row)

        # 4B. Step 5.4 On-the-Fly Treatment Options Row
        otf_row = QtWidgets.QHBoxLayout()
        self.chk_treat_on_the_fly = QtWidgets.QCheckBox(
            "Step 5.4: Treat on the fly & save 4D fitted DHO parameters (Treated H×W bins)"
        )
        self.chk_treat_on_the_fly.setChecked(True)
        self.chk_treat_on_the_fly.toggled.connect(self._update_map_summary)
        otf_row.addWidget(self.chk_treat_on_the_fly)

        otf_row.addSpacing(12)
        self.chk_fast_drift = QtWidgets.QCheckBox("Fast elastic drift tracking per point")
        self.chk_fast_drift.setChecked(True)
        otf_row.addWidget(self.chk_fast_drift)
        otf_row.addStretch()
        layout.addLayout(otf_row)

        # 5. Live Map Summary Banner
        self.lbl_map_summary = QtWidgets.QLabel("")
        self.lbl_map_summary.setStyleSheet(
            "background-color: #1e293b; color: #e2e8f0; font-family: monospace; "
            "font-size: 11px; padding: 6px; border-radius: 4px;"
        )
        self.lbl_map_summary.setWordWrap(True)
        layout.addWidget(self.lbl_map_summary)

        # 6. Execution Controls & Progress Bar
        exec_row = QtWidgets.QHBoxLayout()
        self.btn_start_map = QtWidgets.QPushButton("▶ Start Map Acquisition")
        self.btn_start_map.setMinimumHeight(32)
        self.btn_start_map.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 4px 12px;"
        )
        self.btn_start_map.clicked.connect(self._on_start_map)
        exec_row.addWidget(self.btn_start_map)

        self.btn_pause_map = QtWidgets.QPushButton("⏸ Pause")
        self.btn_pause_map.setMinimumHeight(32)
        self.btn_pause_map.setEnabled(False)
        self.btn_pause_map.clicked.connect(self._on_toggle_pause_map)
        exec_row.addWidget(self.btn_pause_map)

        self.btn_abort_map = QtWidgets.QPushButton("⏹ Abort Map")
        self.btn_abort_map.setMinimumHeight(32)
        self.btn_abort_map.setEnabled(False)
        self.btn_abort_map.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")
        self.btn_abort_map.clicked.connect(self._on_abort_map)
        exec_row.addWidget(self.btn_abort_map)
        layout.addLayout(exec_row)

        self.progress_map = QtWidgets.QProgressBar()
        self.progress_map.setRange(0, 100)
        self.progress_map.setValue(0)
        self.progress_map.setFormat("Idle — %p%")
        layout.addWidget(self.progress_map)

        return box

    # ==========================================================================
    # Event Handlers: HDF5 Session & Calibration Spectra
    # ==========================================================================

    def _on_browse_h5(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select or Create HDF5_BLS Measures File",
            self.txt_h5_path.text().strip() or "Measures.h5",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
            options=QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if path:
            self.txt_h5_path.setText(path)
            self._on_open_or_create_h5()

    def _on_open_or_create_h5(self):
        path = self.txt_h5_path.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Invalid Path", "Please enter a valid HDF5 file path.")
            return
        try:
            info = self.session.open_or_create_h5(path)
            self.txt_h5_path.setText(info["h5_path"])
            maps_str = ", ".join(info["existing_maps"]) if info["existing_maps"] else "none"
            self.lbl_h5_status.setText(
                f"Opened: {os.path.basename(info['h5_path'])}  |  "
                f"Laser: {'✓' if info['has_laser'] else '—'}, "
                f"Rb: {'✓' if info['has_rb'] else '—'}, "
                f"Ref: {'✓' if info['has_ref_sample'] else '—'}  |  "
                f"Maps: {maps_str}"
            )
            self.sig_log.emit(f"Opened HDF5_BLS container: {info['h5_path']}")
            self._update_calibration_badges()
        except Exception as exc:
            self.sig_log.emit(f"ERROR opening HDF5: {exc}")
            QtWidgets.QMessageBox.critical(self, "HDF5 Error", str(exc))

    def _ensure_h5_open(self) -> bool:
        if not self.session.is_open:
            self._on_open_or_create_h5()
        return self.session.is_open

    def _on_acquire_calibration(self, role: str, exp_ms: float, n_avg: int):
        if not self.camera.is_connected:
            QtWidgets.QMessageBox.warning(self, "Camera Not Connected", "Please connect the camera first.")
            return
        if not self._ensure_h5_open():
            return
        if self._is_busy():
            return

        self._stop_live_cb()
        self._set_controls_busy(True)

        self._cal_worker = CalibrationCaptureWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            role=role,
            exposure_ms=exp_ms,
            n_avg=n_avg,
            parent=self,
        )
        self._cal_worker.sig_log.connect(self.sig_log.emit)
        self._cal_worker.sig_captured.connect(self._on_calibration_captured)
        self._cal_worker.sig_error.connect(self._on_worker_error)
        self._cal_worker.finished.connect(lambda: self._set_controls_busy(False))
        self._cal_worker.start()

    @QtCore.Slot(str, object, object, float, int, int)
    def _on_calibration_captured(self, role: str, frame, pos, exp_s: float, n_avg: int, binning: int):
        # Lock camera ROI & hardware binning once calibration spectrum is captured
        self.sig_lock_geometry.emit(True)
        self.sig_camera_updated.emit()
        self.sig_display_frame.emit(frame, pos, exp_s, n_avg, binning)
        self._update_calibration_badges()

        # If Laser Reference was just acquired and auto-calibrate is enabled, trigger Step 5.1
        if role == AcquisitionSession.ROLE_LASER and self.chk_auto_calib_laser.isChecked():
            QtCore.QTimer.singleShot(50, self._on_run_laser_calibration)
        # If Rb Absorption was just acquired and auto-calibrate is enabled, trigger Step 5.2
        elif (
            role == AcquisitionSession.ROLE_RB
            and self.chk_auto_calib_rb.isChecked()
            and self.session.laser_calib_result is not None
        ):
            QtCore.QTimer.singleShot(50, self._on_run_rb_calibration)
        # If Ref Sample was just acquired and auto-fit is enabled, trigger Step 5.3
        elif (
            role == AcquisitionSession.ROLE_REF_SAMPLE
            and self.chk_auto_fit_ref.isChecked()
            and self.session.rb_calib_result is not None
        ):
            QtCore.QTimer.singleShot(50, self._on_run_sample_fit)

    def _on_run_laser_calibration(self):
        if not self._ensure_h5_open():
            return
        if self.session.laser_image is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Laser Reference Missing",
                "Please acquire or load a Laser Reference image first (Section 4B).",
            )
            return
        if self._is_busy():
            return

        params = {
            "nb_laser_modes": int(self.spin_lc_modes.value()),
            "y_center_laser": int(self.spin_lc_ycenter.value()),
            "distance_peaks": int(self.spin_lc_dist.value()),
            "scale_laser": int(self.spin_lc_scale.value()),
            "poly_deg": int(self.spin_lc_deg.value()),
        }
        self._set_controls_busy(True)
        self._laser_calib_worker = LaserCalibrationWorker(
            session=self.session,
            params=params,
            parent=self,
        )
        self._laser_calib_worker.sig_log.connect(self.sig_log.emit)
        self._laser_calib_worker.sig_calibrated.connect(self._on_laser_calibrated)
        self._laser_calib_worker.sig_error.connect(self._on_worker_error)
        self._laser_calib_worker.finished.connect(lambda: self._set_controls_busy(False))
        self._laser_calib_worker.start()

    @QtCore.Slot(object)
    def _on_laser_calibrated(self, res: dict):
        self._update_calibration_badges()
        self.sig_show_laser_diagnostics.emit(res, "fringes_geom")

    def _on_show_laser_diag(self, view_mode: str):
        res = self.session.laser_calib_result
        if res is None:
            QtWidgets.QMessageBox.information(
                self,
                "Not Calibrated Yet",
                "Run 'Calibrate Laser Reference (Steps 2–6)' first to view diagnostic plots.",
            )
            return
        self.sig_show_laser_diagnostics.emit(res, view_mode)

    def _on_run_rb_calibration(self):
        if not self._ensure_h5_open():
            return
        if self.session.laser_calib_result is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Laser Calibration Required",
                "Please complete Step 5.1 (Laser Reference Calibration) before running Step 5.2 (Rb Calibration).",
            )
            return
        if self.session.rb_image is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Rb Absorption Image Missing",
                "Please acquire or load a Rubidium Absorption image first (Section 4B).",
            )
            return
        if self._is_busy():
            return

        params = {
            "t_cell_c": float(self.spin_rb_tcell.value()),
            "l_cell_m": float(self.spin_rb_lcell.value()),
            "rb_lock_offset_ghz": float(self.spin_rb_lock.value()),
            "fit_window_ghz": float(self.spin_rb_fitwin.value()),
            "scale_rb": int(self.spin_rb_scale.value()),
        }
        self._set_controls_busy(True)
        self._rb_calib_worker = RbCalibrationWorker(
            session=self.session,
            params=params,
            parent=self,
        )
        self._rb_calib_worker.sig_log.connect(self.sig_log.emit)
        self._rb_calib_worker.sig_calibrated.connect(self._on_rb_calibrated)
        self._rb_calib_worker.sig_error.connect(self._on_worker_error)
        self._rb_calib_worker.finished.connect(lambda: self._set_controls_busy(False))
        self._rb_calib_worker.start()

    @QtCore.Slot(object)
    def _on_rb_calibrated(self, res: dict):
        self._update_calibration_badges()
        self.sig_show_rb_diagnostics.emit(res, "fsr_fit")

    def _on_show_rb_diag(self, view_mode: str):
        res = self.session.rb_calib_result
        if res is None:
            QtWidgets.QMessageBox.information(
                self,
                "Not Calibrated Yet",
                "Run 'Calibrate Rb & Frequency Map (Steps 7–10B)' first to view diagnostic plots.",
            )
            return
        self.sig_show_rb_diagnostics.emit(res, view_mode)

    def _apply_sample_preset(
        self,
        shift_ghz: float,
        tol_ghz: float,
        lw_ghz: float,
        lw_min: float,
        lw_max: float,
        n_idx: float,
        s_min: float,
        s_max: float,
    ):
        self.spin_dho_shift.setValue(shift_ghz)
        self.spin_dho_tol.setValue(tol_ghz)
        self.spin_dho_lw.setValue(lw_ghz)
        self.spin_dho_lw_min.setValue(lw_min)
        self.spin_dho_lw_max.setValue(lw_max)
        self.spin_q_n.setValue(n_idx)
        self.spin_q_smin.setValue(s_min)
        self.spin_q_smax.setValue(s_max)

    def _on_run_sample_fit(self):
        if not self._ensure_h5_open():
            return
        if self.session.laser_calib_result is None or self.session.rb_calib_result is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration Required",
                "Please complete Steps 5.1 (Laser) and 5.2 (Rubidium) before running Step 5.3.",
            )
            return
        if self.session.ref_sample_image is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Reference Sample Image Missing",
                "Please click '📷 Acquire Ref' in Section 4B (Row 3) to acquire a Reference/Test Sample image first.",
            )
            return
        if self._is_busy():
            return

        params = {
            "bin_val_h": int(self.spin_bin_h.value()),
            "bin_val_w": int(self.spin_bin_w.value()),
            "expected_shift_ghz": float(self.spin_dho_shift.value()),
            "shift_tol_ghz": float(self.spin_dho_tol.value()),
            "expected_lw_ghz": float(self.spin_dho_lw.value()),
            "lw_bounds_ghz": (float(self.spin_dho_lw_min.value()), float(self.spin_dho_lw_max.value())),
            "run_q_calibration": bool(self.chk_run_q_calib.isChecked()),
            "refractive_index": float(self.spin_q_n.value()),
            "shift_filter_bounds": (float(self.spin_q_smin.value()), float(self.spin_q_smax.value())),
        }
        self._set_controls_busy(True)
        self._sample_fit_worker = SampleTreatmentWorker(
            session=self.session,
            params=params,
            parent=self,
        )
        self._sample_fit_worker.sig_log.connect(self.sig_log.emit)
        self._sample_fit_worker.sig_fitted.connect(self._on_sample_fitted)
        self._sample_fit_worker.sig_error.connect(self._on_worker_error)
        self._sample_fit_worker.finished.connect(lambda: self._set_controls_busy(False))
        self._sample_fit_worker.start()

    @QtCore.Slot(object)
    def _on_sample_fitted(self, res: dict):
        self._update_calibration_badges()
        default_view = "q_dispersion" if res.get("q_results") is not None else "dho_maps"
        self.sig_show_sample_diagnostics.emit(res, default_view)

    def _on_show_sample_diag(self, view_mode: str):
        res = self.session.ref_sample_calib_result
        if res is None:
            QtWidgets.QMessageBox.information(
                self,
                "Not Fitted Yet",
                "Run 'Fit Ref / Test Sample (Steps 11–14)' first to view diagnostic plots.",
            )
            return
        self.sig_show_sample_diagnostics.emit(res, view_mode)

    def _on_view_calibration(self, role: str):
        img = None
        meta = {}
        if role == AcquisitionSession.ROLE_LASER:
            img, meta = self.session.laser_image, self.session.laser_meta
        elif role == AcquisitionSession.ROLE_RB:
            img, meta = self.session.rb_image, self.session.rb_meta
        elif role == AcquisitionSession.ROLE_REF_SAMPLE:
            img, meta = self.session.ref_sample_image, self.session.ref_sample_meta

        if img is None:
            QtWidgets.QMessageBox.information(self, "Not Available", f"No '{role}' image has been acquired or loaded yet.")
            return

        exp_s = float(meta.get("exposure_s", 0.05))
        n_avg = int(meta.get("frame_averaging", 1))
        binning = int(meta.get("hardware_binning", 1))
        pos = None
        if "stage_x_um" in meta and "stage_y_um" in meta:
            pos = {"X": float(meta["stage_x_um"]), "Y": float(meta["stage_y_um"]), "Z": meta.get("stage_z_um")}
        self.sig_display_frame.emit(img, pos, exp_s, n_avg, binning)
        self.sig_log.emit(f"Displaying stored '{role}' calibration image ({img.shape[1]}x{img.shape[0]} px).")

    def _update_calibration_badges(self):
        def _style_badge(lbl: QtWidgets.QLabel, has_data: bool, optional: bool = False):
            if has_data:
                lbl.setText("✓ READY (H5)")
                lbl.setStyleSheet(
                    "background-color: #2e7d32; color: white; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
                )
            elif optional:
                lbl.setText("OPTIONAL")
                lbl.setStyleSheet(
                    "background-color: #455a64; color: #cfd8dc; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
                )
            else:
                lbl.setText("MISSING")
                lbl.setStyleSheet(
                    "background-color: #c62828; color: white; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
                )

        has_laser = self.session.laser_image is not None
        has_rb = self.session.rb_image is not None
        has_ref = self.session.ref_sample_image is not None

        _style_badge(self.lbl_status_laser, has_laser, optional=False)
        _style_badge(self.lbl_status_rb, has_rb, optional=False)
        _style_badge(self.lbl_status_ref, has_ref, optional=True)

        self.btn_view_laser.setEnabled(has_laser)
        self.btn_view_rb.setEnabled(has_rb)
        self.btn_view_ref.setEnabled(has_ref)

        # Update Step 5.1 Laser Calibration badge & summary
        lc_res = self.session.laser_calib_result
        has_lc = lc_res is not None
        if has_lc:
            self.lbl_laser_calib_status.setText("✓ CALIBRATED")
            self.lbl_laser_calib_status.setStyleSheet(
                "background-color: #2e7d32; color: white; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
            self.lbl_laser_calib_summary.setText(
                f"Modes: {len(lc_res['refined_fringes'])}  |  "
                f"Optical Axis (x₀, y₀) = ({lc_res['x0_fit']:.2f}, {lc_res['y0_fit']:.2f}) px  |  "
                f"Gnomonic Xc = {lc_res['xc_gnom']:.2f} px\n"
                f"2D Anamorphic Surface Fit RMS: {lc_res['rms_raw_mhz']:.2f} MHz (raw)  /  "
                f"{lc_res['rms_smooth_mhz']:.2f} MHz (smoothed)"
            )
        else:
            self.lbl_laser_calib_status.setText("NOT CALIBRATED")
            self.lbl_laser_calib_status.setStyleSheet(
                "background-color: #455a64; color: #cfd8dc; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
            self.lbl_laser_calib_summary.setText(
                "Acquire or load a Laser Reference image, then click 'Calibrate Laser Reference'."
            )

        self.btn_run_laser_calib.setEnabled(has_laser)
        self.btn_diag_seeds.setEnabled(has_lc)
        self.btn_diag_geom.setEnabled(has_lc)
        self.btn_diag_res.setEnabled(has_lc)

        # Update Step 5.2 Rubidium Calibration badge & summary
        rb_res = self.session.rb_calib_result
        has_rb_cal = rb_res is not None
        if has_rb_cal:
            self.lbl_rb_calib_status.setText("✓ CALIBRATED")
            self.lbl_rb_calib_status.setStyleSheet(
                "background-color: #2e7d32; color: white; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
            fmap_h, fmap_w = rb_res["freq_map"].shape
            self.lbl_rb_calib_summary.setText(
                f"Calibrated VIPA FSR = {rb_res['FSR_cal']:.4f} GHz  |  "
                f"Hyperfine Offset Δν = {rb_res['delta_nu_opt']:+.3f} GHz  |  "
                f"Elastic Zero ξ = {rb_res['xi_elastic']:.4f}\n"
                f"2D Frequency Map: {fmap_w}×{fmap_h} px saved at Brillouin/Calibration/Rb_absorption/Frequency"
            )
        else:
            self.lbl_rb_calib_status.setText("NOT CALIBRATED")
            self.lbl_rb_calib_status.setStyleSheet(
                "background-color: #455a64; color: #cfd8dc; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
            self.lbl_rb_calib_summary.setText(
                "Complete Step 5.1 (Laser Calibration) and acquire/load an Rb Absorption image, then click 'Calibrate Rb & Frequency Map'."
            )

        self.btn_run_rb_calib.setEnabled(has_lc and has_rb)
        self.btn_diag_rb_rad.setEnabled(has_rb_cal)
        self.btn_diag_rb_fsr.setEnabled(has_rb_cal)
        self.btn_diag_rb_fmap.setEnabled(has_rb_cal)

        # Update Step 5.3 Sample Fit & Reference Geometry badge & summary
        sf_res = self.session.ref_sample_calib_result
        has_sf = sf_res is not None
        if has_sf:
            dho = sf_res["dho_results"]
            q_res = sf_res.get("q_results")
            self.lbl_sample_fit_status.setText("✓ FITTED & CALIBRATED")
            self.lbl_sample_fit_status.setStyleSheet(
                "background-color: #2e7d32; color: white; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
            line1 = (
                f"DHO Fit ({dho['bin_val_h']}×{dho['bin_val_w']} bin): {dho['n_fringes']} orders  |  "
                f"Mean Shift = {float(np.nanmean(dho['shift_clipped_averaged'])):.3f} GHz  |  "
                f"Mean LW = {float(np.nanmean(dho['linewidth_clipped_averaged'])):.3f} GHz  |  "
                f"S/AS RMS = {dho['stokes_antistokes_rms_mhz']:.1f} MHz"
            )
            if q_res is not None:
                opt = q_res["opt_params"]
                vs_err = q_res.get("V_s_err") or 0.0
                line2 = (
                    f"\nEllipse Center=({opt['xc']:.1f}, {opt['yc']:.1f}) px, γ={opt['gamma']:.3f}  |  "
                    f"S(0)={q_res['Shift_0']:.4f} GHz  |  M_θ={q_res['M_theta']:.1f} px  |  "
                    f"V_s = {q_res['V_s']:.2f} ± {vs_err:.2f} m/s"
                )
            else:
                line2 = ""
            self.lbl_sample_fit_summary.setText(line1 + line2)
        else:
            self.lbl_sample_fit_status.setText("NOT FITTED")
            self.lbl_sample_fit_status.setStyleSheet(
                "background-color: #455a64; color: #cfd8dc; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
            )
            self.lbl_sample_fit_summary.setText(
                "Complete Step 5.2 (Rb Calibration), acquire a Reference Sample (Row 3 in 4B), then click 'Fit Ref / Test Sample'."
            )

        self.btn_run_sample_fit.setEnabled(has_rb_cal and has_ref)
        self.btn_diag_s_freq.setEnabled(has_sf)
        self.btn_diag_s_dho.setEnabled(has_sf)
        self.btn_diag_s_q.setEnabled(has_sf and (sf_res.get("q_results") is not None))

    # ==========================================================================
    # Event Handlers: Spatial Map Grid & Teaching
    # ==========================================================================

    def get_current_grid_config(self) -> MapGridConfig:
        return MapGridConfig(
            x_start_um=float(self.spin_x_start.value()),
            y_start_um=float(self.spin_y_start.value()),
            dx_um=float(self.spin_dx.value()),
            dy_um=float(self.spin_dy.value()),
            nx=int(self.spin_nx.value()),
            ny=int(self.spin_ny.value()),
            scan_pattern=str(self.combo_pattern.currentData() or "snake"),
        )

    def _on_set_start_from_stage(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Connect the ASI MS-2000 stage first.")
            return
        pos = self.stage.get_position()
        if pos.get("X") is not None:
            self.spin_x_start.setValue(float(pos["X"]))
        if pos.get("Y") is not None:
            self.spin_y_start.setValue(float(pos["Y"]))
        self.sig_log.emit(f"Set Map Start (X₀, Y₀) to ({pos['X']:.2f}, {pos['Y']:.2f}) µm.")

    def _on_teach_corner_a(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Connect the ASI MS-2000 stage first.")
            return
        pos = self.stage.get_position()
        self._corner_a = (float(pos["X"]), float(pos["Y"]))
        self.spin_x_start.setValue(self._corner_a[0])
        self.spin_y_start.setValue(self._corner_a[1])
        self.sig_log.emit(f"Taught Corner A = ({self._corner_a[0]:.2f}, {self._corner_a[1]:.2f}) µm.")

    def _on_teach_corner_b(self):
        if not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(self, "Stage Not Connected", "Connect the ASI MS-2000 stage first.")
            return
        if self._corner_a is None:
            QtWidgets.QMessageBox.information(
                self, "Teach Corner A First", "Please move to the first corner and click 'Teach Corner A' first."
            )
            return
        pos = self.stage.get_position()
        self._corner_b = (float(pos["X"]), float(pos["Y"]))
        cfg = MapGridConfig.from_corners(
            x_a_um=self._corner_a[0],
            y_a_um=self._corner_a[1],
            x_b_um=self._corner_b[0],
            y_b_um=self._corner_b[1],
            dx_um=self.spin_dx.value(),
            dy_um=self.spin_dy.value(),
            scan_pattern=str(self.combo_pattern.currentData() or "snake"),
        )
        self.spin_x_start.setValue(cfg.x_start_um)
        self.spin_y_start.setValue(cfg.y_start_um)
        self.spin_dx.setValue(cfg.dx_um)
        self.spin_dy.setValue(cfg.dy_um)
        self.spin_nx.setValue(cfg.nx)
        self.spin_ny.setValue(cfg.ny)
        self.sig_log.emit(
            f"Taught Corner B = ({self._corner_b[0]:.2f}, {self._corner_b[1]:.2f}) µm -> "
            f"Computed Grid: {cfg.nx}x{cfg.ny} points, step=({cfg.dx_um:.2f}, {cfg.dy_um:.2f}) µm."
        )

    def _update_map_summary(self):
        cfg = self.get_current_grid_config()
        if self.camera.is_connected:
            roi = self.camera.get_roi()
            out_w = (roi[1] - roi[0]) // max(1, roi[4])
            out_h = (roi[3] - roi[2]) // max(1, roi[5])
            frame_shape = (out_h, out_w)
        else:
            frame_shape = (2048, 2048)

        raw_bytes = cfg.estimate_raw_size_bytes(frame_shape)
        bin_h = max(1, int(self.spin_bin_h.value())) if hasattr(self, "spin_bin_h") else 4
        bin_w = max(1, int(self.spin_bin_w.value())) if hasattr(self, "spin_bin_w") else 2
        n_rows = frame_shape[0] // bin_h
        n_orders = 11
        # 6 float64 4D arrays (Shift, Shift_err, LW, LW_err, x_coords, y_coords)
        treated_bytes = cfg.total_points * n_rows * (2 * n_orders) * 8 * 6
        est_sec = cfg.estimate_duration_seconds(
            exposure_s=float(self.spin_map_exp.value()) / 1000.0,
            n_avg=int(self.spin_map_avg.value()),
            settling_s=float(self.spin_map_settling.value()) / 1000.0,
        )
        raw_status = (
            f"SAVED ({_format_bytes(raw_bytes)})"
            if self.chk_save_raw.isChecked()
            else f"NOT SAVED (saves {_format_bytes(raw_bytes)} of disk space)"
        )
        otf_checked = getattr(self, "chk_treat_on_the_fly", None)
        otf_status = (
            f"ENABLED (~{_format_bytes(treated_bytes)} in Treated {bin_h}×{bin_w} bins)"
            if (otf_checked is None or otf_checked.isChecked())
            else "DISABLED"
        )
        self.lbl_map_summary.setText(
            f"Grid: {cfg.nx} × {cfg.ny} = {cfg.total_points} points   |   "
            f"Span: {cfg.span_x_um:.1f} × {cfg.span_y_um:.1f} µm   |   "
            f"Est. Time: {_format_duration(est_sec)}\n"
            f"On-the-Fly Fitted DHO Dataset: {otf_status}   |   "
            f"Raw 4D Dataset: {raw_status}"
        )

    # ==========================================================================
    # Event Handlers: Map Scan Execution
    # ==========================================================================

    def _on_start_map(self):
        if not self.camera.is_connected or not self.stage.is_connected:
            QtWidgets.QMessageBox.warning(
                self,
                "Hardware Not Connected",
                "Please connect both the Camera (Orca Flash 4.0 v3) and the Stage (ASI MS-2000) before starting a map.",
            )
            return

        if not self._ensure_h5_open():
            return

        # Prompt user if Laser Reference or Rb Absorption is missing
        missing = []
        if self.session.laser_image is None:
            missing.append("1. Laser Reference (Brillouin/Calibration/laser_reference/image)")
        if self.session.rb_image is None:
            missing.append("2. Rubidium Absorption (Brillouin/Calibration/Rb_absorption/image)")

        if missing:
            missing_list = "\n  • ".join(missing)
            reply = QtWidgets.QMessageBox.question(
                self,
                "Calibration Spectra Missing",
                f"The following calibration spectra have not been acquired yet:\n  • {missing_list}\n\n"
                "Usually you should acquire Laser Reference and Rb Absorption in Section 4B before starting the map.\n"
                "Do you want to stop and acquire them first (Recommended), or proceed with an uncalibrated raw scan?",
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes,
                QtWidgets.QMessageBox.Cancel,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        if self._is_busy():
            return

        self._stop_live_cb()
        self.sig_lock_geometry.emit(True)
        self._set_controls_busy(True, map_running=True)

        cfg = self.get_current_grid_config()
        self.progress_map.setRange(0, cfg.total_points)
        self.progress_map.setValue(0)
        self.progress_map.setFormat(f"Starting map (0 / {cfg.total_points})...")

        self._map_paused = False
        self.btn_pause_map.setText("⏸ Pause")

        fit_params = {
            "bin_val_h": int(self.spin_bin_h.value()),
            "bin_val_w": int(self.spin_bin_w.value()),
            "expected_shift_ghz": float(self.spin_dho_shift.value()),
            "shift_tol_ghz": float(self.spin_dho_tol.value()),
            "expected_lw_ghz": float(self.spin_dho_lw.value()),
            "lw_bounds_ghz": (float(self.spin_dho_lw_min.value()), float(self.spin_dho_lw_max.value())),
            "refractive_index": float(self.spin_q_n.value()),
            "fast_drift_only": bool(self.chk_fast_drift.isChecked()),
        }

        self._map_worker = MapScanWorker(
            session=self.session,
            camera=self.camera,
            stage=self.stage,
            map_name=self.txt_map_name.text().strip() or "Map_01",
            grid=cfg,
            exposure_ms=self.spin_map_exp.value(),
            n_avg=self.spin_map_avg.value(),
            settling_ms=self.spin_map_settling.value(),
            save_raw_data=self.chk_save_raw.isChecked(),
            on_the_fly_treatment=self.chk_treat_on_the_fly.isChecked(),
            fit_params=fit_params,
            parent=self,
        )
        self._map_worker.sig_log.connect(self.sig_log.emit)
        self._map_worker.sig_progress.connect(self._on_map_progress)
        self._map_worker.sig_stage_updated.connect(self.sig_stage_updated.emit)
        self._map_worker.sig_point_acquired.connect(self.sig_display_frame.emit)
        self._map_worker.sig_live_map_updated.connect(self.sig_update_live_map.emit)
        self._map_worker.sig_finished_map.connect(self._on_map_finished)
        self._map_worker.sig_error.connect(self._on_worker_error)
        self._map_worker.finished.connect(self._on_map_worker_thread_done)
        self._map_worker.start()

    def _on_toggle_pause_map(self):
        if self._map_worker is None or not self._map_worker.isRunning():
            return
        self._map_paused = not self._map_paused
        self._map_worker.request_pause(self._map_paused)
        if self._map_paused:
            self.btn_pause_map.setText("▶ Resume")
            self.sig_log.emit("Map scan PAUSED.")
        else:
            self.btn_pause_map.setText("⏸ Pause")
            self.sig_log.emit("Map scan RESUMED.")

    def _on_abort_map(self):
        if self._map_worker is not None and self._map_worker.isRunning():
            self._map_worker.request_abort()

    @QtCore.Slot(int, int, int, int, float, float)
    def _on_map_progress(self, step_idx: int, total: int, iy: int, ix: int, x_um: float, y_um: float):
        self.progress_map.setValue(step_idx)
        self.progress_map.setFormat(
            f"Point {step_idx}/{total} [row={iy}, col={ix}] @ ({x_um:.1f}, {y_um:.1f}) µm — %p%"
        )

    @QtCore.Slot(str, int)
    def _on_map_finished(self, map_group_path: str, completed: int):
        self.sig_log.emit(f"Finished Map '{map_group_path}' ({completed} point(s) saved to HDF5).")
        self.sig_camera_updated.emit()
        self.sig_stage_updated.emit()
        # Refresh HDF5 status banner
        if self.session.h5_path:
            self._on_open_or_create_h5()

    def _on_map_worker_thread_done(self):
        self._set_controls_busy(False, map_running=False)

    @QtCore.Slot(str, str)
    def _on_worker_error(self, title: str, message: str):
        self.sig_log.emit(f"ERROR ({title}): {message}")
        QtWidgets.QMessageBox.warning(self, title, message)

    def _is_busy(self) -> bool:
        if self._cal_worker is not None and self._cal_worker.isRunning():
            return True
        if self._laser_calib_worker is not None and self._laser_calib_worker.isRunning():
            return True
        if self._rb_calib_worker is not None and self._rb_calib_worker.isRunning():
            return True
        if self._sample_fit_worker is not None and self._sample_fit_worker.isRunning():
            return True
        if self._map_worker is not None and self._map_worker.isRunning():
            return True
        return False

    def _set_controls_busy(self, busy: bool, map_running: bool = False):
        for w in (
            self.btn_browse_h5,
            self.btn_open_h5,
            self.btn_acq_laser,
            self.btn_acq_rb,
            self.btn_acq_ref,
            self.btn_run_laser_calib,
            self.btn_run_rb_calib,
            self.btn_run_sample_fit,
            self.btn_start_map,
            self.btn_set_start_current,
            self.btn_teach_a,
            self.btn_teach_b,
        ):
            w.setEnabled(not busy)
        if not busy:
            self._update_calibration_badges()

        self.btn_pause_map.setEnabled(map_running)
        self.btn_abort_map.setEnabled(map_running)
