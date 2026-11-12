"""
workers.py: Asynchronous Background Execution and Pipeline Workers for arVIPAspectro GUI.
"""

import os
import sys
import io
import time
import traceback
from typing import Dict, Any, Optional, Callable, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PySide6 import QtCore

try:
    from .. import config
    from .. import backend as bck
    from .. import laser_calib
    from .. import Rb_calib
    from .. import spectrum_fitting
    from .. import q_vector
    from .theme import MplTheme, get_mpl_theme, get_system_theme, apply_axis_theme
except ImportError:
    import config
    import backend as bck
    import laser_calib
    import Rb_calib
    import spectrum_fitting
    import q_vector
    from theme import MplTheme, get_mpl_theme, get_system_theme, apply_axis_theme


# ==============================================================================
# Worker Signals & IO Redirector
# ==============================================================================

class WorkerSignals(QtCore.QObject):
    """Signals emitted by the background pipeline thread."""
    started = QtCore.Signal(str)                       # task_name
    progress = QtCore.Signal(int, int, str)            # current, total, status_message
    log = QtCore.Signal(str)                           # streamed text line
    view_ready = QtCore.Signal(str, dict)              # title, {view_name: callable(fig)}
    finished = QtCore.Signal(str, dict)                # task_name, results_dict
    error = QtCore.Signal(str, str)                    # task_name, traceback_str


class StreamToSignal(io.TextIOBase):
    """Redirects writes from sys.stdout / sys.stderr to a Qt Signal."""

    def __init__(self, signal: QtCore.Signal, original_stream):
        super().__init__()
        self.signal = signal
        self.original_stream = original_stream

    def write(self, s: str):
        if s:
            self.signal.emit(s)
            if self.original_stream:
                self.original_stream.write(s)
        return len(s)

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()


# ==============================================================================
# Main Pipeline Worker Thread
# ==============================================================================

class PipelineWorker(QtCore.QThread):
    """
    Background worker thread executing calibration, fitting, and analysis steps
    without blocking the PySide6 UI.
    """

    def __init__(self, task_type: str, h5_path: str, params: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.task_type = task_type
        self.h5_path = h5_path
        self.params = params
        self.signals = WorkerSignals()
        self._is_aborted = False

    def abort(self):
        self._is_aborted = True

    def run(self):
        self.signals.started.emit(self.task_type)
        t_start = time.time()

        # Redirect stdout and stderr during task execution
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirector_out = StreamToSignal(self.signals.log, old_stdout)
        redirector_err = StreamToSignal(self.signals.log, old_stderr)
        sys.stdout = redirector_out
        sys.stderr = redirector_err

        try:
            if not os.path.exists(self.h5_path):
                raise FileNotFoundError(f"HDF5 file does not exist: {self.h5_path}")

            wrapper = bck.create_wrapper(os.path.dirname(self.h5_path), name=os.path.basename(self.h5_path))

            if self.task_type == "laser_all":
                results, views = self._run_laser_all(wrapper)
            elif self.task_type == "rb_all":
                results, views = self._run_rb_all(wrapper)
            elif self.task_type == "peak_fit":
                results, views = self._run_peak_fit(wrapper)
            elif self.task_type == "q_vector":
                results, views = self._run_q_vector(wrapper)
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

            elapsed = time.time() - t_start
            print(f"\n✓ Completed '{self.task_type}' in {elapsed:.2f} seconds.")

            if views:
                self.signals.view_ready.emit(self.task_type, views)
            self.signals.finished.emit(self.task_type, results)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n❌ Error during '{self.task_type}':\n{tb}")
            self.signals.error.emit(self.task_type, tb)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    # --------------------------------------------------------------------------
    # Laser Pipeline (Steps 1 to 6)
    # --------------------------------------------------------------------------

    def _run_laser_all(self, wrapper) -> tuple[Dict[str, Any], Dict[str, Callable[[Figure], None]]]:
        laser_group = self.params.get("laser_group", "laser_reference")
        path_laser = f"Brillouin/{laser_group}"
        scale = int(self.params.get("scale_laser", 5))
        ksize_se = int(self.params.get("ksize_se_laser", 11))
        ksize_blur = int(self.params.get("ksize_blur_laser", 5))
        poly_deg = int(self.params.get("poly_deg", 4))

        self.signals.progress.emit(1, 6, "Step 1: Reading Laser Reference Image...")
        data_laser = wrapper[f"{path_laser}/image"].astype(np.float64)

        self.signals.progress.emit(2, 6, "Step 2: Laser Denoising & Morphological Baseline...")
        data_laser_corr, baseline_laser = laser_calib.denoise_laser(
            wrapper, path_laser, data_laser,
            scale=scale, ksize_structuring_elements=ksize_se, ksize_blur=ksize_blur
        )

        self.signals.progress.emit(3, 6, "Step 3: Extracting Laser Fringes...")
        peaks, average_data, points = laser_calib.extract_laser_fringes(
            wrapper, path_laser, data_laser_corr
        )

        self.signals.progress.emit(4, 6, "Step 4: Unsaturated Sub-pixel Refinement...")
        all_x_laser, all_x_smooth, all_y_laser, all_k_laser = laser_calib.refine_laser_fringes(
            wrapper, path_laser, points, data_laser_corr, baseline_laser
        )

        self.signals.progress.emit(5, 6, "Step 5: Exact 2D Gnomonic VIPA Geometry Fit...")
        geom_results = laser_calib.fit_vipa_exact_gnomonic(
            wrapper, path_laser, all_x_smooth, all_y_laser, all_k_laser, data_laser_corr
        )
        xc_gnom = geom_results["xc_gnom"]
        yc_gnom = geom_results["yc_gnom"]
        y0_fit = geom_results["y0_fit"]
        x0_fit = geom_results["x0_fit"]

        self.signals.progress.emit(6, 6, "Step 6: 2D Anamorphic Surface Calibration...")
        surf_results = laser_calib.fit_2D_anamorphic_calibration(
            wrapper, path_laser, all_x_smooth, all_y_laser, all_k_laser,
            x0_fit=x0_fit, y0_fit=y0_fit, xc_gnom=xc_gnom, yc_gnom=yc_gnom,
            poly_degree=poly_deg
        )

        # Build interactive figure renderers for Left Visualizer
        views: Dict[str, Callable] = {}

        # View 1: Raw & Denoised Image with Detected Fringes
        def render_fringes(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(get_system_theme())
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax = fig.add_subplot(111)
            apply_axis_theme(ax, t, title="Step 3-5: Refined Laser Fringes & Centerline", xlabel="X (px)", ylabel="Y (px)")
            im = ax.imshow(data_laser_corr, origin="lower", aspect="auto", cmap="inferno")
            fringe_color = "#00e5ff" if t.ax_bg != "#fafafa" else "#0284c7"
            for k in np.unique(all_k_laser):
                idx = (all_k_laser == k)
                ax.plot(all_x_smooth[idx], all_y_laser[idx], "-", color=fringe_color, lw=1.2, alpha=0.8)
            ax.plot(xc_gnom, yc_gnom, "r+", markersize=14, markeredgewidth=2, label="VIPA Center")
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.ax.tick_params(colors=t.text_tick)
            cbar.outline.set_edgecolor(t.spine)

        # View 2: Anamorphic Surface Fit Order Prediction
        def render_surface(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(get_system_theme())
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax = fig.add_subplot(111)
            apply_axis_theme(ax, t, title=f"Step 6: 2D Anamorphic Calibration (RMSE={surf_results['rmse_order']:.4f})", xlabel="X (px)", ylabel="Y (px)")
            q_orders = surf_results["q_orders"]
            sc = ax.scatter(all_x_smooth, all_y_laser, c=q_orders, cmap="viridis", s=6, label="Detected Orders")
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, label="VIPA Order q")
            cbar.ax.tick_params(colors=t.text_tick)
            cbar.outline.set_edgecolor(t.spine)

        views["Laser Fringes & Center"] = render_fringes
        views["2D Anamorphic Surface"] = render_surface

        results = {
            "xc_gnom": xc_gnom,
            "yc_gnom": yc_gnom,
            "y0_fit": y0_fit,
            "x0_fit": x0_fit,
            "rmse_order": surf_results["rmse_order"],
        }
        return results, views

    # --------------------------------------------------------------------------
    # Rubidium Pipeline (Steps 7 to 10B)
    # --------------------------------------------------------------------------

    def _run_rb_all(self, wrapper) -> tuple[Dict[str, Any], Dict[str, Callable[[Figure], None]]]:
        rb_group = self.params.get("rb_group", "Rb_absorption")
        laser_group = self.params.get("laser_group", "laser_reference")
        path_rb = f"Brillouin/{rb_group}"
        path_laser = f"Brillouin/{laser_group}"

        t_cell = float(self.params.get("t_cell", 70.0))
        l_cell = float(self.params.get("l_cell", 0.07))
        scale_rb = int(self.params.get("scale_rb", 9))
        ksize_se_rb = int(self.params.get("ksize_se_rb", 3))
        ksize_blur_rb = int(self.params.get("ksize_blur_rb", 11))
        offset_guess = float(self.params.get("laser_offset_guess", 0.0))
        fit_win_ghz = float(self.params.get("fit_window_ghz", 10.0))
        lock_offset = float(self.params.get("rb_lock_offset", 1.323))

        # Retrieve Step 5 & 6 calibration parameters from laser group
        laser_cal = laser_calib.retrieve_anamorphic_surface_calibration(wrapper, path_laser)
        if laser_cal is None:
            raise RuntimeError(f"Laser surface calibration not found in '{path_laser}'. Run Laser Calibration first!")

        xc_gnom = laser_cal["xc_gnom"]
        yc_gnom = laser_cal["yc_gnom"]
        x0_fit = laser_cal["x0_fit"]
        y0_fit = laser_cal["y0_fit"]
        c_2d_order = laser_cal["c_2d_order"]
        g_bar = laser_cal["g_bar"]
        q_orders = laser_cal["q_orders"]
        g_mean_ref = laser_cal["g_mean_ref"]
        g_std_ref = laser_cal["g_std_ref"]
        L_y = laser_cal["L_y"]

        self.signals.progress.emit(1, 5, "Step 7: Creating Theoretical Rubidium D2 Model...")
        det_table, trans_table, trans_fast = Rb_calib.create_rubidium_reference_table(
            T_cell=t_cell, L_cell=l_cell
        )

        self.signals.progress.emit(2, 5, "Step 8: Rb Absorption Denoising...")
        data_raw = wrapper[f"{path_rb}/image"].astype(np.float64)
        data_absorption, baseline_absorption = Rb_calib.denoise_rb(
            wrapper, path_rb, data_raw,
            scale=scale_rb, ksize_structuring_elements=ksize_se_rb, ksize_blur=ksize_blur_rb
        )

        self.signals.progress.emit(3, 5, "Step 9: Radial Profile Extraction...")
        phi_order_r, avg_spectrum_radius = Rb_calib.extract_radial_absorption_profile(
            data_absorption, xc_gnom, yc_gnom, y0_fit,
            q_orders=q_orders, g_bar=g_bar, r_max=1200, dr=1.0
        )

        self.signals.progress.emit(4, 5, "Step 10A: Calibrating FSR & Frequency Offset...")
        fsr_results = Rb_calib.calibrate_fsr_and_offset(
            wrapper, path_rb, g_bar, q_orders, phi_order_r,
            avg_spectrum_radius, trans_fast,
            fit_window_ghz=fit_win_ghz,
            delta_nu_laser_offset=offset_guess
        )
        FSR_cal = fsr_results["FSR_cal"]
        delta_nu_opt = fsr_results["delta_nu_opt"]

        self.signals.progress.emit(5, 5, "Step 10B: Computing 2D Frequency Map...")
        freq_map = Rb_calib.compute_and_save_frequency_map(
            wrapper, path_rb, data_raw.shape, x0_fit, y0_fit, xc_gnom, yc_gnom,
            c_2d_order, g_mean_ref, g_std_ref, L_y, FSR_cal, delta_nu_opt,
            rb_lock_offset_ghz=lock_offset
        )

        views: Dict[str, Callable] = {}

        # View 1: 2D Calibrated Frequency Map
        def render_freq_map(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(get_system_theme())
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax = fig.add_subplot(111)
            apply_axis_theme(ax, t, title=f"Step 10B: Calibrated Frequency Map (FSR = {FSR_cal:.4f} GHz)", xlabel="X (px)", ylabel="Y (px)")
            im = ax.imshow(freq_map, origin="lower", aspect="auto", cmap="turbo")
            cbar = fig.colorbar(im, ax=ax, pad=0.02, label="Frequency Relative to Laser (GHz)")
            cbar.ax.tick_params(colors=t.text_tick)
            cbar.outline.set_edgecolor(t.spine)

        # View 2: FSR Absorption Spectrum Matching
        def render_fsr_match(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(get_system_theme())
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax = fig.add_subplot(111)
            apply_axis_theme(ax, t, title=f"Step 10A: Rb D2 Spectrum Matching (FSR = {FSR_cal:.4f} GHz)", xlabel="Frequency Detuning (GHz)", ylabel="Transmission")
            f_grid = fsr_results["f_grid"]
            order_spectra = fsr_results["order_spectra"]
            f_valid = fsr_results["f_valid"]
            sig_valid = fsr_results["sig_valid"]
            f_dense = fsr_results["f_dense"]
            theo_dense = fsr_results["theo_dense"]

            theo_color = "#374151" if t.ax_bg == "#fafafa" else "#cccccc"
            ax.plot(f_dense, theo_dense, "-", color=theo_color, lw=2.0, alpha=0.7, label="Theoretical Rb D2")
            for ord_idx, s in order_spectra.items():
                ax.plot(f_grid, s, lw=1.0, alpha=0.5, label=f"Order {ord_idx}")
            ax.plot(f_valid, sig_valid, "r.", markersize=4, label="Folded Average")
            ax.legend(loc="best", fontsize=8, facecolor=t.legend_bg, edgecolor=t.legend_edge)

        views["2D Frequency Map"] = render_freq_map
        views["Rb FSR Calibration Fit"] = render_fsr_match

        results = {
            "FSR_cal": FSR_cal,
            "delta_nu_opt": delta_nu_opt,
        }
        return results, views

    # --------------------------------------------------------------------------
    # Sample Peak Fitting Pipeline (Steps 11 to 13)
    # --------------------------------------------------------------------------

    def _run_peak_fit(self, wrapper) -> tuple[Dict[str, Any], Dict[str, Callable[[Figure], None]]]:
        sample_name = self.params.get("sample_name", "water_5s")
        rb_group = self.params.get("rb_group", "Rb_absorption")
        path_sample = f"Brillouin/{sample_name}"
        path_rb = f"Brillouin/{rb_group}"

        bin_h = int(self.params.get("bin_val_h", 4))
        bin_w = int(self.params.get("bin_val_w", 2))
        ds_init = float(self.params.get("expected_shift", 5.1))
        shift_tol = float(self.params.get("shift_tol", 2.0))
        gamma_init = float(self.params.get("expected_lw", 0.8))
        lw_min = float(self.params.get("lw_min", 0.4))
        lw_max = float(self.params.get("lw_max", 1.3))

        # Retrieve frequency map
        freq_results = Rb_calib.retrieve_frequency_map(wrapper, path_rb)
        if freq_results is None:
            raise RuntimeError(f"2D Frequency map not found in '{path_rb}'. Run Rubidium calibration first!")
        freq_map = freq_results["frequency_map"]

        self.signals.progress.emit(1, 3, f"Step 11: Loading & Binning Sample '{sample_name}'...")
        data_sample = wrapper[f"{path_sample}/image"].astype(np.float64)

        img_h, img_w = data_sample.shape
        grid_y, grid_x = np.indices((img_h, img_w))

        self.signals.progress.emit(2, 3, f"Step 12: Vectorized DHO Peak Fitting ({bin_h}x{bin_w} bins)...")
        dho_results = spectrum_fitting.fit_sample_dho(
            data_sample=data_sample,
            freq_map=freq_map,
            xx=grid_x,
            yy=grid_y,
            fsr_val=freq_results.get("FSR_cal", 39.8),
            bin_val_h=bin_h,
            bin_val_w=bin_w,
            ds_init=ds_init,
            shift_tol=shift_tol,
            gamma_init=gamma_init,
            gamma_bounds=(lw_min, lw_max),
            verbose=True
        )

        self.signals.progress.emit(3, 3, "Step 13: Storing Treated Peak Data to HDF5...")
        spectrum_fitting.save_treated_data(
            wrapper=wrapper,
            path_sample=path_sample,
            bin_val_h=bin_h,
            bin_val_w=bin_w,
            shift_rel=dho_results["shift_clipped"],
            shift_err=dho_results["shift_error"],
            lw_rel=dho_results["linewidth_clipped"],
            lw_err=dho_results["linewidth_error"],
            x_coords=dho_results["x_coords"],
            y_coords=dho_results["y_coords"]
        )

        views: Dict[str, Callable[[Figure], None]] = {}

        # View 1: Spatial Maps of Brillouin Shift & Linewidth
        def render_maps(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(get_system_theme())
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            apply_axis_theme(ax1, t, title=f"Brillouin Shift (GHz)\n{sample_name}")
            apply_axis_theme(ax2, t, title=f"Brillouin Linewidth (GHz)\n{sample_name}")

            shift_arr = dho_results["shift_clipped"]
            lw_arr = dho_results["linewidth_clipped"]
            p1_s, p99_s = np.nanpercentile(shift_arr[np.isfinite(shift_arr)], (2, 98))
            p1_l, p99_l = np.nanpercentile(lw_arr[np.isfinite(lw_arr)], (2, 98))

            im1 = ax1.imshow(shift_arr, origin="lower", aspect="auto", cmap="viridis", vmin=p1_s, vmax=p99_s)
            cbar1 = fig.colorbar(im1, ax=ax1, pad=0.03)
            cbar1.ax.tick_params(colors=t.text_tick)
            cbar1.outline.set_edgecolor(t.spine)

            im2 = ax2.imshow(lw_arr, origin="lower", aspect="auto", cmap="inferno", vmin=p1_l, vmax=p99_l)
            cbar2 = fig.colorbar(im2, ax=ax2, pad=0.03)
            cbar2.ax.tick_params(colors=t.text_tick)
            cbar2.outline.set_edgecolor(t.spine)

        views["Shift & Linewidth Maps"] = render_maps

        results = {
            "n_fringes": dho_results["n_fringes"],
            "rms_mhz": dho_results["stokes_antistokes_rms_mhz"],
        }
        return results, views

    # --------------------------------------------------------------------------
    # Acoustic Wavevector (q-vector) Pipeline (Steps 14 & 15)
    # --------------------------------------------------------------------------

    def _run_q_vector(self, wrapper) -> tuple[Dict[str, Any], Dict[str, Callable[[Figure], None]]]:
        sample_name = self.params.get("sample_name", "water_5s")
        path_sample = f"Brillouin/{sample_name}"

        bin_h = int(self.params.get("bin_val_h", 4))
        bin_w = int(self.params.get("bin_val_w", 2))
        n_sample = float(self.params.get("refractive_index", 1.33))
        shift_min = float(self.params.get("shift_min", 4.8))
        shift_max = float(self.params.get("shift_max", 5.3))
        lw_min = float(self.params.get("lw_min", 0.5))
        lw_max = float(self.params.get("lw_max", 1.2))
        use_errors = bool(self.params.get("use_errors", True))
        err_floor = float(self.params.get("error_percentile_floor", 1.0))
        avg_pairs = bool(self.params.get("average_pairs", True))
        init_m_theta = float(self.params.get("initial_m_theta", 3000.0))
        fixed_angle = self.params.get("fixed_ellipse_angle_deg", 90.0)

        self.signals.progress.emit(1, 3, f"Step 14: Loading DHO fit data for '{sample_name}'...")
        dho_results = spectrum_fitting.retrieve_sample_dho_fit(
            wrapper=wrapper,
            path_sample=path_sample,
            bin_val_h=bin_h,
            bin_val_w=bin_w
        )
        if dho_results is None:
            raise RuntimeError(f"Treated DHO fit data not found for '{path_sample}'. Run Peak Fitting first!")

        self.signals.progress.emit(2, 3, "Step 14: Ellipse Optimization & Acoustic Dispersion...")
        q_results = q_vector.extract_q_vector(
            data_source=dho_results,
            path_sample=path_sample,
            wrapper=wrapper,
            bin_val_h=bin_h,
            bin_val_w=bin_w,
            refractive_index=n_sample,
            laser_wavelength=config.lambda_0,
            use_errors=use_errors,
            error_percentile_floor=err_floor,
            average_pairs=avg_pairs,
            shift_bounds=(shift_min, shift_max),
            linewidth_bounds=(lw_min, lw_max),
            initial_M_theta=init_m_theta,
            fixed_ellipse_angle_deg=fixed_angle,
            verbose=True
        )

        self.signals.progress.emit(3, 3, "Step 15: Storing Calibrated Dispersion to HDF5...")
        q_vector.save_q_vector_results(
            wrapper=wrapper,
            path_sample=path_sample,
            bin_val_h=bin_h,
            bin_val_w=bin_w,
            q_results=q_results
        )

        views: Dict[str, Callable] = {}

        # View 1: Dispersion Curves (Shift(q) and Linewidth(q))
        def render_dispersion(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(get_system_theme())
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            apply_axis_theme(ax1, t, title=f"Acoustic Dispersion Shift(q)\n{sample_name}", xlabel="q (rad/µm)", ylabel="Brillouin Shift (GHz)")
            apply_axis_theme(ax2, t, title=f"Brillouin Linewidth(q)\n{sample_name}", xlabel="q (rad/µm)", ylabel="Linewidth FWHM (GHz)")

            qv = q_results["q_vector_um"]
            shift_data = q_results["shift"]
            lw_data = q_results["linewidth"]
            vs = q_results["V_s"]
            vs_err = q_results.get("V_s_err", 0.0)

            # Subplot 1: Shift(q)
            dot_color = "#00bcd4" if t.ax_bg != "#fafafa" else "#0284c7"
            ax1.scatter(qv, shift_data, s=6, color=dot_color, alpha=0.5, label="Data")
            q_line = np.linspace(np.nanmin(qv), np.nanmax(qv), 200)
            shift_line = (vs * (q_line * 1e6) / (2.0 * np.pi)) * 1e-9
            ax1.plot(q_line, shift_line, "r-", lw=1.8, label=f"V_s = {vs:.1f} ± {vs_err:.1f} m/s")
            ax1.grid(True, linestyle=":", alpha=t.grid_alpha, color=t.grid)
            ax1.legend(loc="best", fontsize=9, facecolor=t.legend_bg, edgecolor=t.legend_edge)

            # Subplot 2: Linewidth(q)
            ax2.scatter(qv, lw_data, s=6, color="#f97316", alpha=0.5, label="Linewidth")
            ax2.grid(True, linestyle=":", alpha=t.grid_alpha, color=t.grid)
            ax2.legend(loc="best", fontsize=9, facecolor=t.legend_bg, edgecolor=t.legend_edge)

        # View 2: Complete 4-Panel Diagnostic Suite
        def render_diagnostics(fig: Figure, theme: Optional[MplTheme] = None):
            t = theme or get_mpl_theme(get_system_theme())
            fig.clf()
            fig.patch.set_facecolor(t.fig_bg)
            axs = fig.subplots(2, 2)
            for ax_row in axs:
                for ax in ax_row:
                    apply_axis_theme(ax, t)

            opt = q_results["opt_params"]
            xc = opt["xc"]
            yc = opt["yc"]
            phi_deg = opt["phi_deg"]
            gamma = opt["gamma"]

            # Panel (0,0): Original Shift(x, y) with fitted concentric ellipses
            axs[0, 0].scatter(q_results["x_coords"], q_results["y_coords"], c=q_results["shift"], cmap="viridis", s=5)
            axs[0, 0].plot(xc, yc, "r+", markersize=12, markeredgewidth=2)
            axs[0, 0].set_title(f"Center ({xc:.1f}, {yc:.1f}) px | φ={phi_deg:.1f}° | γ={gamma:.3f}", color=t.text_title, fontsize=10)

            # Panel (0,1): Spherical projection Shift(r_sph)
            sph_dot_color = "#00bcd4" if t.ax_bg != "#fafafa" else "#0284c7"
            axs[0, 1].scatter(q_results["r_sph"], q_results["shift"], s=4, color=sph_dot_color, alpha=0.4)
            axs[0, 1].set_title("Shift vs Spherical Radius r_sph", color=t.text_title, fontsize=10)
            axs[0, 1].set_xlabel("r_sph (px)", color=t.text_label)

            # Panel (1,0): Scattering Angle θ Distribution
            axs[1, 0].hist(q_results["theta_deg"], bins=40, color="#0d9488", edgecolor=t.spine)
            axs[1, 0].set_title("Scattering Angle θ (deg) Distribution", color=t.text_title, fontsize=10)

            # Panel (1,1): Acoustic Dispersion V_s
            vs = q_results["V_s"]
            vs_err = q_results.get("V_s_err", 0.0)
            status_text_color = "#16a34a" if t.ax_bg == "#fafafa" else "#4caf50"
            axs[1, 1].text(0.5, 0.5, f"V_s = {vs:.2f} ± {vs_err:.2f} m/s\nM_θ = {q_results['M_theta']:.1f} ± {q_results.get('M_theta_err', 0.0):.1f} px\nShift(0) = {q_results['Shift_0']:.4f} GHz",
                           ha="center", va="center", color=status_text_color, fontsize=13, fontweight="bold")
            axs[1, 1].axis("off")
            axs[1, 1].axis("off")

        views["Acoustic Dispersion Shift(q)"] = render_dispersion
        views["4-Panel Diagnostic Suite"] = render_diagnostics

        results = {
            "V_s": q_results["V_s"],
            "V_s_err": q_results.get("V_s_err"),
            "M_theta": q_results["M_theta"],
            "Shift_0": q_results["Shift_0"],
            "phi_deg": q_results["opt_params"]["phi_deg"],
            "gamma": q_results["opt_params"]["gamma"],
        }
        return results, views
