"""
Main VIPA Brillouin calibration pipeline orchestrator.
Executes laser reference calibration and Rubidium absorption calibration steps sequentially.
"""

import os
import sys

# Ensure package directory is on path if main.py is run directly
_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import numpy as np

import arVIPAspectro as ar
from arVIPAspectro import config
from arVIPAspectro import backend as bck
from arVIPAspectro import laser_calib
from arVIPAspectro import Rb_calib
from arVIPAspectro import spectrum_fitting
from arVIPAspectro import q_vector
from arVIPAspectro import plotting


# ==============================================================================
# Paths & Execution Configuration
# ==============================================================================

DATA_DIR = '/Volumes/LAUDATE/Data/260916 - arBLS/261105'
TARGET_FILE = 'Measures.h5'
FIGURE_DIRECTORY_NAME = 'figures'
OUT_DIR = os.path.join(DATA_DIR, FIGURE_DIRECTORY_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

FIGURE_PREFIX = ''
SHOW_PLOTS = True
FORCE_ALL_PROCESS = 1
FORCE_PLOT_ALL = 1

# Execution flags per step: (plot_*, stop_*, force_*)
# Step 2: Laser Denoising
plot_denoise_laser = FORCE_PLOT_ALL or 0
stop_denoise_laser = 0
force_denoise_laser = FORCE_ALL_PROCESS or 0

# Step 3: Laser Fringe Detection & Tracing
plot_fringe_laser = FORCE_PLOT_ALL or 0
stop_fringe_laser = 0
force_extract_fringe = FORCE_ALL_PROCESS or 0

# Step 4: Unsaturated Sub-Pixel Refinement
plot_refine_laser = FORCE_PLOT_ALL or 0
stop_refine_laser = 0
force_refine_laser = FORCE_ALL_PROCESS or 0

# Step 5: Exact 2D Gnomonic VIPA Geometry Fit
plot_concentric_fringe = FORCE_PLOT_ALL or 0
stop_concentric_fringe = 0
force_concentric_fringe = FORCE_ALL_PROCESS or 0

# Step 6: 2D Anamorphic Polynomial Order Surface Calibration
lower_order_study = 0
plot_2D_anamorphic_cal = FORCE_PLOT_ALL or 0
stop_2D_anamorphic_cal = 0

# Step 7: Theoretical Rubidium Spectrum Model
plot_Rb_absorption = FORCE_PLOT_ALL or 0
stop_Rb_absorption = 0

# Step 8: Rubidium Absorption Preprocessing & Denoising
plot_denoise_Rb = FORCE_PLOT_ALL or 0
stop_denoise_Rb = 0
force_denoise_Rb = FORCE_ALL_PROCESS or 0

# Step 9: Radial Profile Extraction of Rubidium Absorption
plot_Rb_experimental = FORCE_PLOT_ALL or 0
stop_Rb_experimental = 0

# Step 10: Calibration of VIPA FSR & Rubidium Hyperfine Offset
plot_Rb_calibration = FORCE_PLOT_ALL or 0
stop_Rb_calibration = 0
force_Rb_calibration = FORCE_ALL_PROCESS or 0

# Step 10B: Calibrated 2D Frequency Map Generation
plot_freq_map = FORCE_PLOT_ALL or 0
stop_freq_map = 0

# Step 11: Sample In-Situ Frequency Map Self-Calibration
PATH_SAMPLE = config.PATH_SAMPLE
plot_sample_freq_calib = FORCE_PLOT_ALL or 0
stop_sample_freq_calib = 0

# Step 12 & 13: Sample DHO Peak Fitting & Plotting
force_dho_fit = FORCE_ALL_PROCESS or 0
plot_dho_overview = FORCE_PLOT_ALL or 1
plot_dho_profiles = FORCE_PLOT_ALL or 0
stop_dho_fit = 0

# Step 14: Acoustic Wavevector (q-vector) & Elliptical Geometry Extraction
force_q_vector = FORCE_ALL_PROCESS or 1
plot_q_dispersion = FORCE_PLOT_ALL or 1
plot_q_diagnostics = FORCE_PLOT_ALL or 1
stop_q_vector = 0


# ==============================================================================
# 01 - Open / Create HDF5 Wrapper
# ==============================================================================
wrapper = bck.create_wrapper(DATA_DIR, TARGET_FILE, exclude_dir=[FIGURE_DIRECTORY_NAME])
print('Step 1 - Opened / created H5 file.')

# ==============================================================================
# 02 - Monochromatic Laser Reference Image & Denoising
# ==============================================================================
path_h5_laser = f'{config.GROUP_INTEREST}/{config.PATH_LASER_REF}'
data_laser_reference = wrapper[f'{path_h5_laser}/image'].astype(np.float64)

if not force_denoise_laser and 'image_corrected' in wrapper.get_children_elements(path_h5_laser):
    data_subtracted, baseline_2d = laser_calib.retrieve_denoised_data(wrapper, path_h5_laser)
    print('Step 2 - Retrieved denoised laser data from H5 file.')
else:
    data_subtracted, baseline_2d = laser_calib.denoise_laser(wrapper, path_h5_laser, data_laser_reference)
    print('Step 2 - Subtracted baseline and saved denoised laser data in H5 file.')

if plot_denoise_laser:
    plotting.plot_denoise_laser(data_laser_reference, baseline_2d, data_subtracted, OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS)

if stop_denoise_laser:
    sys.exit(0)


# ==============================================================================
# 03 - Laser Line Detection & Full-Height Fringe Tracing
# ==============================================================================
if not force_extract_fringe and 'fringe_0' in wrapper.get_children_elements(path_h5_laser):
    peaks, points = laser_calib.retrieve_fringes(wrapper, path_h5_laser, y_center_laser=config.Y_CENTER_LASER)
    y_min = max(0, int(config.Y_CENTER_LASER - config.DY_LASER))
    y_max = min(data_subtracted.shape[0], int(config.Y_CENTER_LASER + config.DY_LASER))
    average_data = np.average(data_subtracted[y_min:y_max, :], axis=0)
    print(f'Step 3 - Retrieved {len(points)} fringe positions from H5 file.')
else:
    peaks, points, average_data = laser_calib.extract_laser_fringes(
        wrapper, path_h5_laser, data_subtracted,
        y_center_laser=config.Y_CENTER_LASER,
        dy_laser=config.DY_LASER,
        nb_laser_modes=config.NB_LASER_MODES,
        distance_peaks=config.DISTANCE_PEAKS_LASER,
        dy_trace=config.DY_TRACE,
        var_trace=config.VAR_TRACE
    )
    print(f'Step 3 - Computed {len(points)} fringe positions and saved them in H5 file.')

if plot_fringe_laser:
    plotting.plot_laser_fringes(average_data, peaks, data_subtracted, points,
                                config.Y_CENTER_LASER, config.DY_LASER,
                                OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS)

if stop_fringe_laser:
    sys.exit(0)


# ==============================================================================
# 04 - Unsaturated Sub-Pixel Refinement & Outlier Rejection
# ==============================================================================
laser_attrs = wrapper.get_attributes(path_h5_laser)
fringes_already_smoothed = laser_attrs.get('fringes_smoothed', False)

if not force_refine_laser and fringes_already_smoothed:
    refined_fringes, all_x_laser, all_x_laser_smooth, all_y_laser, all_k_laser = laser_calib.retrieve_refined_fringes(
        wrapper, path_h5_laser
    )
    print(f'Step 4 - Retrieved {len(refined_fringes)} cleaned and smoothed fringes from H5 file.')
else:
    refined_fringes, all_x_laser, all_x_laser_smooth, all_y_laser, all_k_laser = laser_calib.refine_laser_fringes(
        wrapper, path_h5_laser, points, data_laser_reference, data_subtracted,
        w_sub=config.W_SUB,
        saturation_limit=config.SATURATION_LIMIT,
        min_net_amp=config.MIN_NET_AMP
    )
    print(f'Step 4 - Cleaned and smoothed {len(refined_fringes)} fringes and saved them in H5 file.')

if plot_refine_laser:
    plotting.plot_fringe_refinement(all_x_laser, all_x_laser_smooth, all_y_laser,
                                    OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS)

if stop_refine_laser:
    sys.exit(0)


# ==============================================================================
# 05 - Exact 2D Gnomonic VIPA Geometry Fit
# ==============================================================================
if not force_concentric_fringe and 'x0_fit_circular' in wrapper.get_attributes(path_h5_laser):
    x0_fit, y0_fit, xc_gnom, yc_gnom = laser_calib.retrieve_vipa_geometry(wrapper, path_h5_laser)
    print(f'Step 5 - Optical axis: x0 = {x0_fit:.3f} px, y0 = {y0_fit:.3f} px retrieved from H5 file.')
else:
    x0_fit, y0_fit, xc_gnom, yc_gnom = laser_calib.fit_vipa_geometry(
        wrapper, path_h5_laser, refined_fringes,
        all_x_laser, all_y_laser, all_k_laser,
        f_px=config.F_PX_DEFAULT,
        phi_rad=config.VIPA_TILT_ANGLE,
        n_r=config.N_FUSED_SILICA_780
    )
    print(f'Step 5 - Optical axis: x0 = {x0_fit:.3f} px, y0 = {y0_fit:.3f} px fitted and saved in H5 file.')

if plot_concentric_fringe:
    plotting.plot_concentric_fringes_and_geometry(
        data_subtracted, refined_fringes, x0_fit, y0_fit, xc_gnom, yc_gnom,
        OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS
    )

if stop_concentric_fringe:
    sys.exit(0)


# ==============================================================================
# 06 - 2D Anamorphic Polynomial Order Surface Calibration
# ==============================================================================
calib_results = laser_calib.calibrate_anamorphic_order_surface(
    wrapper, path_h5_laser,
    all_x_laser, all_x_laser_smooth, all_y_laser, all_k_laser,
    x0_fit, y0_fit,
    f_px=config.F_PX_DEFAULT,
    phi_rad=config.VIPA_TILT_ANGLE,
    n_r=config.N_FUSED_SILICA_780,
    h_img=data_laser_reference.shape[0],
    nb_laser_modes=config.NB_LASER_MODES,
    fsr_vipa=config.FSR_VIPA,
    deg=4,
    lower_order_study=lower_order_study
)

print(f"Step 6 - 2D Anamorphic Surface RMS: {calib_results['rms_raw_mhz']:.2f} MHz raw | "
      f"{calib_results['rms_smooth_mhz']:.2f} MHz smoothed.")

if lower_order_study:
    print(f"\t 1D Linear RMS: {np.sqrt(np.mean(calib_results['err_lin_mhz']**2)):.2f} MHz")
    print(f"\t 1D Quadratic RMS: {np.sqrt(np.mean(calib_results['err_quad_mhz']**2)):.2f} MHz")

if plot_2D_anamorphic_cal:
    plotting.plot_anamorphic_calibration(
        calib_results['q_orders'],
        calib_results['q_pred_2d'],
        all_k_laser,
        all_x_laser,
        all_y_laser,
        calib_results['err_2d_mhz'],
        calib_results['err_2d_smooth_mhz'],
        config.FSR_VIPA,
        config.NB_LASER_MODES,
        OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS,
        lower_order_study=lower_order_study,
        q_pred_lin=calib_results.get('q_pred_lin'),
        q_pred_quad=calib_results.get('q_pred_quad')
    )

if stop_2D_anamorphic_cal:
    sys.exit(0)


# ==============================================================================
# 07 - Theoretical Natural Rubidium D2 Absorption Spectrum Model
# ==============================================================================
det_table_MHz, trans_table, trans_fast = Rb_calib.create_rubidium_reference_table(
    T_cell=config.T_CELL, L_cell=config.L_CELL
)
print('Step 7 - Created theoretical reference table for Rubidium D2 absorption.')

if plot_Rb_absorption:
    plotting.plot_rb_theoretical_spectrum(det_table_MHz, trans_table, OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS)

if stop_Rb_absorption:
    sys.exit(0)


# ==============================================================================
# 08 - Rubidium Absorption Image Denoising
# ==============================================================================
path_h5_rb = f'{config.GROUP_INTEREST}/{config.PATH_RB_ABSORPTION}'
data_absorption_raw = wrapper[f'{path_h5_rb}/image'].astype(np.float64)

if not force_denoise_Rb and 'image_corrected' in wrapper.get_children_elements(path_h5_rb):
    data_absorption, baseline_absorption = Rb_calib.retrieve_denoised_rb(wrapper, path_h5_rb)
    print('Step 8 - Retrieved denoised Rubidium absorption spectrum from H5 file.')
else:
    data_absorption, baseline_absorption = Rb_calib.denoise_rb(
        wrapper, path_h5_rb, data_absorption_raw,
        scale=config.SCALE_RB,
        ksize_structuring_elements=config.KSIZE_STRUCTURING_ELEMENTS_RB,
        ksize_blur=config.KSIZE_BLUR_RB
    )
    print('Step 8 - Denoised Rubidium absorption spectrum and stored it in H5 file.')

if plot_denoise_Rb:
    plotting.plot_denoise_rb(data_absorption_raw, baseline_absorption, data_absorption,
                             OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS)

if stop_denoise_Rb:
    sys.exit(0)


# ==============================================================================
# 09 - Radial Profile Extraction Around Gnomonic Center
# ==============================================================================
if data_absorption.shape != data_laser_reference.shape:
    raise ValueError("Laser and absorption images must have the same shape.")

profiles_2d, radius_axis, avg_spectrum_radius, phi_order_r = Rb_calib.extract_radial_absorption_profile(
    data_absorption,
    xc_gnom=xc_gnom,
    yc_gnom=yc_gnom,
    y0_fit=y0_fit,
    c_2d_order=calib_results['c_2d_order'],
    g_mean_ref=calib_results['g_mean_ref'],
    g_std_ref=calib_results['g_std_ref'],
    L_y=calib_results['L_y'],
    f_px=config.F_PX_DEFAULT,
    phi_rad=config.VIPA_TILT_ANGLE,
    n_r=config.N_FUSED_SILICA_780,
    num_angles=1000
)
print('Step 9 - Extracted experimental radial absorption profile of the Rb cell.')

if plot_Rb_experimental:
    plotting.plot_rb_experimental_profile(phi_order_r, avg_spectrum_radius, OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS)

if stop_Rb_experimental:
    sys.exit(0)


# ==============================================================================
# 10 - VIPA Free Spectral Range (FSR) & Hyperfine Offset Calibration
# ==============================================================================
rb_attrs = wrapper.get_attributes(path_h5_rb)
if not force_Rb_calibration and 'FSR' in rb_attrs:
    fsr_results = Rb_calib.retrieve_fsr_calibration(
        wrapper, path_h5_rb,
        calib_results['g_bar'],
        calib_results['q_orders'],
        phi_order_r,
        avg_spectrum_radius,
        trans_fast,
        nu_0=config.nu_0,
        fit_window_ghz=config.FIT_WINDOW_GHZ,
        delta_nu_laser_offset=config.DELTA_NU_LASER_OFFSET
    )
    print(f"Step 10 - Retrieved calibrated FSR = {fsr_results['FSR_cal']:.4f} GHz, "
          f"delta_nu = {fsr_results['delta_nu_opt']:.3f} GHz from H5 file.")
else:
    fsr_results = Rb_calib.calibrate_fsr_and_offset(
        wrapper, path_h5_rb,
        calib_results['g_bar'],
        calib_results['q_orders'],
        phi_order_r,
        avg_spectrum_radius,
        trans_fast,
        nu_0=config.nu_0,
        fit_window_ghz=config.FIT_WINDOW_GHZ,
        delta_nu_laser_offset=config.DELTA_NU_LASER_OFFSET
    )
    print(f"Step 10 - Calibrated VIPA FSR = {fsr_results['FSR_cal']:.4f} GHz, "
          f"delta_nu = {fsr_results['delta_nu_opt']:.3f} GHz and saved in H5 file.")

if plot_Rb_calibration:
    plotting.plot_rb_calibration(
        fsr_results['f_grid'],
        fsr_results['orders_used'],
        fsr_results['order_spectra'],
        fsr_results['f_valid'],
        fsr_results['sig_valid'],
        fsr_results['baseline_env'],
        fsr_results['f_fit_min'],
        fsr_results['f_fit_max'],
        fsr_results['delta_nu_opt'],
        fsr_results['delta_nu_fine'],
        fsr_results['FSR_cal'],
        trans_fast,
        OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS
    )

if stop_Rb_calibration:
    sys.exit(0)


# ==============================================================================
# 10B - Full 2D Frequency Map Construction & Save
# ==============================================================================
freq_map, phi_order_map, xi_elastic = Rb_calib.compute_and_save_frequency_map(
    wrapper=wrapper,
    path_group=path_h5_rb,
    img_shape=data_absorption.shape,
    x0_fit=x0_fit,
    y0_fit=y0_fit,
    xc_gnom=xc_gnom,
    yc_gnom=yc_gnom,
    c_2d_order=calib_results['c_2d_order'],
    g_mean_ref=calib_results['g_mean_ref'],
    g_std_ref=calib_results['g_std_ref'],
    L_y=calib_results['L_y'],
    FSR_cal=fsr_results['FSR_cal'],
    delta_nu_opt=fsr_results['delta_nu_opt'],
    rb_lock_offset_ghz=config.RB_LOCK_OFFSET_GHZ,
    f_px=config.F_PX_DEFAULT,
    phi_rad=config.VIPA_TILT_ANGLE,
    n_r=config.N_FUSED_SILICA_780
)
print(f"Step 10B - Saved calibrated {data_absorption.shape[0]}x{data_absorption.shape[1]} 2D frequency map "
      f"zero-referenced to elastic line (xi_elastic = {xi_elastic:.4f}) in H5 file.")

if plot_freq_map:
    plotting.plot_frequency_map(freq_map, OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS)

if stop_freq_map:
    sys.exit(0)


# ==============================================================================
# 11 - Load Sample & Calibrate Sample Frequency Map In-Situ
# ==============================================================================
path_h5_sample = f'{config.GROUP_INTEREST}/{PATH_SAMPLE}'
data_sample = wrapper[f'{path_h5_sample}/image'].astype(np.float64)
yy, xx = np.indices(data_sample.shape, dtype=np.float64)

freq_corrected, df_coarse, dz_map_2d, dinst_disp_map, bls_peaks, avg_data_sample, freq_avg_data_sample = (
    spectrum_fitting.calibrate_sample_frequency_map(
        freq_map_base=freq_map,
        data_clean_in=data_sample,
        xx_in=xx,
        yy_in=yy,
        fsr_val=fsr_results['FSR_cal'],
        y0_center=y0_fit,
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W,
        expected_shift=config.EXPECTED_BRILLOUIN_SHIFT,
        shift_tol=config.BRILLOUIN_SHIFT_TOLERANCE,
    )
)
print(f"Step 11 - In-situ calibrated 2D frequency map for {PATH_SAMPLE}: coarse df = {df_coarse*1e3:+.2f} MHz | "
      f"2D zero-offset RMS = {np.std(dz_map_2d)*1e3:.1f} MHz | higher-order dispersion RMS = {np.std(dinst_disp_map)*1e3:.1f} MHz")

if plot_sample_freq_calib:
    plotting.plot_sample_baseline_and_windows(
        data_sample, freq_avg_data_sample, avg_data_sample,
        bls_peaks, dinst_disp_map, path_h5_sample,
        OUT_DIR, FIGURE_PREFIX, SHOW_PLOTS
    )

if stop_sample_freq_calib:
    sys.exit(0)


# ==============================================================================
# 12 - Bin Data & Fit Brillouin Spectra with DHO
# ==============================================================================
cached_fit = None
if not force_dho_fit:
    cached_fit = spectrum_fitting.retrieve_sample_dho_fit(
        wrapper, path_h5_sample,
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W
    )

if cached_fit is not None:
    dho_results = cached_fit
    print(f"Step 12 - Retrieved cached {config.BIN_VAL_H}x{config.BIN_VAL_W} binned DHO fit for {PATH_SAMPLE} from H5 file.")
else:
    dho_results = spectrum_fitting.fit_sample_dho(
        data_sample=data_sample,
        freq_corrected=freq_corrected,
        xx=xx,
        yy=yy,
        fsr_val=fsr_results['FSR_cal'],
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W,
        ds_init=config.EXPECTED_BRILLOUIN_SHIFT,
        shift_tol=config.BRILLOUIN_SHIFT_TOLERANCE,
        gamma_init=config.EXPECTED_LINEWIDTH,
        gamma_bounds=config.LINEWIDTH_BOUNDS,
        verbose=True
    )
    print(f"Step 12 - Fitted {dho_results['n_fringes']} orders with vectorized DHO for {PATH_SAMPLE} "
          f"({config.BIN_VAL_H}x{config.BIN_VAL_W} binning).")
    print(f"\t Stokes vs Anti-Stokes splitting RMS: {dho_results['stokes_antistokes_rms_mhz']:.2f} MHz")

    # Step 15: Save Treated Data
    spectrum_fitting.save_treated_data(
        wrapper=wrapper,
        path_sample=path_h5_sample,
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W,
        shift_rel=dho_results['shift_clipped'],
        shift_err=dho_results['shift_error'],
        lw_rel=dho_results['linewidth_clipped'],
        lw_err=dho_results['linewidth_error'],
        x_coords=dho_results['x_coords'],
        y_coords=dho_results['y_coords']
    )
    print(f"Step 15 - Saved treated {config.BIN_VAL_H}x{config.BIN_VAL_W} data for {PATH_SAMPLE} in H5 file.")


# ==============================================================================
# 13 - Plot Extracted Brillouin Shift & Linewidth
# ==============================================================================
if plot_dho_overview:
    plotting.plot_dho_overview(
        shift_clipped=dho_results['shift_clipped'],
        shift_clipped_averaged=dho_results['shift_clipped_averaged'],
        linewidth_clipped=dho_results['linewidth_clipped'],
        linewidth_clipped_averaged=dho_results['linewidth_clipped_averaged'],
        x_coords=dho_results['x_coords'],
        y_coords=dho_results['y_coords'],
        x_coords_averaged=dho_results['x_coords_averaged'],
        y_coords_averaged=dho_results['y_coords_averaged'],
        path_sample=path_h5_sample,
        out_dir=OUT_DIR,
        prefix=FIGURE_PREFIX,
        show_plots=SHOW_PLOTS
    )

if plot_dho_profiles:
    plotting.plot_dho_fringe_profiles(
        shift_clipped=dho_results['shift_clipped'],
        linewidth_clipped=dho_results['linewidth_clipped'],
        shift_clipped_averaged=dho_results['shift_clipped_averaged'],
        linewidth_clipped_averaged=dho_results['linewidth_clipped_averaged'],
        y_coords_averaged=dho_results['y_coords_averaged'],
        path_sample=path_h5_sample,
        out_dir=OUT_DIR,
        prefix=FIGURE_PREFIX,
        show_plots=SHOW_PLOTS
    )

if stop_dho_fit:
    sys.exit(0)


# ==============================================================================
# 14 - Acoustic Wavevector (q-vector) & Elliptical Dispersion Extraction
# ==============================================================================
cached_q = None
if not force_q_vector:
    cached_q = q_vector.retrieve_q_vector_results(
        wrapper=wrapper,
        path_sample=path_h5_sample,
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W
    )

if cached_q is not None:
    q_results = cached_q
    print(f"Step 14 - Retrieved cached q-vector results for {PATH_SAMPLE} from H5 file.")
else:
    q_results = q_vector.extract_q_vector(
        data_source=dho_results,
        path_sample=path_h5_sample,
        wrapper=wrapper,
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W,
        refractive_index=config.N_SAMPLE,
        laser_wavelength=config.lambda_0,
        use_errors=config.USE_ERROR_WEIGHTING,
        error_percentile_floor=config.ERROR_PERCENTILE_FLOOR,
        average_pairs=config.AVERAGE_STOKES_ANTISTOKES,
        shift_bounds=config.SHIFT_RANGE_SAMPLE,
        linewidth_bounds=config.LINEWIDTH_RANGE_SAMPLE,
        initial_M_theta=config.INITIAL_M_THETA,
        verbose=True
    )
    q_vector.save_q_vector_results(
        wrapper=wrapper,
        path_sample=path_h5_sample,
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W,
        q_results=q_results
    )
    print(f"Step 14 - Saved q-vector datasets and attributes for {PATH_SAMPLE} in H5 file.")

if plot_q_dispersion:
    plotting.plot_q_vector_dispersion(
        q_vector=q_results["q_vector"],
        shift=q_results["shift"],
        linewidth=q_results["linewidth"],
        theta_deg=q_results["theta_deg"],
        shift_err=q_results.get("shift_err"),
        linewidth_err=q_results.get("linewidth_err"),
        Shift_0=q_results.get("Shift_0"),
        M_theta=q_results.get("M_theta"),
        n_medium=config.N_SAMPLE,
        lambda_vacuum=config.lambda_0,
        V_s=q_results.get("V_s"),
        V_s_err=q_results.get("V_s_err"),
        out_dir=OUT_DIR,
        prefix=FIGURE_PREFIX,
        show_plots=SHOW_PLOTS
    )

if plot_q_diagnostics and "x_sph" in q_results:
    plotting.plot_complete_diagnostic_suite(
        x=q_results["x_coords"],
        y=q_results["y_coords"],
        shift=q_results["shift"],
        linewidth=q_results["linewidth"],
        shift_err=q_results.get("shift_err"),
        linewidth_err=q_results.get("linewidth_err"),
        x_sph=q_results["x_sph"],
        y_sph=q_results["y_sph"],
        r_sph=q_results["r_sph"],
        q_vector=q_results["q_vector"],
        theta_deg=q_results["theta_deg"],
        opt_params=q_results["opt_params"],
        S0_fit=q_results["Shift_0"],
        M_theta_fit=q_results["M_theta"],
        S0_err=q_results.get("Shift_0_err"),
        M_theta_err=q_results.get("M_theta_err"),
        n_medium=config.N_SAMPLE,
        lambda_vacuum=config.lambda_0,
        V_s=q_results.get("V_s"),
        V_s_err=q_results.get("V_s_err"),
        out_dir=OUT_DIR,
        prefix=FIGURE_PREFIX,
        show_plots=SHOW_PLOTS
    )

if stop_q_vector:
    sys.exit(0)

print("\n--- Pipeline completed successfully! ---")
