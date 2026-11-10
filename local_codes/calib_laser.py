"""
calib_laser.py: Laser reference calibration script (Steps 1 to 6).

Performs:
  - Step 1: Open / Create HDF5 container wrapper.
  - Step 2: Monochromatic laser reference image denoising and baseline subtraction.
  - Step 3: Laser line detection and full-height vertical fringe tracing.
  - Step 4: Unsaturated sub-pixel centroid refinement and polynomial outlier filtering.
  - Step 5: Exact 2D gnomonic VIPA geometry fitting (optical axis x0, y0).
  - Step 6: 2D anamorphic polynomial order surface calibration phi_order(g, y),
            and persistence of surface parameters to HDF5.
"""

import os
import sys

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

import arVIPAspectro as ar
from arVIPAspectro import config
from arVIPAspectro import backend as bck
from arVIPAspectro import laser_calib
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

# Persist Step 6 anamorphic surface calibration to HDF5
laser_calib.save_anamorphic_surface_calibration(wrapper, path_h5_laser, calib_results)

print(f"Step 6 - 2D Anamorphic Surface RMS: {calib_results['rms_raw_mhz']:.2f} MHz raw | "
      f"{calib_results['rms_smooth_mhz']:.2f} MHz smoothed.")
print(f"Step 6 - Anamorphic surface parameters saved in H5 file under '{path_h5_laser}'.")

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

print("\n--- Laser calibration (Steps 1-6) completed successfully! ---")
