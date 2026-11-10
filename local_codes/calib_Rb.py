"""
calib_Rb.py: Rubidium absorption and frequency calibration script (Steps 7 to 10B).

Performs:
  - Prerequisite check: loads fitted optical axis and 2D anamorphic surface calibration
    from Step 5 and Step 6 stored in HDF5 by calib_laser.py.
  - Step 7: Theoretical natural Rubidium (85Rb / 87Rb) D2 absorption spectrum model.
  - Step 8: Rubidium absorption image denoising and baseline morphology subtraction.
  - Step 9: Radial absorption profile extraction along gnomonic iso-angle contours.
  - Step 10: Calibration of VIPA Free Spectral Range (FSR) and hyperfine laser detuning.
  - Step 10B: Calibrated 2D master frequency map generation and persistence to HDF5.
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
from arVIPAspectro import Rb_calib
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


# ==============================================================================
# Open HDF5 Container & Check Prerequisite Laser Calibration (Steps 1-6)
# ==============================================================================
wrapper = bck.create_wrapper(DATA_DIR, TARGET_FILE, exclude_dir=[FIGURE_DIRECTORY_NAME])
path_h5_laser = f'{config.GROUP_INTEREST}/{config.PATH_LASER_REF}'

if 'x0_fit_circular' not in wrapper.get_attributes(path_h5_laser):
    sys.exit(
        f"ERROR: Optical axis geometry (Step 5) not found in '{path_h5_laser}'. "
        "Please run 'calib_laser.py' before running 'calib_Rb.py'."
    )

calib_results = laser_calib.retrieve_anamorphic_surface_calibration(wrapper, path_h5_laser)
if calib_results is None:
    sys.exit(
        f"ERROR: 2D anamorphic surface calibration (Step 6) not found in '{path_h5_laser}'. "
        "Please run 'calib_laser.py' before running 'calib_Rb.py'."
    )

x0_fit, y0_fit, xc_gnom, yc_gnom = laser_calib.retrieve_vipa_geometry(wrapper, path_h5_laser)
laser_shape = wrapper[f'{path_h5_laser}/image'].shape
print(f"Loaded laser calibration: optical axis ({x0_fit:.2f}, {y0_fit:.2f}) px | "
      f"gnomonic center ({xc_gnom:.2f}, {yc_gnom:.2f}) px.")


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
if data_absorption.shape != laser_shape:
    raise ValueError(f"Laser ({laser_shape}) and absorption ({data_absorption.shape}) image shapes must match.")

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

print("\n--- Rubidium absorption & frequency calibration (Steps 7-10B) completed successfully! ---")
print("You can now run 'peak_fit.py' autonomously for any sample.")
