"""
peak_fit.py: Sample in-situ frequency calibration and DHO peak fitting (Steps 11 to 13).

Runs AUTONOMOUSLY:
  - Reuses the calibrated master 2D frequency map and FSR from Steps 1-10B directly from the HDF5 file.
  - Does not rerun any laser reference or Rubidium calibration steps.

Performs:
  - Step 11: Sample in-situ two-stage 2D frequency map self-calibration (elastic zero-offset map
            and higher-order dispersion Jacobian scaling using Stokes/Anti-Stokes symmetry).
  - Step 12: 2D spatial block binning (4x2 px) and vectorized 3D batched Levenberg-Marquardt
            Damped Harmonic Oscillator (DHO) peak fitting across all orders.
            Saves treated data (shifts, linewidths, uncertainties, sub-pixel coordinates) to HDF5.
  - Step 13: Spectral diagnostics and overview/fringe profile plotting.
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
from arVIPAspectro import Rb_calib
from arVIPAspectro import spectrum_fitting
from arVIPAspectro import plotting


# ==============================================================================
# Paths & Execution Configuration
# ==============================================================================

DATA_DIR = '/Volumes/LAUDATE/Data/260916 - arBLS/261110'
TARGET_FILE = 'Measures.h5'
FIGURE_DIRECTORY_NAME = 'figures'
OUT_DIR = os.path.join(DATA_DIR, FIGURE_DIRECTORY_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

FIGURE_PREFIX = ''
SHOW_PLOTS = False
FORCE_ALL_PROCESS = 1
FORCE_PLOT_ALL = 1

# ==============================================================================
# HDF5 Hierarchy Defaults
# ==============================================================================
GROUP_INTEREST = 'Brillouin'
PATH_RB_ABSORPTION = 'Rb_absorption'



# ==============================================================================
# Sample Spectrum Fitting (DHO) Parameters (Steps 11 to 13)
# ==============================================================================
BIN_VAL_H = 4                               # Vertical spatial binning factor (pixels per bin)
BIN_VAL_W = 2                               # Horizontal spatial binning factor (pixels per bin)

# --- Polystyrene (Plastic) Reference ---
# EXPECTED_BRILLOUIN_SHIFT = 9.7              # Expected Brillouin shift nu_B in GHz (plastic)
# BRILLOUIN_SHIFT_TOLERANCE = 2.0             # Search half-window around expected shift in GHz
# EXPECTED_LINEWIDTH = 0.8                    # Expected Brillouin linewidth Gamma_B in GHz
# LINEWIDTH_BOUNDS = (0.4, 1.3)               # Acceptable linewidth range (min, max) in GHz

# --- Water Reference ---
EXPECTED_BRILLOUIN_SHIFT = 5.1              # Expected Brillouin shift nu_B in GHz (water)
BRILLOUIN_SHIFT_TOLERANCE = 2.0             # Search half-window around expected shift in GHz
EXPECTED_LINEWIDTH = 0.8                    # Expected Brillouin linewidth Gamma_B in GHz
LINEWIDTH_BOUNDS = (0.4, 1.3)               # Acceptable linewidth range (min, max) in GHz

# --- Glycerol ---
# EXPECTED_BRILLOUIN_SHIFT = 10.8              # Expected Brillouin shift nu_B in GHz (water)
# BRILLOUIN_SHIFT_TOLERANCE = 3                # Search half-window around expected shift in GHz
# EXPECTED_LINEWIDTH = 1.8                     # Expected Brillouin linewidth Gamma_B in GHz
# LINEWIDTH_BOUNDS = (1.5, 3)                   # Acceptable linewidth range (min, max) in GHz

N_ORDERS_SAMPLE = None                      # Number of Brillouin orders to fit (None = automatic detection)
DHO_HW_PX = 20                              # Fitting half-window width in binned pixels
DHO_MIN_AMP = 10.0                          # Minimum peak amplitude above chord baseline
SIGMA_Y_BINS = 6.0                          # Smoothing length along vertical bins for in-situ calibration

# Execution flags per step: (plot_*, stop_*, force_*)
# Step 11: Sample In-Situ Frequency Map Self-Calibration
plot_sample_freq_calib = FORCE_PLOT_ALL or 1
stop_sample_freq_calib = 0

# Step 12 & 13: Sample DHO Peak Fitting & Plotting
force_dho_fit = FORCE_ALL_PROCESS or 1   # Defaults to fitting DHO peaks on execution
plot_dho_overview = FORCE_PLOT_ALL or 1
plot_dho_profiles = FORCE_PLOT_ALL or 0
stop_dho_fit = 0

d_frequency = 0

# ==============================================================================
# Open HDF5 Container & Retrieve Prerequisite Calibration (Steps 1-10B)
# ==============================================================================
wrapper = bck.create_wrapper(DATA_DIR, TARGET_FILE, exclude_dir=[FIGURE_DIRECTORY_NAME])

for i in range(1,23):
    if i<10:
        PATH_SAMPLE = f'0{i}_water_{29+i}'
    elif i<12:
        PATH_SAMPLE = f'{i}_water_{29+i}'
    else:
        PATH_SAMPLE = f'{i}_water_{52-i}'
    print('\n\n\n----------\n',PATH_SAMPLE,'\n----------')

    path_h5_rb = f'{GROUP_INTEREST}/{PATH_RB_ABSORPTION}'
    freq_map, rb_attrs = Rb_calib.retrieve_frequency_map(wrapper, path_h5_rb)
    freq_map += d_frequency

    if freq_map is None or rb_attrs is None or 'FSR' not in rb_attrs or 'y0_fit' not in rb_attrs:
        sys.exit(
            f"ERROR: Calibrated 2D frequency map or FSR attributes not found in '{path_h5_rb}'. "
            "Please run 'calib_laser.py' followed by 'calib_Rb.py' first."
        )

    fsr_val = float(rb_attrs['FSR'])
    y0_fit = float(rb_attrs['y0_fit'])
    print(f"Loaded master calibration from H5: FSR = {fsr_val:.4f} GHz | y0 = {y0_fit:.2f} px | "
        f"Frequency map shape = {freq_map.shape}.")

    path_h5_sample = f'{GROUP_INTEREST}/{PATH_SAMPLE}'
    try:
        data_sample = wrapper[f'{path_h5_sample}/image'].astype(np.float64)
    except Exception:
        sys.exit(f"ERROR: Sample dataset '{path_h5_sample}/image' not found in HDF5 file.")

    yy, xx = np.indices(data_sample.shape, dtype=np.float64)


    # ==============================================================================
    # 11 - Load Sample & Calibrate Sample Frequency Map In-Situ
    # ==============================================================================
    freq_corrected, df_coarse, dz_map_2d, dinst_disp_map, bls_peaks, avg_data_sample, freq_avg_data_sample = (
        spectrum_fitting.calibrate_sample_frequency_map(
            freq_map_base=freq_map,
            data_clean_in=data_sample,
            xx_in=xx,
            yy_in=yy,
            fsr_val=fsr_val,
            y0_center=y0_fit,
            bin_val_h=BIN_VAL_H,
            bin_val_w=BIN_VAL_W,
            expected_shift=EXPECTED_BRILLOUIN_SHIFT,
            shift_tol=BRILLOUIN_SHIFT_TOLERANCE,
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
            bin_val_h=BIN_VAL_H,
            bin_val_w=BIN_VAL_W
        )

    if cached_fit is not None:
        dho_results = cached_fit
        print(f"Step 12 - Retrieved cached {BIN_VAL_H}x{BIN_VAL_W} binned DHO fit for {PATH_SAMPLE} from H5 file.")
    else:
        dho_results = spectrum_fitting.fit_sample_dho(
            data_sample=data_sample,
            freq_corrected=freq_corrected,
            xx=xx,
            yy=yy,
            fsr_val=fsr_val,
            bin_val_h=BIN_VAL_H,
            bin_val_w=BIN_VAL_W,
            ds_init=EXPECTED_BRILLOUIN_SHIFT,
            shift_tol=BRILLOUIN_SHIFT_TOLERANCE,
            gamma_init=EXPECTED_LINEWIDTH,
            gamma_bounds=LINEWIDTH_BOUNDS,
            verbose=True
        )
        print(f"Step 12 - Fitted {dho_results['n_fringes']} orders with vectorized DHO for {PATH_SAMPLE} "
            f"({BIN_VAL_H}x{BIN_VAL_W} binning).")
        print(f"\t Stokes vs Anti-Stokes splitting RMS: {dho_results['stokes_antistokes_rms_mhz']:.2f} MHz")

        # Save Treated Data to HDF5
        spectrum_fitting.save_treated_data(
            wrapper=wrapper,
            path_sample=path_h5_sample,
            bin_val_h=BIN_VAL_H,
            bin_val_w=BIN_VAL_W,
            shift_rel=dho_results['shift_clipped'],
            shift_err=dho_results['shift_error'],
            lw_rel=dho_results['linewidth_clipped'],
            lw_err=dho_results['linewidth_error'],
            x_coords=dho_results['x_coords'],
            y_coords=dho_results['y_coords']
        )
        print(f"Step 12 - Saved treated {BIN_VAL_H}x{BIN_VAL_W} data for {PATH_SAMPLE} in H5 file.")


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

    print(f"\n--- Peak fitting (Steps 11-13) for '{PATH_SAMPLE}' completed successfully! ---")
    print("You can now run 'analysis_results.py' to extract q-vectors and sound velocity.")
