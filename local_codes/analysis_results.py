"""
analysis_results.py: Acoustic wavevector (q-vector) and dispersion analysis (Steps 14 and 15).

Performs:
  - Prerequisite check: retrieves binned DHO fit results (Step 12) from HDF5 for PATH_SAMPLE.
  - Step 14: Concentric coaxial 2D ellipse fitting, aspect ratio optimization,
            spherical coordinate projection, angular magnification (M_theta) kinematic fit,
            and acoustic wavevector q-dispersion extraction with full covariance error propagation.
  - Step 15: Storage of calibrated wavevectors, scattering angles, speed of sound (V_s +/- sigma_Vs),
            and ellipse parameters to HDF5.
  - Diagnostic & dispersion plotting (Figure 14 and 14B).
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
from arVIPAspectro import spectrum_fitting
from arVIPAspectro import q_vector
from arVIPAspectro import plotting


# ==============================================================================
# Paths & Execution Configuration
# ==============================================================================

DATA_DIR = '/Volumes/LAUDATE/Data/260916 - arBLS/261105'
TARGET_FILE = 'Measures.h5'
PATH_SAMPLE = 'water_5s'
FIGURE_DIRECTORY_NAME = 'figures'
OUT_DIR = os.path.join(DATA_DIR, FIGURE_DIRECTORY_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

FIGURE_PREFIX = ''
SHOW_PLOTS = True
FORCE_ALL_PROCESS = 1
FORCE_PLOT_ALL = 1

# Execution flags per step: (plot_*, stop_*, force_*)
# Step 14 & 15: Acoustic Wavevector (q-vector) & Elliptical Geometry Extraction
force_q_vector = FORCE_ALL_PROCESS or 1  # Defaults to extracting q-vector on execution
plot_q_dispersion = FORCE_PLOT_ALL or 1
plot_q_diagnostics = FORCE_PLOT_ALL or 1
stop_q_vector = 0


# ==============================================================================
# Open HDF5 Container & Retrieve Prerequisite DHO Fit Data (Step 12)
# ==============================================================================
wrapper = bck.create_wrapper(DATA_DIR, TARGET_FILE, exclude_dir=[FIGURE_DIRECTORY_NAME])
path_h5_sample = f'{config.GROUP_INTEREST}/{PATH_SAMPLE}'

dho_results = spectrum_fitting.retrieve_sample_dho_fit(
    wrapper=wrapper,
    path_sample=path_h5_sample,
    bin_val_h=config.BIN_VAL_H,
    bin_val_w=config.BIN_VAL_W
)

if dho_results is None:
    sys.exit(
        f"ERROR: Treated DHO fit results for '{path_h5_sample}' not found in HDF5 file. "
        f"Please run 'peak_fit.py' first to perform DHO peak fitting."
    )

print(f"Loaded Step 12 DHO fit for '{PATH_SAMPLE}': {dho_results['n_fringes']} orders | "
      f"Stokes/Anti-Stokes RMS = {dho_results['stokes_antistokes_rms_mhz']:.2f} MHz.")


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

    # ==============================================================================
    # 15 - HDF5 Storage & Persistence of Wavevector and Physical Parameters
    # ==============================================================================
    q_vector.save_q_vector_results(
        wrapper=wrapper,
        path_sample=path_h5_sample,
        bin_val_h=config.BIN_VAL_H,
        bin_val_w=config.BIN_VAL_W,
        q_results=q_results
    )
    print(f"Step 15 - Saved q-vector datasets and attributes for {PATH_SAMPLE} in H5 file.")


# ==============================================================================
# Diagnostics & Dispersion Plots
# ==============================================================================
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

print(f"\n--- Acoustic wavevector analysis (Steps 14-15) for '{PATH_SAMPLE}' completed successfully! ---")
if "V_s" in q_results:
    vs_val = q_results["V_s"]
    vs_err = q_results.get("V_s_err", 0.0)
    print(f"Longitudinal Speed of Sound V_s = {vs_val:.2f} +/- {vs_err:.2f} m/s")
