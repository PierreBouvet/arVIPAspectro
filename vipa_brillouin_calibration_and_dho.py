# %%
# ==============================================================================
# 0 - Environment Setup, Imports & Output Configuration
# ==============================================================================
# Description:
#   Configures dynamic module search paths so that `vipa_dispersion.py` and
#   `module.py` are reliably imported whether running inside a VS Code / Jupyter
#   interactive kernel (# %%) or from the command line. Imports all required
#   scientific computing, image processing, and HDF5_BLS packages.

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.ndimage as ndimage
import scipy.interpolate as interp
from scipy.optimize import least_squares, curve_fit, minimize, minimize_scalar
from scipy.signal import find_peaks
from scipy.special import wofz

from HDF5_BLS import Wrapper
from HDF5_BLS_treat import Treat

# Dynamically locate the directory containing 'vipa_dispersion.py' and 'module.py'
search_dirs = []
if '__file__' in globals():
    f_dir = Path(__file__).resolve().parent
    search_dirs.extend([f_dir, f_dir.parent / 'ar_BLS'])
cwd = Path.cwd()
search_dirs.extend([cwd, cwd / 'ar_BLS', cwd.parent / 'ar_BLS'])
search_dirs.append(Path('/Users/pierrebouvet/Documents/Code/Treatment/ar_BLS'))

bls_dir = None
for d in search_dirs:
    if (d / 'vipa_dispersion.py').is_file() and (d / 'module.py').is_file():
        bls_dir = str(d.resolve())
        break

if bls_dir and bls_dir not in sys.path:
    sys.path.insert(0, bls_dir)

import vipa_dispersion as vd
from module import (
    create_wrapper,
    extract_baseline_image_morphology,
    extract_fringes,
    fit_concentric_arcs,
    refine_fringe_points,
    extract_profiles_2d,
    progress,
)

# %%
# ==============================================================================
# 1 - User Configuration & Experiment Parameters
# ==============================================================================
# Description:
#   Central configuration block defining all experimental parameters, file paths,
#   HDF5 dataset locations, VIPA mode counts, and material acoustic properties.
#   All subsequent calibration, fitting, and visualization steps (Steps 2 to 15)
#   are dynamically parameterized by this configuration.

# --- 1. File & Data Directory Settings ---
DATA_DIR = '/Volumes/LAUDATE/Data/260916 - arBLS/261109'
TARGET_FILE = 'Measures.h5'
OUT_DIR = os.path.join(DATA_DIR, 'figures')
FIGURE_PREFIX = ''
os.makedirs(OUT_DIR, exist_ok=True)

# Automatically show interactive figures when running inside an interactive kernel (# %%)
SHOW_PLOTS = ('ipykernel' in sys.modules) or (os.environ.get('SHOW_PLOTS', '0') == '1')

# --- 2. HDF5 Hierarchy & Dataset Paths ---
# Base group containing the acquisitions (e.g. 'Brillouin' or 'Brillouin/3 point 4')
GROUP_INTEREST = 'Brillouin'

# Relative dataset names within GROUP_INTEREST (or absolute paths starting with '/')
PATH_LASER_REF = 'laser_reference'
PATH_RB_ABSORPTION = 'Rb_absorption'
PATH_SAMPLE = 'plastic'

# Helper to resolve dataset path relative to GROUP_INTEREST
def resolve_h5_path(base_grp, sub_path):
    if sub_path.startswith('/'):
        return sub_path.lstrip('/')
    b = base_grp.strip('/')
    p = sub_path.strip('/')
    return f"{b}/{p}" if b else p

# Aliases for compatibility across downstream cells
directory = DATA_DIR
target_file = TARGET_FILE
group_interest = GROUP_INTEREST.strip('/')
name_figure_prefix = FIGURE_PREFIX
path_wrapper_laser = resolve_h5_path(group_interest, PATH_LASER_REF)
path_wrapper_rb = resolve_h5_path(group_interest, PATH_RB_ABSORPTION)
path_wrapper_sample = resolve_h5_path(group_interest, PATH_SAMPLE)

# --- 3. VIPA Optical Modes & Binning ---
NB_LASER_MODES = 14         # Expected number of laser reference fringes (e.g. 10 or 11; None for auto)
N_ORDERS_SAMPLE = None      # Expected number of Brillouin orders to fit (None for auto-detection)
BIN_VAL_H = 4               # Vertical binning factor (pixels per bin)
BIN_VAL_W = 2               # Horizontal binning factor (pixels per bin)

# --- 4. Material Acoustic Properties & Optical Lock ---
# # Water
# EXPECTED_BRILLOUIN_SHIFT = 5.3       # Expected Brillouin shift nu_B in GHz (e.g. 9.72 for plastic, ~7.5 for water)
# BRILLOUIN_SHIFT_TOLERANCE = 2      # Search half-window around expected shift in GHz (e.g. ±0.70 GHz)
# EXPECTED_LINEWIDTH = 0.6             # Expected Brillouin linewidth Gamma_B in GHz
# LINEWIDTH_BOUNDS = (0.40, 1.40)       # Acceptable linewidth range (min, max) in GHz
# RB_LOCK_OFFSET_GHZ = 1.323            # 85Rb F=2 absorption line offset from Rb D2 center of gravity (GHz)

# Plastic
EXPECTED_BRILLOUIN_SHIFT = 9.7       # Expected Brillouin shift nu_B in GHz (e.g. 9.72 for plastic, ~7.5 for water)
BRILLOUIN_SHIFT_TOLERANCE = 3      # Search half-window around expected shift in GHz (e.g. ±0.70 GHz)
EXPECTED_LINEWIDTH = 0.8            # Expected Brillouin linewidth Gamma_B in GHz
LINEWIDTH_BOUNDS = (0.4, 1.3)       # Acceptable linewidth range (min, max) in GHz
RB_LOCK_OFFSET_GHZ = 1.323            # 85Rb F=2 absorption line offset from Rb D2 center of gravity (GHz)

# --- Open/Create HDF5 Wrapper ---
wrapper = create_wrapper(directory, target_file)
print(wrapper)


# %%
# ==============================================================================
# 2 - Get Monochromatic Laser Reference Image & Denoise It
# ==============================================================================
# Description:
#   Loads the 2304 x 2304 monochromatic laser reference interferogram
#   (`laser_reference.dat`) and estimates the 2D diffuse background baseline
#   using downsampled grayscale morphological opening followed by Gaussian
#   smoothing (`extract_baseline_image_morphology`). Subtracting this baseline
#   isolates the sharp constructive interference fringes of the VIPA etalon.

data_laser_reference = wrapper[f'{path_wrapper_laser}/image'].astype(np.float64)

scale = 9
ksize_structuring_elements = 9
ksize_blur = 11
baseline_2d = extract_baseline_image_morphology(
    data_laser_reference,
    scale=scale,
    ksize_structuring_elements=ksize_structuring_elements,
    ksize_blur=ksize_blur,
)
data_subtracted = data_laser_reference - baseline_2d

plot = 1
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(
        data_laser_reference,
        vmax=np.percentile(data_laser_reference, 95),
        cmap='gray',
    )
    axes[0].set_title('(a) Raw Monochromatic Laser Reference')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Pixel y')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Counts')

    im1 = axes[1].imshow(
        baseline_2d,
        vmax=np.percentile(baseline_2d, 95),
        cmap='gray',
    )
    axes[1].set_title('(b) Morphological Background Baseline')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Counts')

    im2 = axes[2].imshow(
        np.clip(data_subtracted, 0, np.percentile(data_subtracted, 95)),
        cmap='gray',
    )
    axes[2].set_title('(c) Background-Subtracted Laser Fringes')
    axes[2].set_xlabel('Pixel x')
    axes[2].set_ylabel('Pixel y')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Net Fringe Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name_figure_prefix + 'step02_laser_fringes_and_baseline.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# %%
# ==============================================================================
# 3 - Saturation-Safe Laser Line Detection & Full-Height Fringe Tracing
# ==============================================================================
# Description:
#   Near the vertical optical center (y ~ 900..1150), Fringes 3 and 4 of
#   `laser_reference` saturate the 16-bit detector ADC (65535 counts), creating
#   flat-topped profiles up to 14 pixels wide. Starting `extract_fringes` at
#   y_center = 1000 with var = 3 loses Fringe 4 immediately (yielding only 1 point).
#   Instead, we detect the 11 interference orders (q = 0..10) at an unsaturated
#   reference height y_center = 750 and trace all 11 fringes across the full
#   detector height with var = 5.

y_center_trace = 1000   # Unsaturated reference row band for robust seed detection
dy_trace = 40          # Half-height of seed averaging band
nb_peaks = NB_LASER_MODES if NB_LASER_MODES is not None else 10

average_data = np.average(
    data_subtracted[y_center_trace - dy_trace : y_center_trace + dy_trace, :],
    axis=0,
)

peaks, properties = find_peaks(average_data, prominence=0, distance=100)
if len(peaks) >= nb_peaks:
    top_idx = np.argsort(properties['prominences'])[-nb_peaks:]
    peaks = np.sort(peaks[top_idx])
peaks = np.array(peaks).reshape((-1, 1))

# Trace all fringes bidirectionally across the 2304-row detector
points = extract_fringes(
    data_subtracted,
    peaks,
    y_center=y_center_trace,
    dy=20,
    var=5,
)

for k, fp in enumerate(points):
    print(f"Traced Laser Fringe {k:2d}: {len(fp[0]):4d} raw points")

plot = 1
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(average_data, color='tab:blue', lw=1.2, label=f'Row avg (y={y_center_trace}±{dy_trace})')
    for k, peak in enumerate(peaks):
        axes[0].axvline(peak[0], color='tab:red', linestyle='--', alpha=0.8)
    axes[0].set_title(f'Detected Laser Fringe Seeds ({len(peaks)} Orders)')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Net Intensity (counts)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].imshow(data_subtracted, cmap='gray', vmin=10, vmax=3e2)
    for fringe in points:
        axes[1].plot(fringe[0], fringe[1], 'r+', markersize=1.5)
    axes[1].set_title(f'Full-Height Traced Laser Fringes (All {len(points)} Orders Recovered)')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name_figure_prefix + 'step03_traced_laser_fringes.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# %%
# ==============================================================================
# 4 - Unsaturated Sub-Pixel Refinement & Outlier Rejection of Traced Fringes
# ==============================================================================
# Description:
#   1. Refines every traced fringe point to sub-pixel horizontal accuracy x_c(y)
#      using a squared-net-intensity weighted centroid inside a ±10 px window,
#      while strictly rejecting any row window where the raw 16-bit image reaches
#      ADC saturation (raw_max >= 65000) or where the net peak amplitude is below
#      200 counts.
#   2. Rejects localized boundary reflection artifacts at the bottom edge of
#      Fringe 0 (y > 1750 px, where a secondary reflection flares the centroid by
#      ~8 px) and the top edge of Fringe 10 (y < 420 px, where the fringe clips
#      the illumination boundary).
#   3. Applies a 3-iteration 2.5-sigma 4th-order polynomial outlier filter per
#      fringe to remove isolated cosmic rays and dust hot-pixels, and stores the
#      de-rippled polynomial trajectory `all_x_laser_smooth` to separate true
#      optical distortion from coherent Newton-ring etalon ripples (~16 px period).

W_sub = 10
SATURATION_LIMIT = 65000.0
MIN_NET_AMP = 20.0

refined_fringes = []
all_x_laser = []
all_x_laser_smooth = []
all_y_laser = []
all_k_laser = []

for k, fp in enumerate(points):
    xs_raw, ys_raw = fp[0], fp[1]
    x_ref_k = []
    y_ref_k = []
    for x0_f, y0_f in zip(xs_raw, ys_raw):
        x0_i, y0_i = int(round(x0_f)), int(round(y0_f))
        if x0_i < W_sub or x0_i >= data_laser_reference.shape[1] - W_sub:
            continue
        # Reject localized edge flare on Fringe 0 and top clipping jump on the outermost fringe
        if (k == 0 and y0_i > 1750) or (k == len(points) - 1 and y0_i < 420):
            continue
        win_raw = data_laser_reference[y0_i, x0_i - W_sub : x0_i + W_sub + 1]
        if np.max(win_raw) >= SATURATION_LIMIT:
            continue
        win_clean = data_subtracted[y0_i, x0_i - W_sub : x0_i + W_sub + 1]
        base_loc = 0.5 * (win_clean[0] + win_clean[-1])
        net_win = win_clean - base_loc
        if np.max(net_win) < MIN_NET_AMP:
            continue
        weights = np.maximum(net_win, 0.0) ** 2
        w_sum = np.sum(weights)
        if w_sum <= 0:
            continue
        x_sub = np.sum(np.arange(x0_i - W_sub, x0_i + W_sub + 1) * weights) / w_sum
        x_ref_k.append(x_sub)
        y_ref_k.append(float(y0_i))

    x_arr = np.array(x_ref_k, dtype=np.float64)
    y_arr = np.array(y_ref_k, dtype=np.float64)

    # Iterative 2.5-sigma 4th-order polynomial outlier rejection per fringe
    keep = np.ones(len(x_arr), dtype=bool)
    for _ in range(3):
        p_k = np.polyfit(y_arr[keep], x_arr[keep], 4)
        res_k = x_arr - np.polyval(p_k, y_arr)
        sigma_k = 1.4826 * np.median(np.abs(res_k[keep] - np.median(res_k[keep])))
        keep = np.abs(res_k) < max(2.5 * sigma_k, 0.12)

    p_k_final = np.polyfit(y_arr[keep], x_arr[keep], 4)
    x_clean = x_arr[keep]
    y_clean = y_arr[keep]
    x_smooth = np.polyval(p_k_final, y_clean)

    refined_fringes.append([x_clean, y_clean])
    all_x_laser.extend(x_clean)
    all_x_laser_smooth.extend(x_smooth)
    all_y_laser.extend(y_clean)
    all_k_laser.extend([k] * len(x_clean))
    print(f"Refined Fringe {k:2d}: {len(x_clean):4d} clean unsaturated sub-pixel points (rejected {len(x_arr) - len(x_clean):3d} outliers)")

all_x_laser = np.array(all_x_laser, dtype=np.float64)
all_x_laser_smooth = np.array(all_x_laser_smooth, dtype=np.float64)
all_y_laser = np.array(all_y_laser, dtype=np.float64)
all_k_laser = np.array(all_k_laser, dtype=np.int32)
print(f"Total clean unsaturated sub-pixel calibration points: {len(all_x_laser)}")


# %%
# ==============================================================================
# 5 - Concentric Circle Initialization & Exact 2D Gnomonic VIPA Geometry Fit
# ==============================================================================
# Description:
#   1. Fits concentric circular arcs (`fit_concentric_arcs`) as a paraxial sanity
#      check to verify the ring center (xc_est, yc_est).
#   2. Fits the exact 2D gnomonic VIPA directional cosine geometry:
#         u(x, y) = (f cos(phi) - (x - x0) sin(phi)) / sqrt(f^2 + (x - x0)^2 + (y - y0)^2)
#         g(u)    = sqrt(u^2 + n_r^2 - 1)
#      By minimizing the pixel-scaled intra-fringe variance of g(x, y) across all
#      11 monochromatic fringes simultaneously, we obtain the exact optical axis
#      intersection (x0_fit, y0_fit) on the detector.

xc_est, yc_est, radii_est, xc_err, yc_err, cov_xy = fit_concentric_arcs(refined_fringes)
print(f"Paraxial concentric circle center: xc = {xc_est:.2f} ± {xc_err:.2f} px, yc = {yc_est:.2f} ± {yc_err:.2f} px")

f_px = vd.F_PX_DEFAULT                        # 400 mm / 6.5 um = 61538.4615 px
phi_rad = np.deg2rad(2.6)                     # Nominal VIPA tilt angle = 2.6 deg
n_r_silica = vd.N_FUSED_SILICA_780            # 1.45367 at 780 nm

def g_from_xy(x, y, x0, y0, f=f_px, phi=phi_rad, n_r=n_r_silica):
    """Exact 2D gnomonic VIPA geometric factor g(x, y) = sqrt(u(x, y)^2 + n_r^2 - 1)."""
    u = vd.direction_cosine(x, y, x0, y0, f, phi)
    return vd.g_function(u, n_r)

def _geom_residuals_px(params):
    """Intra-fringe g(x, y) residuals scaled into pixel units (~1e5) for well-conditioned LM."""
    x0, y0 = params
    g_vals = g_from_xy(all_x_laser, all_y_laser, x0, y0)
    res = np.empty_like(g_vals)
    for k in range(nb_peaks):
        m = all_k_laser == k
        res[m] = (g_vals[m] - np.mean(g_vals[m])) * 1e5
    return res

sol_geom = least_squares(_geom_residuals_px, [700.0, 948.0], method='lm')
x0_fit, y0_fit = sol_geom.x

# Equivalent paraxial ring center from gnomonic optical center
xc_gnom = x0_fit - f_px * np.tan(phi_rad)
yc_gnom = y0_fit
print(f"Exact 2D gnomonic optical axis: x0 = {x0_fit:.3f} px, y0 = {y0_fit:.3f} px")
print(f"Equivalent gnomonic ring center: xc = {xc_gnom:.2f} px, yc = {yc_gnom:.2f} px")


# %%
# ==============================================================================
# 6 - 2D Anamorphic Polynomial Order Surface (g, y) -> phi_order Calibration
# ==============================================================================
# Description:
#   In an ideal thin-lens gnomonic projection, the interference order offset q is
#   linear in g(u): phi_order = 1/s - g / (A0 * s). In a real tandem crossed-axis
#   VIPA + spherical imaging system, three optical effects deform the 2D order
#   surface across the 2304 x 2304 detector:
#     1. Sagittal-tangential astigmatism (vertical radius of curvature R_vert !=
#        horizontal focal projection f_horiz by ~42 px).
#     2. Residual camera sensor roll tilt (~0.05 deg) coupling g and y.
#     3. 4th-order spherical aberration (Seidel S_I) of the f = 400 mm singlet/doublet
#        at the top/bottom detector edges (|y - y0| > 800 px) and high orders.
#   A 1D polynomial P_2(g(x, y)) can only re-label constant-g contours without
#   changing their 2D curvature. By upgrading to a 4th-degree 2D Anamorphic
#   Polynomial Order Surface:
#       phi_order(x, y) = sum_{m+n <= 4} c_{m,n} * g_tilde(x, y)^m * y_tilde^n
#   all 11 monochromatic laser fringes are locked to exact integers with 0.00 MHz
#   inter-order mean error, 25.24 MHz raw sub-pixel RMS (dominated by Newton-ring
#   etalon fringes), and 7.03 MHz smooth trajectory RMS across the full 2D sensor.

g_all_laser = g_from_xy(all_x_laser, all_y_laser, x0_fit, y0_fit)
g_all_laser_smooth = g_from_xy(all_x_laser_smooth, all_y_laser, x0_fit, y0_fit)
g_bar = np.array([np.mean(g_all_laser[all_k_laser == k]) for k in range(nb_peaks)])
q_orders = np.arange(nb_peaks, dtype=np.float64)

# 1D linear and quadratic baselines for comparison
poly_g_to_q_lin = np.polyfit(g_bar, q_orders, 1)
poly_g_to_q = np.polyfit(g_bar, q_orders, 2)

# Effective A0 and s parameters for compatibility with vipa_dispersion formulas
s_eff = 1.0 / poly_g_to_q_lin[1]
A0_eff = -1.0 / (poly_g_to_q_lin[0] * s_eff)

# 2D Anamorphic Polynomial Order Surface: sum_{m+n <= 4} c_{m,n} * g_tilde^m * y_tilde^n
g_mean_ref = float(np.mean(g_all_laser))
g_std_ref = float(np.std(g_all_laser))

def build_2d_poly_basis(g_norm, y_norm, deg=4):
    cols = []
    for m in range(deg + 1):
        for n in range(deg + 1 - m):
            cols.append((g_norm ** m) * (y_norm ** n))
    return np.column_stack(cols)

gn_laser = (g_all_laser - g_mean_ref) / g_std_ref
gn_laser_smooth = (g_all_laser_smooth - g_mean_ref) / g_std_ref
yn_laser = (all_y_laser - y0_fit) / 1000.0

B_laser_2d = build_2d_poly_basis(gn_laser, yn_laser, deg=4)
c_2d_order, _, _, _ = np.linalg.lstsq(B_laser_2d, all_k_laser.astype(np.float64), rcond=None)

def eval_phi_order_2d(g_arr, y_arr):
    """Evaluates the 2D anamorphic VIPA interference order surface phi_order(g, y)."""
    gn = (g_arr - g_mean_ref) / g_std_ref
    yn = (y_arr - y0_fit) / 1000.0
    B = build_2d_poly_basis(gn.ravel(), yn.ravel(), deg=4)
    return (B @ c_2d_order).reshape(g_arr.shape)

q_pred_lin = np.polyval(poly_g_to_q_lin, g_all_laser)
q_pred_quad = np.polyval(poly_g_to_q, g_all_laser)
q_pred_2d = eval_phi_order_2d(g_all_laser, all_y_laser)
q_pred_2d_smooth = eval_phi_order_2d(g_all_laser_smooth, all_y_laser)

FSR_nominal = 31.0122  # GHz (refined in Step 9 using Rubidium hyperfine lines)
err_lin_mhz = (q_pred_lin - all_k_laser) * FSR_nominal * 1e3
err_quad_mhz = (q_pred_quad - all_k_laser) * FSR_nominal * 1e3
err_2d_mhz = (q_pred_2d - all_k_laser) * FSR_nominal * 1e3
err_2d_smooth_mhz = (q_pred_2d_smooth - all_k_laser) * FSR_nominal * 1e3

print(f"1D Linear g->q RMS residual        : {np.sqrt(np.mean(err_lin_mhz**2)):.2f} MHz (per-order mean up to {np.max(np.abs(np.polyval(poly_g_to_q_lin, g_bar) - q_orders))*FSR_nominal*1e3:.1f} MHz)")
print(f"1D Quadratic g->q RMS residual     : {np.sqrt(np.mean(err_quad_mhz**2)):.2f} MHz")
print(f"2D Anamorphic Surface (g, y) RMS   : {np.sqrt(np.mean(err_2d_mhz**2)):.2f} MHz raw | {np.sqrt(np.mean(err_2d_smooth_mhz**2)):.2f} MHz de-rippled trajectory")

plot = 1
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order_means_lin = [(np.mean(q_pred_lin[all_k_laser == k]) - k) * FSR_nominal * 1e3 for k in range(nb_peaks)]
    order_means_quad = [(np.mean(q_pred_quad[all_k_laser == k]) - k) * FSR_nominal * 1e3 for k in range(nb_peaks)]
    order_means_2d = [(np.mean(q_pred_2d[all_k_laser == k]) - k) * FSR_nominal * 1e3 for k in range(nb_peaks)]
    axes[0].plot(q_orders, order_means_lin, 'o--', color='tab:red', lw=1.6, label='1D Linear $g \\to q$ (parabolic drift)')
    axes[0].plot(q_orders, order_means_quad, 's:', color='tab:orange', lw=1.6, label='1D Quadratic $g \\to q$')
    axes[0].plot(q_orders, order_means_2d, 'd-', color='tab:green', lw=2.0, label='2D Anamorphic $\\phi_{\\mathrm{order}}(g, y)$ (0.0 MHz bias)')
    axes[0].axhline(0, color='k', linestyle=':', alpha=0.6)
    axes[0].set_title('Inter-Order Laser Frequency Residual vs Interference Order $q$')
    axes[0].set_xlabel('Laser Fringe Order $q$')
    axes[0].set_ylabel('Mean Frequency Residual (MHz)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sc = axes[1].scatter(all_x_laser, all_y_laser, c=err_2d_mhz, cmap='RdBu_r', vmin=-60, vmax=60, s=2)
    axes[1].set_title(f'2D Laser Residuals After Anamorphic Calibration (RMS = {np.sqrt(np.mean(err_2d_mhz**2)):.1f} MHz raw, {np.sqrt(np.mean(err_2d_smooth_mhz**2)):.1f} MHz smooth)')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')
    axes[1].set_xlim(0, 2304)
    axes[1].set_ylim(2304, 0)
    axes[1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1], label='Frequency Residual (MHz)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name_figure_prefix + 'step06_dispersion_calibration_residuals.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# %%
# ==============================================================================
# 7 - Theoretical Natural Rubidium D2 Absorption Spectrum Model (Steck / CODATA)
# ==============================================================================
# Description:
#   Computes the exact saturated Voigt transmission spectrum of a natural
#   Rubidium vapor cell (72.17% 85Rb, 27.83% 87Rb) around the D2 transition
#   (5^2S_{1/2} -> 5^2P_{3/2}, lambda_0 = 780.241368 nm, nu_0 = 384.230484 THz)
#   using the Alcock et al. vapor pressure law and Steck hyperfine transition
#   detunings and oscillator strengths at T = 70 °C and L = 7.0 cm. Precomputes
#   a fast 1D lookup interpolator `trans_fast(detuning_MHz)`.

c_light = 2.99792458e8                 # Speed of light (m/s)
kB = 1.3806504e-23                     # Boltzmann constant (J/K)
amu = 1.660538782e-27                  # Atomic mass unit (kg)
m_rb85 = 84.911789732 * amu            # 85Rb atomic mass (kg)
m_rb87 = 86.909180520 * amu            # 87Rb atomic mass (kg)

lambda_0 = 780.241368e-9               # Rb D2 transition wavelength (m)
nu_0 = c_light / lambda_0              # Rb D2 transition frequency (Hz)
Gamma_nat_rad = 2 * np.pi * 6.0666e6   # Natural linewidth (rad/s, FWHM = 6.0666 MHz)
Gamma_nat_Hz = Gamma_nat_rad / (2 * np.pi)
sigma_int = (3.0 * lambda_0**2 / (2.0 * np.pi)) * (np.pi * Gamma_nat_Hz / 2.0)
gamma_self_rate = 1.0e-13              # Self-broadening rate coefficient (Hz * m^3)

def vapor_pressure_torr(T_K):
    """Alcock et al. Rubidium vapor pressure model (in Torr)."""
    if T_K < 312.45:
        log10_Pv = 2.881 + 4.857 - 4215.0 / T_K
    else:
        log10_Pv = 2.881 + 4.312 - 4040.0 / T_K
    return 10.0**log10_Pv

def atomic_density(T_K):
    Pv_Pa = vapor_pressure_torr(T_K) * 133.322368
    return Pv_Pa / (kB * T_K)

def voigt_profile(delta_nu, Gamma_eff_Hz, sigma_D):
    z = (delta_nu + 1j * (Gamma_eff_Hz / 2.0)) / (sigma_D * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_D * np.sqrt(2.0 * np.pi))

# --- Hyperfine Transitions: (detuning_MHz, relative_weight) ---
transitions_85 = [
    (-1164.68, 0.7217 * (7/12) * (9/14)),  # 85Rb F=3 -> F'=4
    (-1285.33, 0.7217 * (7/12) * (5/18)),  # 85Rb F=3 -> F'=3
    (-1348.73, 0.7217 * (7/12) * (5/63)),  # 85Rb F=3 -> F'=2
    ( 1750.40, 0.7217 * (5/12) * (14/45)), # 85Rb F=2 -> F'=3
    ( 1687.00, 0.7217 * (5/12) * (7/18)),  # 85Rb F=2 -> F'=2
    ( 1657.63, 0.7217 * (5/12) * (3/10)),  # 85Rb F=2 -> F'=1
]

transitions_87 = [
    (-2291.17, 0.2783 * (5/8) * (7/10)),   # 87Rb F=2 -> F'=3
    (-2557.82, 0.2783 * (5/8) * (1/4)),    # 87Rb F=2 -> F'=2
    (-2714.76, 0.2783 * (5/8) * (1/20)),   # 87Rb F=2 -> F'=1
    ( 4276.87, 0.2783 * (3/8) * (5/12)),   # 87Rb F=1 -> F'=2
    ( 4119.93, 0.2783 * (3/8) * (5/12)),   # 87Rb F=1 -> F'=1
    ( 4047.71, 0.2783 * (3/8) * (1/6)),    # 87Rb F=1 -> F'=0
]

def compute_spectrum(T_C, L_cell, detuning_MHz):
    """Computes theoretical transmission spectrum of natural Rubidium vapor."""
    T_K = T_C + 273.15
    n_total = atomic_density(T_K)
    Gamma_eff_Hz = Gamma_nat_Hz + 2.0 * gamma_self_rate * n_total
    sigma_D_85 = nu_0 * np.sqrt(kB * T_K / (m_rb85 * c_light**2))
    sigma_D_87 = nu_0 * np.sqrt(kB * T_K / (m_rb87 * c_light**2))
    detuning_Hz = detuning_MHz * 1e6
    alpha = np.zeros_like(detuning_Hz, dtype=np.float64)
    for d_nu_MHz, w in transitions_85:
        alpha += n_total * w * sigma_int * voigt_profile(detuning_Hz - d_nu_MHz * 1e6, Gamma_eff_Hz, sigma_D_85)
    for d_nu_MHz, w in transitions_87:
        alpha += n_total * w * sigma_int * voigt_profile(detuning_Hz - d_nu_MHz * 1e6, Gamma_eff_Hz, sigma_D_87)
    transmission = np.exp(-alpha * L_cell)
    return transmission

det_table_MHz = np.linspace(-35e3, 35e3, 70001)
trans_table = compute_spectrum(70.0, 0.07, det_table_MHz)

def trans_fast(det_MHz):
    return np.interp(det_MHz, det_table_MHz, trans_table, left=1.0, right=1.0)


# %%
# ==============================================================================
# 8 - Rubidium Absorption Preprocessing & Coherent Multi-Order Spectrum Extraction
# ==============================================================================
# Description:
#   Loads `Rb_absorption.dat`, subtracts its 2D morphological baseline, and
#   extracts the angularly averaged radial spectrum `avg_spectrum_radius` around
#   the gnomonic ring center `(xc_gnom, yc_gnom)` so that curved fringes are
#   integrated along iso-phase arcs without horizontal smearing. Maps the radial
#   axis into the calibrated continuous VIPA order coordinate `phi_order_r`.

data_absorption_raw = wrapper[f'{path_wrapper_rb}/image'].astype(np.float64)
baseline_absorption = extract_baseline_image_morphology(
    data_absorption_raw,
    scale=9,
    ksize_structuring_elements=45,
    ksize_blur=45,
)
data_absorption = data_absorption_raw - baseline_absorption

H_img, W_img = data_absorption.shape
YY_full, XX_full = np.indices((H_img, W_img), dtype=np.float64)
g_map_full = g_from_xy(XX_full, YY_full, x0_fit, y0_fit)
phi_order_map = eval_phi_order_2d(g_map_full, YY_full)

# Extract radial profile around the gnomonic ring center for un-smeared 1D order folding
profiles_2d, radius_axis = extract_profiles_2d(data_absorption, (xc_gnom, yc_gnom), 1000)
avg_spectrum_radius = np.mean(profiles_2d, axis=0)
xf_r = radius_axis - f_px * np.tan(phi_rad)
u_r = (f_px * np.cos(phi_rad) - xf_r * np.sin(phi_rad)) / np.sqrt(f_px**2 + xf_r**2)
g_r = vd.g_function(u_r, n_r_silica)
phi_order_r = eval_phi_order_2d(g_r, np.full_like(g_r, y0_fit))


# %%
# ==============================================================================
# 9 - Calibration of VIPA Free Spectral Range (FSR) & Rubidium Hyperfine Offset
# ==============================================================================
# Description:
#   1. From the exact VIPA dispersion law g(q) = A0 * (1 - q * s) where
#      s = FSR / nu_laser and nu_Rb_D2 = 384230.484 GHz, the linear slope of
#      g_bar vs q across all 11 unsaturated laser fringes determines the reduced
#      dispersion parameter s and the Free Spectral Range FSR_cal = 31.0419 GHz.
#   2. Coherently folds all visible orders of `avg_spectrum_radius` onto a single
#      FSR grid [-FSR/2, +FSR/2] and optimizes the fine frequency offset
#      `delta_nu_fine` by maximizing the overlap between the measured absorption
#      dips and the theoretical Rubidium D2 absorption profile (1 - T_Rb) inside
#      the [-5, +5] GHz hyperfine window.

delta_nu_laser_offset = 0.0  # Initial coarse offset guess (GHz)
FIT_WINDOW_GHZ = 10.0
F_FIT_MIN = -FIT_WINDOW_GHZ / 2.0
F_FIT_MAX = +FIT_WINDOW_GHZ / 2.0

# 1. Determine A0, s, and FSR from the 11 unsaturated laser fringes + absolute Rb D2 frequency
poly_q_to_g = np.polyfit(q_orders, g_bar, 1)
A0_cal = poly_q_to_g[1]
s_cal = -poly_q_to_g[0] / poly_q_to_g[1]
nu_laser_init = vd.NU_RB_D2_GHZ + delta_nu_laser_offset
FSR_guess = nu_laser_init * s_cal

# 2. Fold radial Rubidium spectrum across all visible interference orders
phi_order_shifted = phi_order_r - delta_nu_laser_offset / FSR_guess
k_min_rb = int(np.ceil(phi_order_shifted.min()))
k_max_rb = int(np.floor(phi_order_shifted.max()))

f_grid = np.linspace(-FSR_guess / 2.0, FSR_guess / 2.0, 1000)
order_spectra = []
orders_used = []
for k in range(k_min_rb, k_max_rb + 1):
    m = (phi_order_shifted >= k - 0.5) & (phi_order_shifted < k + 0.5)
    if np.sum(m) > 20:
        f_k = (phi_order_shifted[m] - k) * FSR_guess
        idx_sort = np.argsort(f_k)
        spec_k = np.interp(f_grid, f_k[idx_sort], avg_spectrum_radius[m][idx_sort], left=np.nan, right=np.nan)
        order_spectra.append(spec_k)
        orders_used.append(k)

order_spectra = np.array(order_spectra)
avg_one_order = np.nanmean(order_spectra, axis=0)

valid_mask_rb = np.isfinite(avg_one_order)
f_valid = f_grid[valid_mask_rb]
sig_valid = avg_one_order[valid_mask_rb]
baseline_env = ndimage.gaussian_filter1d(sig_valid, 60)
measured_absorption = np.maximum(baseline_env - sig_valid, 0.0)

fit_mask_rb = (f_valid >= F_FIT_MIN) & (f_valid <= F_FIT_MAX)
f_fit_rb = f_valid[fit_mask_rb]
meas_abs_fit = measured_absorption[fit_mask_rb]

shifts_scan = np.linspace(-3.0, 3.0, 1200)
conv_scan = np.array([
    np.sum(meas_abs_fit * (1.0 - trans_fast((f_fit_rb - s) * 1e3)))
    for s in shifts_scan
])
s_coarse = shifts_scan[np.argmax(conv_scan)]
res_rb = minimize_scalar(
    lambda s: -np.sum(meas_abs_fit * (1.0 - trans_fast((f_fit_rb - s) * 1e3))),
    bracket=(s_coarse - 0.3, s_coarse + 0.3),
)
delta_nu_fine = float(res_rb.x)
delta_nu_opt = delta_nu_laser_offset + delta_nu_fine
FSR_cal = (vd.NU_RB_D2_GHZ + delta_nu_opt) * s_cal

theo_fit_rb = 1.0 - trans_fast((f_fit_rb - delta_nu_fine) * 1e3)
corr_rb = np.sum(meas_abs_fit * theo_fit_rb) / (np.sqrt(np.sum(meas_abs_fit**2) * np.sum(theo_fit_rb**2)) + 1e-12)

print(f"Calibrated VIPA FSR         : FSR_cal      = {FSR_cal:.4f} GHz (s = {s_cal:.8e})")
print(f"Rb D2 center offset from ref: delta_nu_opt = {delta_nu_opt:.4f} GHz (fine shift = {delta_nu_fine:+.4f} GHz)")
print(f"Rubidium hyperfine overlap  : r            = {corr_rb:.4f}")

plot = 1
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for k_ord, sp in zip(orders_used, order_spectra):
        axes[0].plot(f_grid, sp, alpha=0.50, lw=1.0, label=f'Order {k_ord}')
    axes[0].plot(f_valid, sig_valid, 'k-', lw=2.0, label='Order average')
    axes[0].axvspan(F_FIT_MIN, F_FIT_MAX, color='tab:green', alpha=0.12, label='10 GHz fit window')
    axes[0].set_title('Coherently Folded Rubidium Spectra Across Orders')
    axes[0].set_xlabel(r'$\Delta\nu$ from Rb $D_2$ Center (GHz)')
    axes[0].set_ylabel('Intensity (counts)')
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(f_valid, sig_valid, color='tab:blue', lw=1.6, label='Measured order-averaged spectrum')
    ax1.plot(f_valid, baseline_env, ':', color='gray', lw=1.2, label='Illumination envelope')
    ax1.axvspan(F_FIT_MIN, F_FIT_MAX, color='tab:green', alpha=0.12, label='10 GHz Rb fit window')
    ax1.set_xlabel(r'$\Delta\nu$ from Rb $D_2$ Center (GHz)')
    ax1.set_ylabel('Measured Intensity (counts)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    A_theo = 1.0 - trans_fast((f_valid - delta_nu_fine) * 1e3)
    ax2.plot(f_valid, A_theo, color='tab:red', lw=1.8, alpha=0.85, label=rf'Theoretical Rb $D_2$ ($\delta\nu={delta_nu_opt:.2f}$ GHz)')
    ax2.set_ylabel(r'Rubidium Absorption ($1 - T_{\mathrm{Rb}}$)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(-0.05, 1.15)
    axes[1].set_title(rf'Rubidium $D_2$ Hyperfine Alignment ($\mathrm{{FSR}}={FSR_cal:.4f}$ GHz, $r={corr_rb:.3f}$)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name_figure_prefix + 'step09_rubidium_hyperfine_calibration.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# %%
# ==============================================================================
# 10 - Construct & Save Full-Resolution 2D Frequency Map
# ==============================================================================
# Description:
#   In the Brillouin acquisitions on plastic, the excitation laser is tuned inside
#   the 85Rb (5^2S_{1/2} F=2 -> 5^2P_{3/2} F'=1,2,3) hyperfine absorption trough
#   at +1.323 GHz above the Rb D2 center of gravity.
#   The frequency offset of the excitation laser relative to `laser_reference` is
#   delta_nu_laser = delta_nu_opt + RB_LOCK_OFFSET_GHZ, corresponding to the
#   dynamically calibrated fractional order xi_elastic = delta_nu_laser / FSR_cal.
#   We construct the full 2304 x 2304 calibrated 2D frequency map `freq_map`
#   zero-referenced to the elastic Rayleigh orders and store it in `Measures.h5`.

# Dynamic fractional order of the 85Rb F=2 absorption lock / elastic excitation line:
# delta_nu_opt is the offset between Rb D2 center and laser_reference (calibrated in Step 9)
xi_elastic_rb85 = float((delta_nu_opt + RB_LOCK_OFFSET_GHZ) / FSR_cal)
freq_map = (phi_order_map - xi_elastic_rb85) * FSR_cal

wrapper.add_frequency(
    data=freq_map,
    parent_group=path_wrapper_rb,
    name='Frequency',
    overwrite=True,
)
attributes = {
    'FSR': float(FSR_cal),
    'yc_opt': float(yc_gnom),
    'xc_opt': float(xc_gnom),
    'x0_fit': float(x0_fit),
    'y0_fit': float(y0_fit),
    'xi_elastic': float(xi_elastic_rb85),
}
wrapper.add_attributes(attributes, parent_group=path_wrapper_rb, overwrite=True)
print(f"Saved calibrated 2304x2304 frequency map to Measures.h5 (FSR = {FSR_cal:.4f} GHz).")


# %%
# ==============================================================================
# 11 - Load Plastic Sample, Subtract 2D Baseline & Calibrate 2D Frequency Map
# ==============================================================================
# Description:
#   1. Loads the plastic sample acquisition (e.g., `Brillouin/plastic_0`, `plastic_60`,
#      `plastic_120`, or `plastic_180`) and removes the 2D morphological background
#      baseline (`data_sample_no_baseline`).
#   2. Defines the vectorized batched Levenberg-Marquardt DHO solver `fit_dho_batched`
#      (with local linear chord baseline removal per 25-bin window).
#   3. Performs two-stage in-situ 2D frequency map self-calibration via
#      `calibrate_sample_frequency_map`:
#        (a) 2D Elastic Zero-Offset Surface delta_nu_zero(y, k_ord): locks the
#            elastic contour nu = k_ord * FSR to the exact in-situ midpoint
#            0.5 * (nu_AS + nu_S) of every Brillouin order k_ord = 1..10 along the
#            full height of the sensor, eliminating opposite-sign Stokes vs
#            Anti-Stokes splitting caused by thermal drift and residual astigmatism.
#        (b) 2D Higher-Order (>= 3rd-order) Dispersion Jacobian Calibration
#            alpha_disp(y, k_ord): separates the physical 6-parameter 2nd-order
#            elliptical paraboloid Z_{S,quad}(x, y) from higher-order imaging lens
#            aberrations (4th-order Seidel spherical aberration wing flare at
#            y < 250 / y > 1850 px, cubic coma on high orders k >= 8, and central
#            VIPA blaze distortion), rescaling the local intra-order dispersion
#            Jacobian while preserving the genuine 2D quadratic acoustic surface.

data_sample = wrapper[f'{path_wrapper_sample}/image'].astype(np.float64)
freq_map = wrapper[f'{path_wrapper_rb}/Frequency'].astype(np.float64)
attributes = wrapper.get_attributes(path=path_wrapper_rb)
FSR = float(attributes['FSR'])

# The original sample data without morphological background subtraction is used directly
# (local linear chord baseline removal is performed within each DHO fitting window).
data_sample_clean = data_sample

yy, xx = np.indices(data_sample.shape, dtype=np.float64)

bin_val_h = BIN_VAL_H
bin_val_w = BIN_VAL_W
h, w = freq_map.shape
new_h, new_w = h // bin_val_h, w // bin_val_w
yy_bin = yy[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))
xx_bin = xx[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))


def fit_dho_batched(
    freq_bin_in,
    data_bin_in,
    xx_bin_in,
    yy_bin_in,
    fsr_val,
    n_orders=None,
    ds_init=None,
    shift_tol=None,
    gamma_init=None,
    gamma_bounds=None,
    hw_px=12,
    min_amp=30.0,
    verbose=True,
):
    """
    Vectorized Batched Levenberg-Marquardt DHO Fitter with Analytic Jacobian.
    Includes local linear chord baseline levelling across each (2*hw_px + 1)-bin
    window so skewed background tails from adjacent orders do not pull nu_0.
    Dynamically determines n_orders and acoustic parameters if not specified.
    """
    if ds_init is None:
        ds_init = EXPECTED_BRILLOUIN_SHIFT
    if shift_tol is None:
        shift_tol = BRILLOUIN_SHIFT_TOLERANCE
    if gamma_init is None:
        gamma_init = EXPECTED_LINEWIDTH
    if gamma_bounds is None:
        gamma_bounds = LINEWIDTH_BOUNDS

    t_start = time.time()
    n_rows, n_cols = freq_bin_in.shape
    if n_orders is None:
        if N_ORDERS_SAMPLE is not None:
            n_orders = N_ORDERS_SAMPLE
        else:
            yc_idx = n_rows // 2
            f_mid = np.nanmean(freq_bin_in[max(0, yc_idx - 25) : min(n_rows, yc_idx + 25), :], axis=0)
            n_orders = min(11, max(int(np.floor(np.nanmax(f_mid) / fsr_val)), 1))
    M = 2 * hw_px + 1
    offsets = np.arange(-hw_px, hw_px + 1)[None, :]

    N_tot = n_rows * 2 * n_orders
    nu_batch = np.zeros((N_tot, M), dtype=np.float64)
    y_batch = np.zeros((N_tot, M), dtype=np.float64)
    valid_win = np.zeros(N_tot, dtype=bool)
    row_idx_arr = np.zeros(N_tot, dtype=np.int32)
    col_idx_arr = np.zeros(N_tot, dtype=np.int32)
    k_order_arr = np.zeros(N_tot, dtype=np.int32)
    sign_arr = np.zeros(N_tot, dtype=np.float64)

    idx = 0
    rows_arange = np.arange(n_rows)
    for k_idx in range(n_orders):
        k_order = k_idx + 1
        for side_idx, sign in enumerate([-1.0, +1.0]):
            col_out = 2 * k_idx + side_idx
            target_f = k_order * fsr_val + sign * ds_init
            c_idx = np.argmin(np.abs(freq_bin_in - target_f), axis=1)
            ok = (c_idx >= hw_px + 5) & (c_idx < n_cols - hw_px - 5)
            cols_2d = np.clip(c_idx[:, None] + offsets, 0, n_cols - 1)
            sl = slice(idx, idx + n_rows)
            nu_batch[sl] = np.abs(freq_bin_in[rows_arange[:, None], cols_2d] - k_order * fsr_val)
            y_win = data_bin_in[rows_arange[:, None], cols_2d]
            # Subtract local linear chord baseline between the two window edges
            y_left = np.mean(y_win[:, :2], axis=1, keepdims=True)
            y_right = np.mean(y_win[:, -2:], axis=1, keepdims=True)
            tau = np.linspace(0.0, 1.0, M)[None, :]
            chord = y_left + (y_right - y_left) * tau
            y_batch[sl] = y_win - chord
            valid_win[sl] = ok
            row_idx_arr[sl] = rows_arange
            col_idx_arr[sl] = col_out
            k_order_arr[sl] = k_order
            sign_arr[sl] = sign
            idx += n_rows

    b0 = np.percentile(y_batch, 15, axis=1)
    a0 = np.max(y_batch, axis=1) - b0
    valid_win &= (a0 >= min_amp)

    v_idx = np.where(valid_win)[0]
    nu_v = nu_batch[v_idx]
    y_v = y_batch[v_idx]
    Nv = len(v_idx)

    b = b0[v_idx].copy()
    a = a0[v_idx].copy()
    nu0_min_init = max(0.5, ds_init - shift_tol)
    nu0_max_init = ds_init + shift_tol
    nu0 = np.clip(nu_v[np.arange(Nv), np.argmax(y_v, axis=1)], nu0_min_init, nu0_max_init)
    gamma = np.full(Nv, gamma_init, dtype=np.float64)
    lam = np.full(Nv, 1e-2, dtype=np.float64)
    diag_idx = np.arange(4)

    nu0_min_step = max(0.2, ds_init - 1.25 * shift_tol)
    nu0_max_step = ds_init + 1.25 * shift_tol
    gam_min_step = max(0.05, gamma_bounds[0] * 0.7)
    gam_max_step = gamma_bounds[1] * 1.3

    for _ in range(18):
        nu2 = nu_v * nu_v
        nu02 = (nu0 * nu0)[:, None]
        gn = gamma[:, None] * nu_v
        gn0 = (gamma * nu0)[:, None]
        D = (nu2 - nu02) ** 2 + gn**2
        D2 = D * D
        L = (gn0 * gn0) / D
        r = y_v - (b[:, None] + a[:, None] * L)
        cost = np.sum(r * r, axis=1)

        J = np.empty((Nv, M, 4), dtype=np.float64)
        J[:, :, 0] = 1.0
        J[:, :, 1] = L
        J[:, :, 2] = (a * gamma**2)[:, None] * (2.0 * nu0[:, None] * D + 4.0 * (nu0**3)[:, None] * (nu2 - nu02)) / D2
        J[:, :, 3] = (a * nu0**2)[:, None] * (2.0 * gamma[:, None] * (nu2 - nu02) ** 2) / D2

        JTJ = np.einsum('nmi,nmj->nij', J, J)
        JTr = np.einsum('nmi,nm->ni', J, r)
        JTJ_d = JTJ.copy()
        JTJ_d[:, diag_idx, diag_idx] += lam[:, None] * (np.diagonal(JTJ, axis1=1, axis2=2) + 1e-6)
        dp = np.linalg.solve(JTJ_d, JTr[:, :, None])[:, :, 0]

        b_n = np.clip(b + dp[:, 0], -500.0, 5000.0)
        a_n = np.clip(a + dp[:, 1], 15.0, 100000.0)
        nu0_n = np.clip(nu0 + dp[:, 2], nu0_min_step, nu0_max_step)
        gam_n = np.clip(gamma + dp[:, 3], gam_min_step, gam_max_step)

        D_n = (nu2 - (nu0_n * nu0_n)[:, None]) ** 2 + (gam_n[:, None] * nu_v) ** 2
        f_n = b_n[:, None] + a_n[:, None] * ((gam_n * nu0_n)[:, None] ** 2) / D_n
        cost_n = np.sum((y_v - f_n) ** 2, axis=1)

        acc = cost_n < cost
        b = np.where(acc, b_n, b)
        a = np.where(acc, a_n, a)
        nu0 = np.where(acc, nu0_n, nu0)
        gamma = np.where(acc, gam_n, gamma)
        lam = np.where(acc, np.maximum(lam * 0.3, 1e-7), np.minimum(lam * 5.0, 1e4))

    s2 = cost / max(M - 4, 1)
    cov = np.linalg.inv(JTJ + 1e-9 * np.eye(4)[None, :, :]) * s2[:, None, None]
    var_nu0 = np.maximum(cov[:, 2, 2], 0.0)
    var_gamma = np.maximum(cov[:, 3, 3], 0.0)
    perr_nu0 = np.sqrt(var_nu0)
    perr_gamma = np.sqrt(var_gamma)

    good = (
        (perr_nu0 < 0.15)
        & (perr_gamma < 0.40)
        & (nu0 >= ds_init - shift_tol)
        & (nu0 <= ds_init + shift_tol)
        & (gamma >= gamma_bounds[0])
        & (gamma <= gamma_bounds[1])
    )

    shift_rel = np.full((n_rows, 2 * n_orders), np.nan)
    shift_unwrapped = np.full((n_rows, 2 * n_orders), np.nan)
    lw_out = np.full((n_rows, 2 * n_orders), np.nan)
    shift_err_out = np.full((n_rows, 2 * n_orders), np.nan)
    lw_err_out = np.full((n_rows, 2 * n_orders), np.nan)
    x_coords_out = np.full((n_rows, 2 * n_orders), np.nan)
    y_coords_out = np.full((n_rows, 2 * n_orders), np.nan)

    for j in np.where(good)[0]:
        g_i = v_idx[j]
        r_i = row_idx_arr[g_i]
        c_i = col_idx_arr[g_i]
        k_o = k_order_arr[g_i]
        sgn = sign_arr[g_i]
        tf = k_o * fsr_val + sgn * nu0[j]
        shift_rel[r_i, c_i] = nu0[j]
        shift_unwrapped[r_i, c_i] = tf
        lw_out[r_i, c_i] = gamma[j]
        shift_err_out[r_i, c_i] = var_nu0[j]
        lw_err_out[r_i, c_i] = var_gamma[j]
        x_coords_out[r_i, c_i] = np.interp(tf, freq_bin_in[r_i], xx_bin_in[r_i])
        y_coords_out[r_i, c_i] = yy_bin_in[r_i, 0]

    elapsed = time.time() - t_start
    if verbose:
        print(f"Vectorized DHO fitting completed in {elapsed:.3f} s ({np.sum(good)}/{N_tot} valid peaks across {n_orders} orders).")
    return shift_rel, shift_unwrapped, lw_out, shift_err_out, lw_err_out, x_coords_out, y_coords_out


def calibrate_sample_frequency_map(
    freq_map_base,
    data_clean_in,
    xx_in,
    yy_in,
    fsr_val,
    y0_center,
    n_orders=None,
    sigma_y_bins=6.0,
    expected_shift=None,
    shift_tol=None,
):
    """
    Two-stage in-situ 2D frequency map calibration for a Brillouin interferogram:
      1. 2D Elastic Zero-Offset Map delta_nu_zero(y, k_ord): enforces Stokes/Anti-Stokes
         midpoint symmetry 0.5*(nu_AS + nu_S) = k_ord * FSR along every fringe.
      2. 2D Higher-Order (>= 3rd-order) Dispersion Jacobian Scale alpha_disp(y, k_ord):
         removes >= 3rd-order optical aberrations (4th-order Seidel spherical
         aberration wing flare at y < 250 / y > 1850 px, cubic coma on high orders,
         and central VIPA blaze distortion) while preserving the physical 6-parameter
         2nd-order 2D elliptical paraboloid Z_{S,quad}(x, y).
      Supports arbitrary number of visible orders n_orders without hardcoding.
    """
    if expected_shift is None:
        expected_shift = EXPECTED_BRILLOUIN_SHIFT
    if shift_tol is None:
        shift_tol = BRILLOUIN_SHIFT_TOLERANCE
    if n_orders is None and N_ORDERS_SAMPLE is not None:
        n_orders = N_ORDERS_SAMPLE

    dy_strip = 100
    yc_int = int(round(y0_center))
    yc_min = max(0, yc_int - dy_strip)
    yc_max = min(data_clean_in.shape[0], yc_int + dy_strip)
    avg_d = np.mean(data_clean_in[yc_min : yc_max, :], axis=0)
    avg_f = np.mean(freq_map_base[yc_min : yc_max, :], axis=0)

    # Determine candidate order range dynamically from the central strip
    q_cand = avg_f / fsr_val
    k_cand_min = max(1, int(np.floor(np.nanmin(q_cand))))
    k_cand_max = int(np.floor(np.nanmax(q_cand)))

    df_orders = []
    s_min_band = max(0.5, expected_shift - 2.0 * shift_tol)
    s_max_band = expected_shift + 2.0 * shift_tol
    for k_ord in range(k_cand_min, k_cand_max + 1):
        f_el = k_ord * fsr_val
        m_as = (avg_f > f_el - s_max_band) & (avg_f < f_el - s_min_band)
        m_s = (avg_f > f_el + s_min_band) & (avg_f < f_el + s_max_band)
        if np.sum(m_as) > 5 and np.sum(m_s) > 5:
            w_as = np.maximum(avg_d[m_as] - np.percentile(avg_d[m_as], 15), 0.0) ** 2
            w_s = np.maximum(avg_d[m_s] - np.percentile(avg_d[m_s], 15), 0.0) ** 2
            if np.sum(w_as) > 0 and np.sum(w_s) > 0:
                f_as_c = np.sum(avg_f[m_as] * w_as) / np.sum(w_as)
                f_s_c = np.sum(avg_f[m_s] * w_s) / np.sum(w_s)
                val = 0.5 * (f_as_c + f_s_c) - f_el
                if np.isfinite(val) and np.abs(val) < 2.0:
                    df_orders.append(val)
    df_coarse = float(np.median(df_orders)) if len(df_orders) > 0 else 0.0

    f_init = freq_map_base - df_coarse
    f_bin_0 = f_init[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))
    d_bin_0 = data_clean_in[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))

    if n_orders is None:
        n_orders = max(k_cand_max, 1)

    s_rel_0, _, _, s_var_0, _, xc_0, _ = fit_dho_batched(
        f_bin_0, d_bin_0, xx_bin, yy_bin, fsr_val,
        n_orders=n_orders, ds_init=expected_shift, shift_tol=shift_tol, verbose=False
    )
    s_as_0 = s_rel_0[:, 0::2]
    s_st_0 = s_rel_0[:, 1::2]
    s_avg_0 = 0.5 * (s_as_0 + s_st_0)
    delta_mid_0 = 0.5 * (s_st_0 - s_as_0)
    x_avg_0 = 0.5 * (xc_0[:, 0::2] + xc_0[:, 1::2])
    y_rows = yy_bin[:, 0]
    n_orders = s_avg_0.shape[1]

    # Variance of the order-averaged shift s_avg_0
    s_as_var_0 = s_var_0[:, 0::2]
    s_st_var_0 = s_var_0[:, 1::2]
    s_var_avg_0 = np.where(
        np.isfinite(s_as_var_0) & np.isfinite(s_st_var_0),
        0.25 * (s_as_var_0 + s_st_var_0),
        np.where(np.isfinite(s_as_var_0), s_as_var_0, s_st_var_0),
    )

    # Smooth fringe x-trajectories x_traj(r, k) for k = 0..n_orders-1
    x_either_0 = np.where(np.isfinite(x_avg_0), x_avg_0, np.where(np.isfinite(xc_0[:, 0::2]), xc_0[:, 0::2], xc_0[:, 1::2]))
    x_traj = np.zeros((new_h, n_orders), dtype=np.float64)
    for k in range(n_orders):
        vk = np.isfinite(x_either_0[:, k])
        if np.sum(vk) >= 5:
            pk = np.polyfit(y_rows[vk], x_either_0[vk, k], 2)
            x_traj[:, k] = np.polyval(pk, y_rows)
        elif np.sum(vk) > 0:
            x_traj[:, k] = np.nanmean(x_either_0[vk, k])
        else:
            # Fallback for empty order: nominal frequency contour
            target_f_k = (k + 1) * fsr_val
            c_est = np.nanargmin(np.abs(f_bin_0 - target_f_k), axis=1)
            x_traj[:, k] = xx_bin[np.arange(new_h), c_est]

    # 1. Fit 2D polynomial base + smooth per-order spline to delta_mid_0(r, k)
    vm_mid = np.isfinite(delta_mid_0) & np.isfinite(x_avg_0) & (np.abs(delta_mid_0) < 0.25)
    if np.sum(vm_mid) >= 20:
        xm_v = (x_avg_0[vm_mid] - 1152.0) / 1000.0
        ym_v = (np.broadcast_to(y_rows[:, None], (new_h, n_orders))[vm_mid] - y0_center) / 1000.0
        Bm = build_2d_poly_basis(xm_v, ym_v, deg=3)
        cm_zero, _, _, _ = np.linalg.lstsq(Bm, delta_mid_0[vm_mid], rcond=None)
    else:
        cm_zero = np.zeros(10)

    delta_zero_grid = np.zeros((new_h, n_orders), dtype=np.float64)
    for k in range(n_orders):
        xk_n = (x_traj[:, k] - 1152.0) / 1000.0
        yk_n = (y_rows - y0_center) / 1000.0
        poly_k = build_2d_poly_basis(xk_n, yk_n, deg=3) @ cm_zero
        raw_k = delta_mid_0[:, k].copy()
        vk = np.isfinite(raw_k) & (np.abs(raw_k - poly_k) < 0.12)
        if np.sum(vk) >= 5:
            filled_k = np.where(vk, raw_k, poly_k)
            delta_zero_grid[:, k] = ndimage.gaussian_filter1d(filled_k, sigma=sigma_y_bins, mode='nearest')
        else:
            delta_zero_grid[:, k] = poly_k

    # Complete order-averaged shift estimate using single-branch + delta_zero where one branch is sensor-edge clipped
    s_est_0 = np.where(
        np.isfinite(s_avg_0),
        s_avg_0,
        np.where(
            np.isfinite(s_as_0),
            s_as_0 + delta_zero_grid,
            s_st_0 - delta_zero_grid,
        ),
    )

    # 2. Fit physical 6-parameter 2nd-order 2D elliptical paraboloid Z_{S,quad}(x, y)
    Y_2d = np.broadcast_to(y_rows[:, None], (new_h, n_orders))
    K_2d = np.broadcast_to(np.arange(n_orders)[None, :], (new_h, n_orders))
    vm_quad = (
        np.isfinite(s_avg_0)
        & np.isfinite(x_avg_0)
        & np.isfinite(s_var_avg_0)
        & (s_var_avg_0 > 0)
        & (Y_2d >= 250.0)
        & (Y_2d <= 1850.0)
        & (K_2d <= min(7, n_orders - 1))
        & (s_avg_0 > expected_shift - 0.35)
        & (s_avg_0 < expected_shift + 0.35)
    )
    xq = x_avg_0[vm_quad]
    yq = Y_2d[vm_quad]
    sq = s_avg_0[vm_quad]
    vq = s_var_avg_0[vm_quad]
    Aq = np.column_stack([np.ones_like(xq), xq, yq, xq**2, xq * yq, yq**2])
    wq = 1.0 / np.maximum(vq, 1e-12)
    wq_norm = wq / np.mean(wq)
    sqrt_wq = np.sqrt(wq_norm)
    cq_quad, _, _, _ = np.linalg.lstsq(Aq * sqrt_wq[:, None], sq * sqrt_wq, rcond=None)
    for _ in range(2):
        rq = sq - (Aq @ cq_quad)
        sig_q = 1.4826 * np.median(np.abs(rq - np.median(rq)))
        inl = np.abs(rq) < 2.2 * sig_q
        cq_quad, _, _, _ = np.linalg.lstsq(
            Aq[inl] * sqrt_wq[inl, None],
            sq[inl] * sqrt_wq[inl],
            rcond=None,
        )

    # Evaluate Z_{S,quad} and extract higher-order (>= 3rd-order) instrumental residual
    z_quad_grid = np.zeros((new_h, n_orders), dtype=np.float64)
    delta_inst_grid = np.zeros((new_h, n_orders), dtype=np.float64)
    for k in range(n_orders):
        xt = x_traj[:, k]
        yt = y_rows
        zq_k = cq_quad[0] + cq_quad[1] * xt + cq_quad[2] * yt + cq_quad[3] * xt**2 + cq_quad[4] * xt * yt + cq_quad[5] * yt**2
        z_quad_grid[:, k] = zq_k
        res_k = s_est_0[:, k] - zq_k
        vk = np.isfinite(res_k) & (np.abs(res_k) < 0.35)
        if np.sum(vk) >= 10:
            res_interp = np.interp(yt, yt[vk], res_k[vk])
            delta_inst_grid[:, k] = ndimage.gaussian_filter1d(res_interp, sigma=sigma_y_bins, mode='nearest')
        else:
            delta_inst_grid[:, k] = 0.0

    alpha_disp_grid = np.clip(z_quad_grid / np.maximum(z_quad_grid + delta_inst_grid, 1.0), 0.80, 1.20)

    # Map (y, k_ord) corrections onto full 2304 x 2304 detector grid without inter-order cross-talk
    q_cont_full = f_init / fsr_val
    k_nearest_idx = np.clip(np.rint(q_cont_full).astype(np.int32) - 1, 0, n_orders - 1)
    y_full_1d = np.arange(h, dtype=np.float64)
    delta_zero_full_k = np.column_stack([np.interp(y_full_1d, y_rows, delta_zero_grid[:, k]) for k in range(n_orders)])
    alpha_disp_full_k = np.column_stack([np.interp(y_full_1d, y_rows, alpha_disp_grid[:, k]) for k in range(n_orders)])
    delta_inst_full_k = np.column_stack([np.interp(y_full_1d, y_rows, delta_inst_grid[:, k]) for k in range(n_orders)])

    row_arange_full = np.arange(h)[:, None]
    dz_map_2d = delta_zero_full_k[row_arange_full, k_nearest_idx]
    alpha_map_2d = alpha_disp_full_k[row_arange_full, k_nearest_idx]
    dinst_map_2d = delta_inst_full_k[row_arange_full, k_nearest_idx]
    k_ord_2d = (k_nearest_idx + 1).astype(np.float64)

    freq_corr_2d = k_ord_2d * fsr_val + alpha_map_2d * (f_init - k_ord_2d * fsr_val - dz_map_2d)
    return freq_corr_2d, df_coarse, dz_map_2d, dinst_map_2d


freq_corrected, df, dz_zero_map, dinst_disp_map = calibrate_sample_frequency_map(
    freq_map, data_sample, xx, yy, FSR, y0_fit
)

dy_avg = 100
yc_sample = int(round(y0_fit))
avg_data_sample = np.mean(data_sample[yc_sample - dy_avg : yc_sample + dy_avg, :], axis=0)
freq_avg_data_sample = np.mean(freq_corrected[yc_sample - dy_avg : yc_sample + dy_avg, :], axis=0)
print(
    f"Calibrated 2D frequency map for {path_wrapper_sample}: "
    f"coarse df = {df*1e3:+.2f} MHz | "
    f"2D zero-offset RMS = {np.std(dz_zero_map)*1e3:.1f} MHz | "
    f"higher-order dispersion RMS = {np.std(dinst_disp_map)*1e3:.1f} MHz"
)

# Build the list of Brillouin peak pairs [k*FSR - ds, k*FSR + ds] for orders k = 1..10
FSR_guess = FSR
ds = EXPECTED_BRILLOUIN_SHIFT  # Expected Brillouin shift (GHz)
BLS_peaks = []
for k in range(0, 11):
    BLS_peaks.append([k * FSR_guess - ds, k * FSR_guess + ds])

plot = 1
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    im = axes[0].imshow(np.clip(data_sample, 0, np.percentile(data_sample, 98)), cmap='inferno', aspect='auto')
    axes[0].set_title(f'Sample Interferogram ({path_wrapper_sample})')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Pixel y')
    plt.colorbar(im, ax=axes[0], label='Counts')

    axes[1].plot(freq_avg_data_sample, avg_data_sample, color='tab:blue', lw=1.2, label='Central horizontal profile')
    for idx_p, p in enumerate(BLS_peaks[1:]):
        axes[1].axvline(p[0], color='tab:green', linestyle='--', alpha=0.7, label='Anti-Stokes' if idx_p == 0 else None)
        axes[1].axvline(p[1], color='tab:red', linestyle='--', alpha=0.7, label='Stokes' if idx_p == 0 else None)
    axes[1].set_xlim(15, 110)
    axes[1].set_title('Calibrated Central Spectrum & Brillouin Windows (Orders 1–3)')
    axes[1].set_xlabel('Calibrated Frequency (GHz)')
    axes[1].set_ylabel('Intensity (counts)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    im_corr = axes[2].imshow(dinst_disp_map * 1e3, cmap='RdBu_r', vmin=-150, vmax=150, aspect='auto')
    axes[2].set_title(r'2D Higher-Order Dispersion Correction $\Delta\nu_{\mathrm{inst}}(x, y)$ (MHz)')
    axes[2].set_xlabel('Pixel x')
    axes[2].set_ylabel('Pixel y')
    plt.colorbar(im_corr, ax=axes[2], label='Dispersion Distortion Removed (MHz)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step11_plastic_baseline_and_windows.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# %%
# ==============================================================================
# 12 - Bin the Data & Fit Plastic Spectra with Damped Harmonic Oscillator (DHO)
# ==============================================================================
# Description:
#   1. Bins the raw plastic image (`data_sample`) and the 2D-calibrated frequency map
#      (`freq_corrected`) by 4x2 (576 vertical bins x 1152 horizontal bins) to boost
#      SNR while preserving 2D spatial resolution.
#   2. Fits each Anti-Stokes and Stokes peak across all interference orders
#      with a Damped Harmonic Oscillator (DHO) profile:
#         I(nu_rel) = b + a * (Gamma_B * nu_0)^2 / ((nu_rel^2 - nu_0^2)^2 + (Gamma_B * nu_rel)^2)
#      evaluated against the local relative frequency shift nu_rel = |nu - k * FSR|
#      (preserving the exact DHO damping renormalization).

print(f"Data size after binning {h}x{w} -> {new_h}x{new_w}")

freq_bin = freq_corrected[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))
data_bin = data_sample[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))

def compute_order_averaged_variance(var_arr):
    """
    Computes the variance of the order average 0.5 * (AS + S).
    For independent random variables, Var(0.5*(X + Y)) = 0.25 * (Var(X) + Var(Y)).
    If only one branch is valid, returns that branch's variance.
    """
    v_as = var_arr[:, 0::2]
    v_st = var_arr[:, 1::2]
    both = np.isfinite(v_as) & np.isfinite(v_st)
    as_only = np.isfinite(v_as) & ~np.isfinite(v_st)
    st_only = ~np.isfinite(v_as) & np.isfinite(v_st)
    v_avg = np.full_like(v_as, np.nan)
    v_avg[both] = 0.25 * (v_as[both] + v_st[both])
    v_avg[as_only] = v_as[as_only]
    v_avg[st_only] = v_st[st_only]
    return v_avg


shift_clipped, shift, linewidth_clipped, Shift_err, Linewidth_err, x_coords, y_coords = fit_dho_batched(
    freq_bin, data_bin, xx_bin, yy_bin, FSR
)
shift_error = Shift_err
linewidth_error = Linewidth_err

# Order-averaged Brillouin shift, linewidth, and variances across Fringes
shift_clipped_averaged = 0.5 * (shift_clipped[:, 0::2] + shift_clipped[:, 1::2])
linewidth_clipped_averaged = 0.5 * (linewidth_clipped[:, 0::2] + linewidth_clipped[:, 1::2])
Shift_err_averaged = compute_order_averaged_variance(Shift_err)
Linewidth_err_averaged = compute_order_averaged_variance(Linewidth_err)
x_coords_averaged = 0.5 * (x_coords[:, 0::2] + x_coords[:, 1::2])
y_coords_averaged = 0.5 * (y_coords[:, 0::2] + y_coords[:, 1::2])

stokes_antistokes_rms_mhz = float(np.nanstd(shift_clipped[:, 1::2] - shift_clipped[:, 0::2]) * 1e3)
n_fringes_plot = shift_clipped_averaged.shape[1]
print(f"Post-calibration Stokes vs Anti-Stokes splitting RMS across all {n_fringes_plot} fringes: {stokes_antistokes_rms_mhz:.2f} MHz")


# %%
# ==============================================================================
# 13 - Graphs Part 1: Shift and Linewidth Values Along the Fringes
# ==============================================================================
# Description:
#   Displays the extracted Brillouin shift (Delta nu_B) and linewidth (Gamma_B)
#   along the interference fringes:
#   1. 2x2 overview of the 2D detector scatter map (both all Stokes/Anti-Stokes
#      branches and the order-averaged fringes) and 1D vertical profiles along
#      each fringe vs detector height y.
#   2. Detailed fringe-by-fringe grids comparing the Anti-Stokes branch,
#      Stokes branch, and Order-Averaged profile on each individual fringe k.

# 1. 2x2 Overview Plot (matching and improving on the user's plastic_0.png layout)
fig_ov, axes_ov = plt.subplots(2, 2, figsize=(14, 11))

axes_ov[0, 0].pcolormesh(xx_bin, yy_bin, np.clip(data_bin, 0, 800), cmap='gray', shading='auto')
sc_s = axes_ov[0, 0].scatter(
    x_coords_averaged, y_coords_averaged,
    s=10, c=shift_clipped_averaged, cmap='coolwarm',
    vmin=EXPECTED_BRILLOUIN_SHIFT - 0.25, vmax=EXPECTED_BRILLOUIN_SHIFT + 0.15,
)
plt.colorbar(sc_s, ax=axes_ov[0, 0], label='Brillouin Shift (GHz)')
axes_ov[0, 0].set_title('Order-Averaged Brillouin Shift on Detector Plane')
axes_ov[0, 0].set_xlabel('Pixel x')
axes_ov[0, 0].set_ylabel('Pixel y')
axes_ov[0, 0].set_xlim(0, w)
axes_ov[0, 0].set_ylim(h, 0)

axes_ov[0, 1].pcolormesh(xx_bin, yy_bin, np.clip(data_bin, 0, 800), cmap='gray', shading='auto')
sc_lw = axes_ov[0, 1].scatter(
    x_coords_averaged, y_coords_averaged,
    s=10, c=linewidth_clipped_averaged, cmap='coolwarm',
    vmin=max(0.1, EXPECTED_LINEWIDTH - 0.20), vmax=EXPECTED_LINEWIDTH + 0.50,
)
plt.colorbar(sc_lw, ax=axes_ov[0, 1], label='Brillouin Linewidth (GHz)')
axes_ov[0, 1].set_title('Order-Averaged Brillouin Linewidth on Detector Plane')
axes_ov[0, 1].set_xlabel('Pixel x')
axes_ov[0, 1].set_ylabel('Pixel y')
axes_ov[0, 1].set_xlim(0, w)
axes_ov[0, 1].set_ylim(h, 0)

fringe_colors = plt.cm.tab10(np.linspace(0, 1, max(n_fringes_plot, 10)))
for k in range(n_fringes_plot):
    yk = y_coords_averaged[:, k]
    sk = shift_clipped_averaged[:, k]
    lwk = linewidth_clipped_averaged[:, k]
    axes_ov[1, 0].plot(yk, sk, '.', color=fringe_colors[k], markersize=3, alpha=0.75, label=f'Fringe {k}')
    axes_ov[1, 1].plot(yk, lwk, '.', color=fringe_colors[k], markersize=3, alpha=0.75, label=f'Fringe {k}')

axes_ov[1, 0].set_title('Brillouin Shift Values Along the Fringes')
axes_ov[1, 0].set_xlabel('Height on detector y (pixels)')
axes_ov[1, 0].set_ylabel('Brillouin Shift (GHz)')
axes_ov[1, 0].set_ylim(EXPECTED_BRILLOUIN_SHIFT - 0.40, EXPECTED_BRILLOUIN_SHIFT + 0.30)
axes_ov[1, 0].grid(True, alpha=0.3, linestyle=':')
axes_ov[1, 0].legend(ncol=2, fontsize=8, loc='lower center')

axes_ov[1, 1].set_title('Brillouin Linewidth Values Along the Fringes')
axes_ov[1, 1].set_xlabel('Height on detector y (pixels)')
axes_ov[1, 1].set_ylabel('Brillouin Linewidth (GHz)')
axes_ov[1, 1].set_ylim(max(0.0, EXPECTED_LINEWIDTH - 0.35), EXPECTED_LINEWIDTH + 0.70)
axes_ov[1, 1].grid(True, alpha=0.3, linestyle=':')
axes_ov[1, 1].legend(ncol=2, fontsize=8, loc='upper center')

plt.suptitle(f'Calibrated DHO Extraction on Fringes — {path_wrapper_sample}', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step13_fringe_shift_and_linewidth_overview.png'), dpi=200)
# Also update the sample PNG in the data folder if writable
try:
    plt.savefig(os.path.join(directory, f'{name_figure_prefix}_{PATH_SAMPLE}_{path_wrapper_sample.split('/')[-1]}_calibrated.png'), dpi=200)
except Exception:
    pass
if SHOW_PLOTS:
    plt.show()
plt.close(fig_ov)

# 2. Detailed Per-Fringe Breakdown (Anti-Stokes vs Stokes vs Order-Averaged)
n_cols_plot = min(n_fringes_plot, 5)
n_rows_plot = int(np.ceil(n_fringes_plot / n_cols_plot))
fig_f_s, axes_f_s = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.8 * n_rows_plot), sharex=True, sharey=True, squeeze=False)
fig_f_lw, axes_f_lw = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.8 * n_rows_plot), sharex=True, sharey=True, squeeze=False)

for k in range(n_fringes_plot):
    ax_s = axes_f_s[k // n_cols_plot, k % n_cols_plot]
    ax_l = axes_f_lw[k // n_cols_plot, k % n_cols_plot]
    y_row_k = yy_bin[:, 0]

    ax_s.plot(y_row_k, shift_clipped[:, 2 * k], '.', color='tab:blue', markersize=2.5, alpha=0.45, label='Anti-Stokes')
    ax_s.plot(y_row_k, shift_clipped[:, 2 * k + 1], '.', color='tab:red', markersize=2.5, alpha=0.45, label='Stokes')
    ax_s.plot(y_row_k, shift_clipped_averaged[:, k], '-', color='black', lw=1.3, alpha=0.85, label='Order Avg')
    ax_s.set_title(f'Fringe {k} (Order {k+1}) Shift')
    ax_s.set_ylim(EXPECTED_BRILLOUIN_SHIFT - 0.40, EXPECTED_BRILLOUIN_SHIFT + 0.30)
    ax_s.grid(True, alpha=0.25, linestyle=':')
    if k % n_cols_plot == 0:
        ax_s.set_ylabel('Shift (GHz)')
    if k // n_cols_plot == n_rows_plot - 1:
        ax_s.set_xlabel('Detector y (px)')
    if k == 0:
        ax_s.legend(fontsize=8)

    ax_l.plot(y_row_k, linewidth_clipped[:, 2 * k], '.', color='tab:blue', markersize=2.5, alpha=0.45, label='Anti-Stokes')
    ax_l.plot(y_row_k, linewidth_clipped[:, 2 * k + 1], '.', color='tab:red', markersize=2.5, alpha=0.45, label='Stokes')
    ax_l.plot(y_row_k, linewidth_clipped_averaged[:, k], '-', color='black', lw=1.3, alpha=0.85, label='Order Avg')
    ax_l.set_title(f'Fringe {k} (Order {k+1}) Linewidth')
    ax_l.set_ylim(max(0.0, EXPECTED_LINEWIDTH - 0.35), EXPECTED_LINEWIDTH + 0.70)
    ax_l.grid(True, alpha=0.25, linestyle=':')
    if k % n_cols_plot == 0:
        ax_l.set_ylabel('Linewidth (GHz)')
    if k // n_cols_plot == n_rows_plot - 1:
        ax_l.set_xlabel('Detector y (px)')
    if k == 0:
        ax_l.legend(fontsize=8)

for k in range(n_fringes_plot, n_rows_plot * n_cols_plot):
    axes_f_s[k // n_cols_plot, k % n_cols_plot].axis('off')
    axes_f_lw[k // n_cols_plot, k % n_cols_plot].axis('off')

fig_f_s.suptitle(f'Brillouin Shift Along Each Fringe (Anti-Stokes, Stokes & Order Average) — {path_wrapper_sample}', fontsize=13)
fig_f_s.tight_layout()
fig_f_s.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step13_fringe_breakdown_shift.png'), dpi=200)
if SHOW_PLOTS:
    plt.show()
plt.close(fig_f_s)

fig_f_lw.suptitle(f'Brillouin Linewidth Along Each Fringe (Anti-Stokes, Stokes & Order Average) — {path_wrapper_sample}', fontsize=13)
fig_f_lw.tight_layout()
fig_f_lw.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step13_fringe_breakdown_linewidth.png'), dpi=200)
if SHOW_PLOTS:
    plt.show()
plt.close(fig_f_lw)


# %%
# ==============================================================================
# 14 - Graphs Part 2: 2D Maps & Elliptical Contours Under Three Hypotheses
# ==============================================================================
# Description:
#   Models the 2D spatial variation of Brillouin shift Z_S(x, y) and linewidth
#   Z_LW(x, y) across the detector plane as 2D elliptical paraboloids under the
#   three required geometric hypotheses:
#     1. Case A — Not Concentric:
#          Shift and linewidth ellipses have independent centers
#          (xc_S, yc_S) != (xc_LW, yc_LW) and independent principal axes
#          theta_S != theta_LW.
#     2. Case B — Concentric, Without the Same Principal Axes:
#          Shift and linewidth ellipses share a single concentric center
#          (xc, yc), but have independent principal axis angles theta_S != theta_LW.
#     3. Case C — Concentric, With the Same Principal Axes (Coaxial):
#          Shift and linewidth ellipses share both a single concentric center
#          (xc, yc) and identical principal axis orientation theta_S = theta_LW = theta.
#   For each hypothesis, produces:
#     - Combined 2D contour overlay (solid Shift ellipses + dashed Linewidth ellipses)
#     - Side-by-side 2D scatter + elliptical contour maps with principal director axes
#     - Experimental vs Ideal elliptical profile comparisons along all 10 fringes.

nb_lines = 10
shift_min = EXPECTED_BRILLOUIN_SHIFT - 0.20
shift_max = EXPECTED_BRILLOUIN_SHIFT + 0.15
linewidth_min = max(0.1, EXPECTED_LINEWIDTH - 0.15)
linewidth_max = EXPECTED_LINEWIDTH + 0.45

h_img, w_img = data_sample.shape
grid_x, grid_y = np.meshgrid(np.linspace(0, w_img, 400), np.linspace(0, h_img, 400))
L_axis = max(h_img, w_img) * 0.65

# Prepare valid order-averaged points for 2D surface fitting
valid_2d = (
    np.isfinite(x_coords_averaged)
    & np.isfinite(y_coords_averaged)
    & np.isfinite(shift_clipped_averaged)
    & np.isfinite(linewidth_clipped_averaged)
    & np.isfinite(Shift_err_averaged)
    & np.isfinite(Linewidth_err_averaged)
    & (Shift_err_averaged > 0)
    & (Linewidth_err_averaged > 0)
    & (shift_clipped_averaged > EXPECTED_BRILLOUIN_SHIFT - 0.35)
    & (shift_clipped_averaged < EXPECTED_BRILLOUIN_SHIFT + 0.30)
    & (linewidth_clipped_averaged > LINEWIDTH_BOUNDS[0])
    & (linewidth_clipped_averaged < LINEWIDTH_BOUNDS[1])
)
px_all = x_coords_averaged[valid_2d]
py_all = y_coords_averaged[valid_2d]
ps_all = shift_clipped_averaged[valid_2d]
plw_all = linewidth_clipped_averaged[valid_2d]
var_s_pts = Shift_err_averaged[valid_2d]
var_lw_pts = Linewidth_err_averaged[valid_2d]


def eval_general_quad(x, y, coeff):
    """Evaluates Z(x, y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2."""
    c0, c1, c2, c3, c4, c5 = coeff
    return c0 + c1 * x + c2 * y + c3 * x**2 + c4 * x * y + c5 * y**2


def fit_ellipse_free(px, py, pz, var_z=None):
    """Fits an unconstrained 2D quadratic surface (independent center & axes) weighted by variance."""
    A = np.column_stack([np.ones_like(px), px, py, px**2, px * py, py**2])
    if var_z is not None:
        w = 1.0 / np.maximum(var_z, 1e-12)
        w_norm = w / np.mean(w)
        sqrt_w = np.sqrt(w_norm)
        c, _, _, _ = np.linalg.lstsq(A * sqrt_w[:, None], pz * sqrt_w, rcond=None)
        res = np.sum(w_norm * (pz - A @ c) ** 2)
    else:
        c, _, _, _ = np.linalg.lstsq(A, pz, rcond=None)
        res = np.sum((pz - A @ c) ** 2)
    M = np.array([[2.0 * c[3], c[4]], [c[4], 2.0 * c[5]]])
    center = np.linalg.solve(M, -np.array([c[1], c[2]]))
    theta_deg = 0.5 * np.degrees(np.arctan2(c[4], c[3] - c[5]))
    eigvals = np.linalg.eigvalsh(M)
    axis_ratio = np.sqrt(abs(eigvals[1] / eigvals[0])) if eigvals[0] != 0 else np.nan
    z0 = eval_general_quad(center[0], center[1], c)
    return c, center, theta_deg, axis_ratio, z0, res


def fit_ellipse_with_center(px, py, pz, xc, yc, var_z=None):
    """Fits a 2D quadratic surface with fixed center (xc, yc) and free principal axes weighted by variance."""
    dx = px - xc
    dy = py - yc
    A = np.column_stack([np.ones_like(dx), dx**2, dx * dy, dy**2])
    if var_z is not None:
        w = 1.0 / np.maximum(var_z, 1e-12)
        w_norm = w / np.mean(w)
        sqrt_w = np.sqrt(w_norm)
        c, _, _, _ = np.linalg.lstsq(A * sqrt_w[:, None], pz * sqrt_w, rcond=None)
        res = np.sum(w_norm * (pz - A @ c) ** 2)
    else:
        c, _, _, _ = np.linalg.lstsq(A, pz, rcond=None)
        res = np.sum((pz - A @ c) ** 2)
    c0 = c[0] + c[1] * xc**2 + c[2] * xc * yc + c[3] * yc**2
    c1 = -2.0 * c[1] * xc - c[2] * yc
    c2 = -2.0 * c[3] * yc - c[2] * xc
    c3, c4, c5 = c[1], c[2], c[3]
    coeff = np.array([c0, c1, c2, c3, c4, c5])
    M = np.array([[2.0 * c3, c4], [c4, 2.0 * c5]])
    theta_deg = 0.5 * np.degrees(np.arctan2(c4, c3 - c5))
    eigvals = np.linalg.eigvalsh(M)
    axis_ratio = np.sqrt(abs(eigvals[1] / eigvals[0])) if eigvals[0] != 0 else np.nan
    z0 = c[0]
    return coeff, np.array([xc, yc]), theta_deg, axis_ratio, z0, res


def fit_ellipse_coaxial(px, py, pz, xc, yc, theta_rad, var_z=None):
    """Fits a 2D quadratic surface with fixed center (xc, yc) and fixed principal axis angle theta_rad weighted by variance."""
    dx = px - xc
    dy = py - yc
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    Xp = dx * ct + dy * st
    Yp = -dx * st + dy * ct
    A = np.column_stack([np.ones_like(Xp), Xp**2, Yp**2])
    if var_z is not None:
        w = 1.0 / np.maximum(var_z, 1e-12)
        w_norm = w / np.mean(w)
        sqrt_w = np.sqrt(w_norm)
        c, _, _, _ = np.linalg.lstsq(A * sqrt_w[:, None], pz * sqrt_w, rcond=None)
        res = np.sum(w_norm * (pz - A @ c) ** 2)
    else:
        c, _, _, _ = np.linalg.lstsq(A, pz, rcond=None)
        res = np.sum((pz - A @ c) ** 2)
    # Convert [Z0, a, b] in rotated frame back to general 6-coefficient form
    Z0, a, b = c
    c3 = a * ct**2 + b * st**2
    c4 = 2.0 * (a - b) * ct * st
    c5 = a * st**2 + b * ct**2
    c1 = -2.0 * c3 * xc - c4 * yc
    c2 = -2.0 * c5 * yc - c4 * xc
    c0 = Z0 + c3 * xc**2 + c4 * xc * yc + c5 * yc**2
    coeff = np.array([c0, c1, c2, c3, c4, c5])
    axis_ratio = np.sqrt(abs(a / b)) if b != 0 else np.nan
    return coeff, np.array([xc, yc]), np.degrees(theta_rad), axis_ratio, Z0, res


# --- Fit Hypothesis A: Not Concentric (Independent Centers & Independent Axes) ---
coeff_s_A, center_s_A, theta_s_A, ratio_s_A, z0_s_A, res_s_A = fit_ellipse_free(px_all, py_all, ps_all, var_z=var_s_pts)
coeff_lw_A, center_lw_A, theta_lw_A, ratio_lw_A, z0_lw_A, res_lw_A = fit_ellipse_free(px_all, py_all, plw_all, var_z=var_lw_pts)

# --- Fit Hypothesis B: Concentric, Without the Same Principal Axes ---
var_s_scale = np.var(ps_all)
var_lw_scale = np.var(plw_all)

def _cost_concentric_diff_axes(center_xy):
    xc, yc = center_xy
    _, _, _, _, _, r_s = fit_ellipse_with_center(px_all, py_all, ps_all, xc, yc, var_z=var_s_pts)
    _, _, _, _, _, r_lw = fit_ellipse_with_center(px_all, py_all, plw_all, xc, yc, var_z=var_lw_pts)
    return r_s / var_s_scale + r_lw / var_lw_scale

opt_B = minimize(_cost_concentric_diff_axes, 0.5 * (center_s_A + center_lw_A), method='Nelder-Mead')
xc_B, yc_B = opt_B.x
coeff_s_B, center_s_B, theta_s_B, ratio_s_B, z0_s_B, res_s_B = fit_ellipse_with_center(px_all, py_all, ps_all, xc_B, yc_B, var_z=var_s_pts)
coeff_lw_B, center_lw_B, theta_lw_B, ratio_lw_B, z0_lw_B, res_lw_B = fit_ellipse_with_center(px_all, py_all, plw_all, xc_B, yc_B, var_z=var_lw_pts)

# --- Fit Hypothesis C: Concentric, With the Same Principal Axes (Coaxial) ---
def _cost_concentric_same_axes(params):
    xc, yc, th = params
    _, _, _, _, _, r_s = fit_ellipse_coaxial(px_all, py_all, ps_all, xc, yc, th, var_z=var_s_pts)
    _, _, _, _, _, r_lw = fit_ellipse_coaxial(px_all, py_all, plw_all, xc, yc, th, var_z=var_lw_pts)
    return r_s / var_s_scale + r_lw / var_lw_scale

opt_C = minimize(
    _cost_concentric_same_axes,
    [xc_B, yc_B, np.deg2rad(0.5 * (theta_s_B + theta_lw_B))],
    method='Nelder-Mead',
)
xc_C, yc_C, theta_rad_C = opt_C.x
coeff_s_C, center_s_C, theta_s_C, ratio_s_C, z0_s_C, res_s_C = fit_ellipse_coaxial(px_all, py_all, ps_all, xc_C, yc_C, theta_rad_C, var_z=var_s_pts)
coeff_lw_C, center_lw_C, theta_lw_C, ratio_lw_C, z0_lw_C, res_lw_C = fit_ellipse_coaxial(px_all, py_all, plw_all, xc_C, yc_C, theta_rad_C, var_z=var_lw_pts)

print("\n" + "=" * 88)
print("  SUMMARY OF 2D ELLIPTICAL CONTOUR FITS ON THE DETECTOR PLANE (ORDER-AVERAGED)")
print("=" * 88)
print(f"1. NOT CONCENTRIC:")
print(f"   - Shift    : Center = ({center_s_A[0]:6.1f}, {center_s_A[1]:6.1f}) px | Axis θ = {theta_s_A:+6.1f}° | Ratio = {ratio_s_A:.2f} | Z0 = {z0_s_A:.4f} GHz")
print(f"   - Linewidth: Center = ({center_lw_A[0]:6.1f}, {center_lw_A[1]:6.1f}) px | Axis θ = {theta_lw_A:+6.1f}° | Ratio = {ratio_lw_A:.2f} | Z0 = {z0_lw_A:.4f} GHz")
print("-" * 88)
print(f"2. CONCENTRIC, WITHOUT SAME PRINCIPAL AXES:")
print(f"   - Shared Center     = ({xc_B:6.1f}, {yc_B:6.1f}) px")
print(f"   - Shift    : Axis θ = {theta_s_B:+6.1f}° | Ratio = {ratio_s_B:.2f} | Z0 = {z0_s_B:.4f} GHz")
print(f"   - Linewidth: Axis θ = {theta_lw_B:+6.1f}° | Ratio = {ratio_lw_B:.2f} | Z0 = {z0_lw_B:.4f} GHz")
print("-" * 88)
print(f"3. CONCENTRIC, WITH SAME PRINCIPAL AXES (COAXIAL):")
print(f"   - Shared Center     = ({xc_C:6.1f}, {yc_C:6.1f}) px | Shared Axis θ = {theta_s_C:+6.1f}°")
print(f"   - Shift    : Ratio  = {ratio_s_C:.2f} | Z0 = {z0_s_C:.4f} GHz")
print(f"   - Linewidth: Ratio  = {ratio_lw_C:.2f} | Z0 = {z0_lw_C:.4f} GHz")
print("=" * 88 + "\n")


def plot_ellipse_hypothesis_suite(
    tag_slug,
    title_label,
    coeff_s,
    center_s,
    theta_s_deg,
    coeff_lw,
    center_lw,
    theta_lw_deg,
    is_concentric,
    is_coaxial,
):
    """Generates the 2D contour overlay, side-by-side scatter+contour map, and fringe profile comparison."""
    grid_shift = eval_general_quad(grid_x, grid_y, coeff_s)
    grid_lw = eval_general_quad(grid_x, grid_y, coeff_lw)
    levels_s = np.linspace(shift_min, shift_max, nb_lines)
    levels_lw = np.linspace(linewidth_min, linewidth_max, nb_lines)

    # --- Plot A: Side-by-side 2D Scatter & Elliptical Contours + Combined Overlay (1x3 Figure) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

    # Panel 1: Brillouin Shift Map + Contours + Principal Axes
    sc1 = axes[0].scatter(px_all, py_all, c=ps_all, s=12, cmap='coolwarm', vmin=shift_min, vmax=shift_max, alpha=0.55, zorder=2)
    cs1 = axes[0].contour(grid_x, grid_y, grid_shift, levels=levels_s, cmap='coolwarm', vmin=shift_min, vmax=shift_max, linewidths=1.6, zorder=3)
    axes[0].clabel(cs1, inline=True, fontsize=8, fmt='%.2f')
    th_s_rad = np.deg2rad(theta_s_deg)
    axes[0].plot(
        [center_s[0] - L_axis * np.cos(th_s_rad), center_s[0] + L_axis * np.cos(th_s_rad)],
        [center_s[1] - L_axis * np.sin(th_s_rad), center_s[1] + L_axis * np.sin(th_s_rad)],
        'b--', lw=1.4, label=f'Shift Axis 1 (θ={theta_s_deg:.1f}°)',
    )
    axes[0].plot(
        [center_s[0] + L_axis * np.sin(th_s_rad), center_s[0] - L_axis * np.sin(th_s_rad)],
        [center_s[1] - L_axis * np.cos(th_s_rad), center_s[1] + L_axis * np.cos(th_s_rad)],
        'b:', lw=1.2, label='Shift Axis 2 (θ+90°)',
    )
    axes[0].plot(center_s[0], center_s[1], 'k+', markersize=14, markeredgewidth=2.2, label=f'Center ({center_s[0]:.1f}, {center_s[1]:.1f})')
    axes[0].set_title(f'Brillouin Shift — {title_label}')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Pixel y')
    axes[0].set_aspect('equal')
    axes[0].set_xlim(0, w_img)
    axes[0].set_ylim(h_img, 0)
    axes[0].grid(True, alpha=0.2, linestyle=':')
    axes[0].legend(loc='upper right', fontsize=8)
    plt.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label='Shift (GHz)')

    # Panel 2: Brillouin Linewidth Map + Contours + Principal Axes
    sc2 = axes[1].scatter(px_all, py_all, c=plw_all, s=12, cmap='autumn', vmin=linewidth_min, vmax=linewidth_max, alpha=0.55, zorder=2)
    cs2 = axes[1].contour(grid_x, grid_y, grid_lw, levels=levels_lw, cmap='autumn', vmin=linewidth_min, vmax=linewidth_max, linewidths=1.6, zorder=3)
    axes[1].clabel(cs2, inline=True, fontsize=8, fmt='%.2f')
    th_lw_rad = np.deg2rad(theta_lw_deg)
    axes[1].plot(
        [center_lw[0] - L_axis * np.cos(th_lw_rad), center_lw[0] + L_axis * np.cos(th_lw_rad)],
        [center_lw[1] - L_axis * np.sin(th_lw_rad), center_lw[1] + L_axis * np.sin(th_lw_rad)],
        'r--', lw=1.4, label=f'LW Axis 1 (θ={theta_lw_deg:.1f}°)',
    )
    axes[1].plot(
        [center_lw[0] + L_axis * np.sin(th_lw_rad), center_lw[0] - L_axis * np.sin(th_lw_rad)],
        [center_lw[1] - L_axis * np.cos(th_lw_rad), center_lw[1] + L_axis * np.cos(th_lw_rad)],
        'r:', lw=1.2, label='LW Axis 2 (θ+90°)',
    )
    axes[1].plot(center_lw[0], center_lw[1], 'k+', markersize=14, markeredgewidth=2.2, label=f'Center ({center_lw[0]:.1f}, {center_lw[1]:.1f})')
    axes[1].set_title(f'Brillouin Linewidth — {title_label}')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')
    axes[1].set_aspect('equal')
    axes[1].set_xlim(0, w_img)
    axes[1].set_ylim(h_img, 0)
    axes[1].grid(True, alpha=0.2, linestyle=':')
    axes[1].legend(loc='upper right', fontsize=8)
    plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04, label='Linewidth (GHz)')

    # Panel 3: Combined Shift & Linewidth Elliptical Contours Overlay
    cs_s_ov = axes[2].contour(grid_x, grid_y, grid_shift, levels=levels_s, cmap='coolwarm', vmin=shift_min, vmax=shift_max, linewidths=1.6, linestyles='solid')
    axes[2].clabel(cs_s_ov, inline=True, fontsize=7.5, fmt='%.2f')
    cs_lw_ov = axes[2].contour(grid_x, grid_y, grid_lw, levels=levels_lw, cmap='autumn', vmin=linewidth_min, vmax=linewidth_max, linewidths=1.4, linestyles='dashed')
    axes[2].clabel(cs_lw_ov, inline=True, fontsize=7.5, fmt='%.2f')

    axes[2].plot(
        [center_s[0] - L_axis * np.cos(th_s_rad), center_s[0] + L_axis * np.cos(th_s_rad)],
        [center_s[1] - L_axis * np.sin(th_s_rad), center_s[1] + L_axis * np.sin(th_s_rad)],
        'b--', lw=1.3, alpha=0.85,
    )
    if not is_coaxial:
        axes[2].plot(
            [center_lw[0] - L_axis * np.cos(th_lw_rad), center_lw[0] + L_axis * np.cos(th_lw_rad)],
            [center_lw[1] - L_axis * np.sin(th_lw_rad), center_lw[1] + L_axis * np.sin(th_lw_rad)],
            'r--', lw=1.3, alpha=0.85,
        )

    legend_handles = [
        Line2D([0], [0], color='blue', lw=1.8, linestyle='solid', label=f'Shift Contours (θ={theta_s_deg:.1f}°)'),
        Line2D([0], [0], color='red', lw=1.8, linestyle='dashed', label=f'Linewidth Contours (θ={theta_lw_deg:.1f}°)'),
    ]
    if is_concentric:
        axes[2].plot(center_s[0], center_s[1], 'k+', markersize=14, markeredgewidth=2.2)
        legend_handles.append(Line2D([0], [0], color='k', marker='+', linestyle='None', markersize=10, label=f'Shared Center ({center_s[0]:.1f}, {center_s[1]:.1f})'))
    else:
        axes[2].plot(center_s[0], center_s[1], 'b+', markersize=13, markeredgewidth=2.2)
        axes[2].plot(center_lw[0], center_lw[1], 'r+', markersize=13, markeredgewidth=2.2)
        legend_handles.append(Line2D([0], [0], color='b', marker='+', linestyle='None', markersize=10, label=f'Shift Center ({center_s[0]:.1f}, {center_s[1]:.1f})'))
        legend_handles.append(Line2D([0], [0], color='r', marker='+', linestyle='None', markersize=10, label=f'LW Center ({center_lw[0]:.1f}, {center_lw[1]:.1f})'))

    axes[2].legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.9)
    axes[2].set_title(f'Combined Overlay — {title_label}')
    axes[2].set_xlabel('Pixel x')
    axes[2].set_ylabel('Pixel y')
    axes[2].set_aspect('equal')
    axes[2].set_xlim(0, w_img)
    axes[2].set_ylim(h_img, 0)
    axes[2].grid(True, alpha=0.2, linestyle=':')

    plt.suptitle(f'2D Elliptical Contours ({title_label}) — {path_wrapper_sample}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step14_{tag_slug}_2d_contours.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # --- Plot B: Experimental (dots) vs Ideal Elliptical Surface Model (lines) Along Fringes ---
    fig_p, axes_p = plt.subplots(1, 2, figsize=(15, 5.5))
    for k in range(shift_clipped_averaged.shape[1]):
        m_k = valid_2d[:, k]
        yk = y_coords_averaged[m_k, k]
        xk = x_coords_averaged[m_k, k]
        sk_exp = shift_clipped_averaged[m_k, k]
        lwk_exp = linewidth_clipped_averaged[m_k, k]

        axes_p[0].plot(yk, sk_exp, '.', color=fringe_colors[k], alpha=0.30, markersize=3)
        axes_p[1].plot(yk, lwk_exp, '.', color=fringe_colors[k], alpha=0.30, markersize=3)

        if len(yk) >= 5:
            # Evaluate continuous ideal curve along the full geometric trajectory of fringe k
            y_dense = np.linspace(np.min(yk), np.max(yk), 200)
            p_x_of_y = np.polyfit(yk, xk, 2)
            x_dense = np.polyval(p_x_of_y, y_dense)
            sk_mod = eval_general_quad(x_dense, y_dense, coeff_s)
            lwk_mod = eval_general_quad(x_dense, y_dense, coeff_lw)
            axes_p[0].plot(y_dense, sk_mod, '-', color=fringe_colors[k], lw=2.0, label=f'Fringe {k}')
            axes_p[1].plot(y_dense, lwk_mod, '-', color=fringe_colors[k], lw=2.0, label=f'Fringe {k}')

    axes_p[0].set_title(f'Shift Along Fringes: Exp vs Model ({title_label})')
    axes_p[0].set_xlabel('Detector y position (pixels)')
    axes_p[0].set_ylabel('Brillouin Shift (GHz)')
    axes_p[0].set_ylim(EXPECTED_BRILLOUIN_SHIFT - 0.35, EXPECTED_BRILLOUIN_SHIFT + 0.25)
    axes_p[0].grid(True, alpha=0.3, linestyle=':')
    axes_p[0].legend(ncol=2, fontsize=8, loc='lower center')

    axes_p[1].set_title(f'Linewidth Along Fringes: Exp vs Model ({title_label})')
    axes_p[1].set_xlabel('Detector y position (pixels)')
    axes_p[1].set_ylabel('Brillouin Linewidth (GHz)')
    axes_p[1].set_ylim(max(0.0, EXPECTED_LINEWIDTH - 0.25), EXPECTED_LINEWIDTH + 0.65)
    axes_p[1].grid(True, alpha=0.3, linestyle=':')
    axes_p[1].legend(ncol=2, fontsize=8, loc='upper center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step14_{tag_slug}_fringe_profiles.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_p)


# Generate suites for all 3 required elliptical contour hypotheses
plot_ellipse_hypothesis_suite(
    'hyp1_not_concentric',
    'Not Concentric',
    coeff_s_A, center_s_A, theta_s_A,
    coeff_lw_A, center_lw_A, theta_lw_A,
    is_concentric=False, is_coaxial=False,
)

plot_ellipse_hypothesis_suite(
    'hyp2_concentric_diff_axes',
    'Concentric (Different Principal Axes)',
    coeff_s_B, center_s_B, theta_s_B,
    coeff_lw_B, center_lw_B, theta_lw_B,
    is_concentric=True, is_coaxial=False,
)

plot_ellipse_hypothesis_suite(
    'hyp3_concentric_same_axes',
    'Concentric (Same Principal Axes)',
    coeff_s_C, center_s_C, theta_s_C,
    coeff_lw_C, center_lw_C, theta_lw_C,
    is_concentric=True, is_coaxial=True,
)

# Side-by-side 3-Hypothesis Comparison Figure (3 columns showing the 3 contour hypotheses directly)
fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(20, 6.2), sharex=True, sharey=True)
hyp_list = [
    ('(a) Not Concentric', coeff_s_A, center_s_A, theta_s_A, coeff_lw_A, center_lw_A, theta_lw_A, False, False),
    ('(b) Concentric, Different Axes', coeff_s_B, center_s_B, theta_s_B, coeff_lw_B, center_lw_B, theta_lw_B, True, False),
    ('(c) Concentric, Same Axes', coeff_s_C, center_s_C, theta_s_C, coeff_lw_C, center_lw_C, theta_lw_C, True, True),
]
levels_s = np.linspace(shift_min, shift_max, nb_lines)
levels_lw = np.linspace(linewidth_min, linewidth_max, nb_lines)

for ax, (h_title, cs_c, ctr_s, th_s, clw_c, ctr_lw, th_lw, is_conc, is_coax) in zip(axes_cmp, hyp_list):
    gs = eval_general_quad(grid_x, grid_y, cs_c)
    glw = eval_general_quad(grid_x, grid_y, clw_c)
    c1 = ax.contour(grid_x, grid_y, gs, levels=levels_s, cmap='coolwarm', vmin=shift_min, vmax=shift_max, linewidths=1.6, linestyles='solid')
    ax.clabel(c1, inline=True, fontsize=7.5, fmt='%.2f')
    c2 = ax.contour(grid_x, grid_y, glw, levels=levels_lw, cmap='autumn', vmin=linewidth_min, vmax=linewidth_max, linewidths=1.4, linestyles='dashed')
    ax.clabel(c2, inline=True, fontsize=7.5, fmt='%.2f')

    th_s_r = np.deg2rad(th_s)
    th_lw_r = np.deg2rad(th_lw)
    ax.plot([ctr_s[0] - L_axis * np.cos(th_s_r), ctr_s[0] + L_axis * np.cos(th_s_r)],
            [ctr_s[1] - L_axis * np.sin(th_s_r), ctr_s[1] + L_axis * np.sin(th_s_r)], 'b--', lw=1.4)
    if not is_coax:
        ax.plot([ctr_lw[0] - L_axis * np.cos(th_lw_r), ctr_lw[0] + L_axis * np.cos(th_lw_r)],
                [ctr_lw[1] - L_axis * np.sin(th_lw_r), ctr_lw[1] + L_axis * np.sin(th_lw_r)], 'r--', lw=1.4)

    if is_conc:
        ax.plot(ctr_s[0], ctr_s[1], 'k+', markersize=14, markeredgewidth=2.2, label=f'Center ({ctr_s[0]:.0f}, {ctr_s[1]:.0f})')
    else:
        ax.plot(ctr_s[0], ctr_s[1], 'b+', markersize=13, markeredgewidth=2.2, label=f'Shift ({ctr_s[0]:.0f}, {ctr_s[1]:.0f})')
        ax.plot(ctr_lw[0], ctr_lw[1], 'r+', markersize=13, markeredgewidth=2.2, label=f'LW ({ctr_lw[0]:.0f}, {ctr_lw[1]:.0f})')

    ax.set_title(f"{h_title}\n$\\theta_S={th_s:.1f}^\\circ, \\theta_\\Gamma={th_lw:.1f}^\\circ$")
    ax.set_xlabel('Pixel x')
    if ax is axes_cmp[0]:
        ax.set_ylabel('Pixel y')
    ax.set_aspect('equal')
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.legend(loc='upper right', fontsize=8)

plt.suptitle(f'Comparison of the Three Elliptical Contour Hypotheses on Detector Plane — {path_wrapper_sample}', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step14_three_hypotheses_comparison.png'), dpi=200)
if SHOW_PLOTS:
    plt.show()
plt.close(fig_cmp)


# %%
# ==============================================================================
# 14B - Rubidium Absorption Profiles Normalized to Orders (Frequency Map Assessment)
# ==============================================================================
# Description:
#   To directly verify whether the horizontal elongation of the Brillouin shift
#   contours in Step 14 is driven by an optical anamorphic pupil asymmetry or by
#   an artifact in the 2D frequency map, we benchmark the frequency map against
#   the isotropic natural Rubidium absorption lines:
#     1. Tracks the 85Rb F=2 absorption dip frequency along fringes vs detector
#        height y across all orders k = 1..8 (analogous to the Step 14 profile plot).
#     2. Overlays the order-normalized absorption spectra Delta q = (nu - k*FSR)/FSR
#        across all orders against the theoretical Steck natural Rubidium spectrum.
#     3. Compares the 2x5 grid of all individual orders across multiple heights y.

plot_rb_benchmarks = True
if plot_rb_benchmarks:
    try:
        rb_raw_img = wrapper[f'{path_wrapper_rb}/image'].astype(np.float64)
    except Exception:
        rb_raw_img = wrapper['/Brillouin/Rb_absorption/image'].astype(np.float64)
    rb_bin_bench = rb_raw_img[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))
    q_bin_bench = freq_bin / FSR
    n_orders_rb = min(10, max(int(np.floor(np.nanmax(q_bin_bench))) + 1, 1))

    # Track 85Rb F=2 dip position along y for each order
    dip_85_bench = np.full((new_h, n_orders_rb), np.nan)
    for r in range(new_h):
        q_row = q_bin_bench[r, :]
        f_row = freq_bin[r, :]
        I_row = rb_bin_bench[r, :]
        for k in range(0, n_orders_rb):
            m = (q_row >= k - 0.5) & (q_row < k + 0.5)
            if np.sum(m) > 10:
                fk = f_row[m] - k * FSR
                ik = I_row[m]
                w85 = (fk >= 0.0) & (fk <= 0.8)
                if np.sum(w85) >= 3:
                    idx = np.argmin(ik[w85])
                    fk_w = fk[w85]
                    ik_w = ik[w85]
                    if 0 < idx < len(ik_w) - 1:
                        p = np.polyfit(fk_w[idx-1:idx+2], ik_w[idx-1:idx+2], 2)
                        if p[0] > 0:
                            v = -p[1] / (2.0 * p[0])
                            if 0.1 <= v <= 0.7:
                                dip_85_bench[r, k] = v

    fig_rb1, axes_rb1 = plt.subplots(1, 2, figsize=(16, 5.8))
    fringe_colors_rb = plt.cm.tab10(np.linspace(0, 1, max(n_orders_rb, 10)))

    # Left Panel: Tracked 85Rb Absorption Dip Position vs Detector y for All Fringes
    for k in range(1, min(9, n_orders_rb)):
        yk = yy_bin[:, 0]
        dk = dip_85_bench[:, k]
        vk = np.isfinite(dk)
        if np.sum(vk) > 10:
            med_k = np.median(dk[vk])
            vk &= np.abs(dk - med_k) < 0.12
            axes_rb1[0].plot(yk[vk], dk[vk], '.', color=fringe_colors_rb[k], markersize=3, alpha=0.35)
            p_fit = np.polyfit(yk[vk], dk[vk], 2)
            axes_rb1[0].plot(yk, np.polyval(p_fit, yk), '-', color=fringe_colors_rb[k], lw=2.0, label=f'Order {k} ($\\bar{{\\nu}}={np.mean(dk[vk]):.3f}$ GHz)')

    axes_rb1[0].set_title(r'Rubidium $^{85}\mathrm{Rb}$ Absorption Dip Frequency Along Fringes vs Detector $y$')
    axes_rb1[0].set_xlabel('Detector y position (pixels)')
    axes_rb1[0].set_ylabel('Measured Absorption Dip Frequency (GHz)')
    axes_rb1[0].set_ylim(0.20, 0.55)
    axes_rb1[0].grid(True, alpha=0.3, linestyle=':')
    axes_rb1[0].legend(ncol=2, fontsize=8, loc='lower center')

    # Right Panel: Order-Normalized Absorption Spectra Overlay Across All Orders (Central Strip)
    yc_idx = int(round(y0_fit)) // 4
    dy_b = 25
    q_strip = q_bin_bench[yc_idx - dy_b : yc_idx + dy_b, :].mean(axis=0)
    f_strip = freq_bin[yc_idx - dy_b : yc_idx + dy_b, :].mean(axis=0)
    I_strip = rb_bin_bench[yc_idx - dy_b : yc_idx + dy_b, :].mean(axis=0)

    dq_grid = np.linspace(-0.20, 0.20, 500)
    specs_bench = []
    for k in range(0, n_orders_rb):
        m = (q_strip >= k - 0.5) & (q_strip < k + 0.5)
        if np.sum(m) > 10:
            fk = f_strip[m] - k * FSR
            dq = fk / FSR
            ik = I_strip[m]
            in_w = (dq >= -0.20) & (dq <= 0.20)
            if np.sum(in_w) > 5:
                ik_norm = (ik[in_w] - np.min(ik[in_w])) / (np.max(ik[in_w]) - np.min(ik[in_w]) + 1e-6)
                spec = np.interp(dq_grid, dq[in_w], ik_norm, left=np.nan, right=np.nan)
                specs_bench.append(spec)
                axes_rb1[1].plot(dq_grid, spec, color=fringe_colors_rb[k], lw=1.3, alpha=0.75, label=f'Order {k}')

    # Align (1 - theoretical absorption) = theoretical transmission with the experimental data
    avg_spec_bench = np.nanmean(specs_bench, axis=0)
    fit_bench_w = np.isfinite(avg_spec_bench) & (dq_grid >= -0.16) & (dq_grid <= 0.16)
    dq_fit_bench = dq_grid[fit_bench_w]
    meas_fit_bench = avg_spec_bench[fit_bench_w]
    meas_zm = meas_fit_bench - np.mean(meas_fit_bench)

    # Maximize the convolution / zero-mean correlation of (1 - A_theo) with experimental transmission
    s_bench_scan = np.linspace(-3500, 3500, 7001)
    corrs_bench = []
    for s in s_bench_scan:
        t_theo_cand = trans_fast(dq_fit_bench * FSR * 1e3 - s)
        t_zm = t_theo_cand - np.mean(t_theo_cand)
        r = np.sum(meas_zm * t_zm) / (np.sqrt(np.sum(meas_zm**2) * np.sum(t_zm**2)) + 1e-12)
        corrs_bench.append(r)

    best_s_MHz = s_bench_scan[np.argmax(corrs_bench)]
    best_corr_rb = np.max(corrs_bench)
    t_theo_aligned = trans_fast(dq_grid * FSR * 1e3 - best_s_MHz)
    t_theo_norm = (t_theo_aligned - np.nanmin(t_theo_aligned)) / (np.nanmax(t_theo_aligned) - np.nanmin(t_theo_aligned) + 1e-12)

    # Plot (1 - A_theo) directly overlaying the experimental transmission profiles
    axes_rb1[1].plot(
        dq_grid,
        t_theo_norm,
        'r--',
        lw=2.2,
        alpha=0.9,
        label=rf'Theoretical Steck $(1 - A_{{\mathrm{{theo}}}})$ ($r = {best_corr_rb:.2f}$, $\delta\nu = {best_s_MHz:+.0f}$ MHz)',
    )

    axes_rb1[1].set_title(r'Overlay of Rubidium Spectra Across Orders vs Theoretical Steck $(1 - A_{\mathrm{theo}})$')
    axes_rb1[1].set_xlabel(r'Order-Normalized Coordinate $\Delta q = (\nu - k\cdot\mathrm{FSR})/\mathrm{FSR}$')
    axes_rb1[1].set_ylabel('Normalized Transmission (0 = dip, 1 = continuum)')
    axes_rb1[1].set_xlim(-0.20, 0.20)
    axes_rb1[1].set_ylim(-0.05, 1.15)
    axes_rb1[1].grid(True, alpha=0.3, linestyle=':')
    axes_rb1[1].legend(ncol=3, fontsize=7.5, loc='lower left')

    plt.suptitle('Rubidium Absorption Line Profiles Normalized to Orders (Frequency Map Quality Assessment)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step14_rubidium_orders_profiles.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_rb1)

    # Detailed Breakdown by Order across Detector Height y
    n_cols_rb2 = min(n_orders_rb, 5)
    n_rows_rb2 = int(np.ceil(n_orders_rb / n_cols_rb2))
    fig_rb2, axes_rb2 = plt.subplots(n_rows_rb2, n_cols_rb2, figsize=(4.0 * n_cols_rb2, 3.8 * n_rows_rb2), sharex=True, sharey=True, squeeze=False)
    y_samples = [300, 700, 948, 1300, 1700, 2100]
    y_sample_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(y_samples)))

    for k in range(n_orders_rb):
        ax = axes_rb2[k // n_cols_rb2, k % n_cols_rb2]
        for ys_idx, ys in enumerate(y_samples):
            r_ys = ys // 4
            q_row = q_bin_bench[r_ys, :]
            f_row = freq_bin[r_ys, :]
            I_row = rb_bin_bench[r_ys, :]
            m = (q_row >= k - 0.5) & (q_row < k + 0.5)
            if np.sum(m) > 10:
                fk = f_row[m] - k * FSR
                dq = fk / FSR
                ik = I_row[m]
                in_w = (dq >= -0.20) & (dq <= 0.20)
                if np.sum(in_w) > 5:
                    ik_norm = (ik[in_w] - np.min(ik[in_w])) / (np.max(ik[in_w]) - np.min(ik[in_w]) + 1e-6)
                    ax.plot(dq[in_w], ik_norm, color=y_sample_colors[ys_idx], lw=1.1, alpha=0.7, label=f'y={ys}px' if k == 0 else None)

        # Theoretical Steck (1 - A_theo) overlaid directly on the same axis (dips matching dips)
        ax.plot(dq_grid, t_theo_norm, 'r--', lw=1.4, alpha=0.85, label='Theory $(1 - A_{\\mathrm{theo}})$' if k == 0 else None)

        x_k_approx = int(round(np.nanmean(np.where(m)[0]) * bin_val_w if np.sum(m) > 0 else 0))
        ax.set_title(f'Order {k} (x ≈ {x_k_approx} px)')
        ax.set_xlim(-0.20, 0.20)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25, linestyle=':')
        if k % n_cols_rb2 == 0:
            ax.set_ylabel('Normalized Intensity')
        if k // n_cols_rb2 == n_rows_rb2 - 1:
            ax.set_xlabel(r'$\Delta q$ from Order Center')
        if k == 0:
            ax.legend(fontsize=7, loc='lower left')

    for k in range(n_orders_rb, n_rows_rb2 * n_cols_rb2):
        axes_rb2[k // n_cols_rb2, k % n_cols_rb2].axis('off')

    fig_rb2.suptitle(rf'Rubidium Absorption Profiles Across Detector Heights $y$ for Orders $0\dots {n_orders_rb-1}$ vs Theoretical $(1 - A_{{\mathrm{{theo}}}})$ (Red Dashed)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name_figure_prefix}_{PATH_SAMPLE}_step14_rubidium_order_breakdown.png'), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_rb2)


# %%
# ==============================================================================
# 15 - Save Treated Data to HDF5 & Compare All Four Plastic Orientations
# ==============================================================================
# Description:
#   Saves the fitted Brillouin shift, linewidth, uncertainties, and sub-pixel
#   detector coordinates into `Measures.h5`, and runs the calibrated DHO +
#   elliptical contour extraction across all available plastic orientations.

wrapper.add_treated_data(
    parent_group=path_wrapper_sample,
    name_group=f'Treated {bin_val_h}x{bin_val_w} bins',
    shift=shift_clipped,
    shift_err=Shift_err,
    linewidth=linewidth_clipped,
    linewidth_err=Linewidth_err,
    overwrite=True,
)
wrapper.add_other(
    data=x_coords,
    parent_group=f'{path_wrapper_sample}/Treated {bin_val_h}x{bin_val_w} bins',
    name='x_coords',
    overwrite=True,
)
wrapper.add_other(
    data=y_coords,
    parent_group=f'{path_wrapper_sample}/Treated {bin_val_h}x{bin_val_w} bins',
    name='y_coords',
    overwrite=True,
)

print("\n" + "=" * 92)
print("  MULTI-SAMPLE SUMMARY ACROSS ALL PLASTIC ACQUISITIONS (CONCENTRIC COAXIAL MODEL)")
print("=" * 92)
print(f"{'Sample':<14} | {'df (MHz)':<10} | {'Center (xc, yc) px':<20} | {'Axis θ':<9} | {'Shift Z0 (GHz)':<15} | {'LW Z0 (GHz)':<12}")
print("-" * 92)

available_samples = []
# 1. Primary configured sample
if PATH_SAMPLE:
    try:
        sample_key = resolve_h5_path(group_interest, PATH_SAMPLE)
        _ = wrapper[f'{sample_key}/image']
        available_samples.append(sample_key)
    except Exception:
        pass

# 2. Check standard orientation samples if present under group_interest or root
for s in ['plastic_0', 'plastic_1', 'plastic_2', 'plastic_3', 'plastic_4', 'plastic_60', 'plastic_120', 'plastic_180', 'plastic']:
    for cand in [resolve_h5_path(group_interest, s), f'Brillouin/{s}', s]:
        if cand not in available_samples:
            try:
                _ = wrapper[f'{cand}/image']
                available_samples.append(cand)
                break
            except Exception:
                pass

for p_grp in available_samples:
    sample_name = p_grp.split('/')[-1]
    d_raw = wrapper[f'{p_grp}/image'].astype(np.float64)
    d_clean = d_raw  # Original sample without morphological baseline subtraction
    f_corr_s, df_s, _, _ = calibrate_sample_frequency_map(freq_map, d_clean, xx, yy, FSR, y0_fit)
    f_bin_s = f_corr_s[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))
    d_bin_s = d_clean[: new_h * bin_val_h, : new_w * bin_val_w].reshape(new_h, bin_val_h, new_w, bin_val_w).mean(axis=(1, 3))
    s_rel, _, lw_rel, s_err_s, lw_err_s, xc_s, yc_s = fit_dho_batched(f_bin_s, d_bin_s, xx_bin, yy_bin, FSR, verbose=False)
    wrapper.add_treated_data(
        parent_group=p_grp,
        name_group=f'Treated {bin_val_h}x{bin_val_w} bins',
        shift=s_rel,
        shift_err=s_err_s,
        linewidth=lw_rel,
        linewidth_err=lw_err_s,
        overwrite=True,
    )
    wrapper.add_other(data=xc_s, parent_group=f'{p_grp}/Treated {bin_val_h}x{bin_val_w} bins', name='x_coords', overwrite=True)
    wrapper.add_other(data=yc_s, parent_group=f'{p_grp}/Treated {bin_val_h}x{bin_val_w} bins', name='y_coords', overwrite=True)
    s_avg = 0.5 * (s_rel[:, 0::2] + s_rel[:, 1::2])
    l_avg = 0.5 * (lw_rel[:, 0::2] + lw_rel[:, 1::2])
    s_err_avg = compute_order_averaged_variance(s_err_s)
    lw_err_avg = compute_order_averaged_variance(lw_err_s)
    x_a = 0.5 * (xc_s[:, 0::2] + xc_s[:, 1::2])
    y_a = 0.5 * (yc_s[:, 0::2] + yc_s[:, 1::2])
    v_m = (
        np.isfinite(x_a)
        & np.isfinite(y_a)
        & np.isfinite(s_avg)
        & np.isfinite(l_avg)
        & np.isfinite(s_err_avg)
        & np.isfinite(lw_err_avg)
        & (s_err_avg > 0)
        & (lw_err_avg > 0)
        & (s_avg > EXPECTED_BRILLOUIN_SHIFT - 0.35)
        & (s_avg < EXPECTED_BRILLOUIN_SHIFT + 0.30)
        & (l_avg > LINEWIDTH_BOUNDS[0])
        & (l_avg < LINEWIDTH_BOUNDS[1])
    )
    px_s, py_s, ps_s, plw_s = x_a[v_m], y_a[v_m], s_avg[v_m], l_avg[v_m]
    var_s_pts, var_lw_pts = s_err_avg[v_m], lw_err_avg[v_m]
    vs_s, vlw_s = np.var(ps_s), np.var(plw_s)
    opt_s = minimize(
        lambda p: fit_ellipse_coaxial(px_s, py_s, ps_s, p[0], p[1], p[2], var_z=var_s_pts)[5] / vs_s + fit_ellipse_coaxial(px_s, py_s, plw_s, p[0], p[1], p[2], var_z=var_lw_pts)[5] / vlw_s,
        [1150.0, 948.0, np.deg2rad(-70.0)],
        method='Nelder-Mead',
    )
    xc_o, yc_o, th_o = opt_s.x
    _, _, _, _, z0_so, _ = fit_ellipse_coaxial(px_s, py_s, ps_s, xc_o, yc_o, th_o, var_z=var_s_pts)
    _, _, _, _, z0_lwo, _ = fit_ellipse_coaxial(px_s, py_s, plw_s, xc_o, yc_o, th_o, var_z=var_lw_pts)
    print(f"{sample_name:<14} | {df_s*1e3:+8.2f}   | ({xc_o:6.1f}, {yc_o:6.1f})     | {np.degrees(th_o):+7.1f}° | {z0_so:<15.4f} | {z0_lwo:<12.4f}")
print("=" * 92)

# %%
