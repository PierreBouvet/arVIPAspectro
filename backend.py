"""
Core mathematical, physical, and signal/image processing functions for VIPA Brillouin calibration.
"""

import os
import time
import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy.special import wofz
from scipy.optimize import minimize, curve_fit
from HDF5_BLS import Wrapper
from HDF5_BLS.errors import WrapperError_Overwrite

import config


# ==============================================================================
# File and Wrapper Management
# ==============================================================================

def create_wrapper(directory, name='Measures.h5', exclude_dir=None):
    """
    Creates or opens an HDF5 Wrapper file at `directory/name`.

    Scans the directory structure and automatically adds any `.dat` files into
    the corresponding HDF5 group hierarchy under `Brillouin/`.
    """
    if exclude_dir is None:
        exclude_dir = []

    def add_dat(file, dir_path, parent_group='Brillouin'):
        if os.path.isdir(file):
            grp_name = os.path.basename(file)
            wrapper.create_group(name=grp_name, parent_group=parent_group)
            for path in os.listdir(file):
                add_dat(
                    os.path.join(file, path),
                    dir_path=os.path.join(dir_path, grp_name),
                    parent_group=f'{parent_group}/{grp_name}'
                )
        elif file.endswith('.dat'):
            data = np.loadtxt(os.path.join(dir_path, file))
            grp_name = os.path.splitext(file)[0]
            wrapper.add_raw_data(data, parent_group=f'{parent_group}/{grp_name}', name='image')

    if not name.endswith('.h5'):
        name += '.h5'
    target_file = os.path.join(directory, name)
    wrapper = Wrapper(target_file)

    for file in os.listdir(directory):
        if file.endswith('.h5') or file in exclude_dir:
            continue
        try:
            add_dat(os.path.join(directory, file), directory)
        except WrapperError_Overwrite:
            pass

    return wrapper


# ==============================================================================
# Morphological Baseline Extraction
# ==============================================================================

def extract_baseline_image_morphology(data, scale=9, ksize_structuring_elements=3, ksize_blur=11):
    """
    Extracts a 2D background baseline from an image using morphological opening.

    The image is downsampled by `scale`, processed with an elliptical structuring
    element of size `ksize_structuring_elements` to remove narrow interference fringes,
    smoothed with a Gaussian blur, and upsampled back to original dimensions.
    """
    h, w = data.shape
    small_w = max(1, w // scale)
    small_h = max(1, h // scale)
    data_small = cv2.resize(data.astype(np.float32), (small_w, small_h), interpolation=cv2.INTER_AREA)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_structuring_elements, ksize_structuring_elements))
    baseline_small = cv2.morphologyEx(data_small, cv2.MORPH_OPEN, kernel)

    if ksize_blur % 2 == 0:
        ksize_blur += 1
    baseline_small = cv2.GaussianBlur(baseline_small, (ksize_blur, ksize_blur), 0)

    baseline_2d = cv2.resize(baseline_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return baseline_2d


# ==============================================================================
# Fringe Tracing & Refinement Algorithms
# ==============================================================================

def extract_fringe_peak(data, i, dy, var, last_px, last_py):
    """
    Sub-pixel peak extraction along row i in the vicinity of last_px using parabolic fit.
    """
    h, w = data.shape
    y_min = max(0, i - dy)
    y_max = min(h, i + dy)
    x_min = max(0, last_px - dy)
    x_max = min(w, last_px + dy)

    arr = np.average(data[y_min:y_max, x_min:x_max], axis=0)
    if len(arr) > 2:
        popt = np.polyfit(np.arange(len(arr)), arr, 2)
        if popt[0] < 0:  # Peak must be concave downward
            pos_max = -popt[1] / (2.0 * popt[0])
            temp_i = pos_max + x_min
            if abs(i - last_py) > var:
                return None
            if abs(temp_i - last_px) < var:
                return temp_i
    return None


def extract_fringes(data, peaks, y_center, dy=20, var=2):
    """
    Extracts the continuous pixel coordinates of fringes across the detector height.

    Starting from horizontal peak positions at row y_center, traces each fringe
    bidirectionally (upwards and downwards) by sub-window parabolic fitting.
    """
    h, _ = data.shape
    points = []
    for peak in peaks:
        for order in range(len(peak)):
            points.append([[float(peak[order])], [float(y_center)]])
            last_px, last_py = int(peak[order]), int(y_center)

            # Traverse downwards
            i = int(y_center)
            while i < h:
                temp_i = extract_fringe_peak(data, i, dy, var, last_px, last_py)
                if temp_i is not None:
                    points[-1][0].append(temp_i)
                    points[-1][1].append(float(i))
                    last_px, last_py = int(round(temp_i)), i
                i += 1

            # Traverse upwards
            i = int(y_center)
            last_px, last_py = int(peak[order]), int(y_center)
            while i > 0:
                temp_i = extract_fringe_peak(data, i, dy, var, last_px, last_py)
                if temp_i is not None:
                    points[-1][0].append(temp_i)
                    points[-1][1].append(float(i))
                    last_px, last_py = int(round(temp_i)), i
                i -= 1

    return points


def subpixel_refine_and_reject_outliers(
    fringe_x,
    fringe_y,
    data_raw,
    data_subtracted,
    w_sub=10,
    saturation_limit=65000.0,
    min_net_amp=200.0,
    poly_deg=4,
    n_sigma=2.5,
    min_sigma=0.12,
    n_iter=3
):
    """
    Applies unsaturated quadratic-weighted centroiding and iterative polynomial
    outlier rejection to a single traced fringe.

    Returns
    -------
    x_clean : numpy.ndarray
        Unsaturated, outlier-rejected sub-pixel x coordinates.
    y_clean : numpy.ndarray
        Corresponding y coordinates.
    x_smooth : numpy.ndarray
        Polynomial-smoothed x coordinates evaluated at y_clean.
    """
    h, w = data_raw.shape
    x_ref = []
    y_ref = []

    for x0_f, y0_f in zip(fringe_x, fringe_y):
        x0_i = int(round(x0_f))
        y0_i = int(round(y0_f))

        if x0_i < w_sub or x0_i >= w - w_sub or y0_i < 0 or y0_i >= h:
            continue

        win_raw = data_raw[y0_i, x0_i - w_sub : x0_i + w_sub + 1]
        if np.max(win_raw) >= saturation_limit:
            continue

        win_clean = data_subtracted[y0_i, x0_i - w_sub : x0_i + w_sub + 1]
        base_loc = 0.5 * (win_clean[0] + win_clean[-1])
        net_win = win_clean - base_loc
        if np.max(net_win) < min_net_amp:
            continue

        weights = np.maximum(net_win, 0.0) ** 2
        w_sum = np.sum(weights)
        if w_sum <= 0:
            continue

        x_sub = np.sum(np.arange(x0_i - w_sub, x0_i + w_sub + 1) * weights) / w_sum
        x_ref.append(x_sub)
        y_ref.append(float(y0_i))

    if len(x_ref) < poly_deg + 2:
        return np.array([]), np.array([]), np.array([])

    x_arr = np.array(x_ref, dtype=np.float64)
    y_arr = np.array(y_ref, dtype=np.float64)

    # Iterative sigma-clipping polynomial fit
    keep = np.ones(len(x_arr), dtype=bool)
    for _ in range(n_iter):
        if np.sum(keep) < poly_deg + 1:
            break
        p_k = np.polyfit(y_arr[keep], x_arr[keep], poly_deg)
        res_k = x_arr - np.polyval(p_k, y_arr)
        sigma_k = 1.4826 * np.median(np.abs(res_k[keep] - np.median(res_k[keep])))
        keep = np.abs(res_k) < max(n_sigma * sigma_k, min_sigma)

    if np.sum(keep) < poly_deg + 1:
        return np.array([]), np.array([]), np.array([])

    p_k_final = np.polyfit(y_arr[keep], x_arr[keep], poly_deg)
    x_clean = x_arr[keep]
    y_clean = y_arr[keep]
    x_smooth = np.polyval(p_k_final, y_clean)

    return x_clean, y_clean, x_smooth


# ==============================================================================
# VIPA Geometric & Optical Functions
# ==============================================================================

def direction_cosine(x, y, x0, y0, f, phi):
    """
    Exact directional cosine u = cos(Theta) = e_n . e_k under gnomonic projection.
    """
    xf = np.asarray(x, float) - x0
    yf = np.asarray(y, float) - y0
    return (f * np.cos(phi) - xf * np.sin(phi)) / np.sqrt(f**2 + xf**2 + yf**2)


def g_function(u, n_r):
    """
    Internal refraction geometric factor g(u) = sqrt(u^2 + n_r^2 - 1).
    """
    return np.sqrt(np.asarray(u, float)**2 + n_r**2 - 1.0)


def g_from_xy(x, y, x0, y0, f, phi, n_r):
    """
    Exact 2D gnomonic VIPA geometric factor g(x, y).
    """
    u = direction_cosine(x, y, x0, y0, f, phi)
    return g_function(u, n_r)


def geom_residuals_px(params, all_x, all_y, all_k, nb_fringes, f, phi, n_r):
    """
    Intra-fringe g(x, y) residuals scaled into pixel units (~1e5) for least-squares optimization.
    """
    x0, y0 = params
    g_vals = g_from_xy(all_x, all_y, x0, y0, f, phi, n_r)
    res = np.empty_like(g_vals)
    for k in range(nb_fringes):
        m = (all_k == k)
        if np.any(m):
            res[m] = (g_vals[m] - np.mean(g_vals[m])) * 1e5
    return res


def fit_concentric_arcs(points):
    """
    Fits multiple concentric circular arcs sharing a common paraxial center (xc, yc).

    Parameters
    ----------
    points : list of [x_coords, y_coords]
        List of fringes with their respective coordinates.

    Returns
    -------
    xc, yc : float
        Fitted paraxial center coordinates.
    radii : list of float
        Fitted radii for each concentric fringe.
    xc_err, yc_err : float
        Standard errors of xc and yc.
    cov_xy : numpy.ndarray, shape (2, 2)
        Estimated covariance matrix for (xc, yc).
    """
    M = len(points)
    N = sum(len(p[0]) for p in points)
    A = np.zeros((N, 2 + M))
    b = np.zeros(N)
    row = 0

    for j, lst in enumerate(points):
        x_arr = np.asarray(lst[0], dtype=np.float64)
        y_arr = np.asarray(lst[1], dtype=np.float64)
        n_j = len(x_arr)
        A[row:row+n_j, 0] = 2.0 * x_arr
        A[row:row+n_j, 1] = 2.0 * y_arr
        A[row:row+n_j, 2+j] = 1.0
        b[row:row+n_j] = x_arr**2 + y_arr**2
        row += n_j

    theta, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = theta[0], theta[1]
    radii = [np.sqrt(max(0.0, theta[2+j] + xc**2 + yc**2)) for j in range(M)]

    if residuals.size > 0:
        sum_sq_res = residuals[0]
    else:
        sum_sq_res = np.sum((b - A @ theta)**2)

    deg_freedom = max(1, N - (2 + M))
    sigma_sq = sum_sq_res / deg_freedom
    cov_matrix = sigma_sq * np.linalg.pinv(A.T @ A)
    std_errs = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
    xc_err, yc_err = std_errs[0], std_errs[1]
    cov_xy = cov_matrix[:2, :2]

    return xc, yc, radii, xc_err, yc_err, cov_xy


# ==============================================================================
# 2D Anamorphic Surface Polynomial Basis & Evaluation
# ==============================================================================

def build_2d_poly_basis(g_norm, y_norm, deg=4):
    """
    Constructs the 2D bivariate polynomial design matrix for surface regression:
    sum_{m+n <= deg} (g_norm^m) * (y_norm^n).
    """
    g_arr = np.asarray(g_norm).ravel()
    y_arr = np.asarray(y_norm).ravel()
    cols = []
    for m in range(deg + 1):
        for n in range(deg + 1 - m):
            cols.append((g_arr ** m) * (y_arr ** n))
    return np.column_stack(cols)


def fit_anamorphic_order_surface(gn_laser, yn_laser, all_k_laser, deg=4):
    """
    Fits the continuous 2D anamorphic interference order surface coefficients c_2d_order.
    """
    B_laser = build_2d_poly_basis(gn_laser, yn_laser, deg=deg)
    c_2d_order, _, _, _ = np.linalg.lstsq(B_laser, all_k_laser.astype(np.float64), rcond=None)
    return c_2d_order


def eval_phi_order_surface(g_arr, y_arr, c_2d_order, y0_fit, L_y, g_mean_ref, g_std_ref, deg=4):
    """
    Evaluates the 2D anamorphic interference order surface phi_order(g, y) using fitted coefficients.
    """
    gn = (g_arr - g_mean_ref) / g_std_ref
    yn = (y_arr - y0_fit) / L_y
    B = build_2d_poly_basis(gn, yn, deg=deg)
    return (B @ c_2d_order).reshape(g_arr.shape)


def eval_phi_order_2d(g_arr, y_arr, L_y, all_k_laser, gn_laser, yn_laser, y0_fit, g_mean_ref, g_std_ref, deg=4):
    """
    Backward-compatible evaluation of 2D anamorphic order surface fitting on the fly.
    """
    c_2d_order = fit_anamorphic_order_surface(gn_laser, yn_laser, all_k_laser, deg=deg)
    return eval_phi_order_surface(g_arr, y_arr, c_2d_order, y0_fit, L_y, g_mean_ref, g_std_ref, deg=deg)


# ==============================================================================
# Radial Profile Extraction
# ==============================================================================

def extract_profile(data_array, center, angle):
    """
    Extracts a 1D radial profile extending outwards from center at `angle` (radians)
    with 1-pixel radial sampling.
    """
    xc, yc = center
    h, w = data_array.shape

    r_min = np.sqrt(xc**2 + yc**2)
    r_max = w - xc
    r = np.arange(int(r_min), int(r_max), 1)

    x_coords = xc + r * np.cos(angle)
    y_coords = yc + r * np.sin(angle)

    profile = ndimage.map_coordinates(data_array, [y_coords, x_coords], order=1, mode='constant', cval=0.0)
    return profile


def extract_profiles_2d(data, center, num_angles=1000):
    """
    Extracts a 2D array of radial profiles along equidistant angles spanning the detector.
    """
    xc, yc = center
    h, w = data.shape

    angle_min = np.arctan2(0.0 - yc, (w - 1.0) - xc)
    angle_max = np.arctan2((h - 1.0) - yc, (w - 1.0) - xc)
    angles = np.linspace(angle_min, angle_max, num_angles)

    profiles = [extract_profile(data, center, a) for a in angles]

    r_min = np.sqrt(xc**2 + yc**2)
    r_max = w - xc
    radius_axis = np.arange(int(r_min), int(r_max), 1)

    return np.array(profiles), radius_axis


# ==============================================================================
# Rubidium Vapor Physics & Spectrum Modelling
# ==============================================================================

def vapor_pressure_torr(T_K):
    """
    Alcock et al. Rubidium vapor pressure model (Torr).
    """
    if T_K < 312.45:
        log10_Pv = 2.881 + 4.857 - 4215.0 / T_K
    else:
        log10_Pv = 2.881 + 4.312 - 4040.0 / T_K
    return 10.0**log10_Pv


def atomic_density(T_K):
    """
    Atomic number density of Rubidium vapor (atoms / m^3).
    """
    pv_pa = vapor_pressure_torr(T_K) * 133.322368
    return pv_pa / (config.kB * T_K)


def voigt_profile(delta_nu, Gamma_eff_Hz, sigma_D):
    """
    Normalized Voigt spectral profile via Faddeeva (wofz) function.
    """
    z = (delta_nu + 1j * (Gamma_eff_Hz / 2.0)) / (sigma_D * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_D * np.sqrt(2.0 * np.pi))


def compute_spectrum(T_C, L_cell, detuning_MHz):
    """
    Computes theoretical transmission spectrum of natural Rubidium vapor (85Rb + 87Rb D2 line).
    """
    T_K = T_C + 273.15
    n_total = atomic_density(T_K)
    Gamma_eff_Hz = config.Gamma_nat_Hz + 2.0 * config.gamma_self_rate * n_total
    sigma_D_85 = config.nu_0 * np.sqrt(config.kB * T_K / (config.m_rb85 * config.c_light**2))
    sigma_D_87 = config.nu_0 * np.sqrt(config.kB * T_K / (config.m_rb87 * config.c_light**2))

    detuning_Hz = detuning_MHz * 1e6
    alpha = np.zeros_like(detuning_Hz, dtype=np.float64)

    for d_nu_MHz, w in config.transitions_85:
        alpha += n_total * w * config.sigma_int * voigt_profile(detuning_Hz - d_nu_MHz * 1e6, Gamma_eff_Hz, sigma_D_85)
    for d_nu_MHz, w in config.transitions_87:
        alpha += n_total * w * config.sigma_int * voigt_profile(detuning_Hz - d_nu_MHz * 1e6, Gamma_eff_Hz, sigma_D_87)

    return np.exp(-alpha * L_cell)


def fold_spectrum_across_orders(phi_order, spectrum, fsr, delta_nu_offset, f_grid, min_points_per_order=20):
    """
    Folds a radial spectrum across its interference orders onto a common detuning grid.

    Parameters
    ----------
    phi_order : numpy.ndarray
        Interference order array.
    spectrum : numpy.ndarray
        Spectral intensity array.
    fsr : float
        VIPA Free Spectral Range in GHz.
    delta_nu_offset : float
        Frequency offset guess in GHz.
    f_grid : numpy.ndarray
        Target detuning frequency grid in GHz.
    min_points_per_order : int
        Minimum points required to accept an order.

    Returns
    -------
    order_spectra : numpy.ndarray, shape (M, len(f_grid))
        Interpolated spectrum for each valid order.
    orders_used : list of int
        List of order indices included.
    avg_one_order : numpy.ndarray, shape (len(f_grid),)
        Averaged spectrum across all valid orders.
    """
    phi_shifted = phi_order - delta_nu_offset / fsr
    k_min = int(np.ceil(np.nanmin(phi_shifted)))
    k_max = int(np.floor(np.nanmax(phi_shifted)))

    order_spectra = []
    orders_used = []

    for k in range(k_min, k_max + 1):
        m = (phi_shifted >= k - 0.5) & (phi_shifted < k + 0.5)
        if np.sum(m) > min_points_per_order:
            f_k = (phi_shifted[m] - k) * fsr
            idx_sort = np.argsort(f_k)
            spec_k = np.interp(f_grid, f_k[idx_sort], spectrum[m][idx_sort], left=np.nan, right=np.nan)
            order_spectra.append(spec_k)
            orders_used.append(k)

    order_spectra = np.array(order_spectra) if len(order_spectra) > 0 else np.empty((0, len(f_grid)))
    avg_one_order = np.nanmean(order_spectra, axis=0) if len(order_spectra) > 0 else np.full_like(f_grid, np.nan)

    return order_spectra, orders_used, avg_one_order


# ==============================================================================
# Spectral Binning & Vectorized DHO Fitting Algorithms
# ==============================================================================

def bin_2d_array(arr, bin_h=4, bin_w=2):
    """
    Spatially bins a 2D array by averaging over blocks of size (bin_h, bin_w).
    """
    h, w = arr.shape
    new_h, new_w = h // bin_h, w // bin_w
    return arr[:new_h * bin_h, :new_w * bin_w].reshape(new_h, bin_h, new_w, bin_w).mean(axis=(1, 3))


def compute_order_averaged_variance(var_arr):
    """
    Computes the variance of the order average 0.5 * (AS + S).
    For independent random variables, Var(0.5*(X + Y)) = 0.25 * (Var(X) + Var(Y)).
    If only one branch is finite, returns that branch's variance.
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
        ds_init = config.EXPECTED_BRILLOUIN_SHIFT
    if shift_tol is None:
        shift_tol = config.BRILLOUIN_SHIFT_TOLERANCE
    if gamma_init is None:
        gamma_init = config.EXPECTED_LINEWIDTH
    if gamma_bounds is None:
        gamma_bounds = config.LINEWIDTH_BOUNDS

    t_start = time.time()
    n_rows, n_cols = freq_bin_in.shape
    if n_orders is None:
        if config.N_ORDERS_SAMPLE is not None:
            n_orders = config.N_ORDERS_SAMPLE
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


# ==============================================================================
# 2D Coaxial Ellipses & Acoustic Wavevector (q-vector) Geometry
# ==============================================================================

def eval_coaxial_surface(x, y, xc, yc, phi_rad):
    """
    Transforms (x, y) into centered coordinates (Xp, Yp) rotated by phi_rad,
    and returns the design matrix [1, Xp**2, Yp**2] for concentric coaxial quadrics.
    """
    dx = x - xc
    dy = y - yc
    ct = np.cos(phi_rad)
    st = np.sin(phi_rad)
    Xp = dx * ct + dy * st
    Yp = -dx * st + dy * ct
    A = np.column_stack([np.ones_like(Xp), Xp**2, Yp**2])
    return Xp, Yp, A


def fit_concentric_ellipses_coaxial(
    x,
    y,
    shift,
    linewidth,
    shift_err=None,
    linewidth_err=None,
    initial_guess=None,
):
    """
    Fits 2D concentric ellipses with shared director axes (coaxial center xc, yc
    and principal rotation angle phi) simultaneously to both Shift(x, y) and
    Linewidth(x, y) quadratic surfaces using Weighted Least Squares (WLS):
        Shift(X', Y')     = S0  + a_S * X'^2  + b_S * Y'^2
        Linewidth(X', Y') = LW0 + a_LW * X'^2 + b_LW * Y'^2
    """
    w_s = 1.0 / np.asarray(shift_err, dtype=np.float64) if shift_err is not None else np.ones_like(shift, dtype=np.float64)
    w_lw = 1.0 / np.asarray(linewidth_err, dtype=np.float64) if linewidth_err is not None else np.ones_like(linewidth, dtype=np.float64)

    mean_w_s = np.sum(w_s**2 * shift) / np.sum(w_s**2)
    chi2_0_s = float(np.sum((w_s * (shift - mean_w_s)) ** 2))

    mean_w_lw = np.sum(w_lw**2 * linewidth) / np.sum(w_lw**2)
    chi2_0_lw = float(np.sum((w_lw * (linewidth - mean_w_lw)) ** 2))

    if initial_guess is None:
        xc_init = float(np.average(x, weights=w_s))
        yc_init = float(np.average(y, weights=w_s))
        phi_init = float(np.deg2rad(-70.0))
    else:
        xc_init, yc_init, phi_init = initial_guess

    def _eval_weighted_solution(xc, yc, phi):
        _, _, A = eval_coaxial_surface(x, y, xc, yc, phi)
        Aw_s = w_s[:, None] * A
        zw_s = w_s * shift
        cs, _, _, _ = np.linalg.lstsq(Aw_s, zw_s, rcond=None)
        chi2_s = np.sum((zw_s - Aw_s @ cs) ** 2)

        Aw_lw = w_lw[:, None] * A
        zw_lw = w_lw * linewidth
        clw, _, _, _ = np.linalg.lstsq(Aw_lw, zw_lw, rcond=None)
        chi2_lw = np.sum((zw_lw - Aw_lw @ clw) ** 2)

        return cs, clw, chi2_s, chi2_lw

    def _joint_weighted_cost(params):
        xc, yc, phi = params
        _, _, chi2_s, chi2_lw = _eval_weighted_solution(xc, yc, phi)
        return float(chi2_s / chi2_0_s + chi2_lw / chi2_0_lw)

    opt_result = minimize(
        _joint_weighted_cost,
        x0=np.array([xc_init, yc_init, phi_init]),
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-5},
    )

    xc_opt, yc_opt, phi_opt = opt_result.x
    cs_opt, clw_opt, _, _ = _eval_weighted_solution(xc_opt, yc_opt, phi_opt)

    S0, a_s, b_s = cs_opt
    LW0, a_lw, b_lw = clw_opt

    gamma_s = np.sqrt(abs(a_s / b_s)) if b_s != 0 else 1.0
    gamma_lw = np.sqrt(abs(a_lw / b_lw)) if b_lw != 0 else 1.0
    gamma_mean = float(0.5 * (gamma_s + gamma_lw))

    return {
        "xc": float(xc_opt),
        "yc": float(yc_opt),
        "phi_rad": float(phi_opt),
        "phi_deg": float(np.degrees(phi_opt)),
        "gamma_shift": float(gamma_s),
        "gamma_linewidth": float(gamma_lw),
        "gamma": gamma_mean,
        "coeffs_shift": cs_opt,
        "coeffs_linewidth": clw_opt,
        "chi2_0_s": chi2_0_s,
        "chi2_0_lw": chi2_0_lw,
    }


def optimize_elliptical_symmetry_variance(
    x,
    y,
    shift,
    linewidth,
    shift_err,
    linewidth_err,
    initial_params,
):
    """
    Refines 2D concentric ellipse parameters (xc, yc, phi, gamma) by minimizing
    the error-weighted residual variance around 1D radial profiles.
    """
    w_s = 1.0 / np.asarray(shift_err, dtype=np.float64) if shift_err is not None else np.ones_like(shift, dtype=np.float64)
    w_lw = 1.0 / np.asarray(linewidth_err, dtype=np.float64) if linewidth_err is not None else np.ones_like(linewidth, dtype=np.float64)

    chi2_0_s = float(initial_params.get("chi2_0_s", np.sum((w_s * (shift - np.mean(shift))) ** 2)))
    chi2_0_lw = float(initial_params.get("chi2_0_lw", np.sum((w_lw * (linewidth - np.mean(linewidth))) ** 2)))

    xc0 = float(initial_params["xc"])
    yc0 = float(initial_params["yc"])
    phi0 = float(initial_params["phi_rad"])
    gamma0 = float(initial_params["gamma"])

    p0 = np.array([xc0, yc0, phi0, gamma0])
    R0 = 1000.0

    def _weighted_radial_variance_cost(params):
        xc, yc, phi, gamma = params
        if gamma <= 0.2 or gamma >= 4.0:
            return 1e10

        dx = x - xc
        dy = y - yc
        ct = np.cos(phi)
        st = np.sin(phi)
        Xp = dx * ct + dy * st
        Yp = -dx * st + dy * ct

        r_sph2 = Xp**2 + (Yp / gamma)**2
        u = r_sph2 / (R0**2)
        A_rad = np.column_stack([np.ones_like(u), u, u**2])

        Aw_s = w_s[:, None] * A_rad
        zw_s = w_s * shift
        c_s, _, _, _ = np.linalg.lstsq(Aw_s, zw_s, rcond=None)
        chi2_res_s = np.sum((zw_s - Aw_s @ c_s) ** 2)

        Aw_lw = w_lw[:, None] * A_rad
        zw_lw = w_lw * linewidth
        c_lw, _, _, _ = np.linalg.lstsq(Aw_lw, zw_lw, rcond=None)
        chi2_res_lw = np.sum((zw_lw - Aw_lw @ c_lw) ** 2)

        return float(chi2_res_s / chi2_0_s + chi2_res_lw / chi2_0_lw)

    bounds = [
        (xc0 - 250.0, xc0 + 250.0),
        (yc0 - 250.0, yc0 + 250.0),
        (phi0 - np.deg2rad(25.0), phi0 + np.deg2rad(25.0)),
        (max(0.3, gamma0 - 0.8), min(3.5, gamma0 + 0.8)),
    ]

    opt_result = minimize(
        _weighted_radial_variance_cost,
        x0=p0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-7},
    )

    xc_ref, yc_ref, phi_ref, gamma_ref = opt_result.x
    initial_cost = _weighted_radial_variance_cost(p0)
    final_cost = float(opt_result.fun)
    improvement = (initial_cost - final_cost) / initial_cost * 100.0

    return {
        "xc": float(xc_ref),
        "yc": float(yc_ref),
        "phi_rad": float(phi_ref),
        "phi_deg": float(np.degrees(phi_ref)),
        "gamma": float(gamma_ref),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "improvement_pct": improvement,
    }


def transform_to_spherical_coordinates(x, y, xc, yc, phi_rad, gamma):
    """
    Transforms detector coordinates (x, y) into spherical coordinates (x_sph, y_sph, r_sph)
    where Shift and Linewidth exhibit circular/spherical symmetry:
        X' =  (x - xc) * cos(phi) + (y - yc) * sin(phi)
        Y' = -(x - xc) * sin(phi) + (y - yc) * cos(phi)
        x_sph = X'
        y_sph = Y' / gamma
        r_sph = sqrt(x_sph**2 + y_sph**2)
    """
    dx = x - xc
    dy = y - yc
    ct = np.cos(phi_rad)
    st = np.sin(phi_rad)
    Xp = dx * ct + dy * st
    Yp = -dx * st + dy * ct

    x_sph = Xp
    y_sph = Yp / gamma
    r_sph = np.sqrt(x_sph**2 + y_sph**2)
    return x_sph, y_sph, r_sph


def fit_magnification_factor_M_theta(r_sph, shift, shift_err=None, initial_M_theta=3000.0):
    """
    Fits the angular magnification factor M_theta and normal backscattering shift Shift(0):
        Shift(r_sph) = Shift(0) * cos(arctan(r_sph / M_theta) / 2)
    weighted by experimental errors (sigma = shift_err).
    """
    def _shift_vs_r_sph_model(r, S0, M):
        return S0 * np.cos(np.arctan(r / M) / 2.0)

    S0_guess = float(np.percentile(shift, 98))
    p0 = [S0_guess, initial_M_theta]
    bounds = ([0.0, 100.0], [50.0, 100000.0])

    use_sig = shift_err is not None and not np.all(shift_err == 1.0)
    popt, pcov = curve_fit(
        _shift_vs_r_sph_model,
        r_sph,
        shift,
        p0=p0,
        sigma=shift_err if use_sig else None,
        absolute_sigma=use_sig,
        bounds=bounds,
    )
    S0_fit, M_theta_fit = popt
    perr = np.sqrt(np.diag(pcov))

    return float(S0_fit), float(M_theta_fit), (float(perr[0]), float(perr[1]))


def compute_theta_and_q_vector(r_sph, M_theta, n_medium, lambda_vacuum, Shift_0=None):
    """
    Deduces scattering angle theta, acoustic wavevector magnitude q_vector, and longitudinal sound velocity V_s:
        theta    = np.arctan(r_sph / M_theta)
        q_vector = 2 * n * (2 * np.pi / lambda) * np.cos(theta / 2)
        V_s      = (lambda * Shift_0) / (2 * n)
    """
    theta = np.arctan(r_sph / M_theta)
    theta_deg = np.degrees(theta)

    k0 = 2.0 * np.pi / lambda_vacuum
    q_vector = 2.0 * n_medium * k0 * np.cos(theta / 2.0)
    q_vector_um = q_vector * 1e-6

    results = {
        "theta": theta,
        "theta_deg": theta_deg,
        "q_vector": q_vector,
        "q_vector_um": q_vector_um,
    }

    if Shift_0 is not None:
        V_s = (lambda_vacuum * (Shift_0 * 1e9)) / (2.0 * n_medium)
        results["V_s"] = float(V_s)

    return results


def filter_and_prepare_q_vector_data(
    x_raw,
    y_raw,
    shift_raw,
    linewidth_raw,
    shift_err_raw=None,
    linewidth_err_raw=None,
    average_stokes_antistokes=True,
    shift_range=(None, None),
    linewidth_range=(None, None),
    error_percentile_floor=1.0,
):
    """
    Cleans, filters, averages Stokes/Anti-Stokes orders, and flattens arrays for q-vector fitting.
    """
    has_errors = (shift_err_raw is not None) and (linewidth_err_raw is not None)
    if not has_errors:
        shift_err_raw = np.ones_like(shift_raw)
        linewidth_err_raw = np.ones_like(linewidth_raw)

    if average_stokes_antistokes and shift_raw.ndim == 2 and shift_raw.shape[1] > 2 and (shift_raw.shape[1] % 2 == 0):
        shift_proc = 0.5 * (shift_raw[:, 0::2] + shift_raw[:, 1::2])
        linewidth_proc = 0.5 * (linewidth_raw[:, 0::2] + linewidth_raw[:, 1::2])
        x_coords_proc = 0.5 * (x_raw[:, 0::2] + x_raw[:, 1::2])
        y_coords_proc = 0.5 * (y_raw[:, 0::2] + y_raw[:, 1::2])

        if has_errors:
            shift_err_proc = 0.5 * np.sqrt(shift_err_raw[:, 0::2] ** 2 + shift_err_raw[:, 1::2] ** 2)
            linewidth_err_proc = 0.5 * np.sqrt(linewidth_err_raw[:, 0::2] ** 2 + linewidth_err_raw[:, 1::2] ** 2)
        else:
            shift_err_proc = np.ones_like(shift_proc)
            linewidth_err_proc = np.ones_like(linewidth_proc)
    else:
        shift_proc = shift_raw
        linewidth_proc = linewidth_raw
        x_coords_proc = x_raw
        y_coords_proc = y_raw
        shift_err_proc = shift_err_raw
        linewidth_err_proc = linewidth_err_raw

    mask = (
        np.isfinite(shift_proc)
        & np.isfinite(linewidth_proc)
        & np.isfinite(x_coords_proc)
        & np.isfinite(y_coords_proc)
        & np.isfinite(shift_err_proc)
        & np.isfinite(linewidth_err_proc)
        & (shift_proc > 0)
        & (linewidth_proc > 0)
        & (shift_err_proc > 0)
        & (linewidth_err_proc > 0)
    )

    s_min, s_max = shift_range
    if s_min is not None:
        mask &= (shift_proc >= s_min)
    if s_max is not None:
        mask &= (shift_proc <= s_max)

    lw_min, lw_max = linewidth_range
    if lw_min is not None:
        mask &= (linewidth_proc >= lw_min)
    if lw_max is not None:
        mask &= (linewidth_proc <= lw_max)

    n_valid = int(np.sum(mask))
    if n_valid == 0:
        raise ValueError("No valid data points remaining after filtering for q-vector extraction.")

    x_flat = x_coords_proc[mask].flatten()
    y_flat = y_coords_proc[mask].flatten()
    shift_flat = shift_proc[mask].flatten()
    linewidth_flat = linewidth_proc[mask].flatten()
    shift_err_flat = shift_err_proc[mask].flatten()
    linewidth_err_flat = linewidth_err_proc[mask].flatten()

    if has_errors and error_percentile_floor > 0:
        p_floor_s = float(np.percentile(shift_err_flat, error_percentile_floor))
        p_floor_lw = float(np.percentile(linewidth_err_flat, error_percentile_floor))
        shift_err_flat = np.maximum(shift_err_flat, p_floor_s)
        linewidth_err_flat = np.maximum(linewidth_err_flat, p_floor_lw)

    return x_flat, y_flat, shift_flat, linewidth_flat, shift_err_flat, linewidth_err_flat, has_errors
