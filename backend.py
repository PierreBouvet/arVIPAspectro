"""
Core mathematical, physical, and signal/image processing functions for VIPA Brillouin calibration.
"""

import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy.special import wofz
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
