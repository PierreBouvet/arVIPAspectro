"""
Rubidium absorption calibration pipeline: theoretical spectrum modeling,
absorption denoising, radial profile extraction, FSR calibration, and 2D frequency map construction.
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import minimize_scalar

import config
import backend as bck


# ==============================================================================
# Step 7: Theoretical Rubidium D2 Absorption Spectrum Model
# ==============================================================================

def create_rubidium_reference_table(T_cell=config.T_CELL, L_cell=config.L_CELL,
                                    det_min_MHz=-35e3, det_max_MHz=35e3, n_points=70001):
    """
    Computes a high-resolution theoretical natural Rubidium D2 transmission lookup table
    and returns (det_table_MHz, trans_table, trans_fast).
    """
    det_table_MHz = np.linspace(det_min_MHz, det_max_MHz, n_points)
    trans_table = bck.compute_spectrum(T_cell, L_cell, det_table_MHz)

    def trans_fast(det_MHz):
        return np.interp(det_MHz, det_table_MHz, trans_table, left=1.0, right=1.0)

    return det_table_MHz, trans_table, trans_fast


# ==============================================================================
# Step 8: Rubidium Absorption Preprocessing & Denoising
# ==============================================================================

def retrieve_denoised_rb(wrapper, path_group, name='image_corrected'):
    """
    Retrieves the denoised Rubidium absorption image and baseline from HDF5.
    """
    path_corrected = f'{path_group}/{name}'
    path_raw = f'{path_group}/image'
    data_absorption = wrapper[path_corrected].astype(np.float64)
    data_raw = wrapper[path_raw].astype(np.float64)
    baseline_absorption = data_raw - data_absorption
    return data_absorption, baseline_absorption


def denoise_rb(wrapper, path_group, data_absorption_raw,
               scale=config.SCALE_RB,
               ksize_structuring_elements=config.KSIZE_STRUCTURING_ELEMENTS_RB,
               ksize_blur=config.KSIZE_BLUR_RB):
    """
    Denoises the Rubidium absorption figure using morphological opening,
    saves 'image_corrected' to HDF5, and returns (data_absorption, baseline_absorption).
    """
    baseline_absorption = bck.extract_baseline_image_morphology(
        data_absorption_raw,
        scale=scale,
        ksize_structuring_elements=ksize_structuring_elements,
        ksize_blur=ksize_blur
    )
    data_absorption = data_absorption_raw - baseline_absorption
    wrapper.add_other(data_absorption, path_group, 'image_corrected', overwrite=True)
    return data_absorption, baseline_absorption


# ==============================================================================
# Step 9: Extraction of Radial Absorption Profile
# ==============================================================================

def extract_radial_absorption_profile(data_absorption, xc_gnom, yc_gnom, y0_fit,
                                      c_2d_order, g_mean_ref, g_std_ref, L_y,
                                      f_px=config.F_PX_DEFAULT,
                                      phi_rad=config.VIPA_TILT_ANGLE,
                                      n_r=config.N_FUSED_SILICA_780,
                                      num_angles=1000):
    """
    Extracts 2D radial profiles centered at the gnomonic ring center, averages across angles,
    and maps the radial axis to 2D continuous interference orders.
    """
    profiles_2d, radius_axis = bck.extract_profiles_2d(data_absorption, (xc_gnom, yc_gnom), num_angles=num_angles)
    avg_spectrum_radius = np.mean(profiles_2d, axis=0)

    # Convert radius to directional cosine u and internal refraction factor g(u)
    xf_r = radius_axis - f_px * np.tan(phi_rad)
    u_r = (f_px * np.cos(phi_rad) - xf_r * np.sin(phi_rad)) / np.sqrt(f_px**2 + xf_r**2)
    g_r = bck.g_function(u_r, n_r)

    # Evaluate continuous 2D interference orders along the radial profile
    phi_order_r = bck.eval_phi_order_surface(
        g_r,
        np.full_like(g_r, y0_fit),
        c_2d_order,
        y0_fit=y0_fit,
        L_y=L_y,
        g_mean_ref=g_mean_ref,
        g_std_ref=g_std_ref
    )

    return profiles_2d, radius_axis, avg_spectrum_radius, phi_order_r


# ==============================================================================
# Step 10: Calibration of VIPA FSR & Rubidium Hyperfine Offset
# ==============================================================================

def retrieve_fsr_calibration(wrapper, path_group, g_bar, q_orders, phi_order_r,
                             avg_spectrum_radius, trans_fast,
                             nu_0=config.nu_0, fit_window_ghz=config.FIT_WINDOW_GHZ,
                             delta_nu_laser_offset=config.DELTA_NU_LASER_OFFSET):
    """
    Retrieves calibrated FSR and delta_nu from HDF5 attributes and reconstructs folded spectrum data.
    """
    attributes = wrapper.get_attributes(path=path_group)
    FSR_cal = float(attributes['FSR'])
    delta_nu_opt = float(attributes['delta_nu'])
    delta_nu_fine = delta_nu_opt - delta_nu_laser_offset

    f_grid = np.linspace(-FSR_cal / 2.0, FSR_cal / 2.0, 1000)
    order_spectra, orders_used, avg_one_order = bck.fold_spectrum_across_orders(
        phi_order_r, avg_spectrum_radius, FSR_cal, delta_nu_opt, f_grid
    )

    valid_mask = np.isfinite(avg_one_order)
    f_valid = f_grid[valid_mask]
    sig_valid = avg_one_order[valid_mask]
    baseline_env = ndimage.gaussian_filter1d(sig_valid, 60) if len(sig_valid) > 60 else np.zeros_like(sig_valid)
    measured_absorption = np.maximum(baseline_env - sig_valid, 0.0)

    f_fit_min = -fit_window_ghz / 2.0
    f_fit_max = +fit_window_ghz / 2.0
    fit_mask = (f_valid >= f_fit_min) & (f_valid <= f_fit_max)
    f_fit_rb = f_valid[fit_mask]
    meas_abs_fit = measured_absorption[fit_mask]
    theo_fit_rb = 1.0 - trans_fast((f_fit_rb - delta_nu_fine) * 1e3)

    return {
        'FSR_cal': FSR_cal,
        'delta_nu_opt': delta_nu_opt,
        'delta_nu_fine': delta_nu_fine,
        'f_grid': f_grid,
        'order_spectra': order_spectra,
        'orders_used': orders_used,
        'f_valid': f_valid,
        'sig_valid': sig_valid,
        'baseline_env': baseline_env,
        'f_fit_min': f_fit_min,
        'f_fit_max': f_fit_max,
        'f_fit_rb': f_fit_rb,
        'meas_abs_fit': meas_abs_fit,
        'theo_fit_rb': theo_fit_rb
    }


def calibrate_fsr_and_offset(wrapper, path_group, g_bar, q_orders, phi_order_r,
                             avg_spectrum_radius, trans_fast,
                             nu_0=config.nu_0, fit_window_ghz=config.FIT_WINDOW_GHZ,
                             delta_nu_laser_offset=config.DELTA_NU_LASER_OFFSET):
    """
    Calibrates the absolute Free Spectral Range (FSR) and frequency offset by matching
    coherently folded multi-order absorption spectra with the theoretical Rb D2 model.
    """
    ok = ~np.isnan(g_bar)
    poly_q_to_g = np.polyfit(q_orders[ok], g_bar[ok], 1)
    s_cal = -poly_q_to_g[0] / poly_q_to_g[1]
    nu_laser_init = nu_0 * 1e-9 + delta_nu_laser_offset
    FSR_guess = nu_laser_init * s_cal

    f_grid = np.linspace(-FSR_guess / 2.0, FSR_guess / 2.0, 1000)
    order_spectra, orders_used, avg_one_order = bck.fold_spectrum_across_orders(
        phi_order_r, avg_spectrum_radius, FSR_guess, delta_nu_laser_offset, f_grid
    )

    valid_mask = np.isfinite(avg_one_order)
    f_valid = f_grid[valid_mask]
    sig_valid = avg_one_order[valid_mask]
    baseline_env = ndimage.gaussian_filter1d(sig_valid, 60)
    measured_absorption = np.maximum(baseline_env - sig_valid, 0.0)

    f_fit_min = -fit_window_ghz / 2.0
    f_fit_max = +fit_window_ghz / 2.0
    fit_mask = (f_valid >= f_fit_min) & (f_valid <= f_fit_max)
    f_fit_rb = f_valid[fit_mask]
    meas_abs_fit = measured_absorption[fit_mask]

    # Coarse grid scan followed by scalar minimization
    shifts_scan = np.linspace(-3.0, 3.0, 1200)
    conv_scan = np.array([
        np.sum(meas_abs_fit * (1.0 - trans_fast((f_fit_rb - s) * 1e3)))
        for s in shifts_scan
    ])
    s_coarse = shifts_scan[np.argmax(conv_scan)]

    res_rb = minimize_scalar(
        lambda s: -np.sum(meas_abs_fit * (1.0 - trans_fast((f_fit_rb - s) * 1e3))),
        bracket=(s_coarse - 0.3, s_coarse + 0.3)
    )
    delta_nu_fine = float(res_rb.x)
    delta_nu_opt = float(delta_nu_laser_offset + delta_nu_fine)
    FSR_cal = float((nu_0 * 1e-9 + delta_nu_opt) * s_cal)

    theo_fit_rb = 1.0 - trans_fast((f_fit_rb - delta_nu_fine) * 1e3)

    attributes = {
        'FSR': FSR_cal,
        'delta_nu': delta_nu_opt,
    }
    wrapper.add_attributes(attributes, parent_group=path_group, overwrite=True)

    return {
        'FSR_cal': FSR_cal,
        'delta_nu_opt': delta_nu_opt,
        'delta_nu_fine': delta_nu_fine,
        'f_grid': f_grid,
        'order_spectra': order_spectra,
        'orders_used': orders_used,
        'f_valid': f_valid,
        'sig_valid': sig_valid,
        'baseline_env': baseline_env,
        'f_fit_min': f_fit_min,
        'f_fit_max': f_fit_max,
        'f_fit_rb': f_fit_rb,
        'meas_abs_fit': meas_abs_fit,
        'theo_fit_rb': theo_fit_rb
    }


# ==============================================================================
# Frequency Map Construction & Storage
# ==============================================================================

def compute_and_save_frequency_map(wrapper, path_group, img_shape, x0_fit, y0_fit, xc_gnom, yc_gnom,
                                   c_2d_order, g_mean_ref, g_std_ref, L_y, FSR_cal, delta_nu_opt,
                                   rb_lock_offset_ghz=config.RB_LOCK_OFFSET_GHZ,
                                   f_px=config.F_PX_DEFAULT,
                                   phi_rad=config.VIPA_TILT_ANGLE,
                                   n_r=config.N_FUSED_SILICA_780):
    """
    Constructs the full-resolution 2D calibrated frequency map zero-referenced to the
    elastic Rayleigh lines (85Rb F=2 absorption lock) and stores it in HDF5.
    """
    h, w = img_shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    g_map = bck.g_from_xy(xx, yy, x0_fit, y0_fit, f=f_px, phi=phi_rad, n_r=n_r)
    phi_order_map = bck.eval_phi_order_surface(g_map, yy, c_2d_order, y0_fit, L_y, g_mean_ref, g_std_ref)

    xi_elastic_rb85 = float((delta_nu_opt + rb_lock_offset_ghz) / FSR_cal)
    freq_map = (phi_order_map - xi_elastic_rb85) * FSR_cal

    wrapper.add_frequency(
        data=freq_map,
        parent_group=path_group,
        name='Frequency',
        overwrite=True
    )
    attributes = {
        'FSR': float(FSR_cal),
        'yc_opt': float(yc_gnom),
        'xc_opt': float(xc_gnom),
        'x0_fit': float(x0_fit),
        'y0_fit': float(y0_fit),
        'xi_elastic': float(xi_elastic_rb85),
    }
    wrapper.add_attributes(attributes, parent_group=path_group, overwrite=True)

    return freq_map, phi_order_map, xi_elastic_rb85
