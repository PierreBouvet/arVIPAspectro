"""
Laser image calibration pipeline: denoising, fringe tracing, sub-pixel refinement,
gnomonic geometry fitting, and 2D anamorphic order surface calibration.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares

import config
import backend as bck


# ==============================================================================
# Step 2: Laser Reference Denoising
# ==============================================================================

def retrieve_denoised_data(wrapper, path_group, name='image_corrected'):
    """
    Retrieves the background-subtracted laser reference image and reconstructed baseline from HDF5.
    """
    path_corrected = f'{path_group}/{name}'
    path_raw = f'{path_group}/image'
    data_subtracted = wrapper[path_corrected].astype(np.float64)
    data_raw = wrapper[path_raw].astype(np.float64)
    baseline_2d = data_raw - data_subtracted
    return data_subtracted, baseline_2d


def denoise_laser(wrapper, path_group, data_laser_reference, scale=config.SCALE_LASER,
                  ksize_structuring_elements=config.KSIZE_STRUCTURING_ELEMENTS_LASER,
                  ksize_blur=config.KSIZE_BLUR_LASER):
    """
    Denoises the monochromatic laser reference image using morphological opening,
    saves 'image_corrected' to HDF5, and returns (data_subtracted, baseline_2d).
    """
    baseline_2d = bck.extract_baseline_image_morphology(
        data_laser_reference,
        scale=scale,
        ksize_structuring_elements=ksize_structuring_elements,
        ksize_blur=ksize_blur
    )
    data_subtracted = data_laser_reference - baseline_2d
    wrapper.add_other(data_subtracted, path_group, 'image_corrected', overwrite=True)
    return data_subtracted, baseline_2d


# ==============================================================================
# Step 3: Laser Line Detection & Fringe Tracing
# ==============================================================================

def retrieve_fringes(wrapper, path_group, y_center_laser=config.Y_CENTER_LASER):
    """
    Retrieves traced fringes and central peak seeds from HDF5.
    """
    nb_fringe = 0
    points = []
    children = wrapper.get_children_elements(path_group)
    while f'fringe_{nb_fringe}' in children:
        fringe = wrapper[f'{path_group}/fringe_{nb_fringe}']
        points.append(fringe)
        nb_fringe += 1
    peaks = [[fringe[0][np.argmin(np.abs(fringe[1] - y_center_laser))]] for fringe in points]
    return peaks, points


def extract_laser_fringes(wrapper, path_group, data_subtracted,
                          y_center_laser=config.Y_CENTER_LASER,
                          dy_laser=config.DY_LASER,
                          nb_laser_modes=config.NB_LASER_MODES,
                          distance_peaks=config.DISTANCE_PEAKS_LASER,
                          dy_trace=config.DY_TRACE,
                          var_trace=config.VAR_TRACE):
    """
    Detects seed peaks along the central symmetry rows and traces fringes vertically.
    Stores raw fringes in HDF5 and sets 'fringes_smoothed': False.
    """
    h, _ = data_subtracted.shape
    y_min = max(0, int(y_center_laser - dy_laser))
    y_max = min(h, int(y_center_laser + dy_laser))
    average_data = np.average(data_subtracted[y_min:y_max, :], axis=0)

    peaks, properties = find_peaks(average_data, prominence=0, distance=distance_peaks)
    if len(peaks) >= nb_laser_modes:
        top_idx = np.argsort(properties['prominences'])[-nb_laser_modes:]
        peaks = np.sort(peaks[top_idx])
    peaks = np.array(peaks).reshape((-1, 1))

    # Bidirectional vertical fringe tracing
    points = bck.extract_fringes(data_subtracted, peaks, y_center=y_center_laser, dy=dy_trace, var=var_trace)
    points = [np.array(fringe) for fringe in points]

    for nb_fringe, fringe in enumerate(points):
        wrapper.add_other(fringe, path_group, f'fringe_{nb_fringe}', overwrite=True)

    attributes = {'fringes_smoothed': False}
    wrapper.add_attributes(attributes, path_group, overwrite=True)

    return peaks, points, average_data


# ==============================================================================
# Step 4: Unsaturated Sub-Pixel Refinement & Outlier Rejection
# ==============================================================================

def retrieve_refined_fringes(wrapper, path_group):
    """
    Retrieves refined and smoothed fringes from HDF5.
    """
    nb_fringe = 0
    refined_fringes = []
    all_x_laser = []
    all_x_laser_smooth = []
    all_y_laser = []
    all_k_laser = []

    children = wrapper.get_children_elements(path_group)
    while f'fringe_{nb_fringe}_smoothed' in children:
        fringe = wrapper[f'{path_group}/fringe_{nb_fringe}']
        fringe_smoothed = wrapper[f'{path_group}/fringe_{nb_fringe}_smoothed']

        refined_fringes.append([fringe[0], fringe[1]])
        all_x_laser.extend(fringe[0])
        all_x_laser_smooth.extend(fringe_smoothed[0])
        all_y_laser.extend(fringe_smoothed[1])
        all_k_laser.extend([nb_fringe] * len(fringe[0]))
        nb_fringe += 1

    return (
        refined_fringes,
        np.array(all_x_laser, dtype=np.float64),
        np.array(all_x_laser_smooth, dtype=np.float64),
        np.array(all_y_laser, dtype=np.float64),
        np.array(all_k_laser, dtype=np.int32)
    )


def refine_laser_fringes(wrapper, path_group, points, data_laser_reference, data_subtracted,
                         w_sub=config.W_SUB,
                         saturation_limit=config.SATURATION_LIMIT,
                         min_net_amp=config.MIN_NET_AMP):
    """
    Performs quadratic-weighted sub-pixel centroiding and polynomial outlier rejection.
    Saves cleaned and smoothed fringes in HDF5 and sets 'fringes_smoothed': True.
    """
    refined_fringes = []
    all_x_laser = []
    all_x_laser_smooth = []
    all_y_laser = []
    all_k_laser = []

    for nb_fringe, fringe in enumerate(points):
        x_clean, y_clean, x_smooth = bck.subpixel_refine_and_reject_outliers(
            fringe[0], fringe[1],
            data_laser_reference,
            data_subtracted,
            w_sub=w_sub,
            saturation_limit=saturation_limit,
            min_net_amp=min_net_amp
        )
        if len(x_clean) == 0:
            continue

        new_fringe = np.array([x_clean, y_clean])
        wrapper.add_other(new_fringe, path_group, f'fringe_{nb_fringe}', overwrite=True)
        new_fringe_smooth = np.array([x_smooth, y_clean])
        wrapper.add_other(new_fringe_smooth, path_group, f'fringe_{nb_fringe}_smoothed', overwrite=True)

        refined_fringes.append([x_clean, y_clean])
        all_x_laser.extend(x_clean)
        all_x_laser_smooth.extend(x_smooth)
        all_y_laser.extend(y_clean)
        all_k_laser.extend([nb_fringe] * len(x_clean))

    attributes = {'fringes_smoothed': True}
    wrapper.add_attributes(attributes, path_group, overwrite=True)

    return (
        refined_fringes,
        np.array(all_x_laser, dtype=np.float64),
        np.array(all_x_laser_smooth, dtype=np.float64),
        np.array(all_y_laser, dtype=np.float64),
        np.array(all_k_laser, dtype=np.int32)
    )


# ==============================================================================
# Step 5: Exact 2D Gnomonic VIPA Geometry Fitting
# ==============================================================================

def retrieve_vipa_geometry(wrapper, path_group, f_px=config.F_PX_DEFAULT, phi_rad=config.VIPA_TILT_ANGLE):
    """
    Retrieves fitted optical center (x0_fit, y0_fit) from HDF5 attributes and computes gnomonic center.
    """
    attributes = wrapper.get_attributes(path=path_group)
    x0_fit = float(attributes['x0_fit_circular'])
    y0_fit = float(attributes['y0_fit_circular'])
    xc_gnom = x0_fit - f_px * np.tan(phi_rad)
    yc_gnom = y0_fit
    return x0_fit, y0_fit, xc_gnom, yc_gnom


def fit_vipa_geometry(wrapper, path_group, refined_fringes, all_x_laser, all_y_laser, all_k_laser,
                      f_px=config.F_PX_DEFAULT, phi_rad=config.VIPA_TILT_ANGLE, n_r=config.N_FUSED_SILICA_780):
    """
    Fits concentric circular arcs for initial guess, optimizes exact gnomonic optical axis (x0, y0),
    and stores results in HDF5 attributes.
    """
    xc_est, yc_est, _, _, _, _ = bck.fit_concentric_arcs(refined_fringes)
    x_guess = xc_est + f_px * np.tan(phi_rad)

    nb_fringes = len(refined_fringes)
    sol_geom = least_squares(
        bck.geom_residuals_px,
        [x_guess, yc_est],
        args=(all_x_laser, all_y_laser, all_k_laser, nb_fringes, f_px, phi_rad, n_r),
        method='lm'
    )
    x0_fit, y0_fit = float(sol_geom.x[0]), float(sol_geom.x[1])

    attributes = {
        'x0_fit_circular': x0_fit,
        'y0_fit_circular': y0_fit,
    }
    wrapper.add_attributes(attributes=attributes, parent_group=path_group, overwrite=True)

    xc_gnom = x0_fit - f_px * np.tan(phi_rad)
    yc_gnom = y0_fit
    return x0_fit, y0_fit, xc_gnom, yc_gnom


# ==============================================================================
# Step 6: 2D Anamorphic Polynomial Order Surface Calibration
# ==============================================================================

def calibrate_anamorphic_order_surface(wrapper, path_group,
                                       all_x_laser, all_x_laser_smooth, all_y_laser, all_k_laser,
                                       x0_fit, y0_fit,
                                       f_px=config.F_PX_DEFAULT,
                                       phi_rad=config.VIPA_TILT_ANGLE,
                                       n_r=config.N_FUSED_SILICA_780,
                                       h_img=2304,
                                       nb_laser_modes=config.NB_LASER_MODES,
                                       fsr_vipa=config.FSR_VIPA,
                                       deg=4,
                                       lower_order_study=False):
    """
    Fits continuous 2D anamorphic polynomial surface phi_order(g, y) to eliminate
    imaging lens field curvature and astigmatism.
    """
    g_all_laser = bck.g_from_xy(all_x_laser, all_y_laser, x0_fit, y0_fit, f=f_px, phi=phi_rad, n_r=n_r)
    g_all_laser_smooth = bck.g_from_xy(all_x_laser_smooth, all_y_laser, x0_fit, y0_fit, f=f_px, phi=phi_rad, n_r=n_r)

    g_bar = np.array([
        np.mean(g_all_laser[all_k_laser == k]) if np.any(all_k_laser == k) else np.nan
        for k in range(nb_laser_modes)
    ])
    q_orders = np.arange(nb_laser_modes, dtype=np.float64)

    g_mean_ref = float(np.mean(g_all_laser))
    g_std_ref = float(np.std(g_all_laser))
    L_y = h_img / 2.0

    gn_laser = (g_all_laser - g_mean_ref) / g_std_ref
    gn_laser_smooth = (g_all_laser_smooth - g_mean_ref) / g_std_ref
    yn_laser = (all_y_laser - y0_fit) / L_y

    c_2d_order = bck.fit_anamorphic_order_surface(gn_laser, yn_laser, all_k_laser, deg=deg)

    q_pred_2d = bck.eval_phi_order_surface(g_all_laser, all_y_laser, c_2d_order, y0_fit, L_y, g_mean_ref, g_std_ref, deg=deg)
    q_pred_2d_smooth = bck.eval_phi_order_surface(g_all_laser_smooth, all_y_laser, c_2d_order, y0_fit, L_y, g_mean_ref, g_std_ref, deg=deg)

    err_2d_mhz = (q_pred_2d - all_k_laser) * fsr_vipa * 1e3
    err_2d_smooth_mhz = (q_pred_2d_smooth - all_k_laser) * fsr_vipa * 1e3

    results = {
        'c_2d_order': c_2d_order,
        'g_mean_ref': g_mean_ref,
        'g_std_ref': g_std_ref,
        'L_y': L_y,
        'gn_laser': gn_laser,
        'yn_laser': yn_laser,
        'g_all_laser': g_all_laser,
        'g_bar': g_bar,
        'q_orders': q_orders,
        'q_pred_2d': q_pred_2d,
        'q_pred_2d_smooth': q_pred_2d_smooth,
        'err_2d_mhz': err_2d_mhz,
        'err_2d_smooth_mhz': err_2d_smooth_mhz,
        'rms_raw_mhz': float(np.sqrt(np.mean(err_2d_mhz**2))),
        'rms_smooth_mhz': float(np.sqrt(np.mean(err_2d_smooth_mhz**2)))
    }

    if lower_order_study:
        ok = ~np.isnan(g_bar)
        poly_g_to_q_lin = np.polyfit(g_bar[ok], q_orders[ok], 1)
        poly_g_to_q = np.polyfit(g_bar[ok], q_orders[ok], 2)
        q_pred_lin = np.polyval(poly_g_to_q_lin, g_all_laser)
        q_pred_quad = np.polyval(poly_g_to_q, g_all_laser)
        results['q_pred_lin'] = q_pred_lin
        results['q_pred_quad'] = q_pred_quad
        results['err_lin_mhz'] = (q_pred_lin - all_k_laser) * fsr_vipa * 1e3
        results['err_quad_mhz'] = (q_pred_quad - all_k_laser) * fsr_vipa * 1e3

    return results
