"""
Sample spectrum fitting module for VIPA Brillouin spectroscopy:
In-situ 2D frequency map calibration, spatial binning, vectorized DHO peak fitting,
and HDF5 treated data persistence.
"""

import numpy as np
import scipy.ndimage as ndimage

import config
import backend as bck


# ==============================================================================
# In-Situ 2D Frequency Map Calibration
# ==============================================================================

def calibrate_sample_frequency_map(
    freq_map_base,
    data_clean_in,
    xx_in,
    yy_in,
    fsr_val,
    y0_center,
    bin_val_h=config.BIN_VAL_H,
    bin_val_w=config.BIN_VAL_W,
    n_orders=None,
    sigma_y_bins=config.SIGMA_Y_BINS,
    expected_shift=config.EXPECTED_BRILLOUIN_SHIFT,
    shift_tol=config.BRILLOUIN_SHIFT_TOLERANCE,
):
    """
    Two-stage in-situ 2D frequency map calibration for a Brillouin sample interferogram:
      1. 2D Elastic Zero-Offset Map delta_nu_zero(y, k_ord): locks midpoint symmetry
         0.5 * (nu_AS + nu_S) = k_ord * FSR along every fringe.
      2. 2D Higher-Order (>= 3rd-order) Dispersion Jacobian Scale alpha_disp(y, k_ord):
         removes Seidel spherical aberrations and high-order coma while preserving
         the physical 6-parameter 2D elliptical paraboloid Z_{S,quad}(x, y).

    Returns
    -------
    freq_corr_2d : numpy.ndarray
        Calibrated 2D frequency map for the sample.
    df_coarse : float
        Coarse global frequency offset in GHz.
    dz_map_2d : numpy.ndarray
        2D zero-offset correction map (GHz).
    dinst_map_2d : numpy.ndarray
        2D higher-order dispersion residual map (GHz).
    bls_peaks : list of [float, float]
        Expected [Anti-Stokes, Stokes] peak frequency pairs per order.
    avg_data_sample : numpy.ndarray
        Row-averaged sample intensity around the optical center.
    freq_avg_data_sample : numpy.ndarray
        Corresponding row-averaged calibrated frequency axis.
    """
    h, w = data_clean_in.shape
    new_h, new_w = h // bin_val_h, w // bin_val_w

    xx_bin = bck.bin_2d_array(xx_in, bin_val_h, bin_val_w)
    yy_bin = bck.bin_2d_array(yy_in, bin_val_h, bin_val_w)

    dy_strip = 100
    yc_int = int(round(y0_center))
    yc_min = max(0, yc_int - dy_strip)
    yc_max = min(h, yc_int + dy_strip)
    avg_d = np.mean(data_clean_in[yc_min:yc_max, :], axis=0)
    avg_f = np.mean(freq_map_base[yc_min:yc_max, :], axis=0)

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
    f_bin_0 = bck.bin_2d_array(f_init, bin_val_h, bin_val_w)
    d_bin_0 = bck.bin_2d_array(data_clean_in, bin_val_h, bin_val_w)

    if n_orders is None:
        n_orders = max(k_cand_max, 1)

    s_rel_0, _, _, s_var_0, _, xc_0, _ = bck.fit_dho_batched(
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

    s_as_var_0 = s_var_0[:, 0::2]
    s_st_var_0 = s_var_0[:, 1::2]
    s_var_avg_0 = np.where(
        np.isfinite(s_as_var_0) & np.isfinite(s_st_var_0),
        0.25 * (s_as_var_0 + s_st_var_0),
        np.where(np.isfinite(s_as_var_0), s_as_var_0, s_st_var_0),
    )

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
            target_f_k = (k + 1) * fsr_val
            c_est = np.nanargmin(np.abs(f_bin_0 - target_f_k), axis=1)
            x_traj[:, k] = xx_bin[np.arange(new_h), c_est]

    # Stage 1: Fit 2D polynomial base + smooth per-order spline to delta_mid_0
    vm_mid = np.isfinite(delta_mid_0) & np.isfinite(x_avg_0) & (np.abs(delta_mid_0) < 0.25)
    if np.sum(vm_mid) >= 20:
        xm_v = (x_avg_0[vm_mid] - 1152.0) / 1000.0
        ym_v = (np.broadcast_to(y_rows[:, None], (new_h, n_orders))[vm_mid] - y0_center) / 1000.0
        bm = bck.build_2d_poly_basis(xm_v, ym_v, deg=3)
        cm_zero, _, _, _ = np.linalg.lstsq(bm, delta_mid_0[vm_mid], rcond=None)
    else:
        cm_zero = np.zeros(10)

    delta_zero_grid = np.zeros((new_h, n_orders), dtype=np.float64)
    for k in range(n_orders):
        xk_n = (x_traj[:, k] - 1152.0) / 1000.0
        yk_n = (y_rows - y0_center) / 1000.0
        poly_k = bck.build_2d_poly_basis(xk_n, yk_n, deg=3) @ cm_zero
        raw_k = delta_mid_0[:, k].copy()
        vk = np.isfinite(raw_k) & (np.abs(raw_k - poly_k) < 0.12)
        if np.sum(vk) >= 5:
            filled_k = np.where(vk, raw_k, poly_k)
            delta_zero_grid[:, k] = ndimage.gaussian_filter1d(filled_k, sigma=sigma_y_bins, mode='nearest')
        else:
            delta_zero_grid[:, k] = poly_k

    s_est_0 = np.where(
        np.isfinite(s_avg_0),
        s_avg_0,
        np.where(
            np.isfinite(s_as_0),
            s_as_0 + delta_zero_grid,
            s_st_0 - delta_zero_grid,
        ),
    )

    # Stage 2: Fit physical 6-parameter 2D elliptical paraboloid Z_{S,quad}(x, y)
    y_2d = np.broadcast_to(y_rows[:, None], (new_h, n_orders))
    k_2d = np.broadcast_to(np.arange(n_orders)[None, :], (new_h, n_orders))
    vm_quad = (
        np.isfinite(s_avg_0)
        & np.isfinite(x_avg_0)
        & np.isfinite(s_var_avg_0)
        & (s_var_avg_0 > 0)
        & (y_2d >= 250.0)
        & (y_2d <= 1850.0)
        & (k_2d <= min(7, n_orders - 1))
        & (s_avg_0 > expected_shift - 0.35)
        & (s_avg_0 < expected_shift + 0.35)
    )
    xq = x_avg_0[vm_quad]
    yq = y_2d[vm_quad]
    sq = s_avg_0[vm_quad]
    vq = s_var_avg_0[vm_quad]
    aq = np.column_stack([np.ones_like(xq), xq, yq, xq**2, xq * yq, yq**2])
    wq = 1.0 / np.maximum(vq, 1e-12)
    wq_norm = wq / np.mean(wq)
    sqrt_wq = np.sqrt(wq_norm)
    cq_quad, _, _, _ = np.linalg.lstsq(aq * sqrt_wq[:, None], sq * sqrt_wq, rcond=None)
    for _ in range(2):
        rq = sq - (aq @ cq_quad)
        sig_q = 1.4826 * np.median(np.abs(rq - np.median(rq)))
        inl = np.abs(rq) < 2.2 * sig_q
        cq_quad, _, _, _ = np.linalg.lstsq(
            aq[inl] * sqrt_wq[inl, None],
            sq[inl] * sqrt_wq[inl],
            rcond=None,
        )

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

    # Map corrections onto full detector grid
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

    # Build reference Brillouin peak pairs
    bls_peaks = [[k * fsr_val - expected_shift, k * fsr_val + expected_shift] for k in range(1, n_orders + 1)]

    # Central profiles
    dy_avg = 100
    yc_sample = int(round(y0_center))
    avg_data_sample = np.mean(data_clean_in[max(0, yc_sample - dy_avg) : min(h, yc_sample + dy_avg), :], axis=0)
    freq_avg_data_sample = np.mean(freq_corr_2d[max(0, yc_sample - dy_avg) : min(h, yc_sample + dy_avg), :], axis=0)

    return freq_corr_2d, df_coarse, dz_map_2d, dinst_map_2d, bls_peaks, avg_data_sample, freq_avg_data_sample


# ==============================================================================
# Spectral DHO Fitting & Averaging
# ==============================================================================

def fit_sample_dho(
    data_sample,
    freq_corrected,
    xx,
    yy,
    fsr_val,
    bin_val_h=config.BIN_VAL_H,
    bin_val_w=config.BIN_VAL_W,
    n_orders=config.N_ORDERS_SAMPLE,
    ds_init=config.EXPECTED_BRILLOUIN_SHIFT,
    shift_tol=config.BRILLOUIN_SHIFT_TOLERANCE,
    gamma_init=config.EXPECTED_LINEWIDTH,
    gamma_bounds=config.LINEWIDTH_BOUNDS,
    verbose=True,
):
    """
    Spatially bins the sample interferogram and frequency map, executes vectorized
    DHO fitting across all orders, and calculates order-averaged shift and linewidth maps.
    """
    freq_bin = bck.bin_2d_array(freq_corrected, bin_val_h, bin_val_w)
    data_bin = bck.bin_2d_array(data_sample, bin_val_h, bin_val_w)
    xx_bin = bck.bin_2d_array(xx, bin_val_h, bin_val_w)
    yy_bin = bck.bin_2d_array(yy, bin_val_h, bin_val_w)

    shift_clipped, shift_unwrapped, linewidth_clipped, shift_err, linewidth_err, x_coords, y_coords = bck.fit_dho_batched(
        freq_bin, data_bin, xx_bin, yy_bin, fsr_val,
        n_orders=n_orders, ds_init=ds_init, shift_tol=shift_tol,
        gamma_init=gamma_init, gamma_bounds=gamma_bounds,
        hw_px=config.DHO_HW_PX, min_amp=config.DHO_MIN_AMP, verbose=verbose
    )

    shift_clipped_averaged = 0.5 * (shift_clipped[:, 0::2] + shift_clipped[:, 1::2])
    linewidth_clipped_averaged = 0.5 * (linewidth_clipped[:, 0::2] + linewidth_clipped[:, 1::2])
    shift_err_averaged = bck.compute_order_averaged_variance(shift_err)
    linewidth_err_averaged = bck.compute_order_averaged_variance(linewidth_err)
    x_coords_averaged = 0.5 * (x_coords[:, 0::2] + x_coords[:, 1::2])
    y_coords_averaged = 0.5 * (y_coords[:, 0::2] + y_coords[:, 1::2])

    stokes_antistokes_rms_mhz = float(np.nanstd(shift_clipped[:, 1::2] - shift_clipped[:, 0::2]) * 1e3)

    return {
        'shift_clipped': shift_clipped,
        'shift_unwrapped': shift_unwrapped,
        'linewidth_clipped': linewidth_clipped,
        'shift_error': shift_err,
        'linewidth_error': linewidth_err,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'shift_clipped_averaged': shift_clipped_averaged,
        'linewidth_clipped_averaged': linewidth_clipped_averaged,
        'shift_err_averaged': shift_err_averaged,
        'linewidth_err_averaged': linewidth_err_averaged,
        'x_coords_averaged': x_coords_averaged,
        'y_coords_averaged': y_coords_averaged,
        'stokes_antistokes_rms_mhz': stokes_antistokes_rms_mhz,
        'n_fringes': shift_clipped_averaged.shape[1],
        'bin_val_h': bin_val_h,
        'bin_val_w': bin_val_w,
    }


# ==============================================================================
# Persistence & Retrieval (HDF5)
# ==============================================================================

def retrieve_sample_dho_fit(wrapper, path_sample, bin_val_h=config.BIN_VAL_H, bin_val_w=config.BIN_VAL_W):
    """
    Retrieves previously fitted DHO datasets from HDF5 if present.
    """
    group_treated = f'{path_sample}/Treated {bin_val_h}x{bin_val_w} bins'
    children = wrapper.get_children_elements(path_sample)
    if f'Treated {bin_val_h}x{bin_val_w} bins' not in children:
        return None

    try:
        shift_clipped = wrapper[f'{group_treated}/shift'].astype(np.float64)
        shift_err = wrapper[f'{group_treated}/shift_err'].astype(np.float64)
        linewidth_clipped = wrapper[f'{group_treated}/linewidth'].astype(np.float64)
        linewidth_err = wrapper[f'{group_treated}/linewidth_err'].astype(np.float64)
        x_coords = wrapper[f'{group_treated}/x_coords'].astype(np.float64)
        y_coords = wrapper[f'{group_treated}/y_coords'].astype(np.float64)

        shift_clipped_averaged = 0.5 * (shift_clipped[:, 0::2] + shift_clipped[:, 1::2])
        linewidth_clipped_averaged = 0.5 * (linewidth_clipped[:, 0::2] + linewidth_clipped[:, 1::2])
        shift_err_averaged = bck.compute_order_averaged_variance(shift_err)
        linewidth_err_averaged = bck.compute_order_averaged_variance(linewidth_err)
        x_coords_averaged = 0.5 * (x_coords[:, 0::2] + x_coords[:, 1::2])
        y_coords_averaged = 0.5 * (y_coords[:, 0::2] + y_coords[:, 1::2])
        stokes_antistokes_rms_mhz = float(np.nanstd(shift_clipped[:, 1::2] - shift_clipped[:, 0::2]) * 1e3)

        return {
            'shift_clipped': shift_clipped,
            'linewidth_clipped': linewidth_clipped,
            'shift_error': shift_err,
            'linewidth_error': linewidth_err,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'shift_clipped_averaged': shift_clipped_averaged,
            'linewidth_clipped_averaged': linewidth_clipped_averaged,
            'shift_err_averaged': shift_err_averaged,
            'linewidth_err_averaged': linewidth_err_averaged,
            'x_coords_averaged': x_coords_averaged,
            'y_coords_averaged': y_coords_averaged,
            'stokes_antistokes_rms_mhz': stokes_antistokes_rms_mhz,
            'n_fringes': shift_clipped_averaged.shape[1],
            'bin_val_h': bin_val_h,
            'bin_val_w': bin_val_w,
        }
    except Exception:
        return None


def save_treated_data(wrapper, path_sample, bin_val_h, bin_val_w,
                      shift_rel, shift_err, lw_rel, lw_err, x_coords, y_coords):
    """
    Saves the treated Brillouin shift, linewidth, uncertainties, and sub-pixel coordinates to HDF5.
    """
    wrapper.add_treated_data(
        parent_group=path_sample,
        name_group=f'Treated {bin_val_h}x{bin_val_w} bins',
        shift=shift_rel,
        shift_err=shift_err,
        linewidth=lw_rel,
        linewidth_err=lw_err,
        overwrite=True,
    )
    wrapper.add_other(
        data=x_coords,
        parent_group=f'{path_sample}/Treated {bin_val_h}x{bin_val_w} bins',
        name='x_coords',
        overwrite=True,
    )
    wrapper.add_other(
        data=y_coords,
        parent_group=f'{path_sample}/Treated {bin_val_h}x{bin_val_w} bins',
        name='y_coords',
        overwrite=True,
    )
