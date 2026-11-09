"""
Visualization functions for the VIPA calibration pipeline.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_denoise_laser(data_laser_reference, baseline_2d, data_subtracted,
                       out_dir, prefix='', show_plots=False):
    """
    Plots the raw laser reference, morphological baseline, and background-subtracted fringes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(data_laser_reference, vmax=np.percentile(data_laser_reference, 95), cmap='gray')
    axes[0].set_title('(a) Raw Monochromatic Laser Reference')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Pixel y')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Counts')

    im1 = axes[1].imshow(baseline_2d, vmax=np.percentile(baseline_2d, 95), cmap='gray')
    axes[1].set_title('(b) Morphological Background Baseline')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Counts')

    im2 = axes[2].imshow(np.clip(data_subtracted, 0, np.percentile(data_subtracted, 95)), cmap='gray')
    axes[2].set_title('(c) Background-Subtracted Laser Fringes')
    axes[2].set_xlabel('Pixel x')
    axes[2].set_ylabel('Pixel y')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Net Fringe Counts')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}2 - Cleaning laser lines.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_laser_fringes(average_data, peaks, data_subtracted, points,
                       y_center_laser, dy_laser, out_dir, prefix='', show_plots=False):
    """
    Plots detected fringe seeds along the center row-average and full-height traced fringes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(average_data, color='tab:blue', lw=1.2, label=f'Row avg (y={y_center_laser}±{dy_laser})')
    for peak in peaks:
        axes[0].axvline(peak[0], color='tab:red', linestyle='--', alpha=0.8)
    axes[0].set_title(f'Detected Laser Fringe Seeds ({len(peaks)} Orders)')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Net Intensity (counts)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].imshow(data_subtracted, cmap='gray',
                        vmin=np.percentile(average_data, 20),
                        vmax=np.percentile(average_data, 90))
    for fringe in points:
        axes[1].plot(fringe[0], fringe[1], 'r+', markersize=1.5)
    axes[1].set_title(f'Full-Height Traced Laser Fringes (All {len(points)} Orders Recovered)')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}3 - Laser fringe identification.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_fringe_refinement(all_x_laser, all_x_laser_smooth, all_y_laser,
                           out_dir, prefix='', show_plots=False):
    """
    Plots the raw sub-pixel centroid coordinates versus polynomial-smoothed coordinates.
    """
    fig = plt.figure(figsize=(8, 7))
    plt.scatter(all_x_laser, all_y_laser, c='b', marker='+', s=5, alpha=0.6, label='Sub-pixel centroided')
    plt.scatter(all_x_laser_smooth, all_y_laser, c='r', marker='.', s=2, alpha=0.6, label='Polynomial-smoothed')
    plt.title('Laser Fringe Sub-Pixel Refinement & Smoothing')
    plt.xlabel('Pixel x')
    plt.ylabel('Pixel y')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}4 - Laser fringe refinement.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_concentric_fringes_and_geometry(data_subtracted, refined_fringes,
                                         x0_fit, y0_fit, xc_gnom, yc_gnom,
                                         out_dir, prefix='', show_plots=False):
    """
    Plots refined fringes overlaid with the fitted optical axis (x0, y0) and gnomonic ring center (xc, yc).
    """
    fig = plt.figure(figsize=(9, 8))
    plt.imshow(data_subtracted, cmap='gray',
               vmin=np.percentile(data_subtracted, 20),
               vmax=np.percentile(data_subtracted, 90))
    for fringe in refined_fringes:
        sort_idx = np.argsort(fringe[1])
        plt.plot(fringe[0][sort_idx], fringe[1][sort_idx], 'r--', lw=0.8)

    plt.scatter(x0_fit, y0_fit, c='white', marker='x', s=120, lw=2, label=f'Optical axis ({x0_fit:.1f}, {y0_fit:.1f})')
    plt.scatter(xc_gnom, yc_gnom, c='cyan', marker='+', s=120, lw=2, label=f'Gnomonic center ({xc_gnom:.1f}, {yc_gnom:.1f})')

    plt.title('Gnomonic Projection & Optical Axis Identification')
    plt.xlabel('Pixel x')
    plt.ylabel('Pixel y')
    plt.legend(loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}5 - Identification gnomonic projection.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_anamorphic_calibration(q_orders, q_pred_2d, all_k_laser, all_x_laser, all_y_laser,
                                err_2d_mhz, err_2d_smooth_mhz, fsr_vipa, nb_laser_modes,
                                out_dir, prefix='', show_plots=False,
                                lower_order_study=False, q_pred_lin=None, q_pred_quad=None):
    """
    Plots inter-order frequency residuals and 2D spatial residual map after anamorphic calibration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order_means_2d = [
        (np.mean(q_pred_2d[all_k_laser == k]) - k) * fsr_vipa * 1e3
        if np.any(all_k_laser == k) else 0.0
        for k in range(nb_laser_modes)
    ]
    axes[0].plot(q_orders, order_means_2d, 'd-', color='tab:green', lw=2.0,
                 label='2D Anamorphic $\\phi_{\\mathrm{order}}(g, y)$')
    axes[0].axhline(0, color='k', linestyle=':', alpha=0.6)

    if lower_order_study and q_pred_lin is not None and q_pred_quad is not None:
        order_means_lin = [
            (np.mean(q_pred_lin[all_k_laser == k]) - k) * fsr_vipa * 1e3
            if np.any(all_k_laser == k) else 0.0
            for k in range(nb_laser_modes)
        ]
        order_means_quad = [
            (np.mean(q_pred_quad[all_k_laser == k]) - k) * fsr_vipa * 1e3
            if np.any(all_k_laser == k) else 0.0
            for k in range(nb_laser_modes)
        ]
        axes[0].plot(q_orders, order_means_lin, 'o--', color='tab:red', lw=1.6, label='1D Linear $g \\to q$')
        axes[0].plot(q_orders, order_means_quad, 's:', color='tab:orange', lw=1.6, label='1D Quadratic $g \\to q$')

    axes[0].set_title('Inter-Order Frequency Residual vs Order $q$')
    axes[0].set_xlabel('Fringe Order $q$')
    axes[0].set_ylabel('Mean Frequency Residual (MHz)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sc = axes[1].scatter(all_x_laser, all_y_laser, c=err_2d_mhz, cmap='RdBu_r', vmin=-60, vmax=60, s=2)
    rms_raw = np.sqrt(np.mean(err_2d_mhz**2))
    rms_smooth = np.sqrt(np.mean(err_2d_smooth_mhz**2))
    axes[1].set_title(f'2D Residuals (RMS: {rms_raw:.1f} MHz raw, {rms_smooth:.1f} MHz smooth)')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')
    axes[1].set_xlim(0, 2304)
    axes[1].set_ylim(2304, 0)
    axes[1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1], label='Residual (MHz)')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}6 - Dispersion calibration residuals.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_rb_theoretical_spectrum(det_table_MHz, trans_table,
                                 out_dir, prefix='', show_plots=False):
    """
    Plots the theoretical natural Rubidium D2 transmission spectrum.
    """
    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(det_table_MHz * 1e-3, trans_table, color='tab:red', lw=1.5)
    plt.title('Theoretical Transmission Spectrum of Rubidium D2 Line')
    plt.xlabel('Detuning from Center of Gravity (GHz)')
    plt.ylabel('Transmittivity')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}7 - Absorption profile of the Rubidium cell.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_denoise_rb(data_raw, baseline_absorption, data_absorption,
                    out_dir, prefix='', show_plots=False):
    """
    Plots the raw Rubidium absorption image, morphological baseline, and background-subtracted absorption.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(data_raw, vmax=np.percentile(data_raw, 95), cmap='gray')
    axes[0].set_title('(a) Raw Rubidium Absorption')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Pixel y')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Counts')

    im1 = axes[1].imshow(baseline_absorption, vmax=np.percentile(baseline_absorption, 95), cmap='gray')
    axes[1].set_title('(b) Morphological Background Baseline')
    axes[1].set_xlabel('Pixel x')
    axes[1].set_ylabel('Pixel y')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Counts')

    im2 = axes[2].imshow(np.clip(data_absorption, 0, np.percentile(data_absorption, 95)), cmap='gray')
    axes[2].set_title('(c) Background-Subtracted Absorption Image')
    axes[2].set_xlabel('Pixel x')
    axes[2].set_ylabel('Pixel y')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Counts')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}8 - Cleaning Rubidium absorption figure.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_rb_experimental_profile(phi_order_r, avg_spectrum_radius,
                                 out_dir, prefix='', show_plots=False):
    """
    Plots the experimentally extracted radial absorption profile vs interference orders.
    """
    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(phi_order_r, avg_spectrum_radius, color='tab:blue', lw=1.2)
    plt.title('Experimental Rubidium Absorption Profile vs Interference Order')
    plt.xlabel('Interference Order $\\phi_{\\mathrm{order}}$')
    plt.ylabel('Intensity (counts)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}9 - Absorption profile versus orders of interference.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_rb_calibration(f_grid, orders_used, order_spectra, f_valid, sig_valid,
                        baseline_env, f_fit_min, f_fit_max, delta_nu_opt,
                        delta_nu_fine, fsr_cal, trans_fast,
                        out_dir, prefix='', show_plots=False):
    """
    Plots the coherently folded multi-order spectra and the hyperfine theoretical alignment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for k_ord, sp in zip(orders_used, order_spectra):
        axes[0].plot(f_grid, sp, alpha=0.45, lw=1.0, label=f'Order {k_ord}')
    axes[0].plot(f_valid, sig_valid, 'k-', lw=2.0, label='Order average')
    axes[0].axvspan(f_fit_min, f_fit_max, color='tab:green', alpha=0.12, label='Fit window')
    axes[0].set_title('Coherently Folded Rubidium Spectra Across Orders')
    axes[0].set_xlabel(r'$\Delta\nu$ from Rb $D_2$ Center (GHz)')
    axes[0].set_ylabel('Intensity (counts)')
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(f_valid, sig_valid, color='tab:blue', lw=1.6, label='Order-averaged spectrum')
    ax1.plot(f_valid, baseline_env, ':', color='gray', lw=1.2, label='Illumination envelope')
    ax1.axvspan(f_fit_min, f_fit_max, color='tab:green', alpha=0.12, label='Fit window')
    ax1.set_xlabel(r'$\Delta\nu$ from Rb $D_2$ Center (GHz)')
    ax1.set_ylabel('Measured Intensity (counts)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    a_theo = 1.0 - trans_fast((f_valid - delta_nu_fine) * 1e3)
    ax2.plot(f_valid, a_theo, color='tab:red', lw=1.8, alpha=0.85,
             label=rf'Theoretical Rb $D_2$ ($\delta\nu={delta_nu_opt:.2f}$ GHz)')
    ax2.set_ylabel(r'Rubidium Absorption ($1 - T_{\mathrm{Rb}}$)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(-0.05, 1.15)
    axes[1].set_title(rf'Rubidium $D_2$ Hyperfine Alignment ($\mathrm{{FSR}}={fsr_cal:.4f}$ GHz)')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}10 - Calibration VIPA Free Spectral Range on Rb absorption.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_frequency_map(freq_map, out_dir, prefix='', show_plots=False):
    """
    Plots the calibrated 2D frequency map zero-referenced to the elastic line.
    """
    fig = plt.figure(figsize=(9, 8))
    im = plt.imshow(freq_map, cmap='viridis')
    plt.title('Calibrated 2D VIPA Frequency Map (GHz)')
    plt.xlabel('Pixel x')
    plt.ylabel('Pixel y')
    plt.colorbar(im, label='Frequency Relative to Elastic Line (GHz)')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}10B - Full 2D Frequency Map.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_sample_baseline_and_windows(data_sample, freq_avg_data_sample, avg_data_sample,
                                     bls_peaks, dinst_disp_map, path_sample,
                                     out_dir, prefix='', show_plots=False):
    """
    Plots sample interferogram, central row-averaged spectrum with Anti-Stokes/Stokes windows,
    and 2D higher-order dispersion distortion correction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    im = axes[0].imshow(np.clip(data_sample, 0, np.percentile(data_sample, 98)), cmap='inferno', aspect='auto')
    sample_name = os.path.basename(path_sample)
    axes[0].set_title(f'Sample Interferogram ({sample_name})')
    axes[0].set_xlabel('Pixel x')
    axes[0].set_ylabel('Pixel y')
    plt.colorbar(im, ax=axes[0], label='Counts')

    axes[1].plot(freq_avg_data_sample, avg_data_sample, color='tab:blue', lw=1.2, label='Central profile')
    for idx_p, p in enumerate(bls_peaks[:3]):
        axes[1].axvline(p[0], color='tab:green', linestyle='--', alpha=0.7, label='Anti-Stokes' if idx_p == 0 else None)
        axes[1].axvline(p[1], color='tab:red', linestyle='--', alpha=0.7, label='Stokes' if idx_p == 0 else None)
    axes[1].set_xlim(15, 110)
    axes[1].set_title('Calibrated Central Spectrum & Brillouin Windows')
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
    save_path = os.path.join(out_dir, f'{prefix}11 - {sample_name} baseline and windows.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dho_overview(shift_clipped, shift_clipped_averaged,
                      linewidth_clipped, linewidth_clipped_averaged,
                      x_coords, y_coords, x_coords_averaged, y_coords_averaged,
                      path_sample, out_dir, prefix='', show_plots=False):
    """
    2x2 overview plot of extracted Brillouin shift and linewidth across the detector.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sample_name = os.path.basename(path_sample)

    # 1. 2D Scatter Shift
    sc0 = axes[0, 0].scatter(x_coords.ravel(), y_coords.ravel(), c=shift_clipped.ravel(),
                             cmap='plasma', s=3, alpha=0.6)
    axes[0, 0].set_title(f'Brillouin Shift $\\Delta\\nu_B$ on Detector ({sample_name})')
    axes[0, 0].set_xlabel('Pixel x')
    axes[0, 0].set_ylabel('Pixel y')
    axes[0, 0].invert_yaxis()
    plt.colorbar(sc0, ax=axes[0, 0], label='Shift (GHz)')

    # 2. Shift Profiles vs y
    n_orders = shift_clipped_averaged.shape[1]
    for k in range(n_orders):
        valid = np.isfinite(shift_clipped_averaged[:, k])
        if np.any(valid):
            axes[0, 1].plot(y_coords_averaged[valid, k], shift_clipped_averaged[valid, k],
                            lw=1.2, label=f'Order {k+1}')
    axes[0, 1].set_title('Order-Averaged Brillouin Shift vs Detector Height y')
    axes[0, 1].set_xlabel('Pixel y')
    axes[0, 1].set_ylabel('Shift (GHz)')
    axes[0, 1].grid(True, alpha=0.3)
    if n_orders <= 10:
        axes[0, 1].legend(fontsize=8, ncol=2)

    # 3. 2D Scatter Linewidth
    sc1 = axes[1, 0].scatter(x_coords.ravel(), y_coords.ravel(), c=linewidth_clipped.ravel(),
                             cmap='viridis', s=3, alpha=0.6)
    axes[1, 0].set_title(f'Brillouin Linewidth $\\Gamma_B$ on Detector ({sample_name})')
    axes[1, 0].set_xlabel('Pixel x')
    axes[1, 0].set_ylabel('Pixel y')
    axes[1, 0].invert_yaxis()
    plt.colorbar(sc1, ax=axes[1, 0], label='Linewidth (GHz)')

    # 4. Linewidth Profiles vs y
    for k in range(n_orders):
        valid = np.isfinite(linewidth_clipped_averaged[:, k])
        if np.any(valid):
            axes[1, 1].plot(y_coords_averaged[valid, k], linewidth_clipped_averaged[valid, k],
                            lw=1.2, label=f'Order {k+1}')
    axes[1, 1].set_title('Order-Averaged Brillouin Linewidth vs Detector Height y')
    axes[1, 1].set_xlabel('Pixel y')
    axes[1, 1].set_ylabel('Linewidth (GHz)')
    axes[1, 1].grid(True, alpha=0.3)
    if n_orders <= 10:
        axes[1, 1].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}13 - {sample_name} DHO shift and linewidth overview.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_dho_fringe_profiles(shift_clipped, linewidth_clipped,
                             shift_clipped_averaged, linewidth_clipped_averaged,
                             y_coords_averaged, path_sample,
                             out_dir, prefix='', show_plots=False):
    """
    Per-fringe comparison plot: Anti-Stokes vs Stokes vs Order-Averaged.
    """
    sample_name = os.path.basename(path_sample)
    n_orders = shift_clipped_averaged.shape[1]
    n_cols = min(5, n_orders)
    n_rows = int(np.ceil(n_orders / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for k in range(n_orders):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        y_k = y_coords_averaged[:, k]

        s_as = shift_clipped[:, 2 * k]
        s_st = shift_clipped[:, 2 * k + 1]
        s_avg = shift_clipped_averaged[:, k]

        ax.plot(y_k, s_as, 'g.', markersize=2, alpha=0.5, label='Anti-Stokes')
        ax.plot(y_k, s_st, 'r.', markersize=2, alpha=0.5, label='Stokes')
        ax.plot(y_k, s_avg, 'b-', lw=1.2, label='Order Avg')
        ax.set_title(f'Order {k+1}')
        ax.set_xlabel('Pixel y')
        ax.set_ylabel('Shift (GHz)')
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for k in range(n_orders, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}13B - {sample_name} per-fringe breakdown.png')
    plt.savefig(save_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_q_vector_dispersion(
    q_vector,
    shift,
    linewidth,
    theta_deg,
    shift_err=None,
    linewidth_err=None,
    Shift_0=None,
    M_theta=None,
    n_medium=None,
    lambda_vacuum=None,
    V_s=None,
    V_s_err=None,
    out_dir=None,
    prefix='',
    show_plots=False,
):
    """
    Displays Shift(q_vector) and Linewidth(q_vector) with dual axes (q in rad/µm and theta in degrees).
    """
    q_um = q_vector * 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=False)

    # Panel 1: Brillouin Shift(q_vector)
    ax1 = axes[0]
    sc1 = ax1.scatter(q_um, shift, c=theta_deg, cmap="viridis", s=14, alpha=0.55, edgecolors="none", label="Experimental Data")
    cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r"Scattering Angle $\theta$ (°)", fontsize=10)

    if V_s is not None:
        q_model_um = np.linspace(np.min(q_um), np.max(q_um), 200)
        q_model_si = q_model_um * 1e6
        shift_model_ghz = (V_s * q_model_si / (2.0 * np.pi)) * 1e-9

        if V_s_err is not None and np.isfinite(V_s_err):
            shift_model_low = ((V_s - V_s_err) * q_model_si / (2.0 * np.pi)) * 1e-9
            shift_model_high = ((V_s + V_s_err) * q_model_si / (2.0 * np.pi)) * 1e-9
            ax1.fill_between(
                q_model_um,
                shift_model_low,
                shift_model_high,
                color="red",
                alpha=0.18,
                label=r"$\pm 1\sigma$ Dispersion Band",
            )
            line_lbl = rf"Linear Dispersion $\Omega = V_s q$ ($V_s = {V_s:.1f} \pm {V_s_err:.1f}\text{{ m/s}}$)"
        else:
            line_lbl = rf"Linear Dispersion $\Omega = V_s q$ ($V_s = {V_s:.1f}\text{{ m/s}}$)"

        ax1.plot(q_model_um, shift_model_ghz, "r-", lw=2.2, label=line_lbl)

    ax1.set_title(r"Brillouin Frequency Shift $\mathrm{Shift}(q)$", fontsize=12, pad=10)
    ax1.set_xlabel(r"Acoustic Wavevector $q$ ($\mathrm{rad}/\mu\mathrm{m}$)", fontsize=11)
    ax1.set_ylabel("Brillouin Shift (GHz)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)

    if lambda_vacuum is not None and n_medium is not None:
        ax1_top = ax1.twiny()
        k0 = 2.0 * np.pi / lambda_vacuum
        th_ticks = np.linspace(np.min(theta_deg), np.max(theta_deg), 5)
        q_ticks_for_th = 2.0 * n_medium * k0 * np.cos(np.deg2rad(th_ticks) / 2.0) * 1e-6
        ax1_top.set_xlim(ax1.get_xlim())
        ax1_top.set_xticks(q_ticks_for_th)
        ax1_top.set_xticklabels([f"{t:.1f}°" for t in th_ticks])
        ax1_top.set_xlabel(r"Scattering Angle $\theta$", fontsize=10, color="dimgray")

    # Panel 2: Brillouin Linewidth(q_vector) with Weighted Binned Statistics
    ax2 = axes[1]
    sc2 = ax2.scatter(q_um, linewidth, c=theta_deg, cmap="magma", s=14, alpha=0.55, edgecolors="none", label="Experimental Linewidth")
    cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(r"Scattering Angle $\theta$ (°)", fontsize=10)

    nb_bins = 25
    bin_edges = np.linspace(np.min(q_um), np.max(q_um), nb_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_lw_mean = np.zeros(nb_bins)
    bin_lw_err = np.zeros(nb_bins)
    w_lw = 1.0 / linewidth_err if linewidth_err is not None else np.ones_like(linewidth)

    for b in range(nb_bins):
        in_b = (q_um >= bin_edges[b]) & (q_um < bin_edges[b + 1])
        if np.sum(in_b) > 3:
            wb = w_lw[in_b]
            lwb = linewidth[in_b]
            mean_wb = np.sum(wb * lwb) / np.sum(wb)
            bin_lw_mean[b] = mean_wb
            v1 = np.sum(wb)
            v2 = np.sum(wb**2)
            if v1**2 > v2:
                var_wb = (v1 / (v1**2 - v2)) * np.sum(wb * (lwb - mean_wb) ** 2)
                bin_lw_err[b] = np.sqrt(max(0.0, var_wb) / np.sum(in_b))
            else:
                bin_lw_err[b] = np.std(lwb) / np.sqrt(np.sum(in_b))
        else:
            bin_lw_mean[b] = np.nan
            bin_lw_err[b] = np.nan

    valid_bins = np.isfinite(bin_lw_mean)
    ax2.errorbar(
        bin_centers[valid_bins],
        bin_lw_mean[valid_bins],
        yerr=bin_lw_err[valid_bins],
        fmt="o-",
        color="crimson",
        lw=1.8,
        markersize=4.5,
        capsize=3,
        label=r"Error-Weighted Mean $\pm 1\mathrm{SE}$",
    )

    ax2.set_title(r"Brillouin Linewidth $\mathrm{Linewidth}(q)$", fontsize=12, pad=10)
    ax2.set_xlabel(r"Acoustic Wavevector $q$ ($\mathrm{rad}/\mu\mathrm{m}$)", fontsize=11)
    ax2.set_ylabel("Brillouin Linewidth FWHM (GHz)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.35)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

    if lambda_vacuum is not None and n_medium is not None:
        ax2_top = ax2.twiny()
        ax2_top.set_xlim(ax2.get_xlim())
        ax2_top.set_xticks(q_ticks_for_th)
        ax2_top.set_xticklabels([f"{t:.1f}°" for t in th_ticks])
        ax2_top.set_xlabel(r"Scattering Angle $\theta$", fontsize=10, color="dimgray")

    if lambda_vacuum is not None and n_medium is not None:
        plt.suptitle(
            rf"Angle-Resolved BLS Acoustic Dispersion — $\lambda={lambda_vacuum*1e9:.2f}\text{{ nm}}, n={n_medium:.2f}$",
            fontsize=13, y=0.99
        )
    plt.tight_layout()

    if out_dir is not None:
        save_path = os.path.join(out_dir, f"{prefix}14 - Shift and linewidth vs q_vector.png")
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_complete_diagnostic_suite(
    x,
    y,
    shift,
    linewidth,
    shift_err,
    linewidth_err,
    x_sph,
    y_sph,
    r_sph,
    q_vector,
    theta_deg,
    opt_params,
    S0_fit,
    M_theta_fit,
    S0_err=None,
    M_theta_err=None,
    n_medium=None,
    lambda_vacuum=None,
    V_s=None,
    V_s_err=None,
    out_dir=None,
    prefix='',
    show_plots=False,
):
    """
    Generates a 6-panel comprehensive diagnostic figure illustrating:
        1. 2D detector coordinates (x, y) with fitted concentric ellipses & axes.
        2. Spherically symmetric coordinates (x_sph, y_sph) confirming circularity.
        3. Shift(r_sph) vs theoretical M_theta kinematic fit with explicit formula.
        4. Linewidth(r_sph) vs r_sph.
        5. Shift(q_vector) dispersion.
        6. Linewidth(q_vector) broadening.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    xc = opt_params["xc"]
    yc = opt_params["yc"]
    phi = opt_params["phi_rad"]
    gamma = opt_params["gamma"]

    # Panel 1: Detector Space with Fitted Concentric Ellipses
    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=shift, cmap="coolwarm", s=8, alpha=0.55)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Shift (GHz)")

    th_dense = np.linspace(0, 2 * np.pi, 200)
    for r_level in [400, 800, 1200, 1600]:
        x_ell = r_level * np.cos(th_dense)
        y_ell = (gamma * r_level) * np.sin(th_dense)
        x_rot = xc + x_ell * np.cos(phi) - y_ell * np.sin(phi)
        y_rot = yc + x_ell * np.sin(phi) + y_ell * np.cos(phi)
        ax.plot(x_rot, y_rot, "k--", lw=1.2, alpha=0.7)

    L_axis = 1000.0
    ax.plot([xc - L_axis * np.cos(phi), xc + L_axis * np.cos(phi)],
            [yc - L_axis * np.sin(phi), yc + L_axis * np.sin(phi)],
            "b-", lw=1.8, label=f"Principal Axis 1 (θ={np.degrees(phi):.1f}°)")
    ax.plot([xc + L_axis * np.sin(phi), xc - L_axis * np.sin(phi)],
            [yc - L_axis * np.cos(phi), yc + L_axis * np.cos(phi)],
            "b:", lw=1.5, label="Principal Axis 2")
    ax.plot(xc, yc, "r+", markersize=14, markeredgewidth=2.2, label=f"Center ({xc:.0f}, {yc:.0f})")
    ax.set_title("1. Detector Coordinates & Ellipses", fontsize=11)
    ax.set_xlabel("Detector x (px)")
    ax.set_ylabel("Detector y (px)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel 2: Spherical Symmetry
    ax = axes[0, 1]
    sc = ax.scatter(x_sph, y_sph, c=shift, cmap="coolwarm", s=8, alpha=0.55)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Shift (GHz)")
    for r_level in [400, 800, 1200, 1600]:
        circ = plt.Circle((0, 0), r_level, color="k", fill=False, linestyle="--", lw=1.2, alpha=0.7)
        ax.add_patch(circ)
    ax.plot(0, 0, "r+", markersize=14, markeredgewidth=2.2)
    ax.set_title(f"2. Transformed Coordinates (γ={gamma:.3f})", fontsize=11)
    ax.set_xlabel(r"$x_{\mathrm{sph}}$ (px)")
    ax.set_ylabel(r"$y_{\mathrm{sph}}$ (px)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel 3: Shift(r_sph) & M_theta Kinematic Fit with Explicit Formula
    ax = axes[0, 2]
    ax.scatter(r_sph, shift, c="royalblue", s=8, alpha=0.45, label="Experimental Shift")
    r_arr = np.linspace(0, np.max(r_sph), 200)
    s_model = S0_fit * np.cos(np.arctan(r_arr / M_theta_fit) / 2.0)
    ax.plot(
        r_arr,
        s_model,
        "r-",
        lw=2.2,
        label=rf"Kinematic Fit ($S_0={S0_fit:.3f}\text{{ GHz}}, M_\theta={M_theta_fit:.0f}\text{{ px}}$)"
    )

    # Text box displaying the explicit mathematical expression and fitted parameters
    if S0_err is not None and np.isfinite(S0_err):
        s0_text = rf"$S_0 = {S0_fit:.4f} \pm {S0_err:.4f}\text{{ GHz}}$"
    else:
        s0_text = rf"$S_0 = {S0_fit:.4f}\text{{ GHz}}$"

    if M_theta_err is not None and np.isfinite(M_theta_err):
        m_text = rf"$M_\theta = {M_theta_fit:.1f} \pm {M_theta_err:.1f}\text{{ px}}$"
    else:
        m_text = rf"$M_\theta = {M_theta_fit:.1f}\text{{ px}}$"

    formula_box = (
        r"$\mathbf{Kinematic\ Model:}$" + "\n"
        r"$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cdot \cos\left(\frac{1}{2}\arctan\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$" + "\n"
        f"{s0_text}\n"
        f"{m_text}"
    )
    ax.text(
        0.96,
        0.95,
        formula_box,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
    )

    ax.set_title(r"3. $\mathrm{Shift}(r_{\mathrm{sph}})$ & Kinematic Fit", fontsize=11)
    ax.set_xlabel(r"Spherical Radius $r_{\mathrm{sph}}$ (px)")
    ax.set_ylabel("Shift (GHz)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel 4: Linewidth(r_sph)
    ax = axes[1, 0]
    ax.scatter(r_sph, linewidth, c="seagreen", s=8, alpha=0.45)
    ax.set_title(r"4. $\mathrm{Linewidth}(r_{\mathrm{sph}})$", fontsize=11)
    ax.set_xlabel(r"Spherical Radius $r_{\mathrm{sph}}$ (px)")
    ax.set_ylabel("Linewidth (GHz)")
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel 5: Shift(q) Acoustic Dispersion with Propagated Uncertainty
    ax = axes[1, 1]
    q_um = q_vector * 1e-6
    sc5 = ax.scatter(q_um, shift, c=theta_deg, cmap="viridis", s=10, alpha=0.55, label="Experimental Data")
    plt.colorbar(sc5, ax=ax, fraction=0.046, pad=0.04, label=r"Angle $\theta$ (°)")

    if V_s is not None:
        q_lin_um = np.linspace(np.min(q_um), np.max(q_um), 200)
        q_lin_si = q_lin_um * 1e6
        shift_model = (V_s * q_lin_si / (2.0 * np.pi)) * 1e-9

        if V_s_err is not None and np.isfinite(V_s_err):
            shift_model_low = ((V_s - V_s_err) * q_lin_si / (2.0 * np.pi)) * 1e-9
            shift_model_high = ((V_s + V_s_err) * q_lin_si / (2.0 * np.pi)) * 1e-9
            ax.fill_between(
                q_lin_um,
                shift_model_low,
                shift_model_high,
                color="red",
                alpha=0.18,
                label=r"$\pm 1\sigma$ Dispersion Band",
            )
            line_label = rf"Linear Fit: $V_s = {V_s:.1f} \pm {V_s_err:.1f}\text{{ m/s}}$"
            vs_val_text = rf"$V_s = {V_s:.1f} \pm {V_s_err:.1f}\text{{ m/s}}$"
        else:
            line_label = rf"Linear Fit: $V_s = {V_s:.1f}\text{{ m/s}}$"
            vs_val_text = rf"$V_s = {V_s:.1f}\text{{ m/s}}$"

        ax.plot(q_lin_um, shift_model, "r-", lw=2.2, label=line_label)

        # Text box with speed of sound expression and propagated error
        dispersion_box = (
            r"$\mathbf{Speed\ of\ Sound:}$" + "\n"
            r"$\Omega(q) = V_s \cdot q \iff \mathrm{Shift} = \frac{V_s}{2\pi} q$" + "\n"
            f"{vs_val_text}\n"
            r"$V_s = \frac{\lambda_0 S_0}{2n}$" + (r" $\pm \sigma_{V_s}(S_0, M_\theta)$" if (V_s_err is not None and np.isfinite(V_s_err)) else "")
        )
        ax.text(
            0.05,
            0.95,
            dispersion_box,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.92,
            ),
        )

        ax.legend(loc="lower right", fontsize=8)

    ax.set_title(r"5. $\mathrm{Shift}(q)$ Acoustic Dispersion", fontsize=11)
    ax.set_xlabel(r"Acoustic Wavevector $q$ ($\mathrm{rad}/\mu\mathrm{m}$)")
    ax.set_ylabel("Shift (GHz)")
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel 6: Linewidth(q)
    ax = axes[1, 2]
    ax.scatter(q_um, linewidth, c=theta_deg, cmap="magma", s=10, alpha=0.55)
    ax.set_title(r"6. $\mathrm{Linewidth}(q)$ Broadening", fontsize=11)
    ax.set_xlabel(r"$q$ ($\mathrm{rad}/\mu\mathrm{m}$)")
    ax.set_ylabel("Linewidth (GHz)")
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.suptitle("Complete 2D Elliptical Geometry & Acoustic Wavevector Diagnostics", fontsize=14, y=0.995)
    plt.tight_layout()

    if out_dir is not None:
        save_path = os.path.join(out_dir, f"{prefix}14B - Complete elliptical qvector diagnostics.png")
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
