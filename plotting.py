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
