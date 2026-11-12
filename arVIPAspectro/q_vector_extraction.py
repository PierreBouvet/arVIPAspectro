"""
Angle-Resolved Brillouin Light Scattering (ar-BLS):
2D Concentric Ellipse Fitting with Error-Weighting, Variance Optimization,
and Acoustic Wavevector (q-vector) Extraction.

This script provides a standalone entry point and CLI wrapper around the `q_vector` module.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    from . import config
    from . import backend as bck
    from . import q_vector
    from . import plotting
except ImportError:
    import config
    import backend as bck
    import q_vector
    import plotting


def run_q_vector_extraction(
    path_h5: Union[str, Path] = None,
    group: str = None,
    refractive_index: float = 1.33,
    laser_wavelength: float = config.lambda_0,
    use_errors: bool = True,
    error_percentile_floor: float = 1.0,
    average_pairs: bool = True,
    shift_bounds: Tuple[Optional[float], Optional[float]] = (4.8, 5.3),
    linewidth_bounds: Tuple[Optional[float], Optional[float]] = (0.5, 1.2),
    fixed_ellipse_angle_deg: Optional[float] = 90.0,
    out_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = True,
    save_plots: bool = True,
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Executes the complete error-weighted q-vector extraction pipeline.
    """
    if path_h5 is None:
        path_h5 = os.path.join(config.DATA_DIR, config.TARGET_FILE) if hasattr(config, "DATA_DIR") else "Measures.h5"
    if group is None:
        group = "Brillouin/water_5s/Treated 4x2 bins"

    wrapper = bck.create_wrapper(os.path.dirname(str(path_h5)), name=os.path.basename(str(path_h5)))

    # Path to sample parent group
    path_sample = "/".join(group.split("/")[:2])
    parts = group.split("/")
    bin_h = 4
    bin_w = 2
    if len(parts) >= 3 and "Treated" in parts[2]:
        try:
            dims = parts[2].replace("Treated ", "").replace(" bins", "").split("x")
            bin_h = int(dims[0])
            bin_w = int(dims[1])
        except Exception:
            pass

    q_results = q_vector.extract_q_vector(
        data_source=group,
        path_sample=path_sample,
        wrapper=wrapper,
        bin_val_h=bin_h,
        bin_val_w=bin_w,
        refractive_index=refractive_index,
        laser_wavelength=laser_wavelength,
        use_errors=use_errors,
        error_percentile_floor=error_percentile_floor,
        average_pairs=average_pairs,
        shift_bounds=shift_bounds,
        linewidth_bounds=linewidth_bounds,
        initial_M_theta=3000.0,
        fixed_ellipse_angle_deg=fixed_ellipse_angle_deg,
        verbose=True,
    )

    if save_plots and out_dir is None:
        out_dir = os.path.join(os.path.dirname(str(path_h5)), "figures")

    # Plot 1: Dispersion (Shift(q) and Linewidth(q))
    plotting.plot_q_vector_dispersion(
        q_vector=q_results["q_vector"],
        shift=q_results["shift"],
        linewidth=q_results["linewidth"],
        theta_deg=q_results["theta_deg"],
        shift_err=q_results["shift_err"],
        linewidth_err=q_results["linewidth_err"],
        Shift_0=q_results["Shift_0"],
        M_theta=q_results["M_theta"],
        n_medium=refractive_index,
        lambda_vacuum=laser_wavelength,
        V_s=q_results["V_s"],
        V_s_err=q_results.get("V_s_err"),
        out_dir=out_dir if save_plots else None,
        show_plots=show_plots,
    )

    # Plot 2: Complete diagnostic suite
    plotting.plot_complete_diagnostic_suite(
        x=q_results["x_coords"],
        y=q_results["y_coords"],
        shift=q_results["shift"],
        linewidth=q_results["linewidth"],
        shift_err=q_results["shift_err"],
        linewidth_err=q_results["linewidth_err"],
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
        n_medium=refractive_index,
        lambda_vacuum=laser_wavelength,
        V_s=q_results["V_s"],
        V_s_err=q_results.get("V_s_err"),
        out_dir=out_dir if save_plots else None,
        show_plots=show_plots,
    )

    # Save to HDF5
    q_vector.save_q_vector_results(
        wrapper=wrapper,
        path_sample=path_sample,
        bin_val_h=bin_h,
        bin_val_w=bin_w,
        q_results=q_results,
    )

    return q_results


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    group_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run_q_vector_extraction(path_h5=path_arg, group=group_arg)
