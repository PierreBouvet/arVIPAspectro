"""
Acoustic wavevector (q-vector) and 2D concentric coaxial ellipse dispersion pipeline.
Calculates angle-resolved scattering angle theta, acoustic wavevector q, and sound velocity Vs.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

import config
import backend as bck


# ==============================================================================
# Q-Vector Extraction Pipeline
# ==============================================================================

def extract_q_vector(
    data_source: Union[Dict, str, Path],
    path_sample: Optional[str] = None,
    wrapper=None,
    bin_val_h: int = config.BIN_VAL_H,
    bin_val_w: int = config.BIN_VAL_W,
    refractive_index: float = config.N_SAMPLE,
    laser_wavelength: float = config.lambda_0,
    use_errors: bool = config.USE_ERROR_WEIGHTING,
    error_percentile_floor: float = config.ERROR_PERCENTILE_FLOOR,
    average_pairs: bool = config.AVERAGE_STOKES_ANTISTOKES,
    shift_bounds: Tuple[Optional[float], Optional[float]] = config.SHIFT_RANGE_SAMPLE,
    linewidth_bounds: Tuple[Optional[float], Optional[float]] = config.LINEWIDTH_RANGE_SAMPLE,
    initial_M_theta: float = config.INITIAL_M_THETA,
    verbose: bool = True,
) -> Dict[str, Union[np.ndarray, float, Dict]]:
    """
    Executes the error-weighted q-vector extraction and 2D concentric ellipse pipeline.

    Parameters
    ----------
    data_source : dict or path-like
        Either a dictionary containing fitted DHO arrays (from Step 12),
        or a file path / string to an HDF5 file.
    path_sample : str, optional
        Internal HDF5 group path to the sample if data_source is an HDF5 wrapper.
    wrapper : Wrapper, optional
        Open HDF5 wrapper instance if extracting from an existing open file.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("  ANGLE-RESOLVED BLS: ACOUSTIC WAVEVECTOR (Q-VECTOR) EXTRACTION")
        print("=" * 80)

    # 1. Retrieve data arrays from in-memory dictionary or HDF5
    if isinstance(data_source, dict):
        x_raw = data_source["x_coords"]
        y_raw = data_source["y_coords"]
        shift_raw = data_source["shift_clipped"]
        linewidth_raw = data_source["linewidth_clipped"]
        shift_err_raw = data_source.get("shift_error")
        linewidth_err_raw = data_source.get("linewidth_error")
    elif wrapper is not None and path_sample is not None:
        group_treated = f"{path_sample}/Treated {bin_val_h}x{bin_val_w} bins"
        x_raw = wrapper[f"{group_treated}/x_coords"].astype(np.float64)
        y_raw = wrapper[f"{group_treated}/y_coords"].astype(np.float64)
        shift_raw = wrapper[f"{group_treated}/shift"].astype(np.float64)
        linewidth_raw = wrapper[f"{group_treated}/linewidth"].astype(np.float64)
        shift_err_raw = wrapper[f"{group_treated}/shift_err"].astype(np.float64)
        linewidth_err_raw = wrapper[f"{group_treated}/linewidth_err"].astype(np.float64)
    else:
        raise ValueError("Must provide either an in-memory dictionary or wrapper + path_sample.")

    # 2. Filter, clean, and regularize data points
    x_flat, y_flat, shift_flat, linewidth_flat, shift_err_flat, linewidth_err_flat, has_errors = (
        bck.filter_and_prepare_q_vector_data(
            x_raw=x_raw,
            y_raw=y_raw,
            shift_raw=shift_raw,
            linewidth_raw=linewidth_raw,
            shift_err_raw=shift_err_raw if use_errors else None,
            linewidth_err_raw=linewidth_err_raw if use_errors else None,
            average_stokes_antistokes=average_pairs,
            shift_range=shift_bounds,
            linewidth_range=linewidth_bounds,
            error_percentile_floor=error_percentile_floor,
        )
    )

    if verbose:
        std_s_sample = float(np.std(shift_flat))
        std_lw_sample = float(np.std(linewidth_flat))
        print(f"  • Retained {len(x_flat):,} valid data points after filtering.")
        print(f"  • Error-Weighting Active: {has_errors}")
        print(f"  • Balancing Scales (std): Shift std = {std_s_sample:.4f} GHz, Linewidth std = {std_lw_sample:.4f} GHz")

    # 3. Fit 2D concentric coaxial ellipses with error weighting
    coaxial_fit = bck.fit_concentric_ellipses_coaxial(
        x=x_flat,
        y=y_flat,
        shift=shift_flat,
        linewidth=linewidth_flat,
        shift_err=shift_err_flat if has_errors else None,
        linewidth_err=linewidth_err_flat if has_errors else None,
    )

    # 4. Refine ellipse parameters by minimizing radial residual variance
    opt_params = bck.optimize_elliptical_symmetry_variance(
        x=x_flat,
        y=y_flat,
        shift=shift_flat,
        linewidth=linewidth_flat,
        shift_err=shift_err_flat if has_errors else None,
        linewidth_err=linewidth_err_flat if has_errors else None,
        initial_params=coaxial_fit,
    )

    # 5. Transform coordinates to spherical system
    x_sph, y_sph, r_sph = bck.transform_to_spherical_coordinates(
        x=x_flat,
        y=y_flat,
        xc=opt_params["xc"],
        yc=opt_params["yc"],
        phi_rad=opt_params["phi_rad"],
        gamma=opt_params["gamma"],
    )

    # 6. Fit angular magnification factor M_theta
    S0_fit, M_theta_fit, (S0_err, M_theta_err) = bck.fit_magnification_factor_M_theta(
        r_sph=r_sph,
        shift=shift_flat,
        shift_err=shift_err_flat if has_errors else None,
        initial_M_theta=initial_M_theta,
    )

    # 7. Deduce theta and q_vector
    q_results = bck.compute_theta_and_q_vector(
        r_sph=r_sph,
        M_theta=M_theta_fit,
        n_medium=refractive_index,
        lambda_vacuum=laser_wavelength,
        Shift_0=S0_fit,
    )

    results = {
        "x_coords": x_flat,
        "y_coords": y_flat,
        "shift": shift_flat,
        "linewidth": linewidth_flat,
        "shift_err": shift_err_flat,
        "linewidth_err": linewidth_err_flat,
        "has_errors": has_errors,
        "x_sph": x_sph,
        "y_sph": y_sph,
        "r_sph": r_sph,
        "theta": q_results["theta"],
        "theta_deg": q_results["theta_deg"],
        "q_vector": q_results["q_vector"],
        "q_vector_um": q_results["q_vector_um"],
        "M_theta": M_theta_fit,
        "M_theta_err": M_theta_err,
        "Shift_0": S0_fit,
        "Shift_0_err": S0_err,
        "V_s": q_results["V_s"],
        "opt_params": opt_params,
        "coaxial_fit": coaxial_fit,
    }

    if verbose:
        print(f"  • Refined Ellipse Center : ({opt_params['xc']:.2f}, {opt_params['yc']:.2f}) px")
        print(f"  • Principal Director Axis: {opt_params['phi_deg']:+.2f}°")
        print(f"  • Aspect Ratio γ         : {opt_params['gamma']:.4f}")
        print(f"  • Normal Shift S(0)      : {S0_fit:.4f} ± {S0_err:.4f} GHz")
        print(f"  • Magnification M_theta  : {M_theta_fit:.1f} ± {M_theta_err:.1f} px")
        print(f"  • Acoustic Velocity V_s  : {q_results['V_s']:.2f} m/s")
        print("=" * 80 + "\n")

    return results


# ==============================================================================
# HDF5 Persistence & Retrieval
# ==============================================================================

def save_q_vector_results(wrapper, path_sample, bin_val_h, bin_val_w, q_results):
    """
    Saves the extracted q-vector arrays and geometric attributes to HDF5.
    """
    group_treated = f"{path_sample}/Treated {bin_val_h}x{bin_val_w} bins"

    wrapper.add_other(
        data=q_results["q_vector"],
        parent_group=group_treated,
        name="q_vector",
        overwrite=True,
    )
    wrapper.add_other(
        data=q_results["theta"],
        parent_group=group_treated,
        name="theta",
        overwrite=True,
    )
    wrapper.add_other(
        data=q_results["r_sph"],
        parent_group=group_treated,
        name="r_sph",
        overwrite=True,
    )

    opt = q_results["opt_params"]
    attributes = {
        "M_theta": float(q_results["M_theta"]),
        "Shift_0": float(q_results["Shift_0"]),
        "V_s": float(q_results["V_s"]),
        "xc_ellipse": float(opt["xc"]),
        "yc_ellipse": float(opt["yc"]),
        "phi_ellipse_deg": float(opt["phi_deg"]),
        "gamma_ellipse": float(opt["gamma"]),
    }
    wrapper.add_attributes(attributes, parent_group=group_treated, overwrite=True)


def retrieve_q_vector_results(wrapper, path_sample, bin_val_h, bin_val_w):
    """
    Retrieves previously stored q-vector arrays and attributes from HDF5 if available.
    """
    group_treated = f"{path_sample}/Treated {bin_val_h}x{bin_val_w} bins"
    children = wrapper.get_children_elements(group_treated)
    if "q_vector" not in children or "theta" not in children:
        return None

    try:
        q_vector = wrapper[f"{group_treated}/q_vector"].astype(np.float64)
        theta = wrapper[f"{group_treated}/theta"].astype(np.float64)
        r_sph = wrapper[f"{group_treated}/r_sph"].astype(np.float64)
        attrs = wrapper.get_attributes(group_treated)

        return {
            "q_vector": q_vector,
            "q_vector_um": q_vector * 1e-6,
            "theta": theta,
            "theta_deg": np.degrees(theta),
            "r_sph": r_sph,
            "M_theta": float(attrs.get("M_theta", np.nan)),
            "Shift_0": float(attrs.get("Shift_0", np.nan)),
            "V_s": float(attrs.get("V_s", np.nan)),
            "opt_params": {
                "xc": float(attrs.get("xc_ellipse", np.nan)),
                "yc": float(attrs.get("yc_ellipse", np.nan)),
                "phi_deg": float(attrs.get("phi_ellipse_deg", np.nan)),
                "phi_rad": np.deg2rad(float(attrs.get("phi_ellipse_deg", 0.0))),
                "gamma": float(attrs.get("gamma_ellipse", np.nan)),
            },
        }
    except Exception:
        return None
