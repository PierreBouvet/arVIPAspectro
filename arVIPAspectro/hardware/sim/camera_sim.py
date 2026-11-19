"""
Synthetic 2D frame generators for simulated Orca (ar-VIPA Brillouin) and Teledyne (brightfield) cameras.
Separated from hardware drivers so physical HAL modules stay decoupled from spectroscopy physics.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def generate_vipa_frame(
    detector_size: Tuple[int, int],
    roi: Tuple[int, int, int, int, int, int],
    exposure_s: float,
    readout_speed: str,
    frame_counter: int,
    grid_cache: Optional[Tuple[int, int, np.ndarray, np.ndarray]] = None,
    stage_pos_um: Optional[Dict[str, Optional[float]]] = None,
    spectrum_type: str = "sample",
) -> Tuple[np.ndarray, Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Generates a synthetic 2D ar-VIPA Brillouin interferogram (`uint16`) matching the active ROI & binning.
    Returns `(frame, updated_grid_cache)`.
    """
    det_w, det_h = detector_size
    hstart, hend, vstart, vend, hbin, vbin = roi
    w = max(1, (hend - hstart) // max(1, hbin))
    h = max(1, (vend - vstart) // max(1, vbin))

    if grid_cache is None or grid_cache[0] != h or grid_cache[1] != w:
        y_sensor = vstart + (np.arange(h, dtype=np.float32) + 0.5) * vbin
        x_sensor = hstart + (np.arange(w, dtype=np.float32) + 0.5) * hbin
        yy, xx = np.meshgrid(y_sensor, x_sensor, indexing="ij")
        grid_cache = (h, w, xx, yy)
    else:
        _, _, xx, yy = grid_cache

    x_um = float((stage_pos_um or {}).get("X") or 0.0)
    y_um = float((stage_pos_um or {}).get("Y") or 0.0)

    xn = (xx - 0.5 * det_w) / (0.5 * det_w)
    yn = (yy - 0.5 * det_h) / (0.5 * det_h)

    try:
        from ... import config, backend as bck
        g_2d = bck.g_from_xy(
            xx,
            yy,
            x0=1800.0,
            y0=1150.0,
            f=config.F_PX_DEFAULT,
            phi=config.VIPA_TILT_ANGLE,
            n_r=config.N_FUSED_SILICA_780,
        )
        dg_order = 1.45265 * (30.0 / (config.nu_0 * 1e-9))
        order_phase = (float(g_2d[min(1150, h - 1), 0]) - g_2d) / dg_order + 0.42
    except Exception:
        order_phase = 6.0 * (xn + 1.05) - 0.45 * (xn + 1.05) ** 2 + 0.18 * (yn ** 2)

    frac_signed = (order_phase + 0.5) - np.floor(order_phase + 0.5) - 0.5
    f_rel_ghz = frac_signed * 30.0

    mode = str(spectrum_type).lower().strip()
    if mode == "laser":
        spectral_profile = 1.4 * np.exp(-0.5 * (frac_signed / 0.016) ** 2)
        env_x0, env_y0 = 0.0, 0.0
    elif mode == "rb":
        f_rb = f_rel_ghz + 1.323
        continuum = 0.85 * np.exp(-0.5 * (frac_signed / 0.28) ** 2) + 0.15
        dip_87_a = 0.32 * np.exp(-0.5 * ((f_rb - (-2.56)) / 0.32) ** 2)
        dip_85_a = 0.62 * np.exp(-0.5 * ((f_rb - (-1.22)) / 0.35) ** 2)
        dip_85_b = 0.48 * np.exp(-0.5 * ((f_rb - (+1.70)) / 0.35) ** 2)
        dip_87_b = 0.22 * np.exp(-0.5 * ((f_rb - (+4.15)) / 0.32) ** 2)
        trans_rb = np.clip(1.0 - (dip_87_a + dip_85_a + dip_85_b + dip_87_b), 0.08, 1.0)
        spectral_profile = continuum * trans_rb
        env_x0, env_y0 = 0.0, 0.0
    else:
        dx_c = xx - 0.5 * det_w
        dy_c = yy - 1150.0
        r_sph = np.sqrt(dy_c ** 2 + (dx_c / 1.25) ** 2)
        u_ang = np.clip(r_sph / 3200.0, 0.0, 0.98)

        s0_ghz = 5.10 + 0.15 * np.sin(x_um / 60.0) * np.cos(y_um / 60.0)
        nu_b = s0_ghz * np.cos(0.5 * np.arcsin(u_ang))
        gamma_b = 0.78 + 0.06 * (u_ang ** 2)

        elastic = np.exp(-0.5 * (f_rel_ghz / 0.42) ** 2)
        nu_abs = np.abs(f_rel_ghz)
        dho_num = (gamma_b * nu_b) ** 2
        dho_den = (nu_abs ** 2 - nu_b ** 2) ** 2 + (gamma_b * nu_abs) ** 2 + 1e-9
        brillouin_dho = 0.38 * (dho_num / dho_den)
        spectral_profile = elastic + brillouin_dho
        env_x0 = 0.04 * np.sin(x_um / 120.0)
        env_y0 = 0.04 * np.cos(y_um / 120.0)

    envelope = np.exp(-0.95 * ((xn - env_x0) ** 2) - 0.55 * ((yn - env_y0) ** 2))
    exp_scale = np.clip(exposure_s / 0.050, 0.02, 20.0) * (hbin * vbin)
    signal = (120.0 + 18000.0 * exp_scale * envelope * spectral_profile).astype(np.float32)

    noise_std = 12.0 if readout_speed == "slow" else 25.0
    rng = np.random.default_rng(seed=1000 + frame_counter)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(h, w)).astype(np.float32)
    frame = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    return frame, grid_cache


def generate_brightfield_frame(
    detector_size: Tuple[int, int],
    roi: Tuple[int, int, int, int],
    pixel_size_um: float,
    exposure_s: float,
    gain_db: float,
    frame_counter: int,
    grid_cache: Optional[Tuple[int, int, np.ndarray, np.ndarray]] = None,
    stage_pos_um: Optional[Dict[str, Optional[float]]] = None,
) -> Tuple[np.ndarray, Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Generates a synthetic brightfield microscopy image (`uint16`) matching the active ROI.
    Returns `(frame, updated_grid_cache)`.
    """
    det_w, det_h = detector_size
    hstart, hend, vstart, vend = roi
    w = max(1, hend - hstart)
    h = max(1, vend - vstart)

    if grid_cache is None or grid_cache[0] != h or grid_cache[1] != w:
        yy, xx = np.mgrid[vstart:vend, hstart:hend].astype(np.float32)
        grid_cache = (h, w, xx, yy)
    else:
        _, _, xx, yy = grid_cache

    xn = (xx - 0.5 * det_w) / (0.5 * det_w)
    yn = (yy - 0.5 * det_h) / (0.5 * det_h)

    illum = np.exp(-0.35 * (xn ** 2 + yn ** 2))
    if stage_pos_um is not None:
        sim_um_per_px = pixel_size_um / 10.0
        sx_um = float(stage_pos_um.get("X") or 0.0)
        sy_um = float(stage_pos_um.get("Y") or 0.0)
        drift_x = (sx_um / sim_um_per_px) / (0.5 * det_w)
        drift_y = (sy_um / sim_um_per_px) / (0.5 * det_h)
    else:
        drift_x = 0.04 * np.sin(frame_counter * 0.08)
        drift_y = 0.04 * np.cos(frame_counter * 0.06)

    obj1 = 0.45 * np.exp(-(((xn - 0.15 - drift_x) / 0.18) ** 2 + ((yn + 0.10 - drift_y) / 0.12) ** 2))
    obj2 = 0.35 * np.exp(-(((xn + 0.25 - drift_x) / 0.10) ** 2 + ((yn - 0.20 - drift_y) / 0.10) ** 2))
    rings = 0.12 * (0.5 + 0.5 * np.cos(28.0 * np.hypot(xn - drift_x, yn - drift_y)))

    exp_scale = np.clip(exposure_s / 0.020, 0.05, 10.0)
    gain_scale = 10.0 ** (gain_db / 20.0)
    signal = 42000.0 * exp_scale * gain_scale * illum * (1.0 - obj1 - obj2 - rings * 0.2)

    rng = np.random.default_rng(seed=2000 + frame_counter)
    noise = rng.normal(loc=0.0, scale=220.0 * gain_scale, size=(h, w)).astype(np.float32)
    frame = np.clip(signal + noise, 0, 65535).astype(np.uint16)
    return frame, grid_cache
