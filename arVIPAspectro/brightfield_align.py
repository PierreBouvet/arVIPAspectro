"""
Brightfield Spatial Calibration & Sub-Pixel Image Registration (`arVIPAspectro/brightfield_align.py`):
  - Sub-pixel 2D FFT phase cross-correlation with vignetting removal for brightfield microscopy
  - 3-point ASI MS-2000 stage jog calibration -> full 2x2 pixel-to-stage affine Jacobian (um/px & rotation)
  - Translation of measured pixel offsets (dcol_px, drow_px) into stage / map center corrections (dX_um, dY_um)
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _prepare_registration_image(img: np.ndarray) -> np.ndarray:
    """
    Converts a 2D image to float64, suppresses multiplicative illumination vignetting/tilt
    via flat-field division (`img / G_large(img) - 1`), attenuates pixel shot noise with
    a narrow Gaussian filter, zero-centers, and applies a 2D Hanning window.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image for registration, got shape {arr.shape}.")
    h, w = arr.shape
    sigma_bg = max(12.0, min(h, w) / 8.0)
    if min(h, w) >= 256:
        small = arr[::4, ::4]
        illum_small = gaussian_filter(small, sigma=sigma_bg / 4.0, mode="nearest")
        illum = zoom(
            illum_small,
            (h / illum_small.shape[0], w / illum_small.shape[1]),
            order=1,
            mode="nearest",
        )[:h, :w]
    else:
        illum = gaussian_filter(arr, sigma=sigma_bg, mode="nearest")
    illum_floor = max(float(np.mean(illum)) * 0.1, 1.0)
    flat = (arr / np.maximum(illum, illum_floor)) - 1.0
    bp = gaussian_filter(flat, sigma=2.0, mode="nearest")
    bp -= np.mean(bp)
    win_y = np.hanning(h)[:, None]
    win_x = np.hanning(w)[None, :]
    return bp * (win_y * win_x)


def _extract_subpixel_peak(cc: np.ndarray) -> Tuple[float, float, float]:
    """Extracts wrapped sub-pixel shift `(dcol_px, drow_px, peak_val)` from a 2D correlation map."""
    h, w = cc.shape
    peak_idx = int(np.argmax(cc))
    py, px = divmod(peak_idx, w)
    peak_val = float(cc[py, px])

    ym = (py - 1) % h
    yp = (py + 1) % h
    xm = (px - 1) % w
    xp = (px + 1) % w

    denom_y = 2.0 * (cc[ym, px] - 2.0 * cc[py, px] + cc[yp, px])
    dy_sub = (cc[ym, px] - cc[yp, px]) / denom_y if abs(denom_y) > 1e-12 else 0.0
    dy_sub = float(np.clip(dy_sub, -0.5, 0.5))

    denom_x = 2.0 * (cc[py, xm] - 2.0 * cc[py, px] + cc[py, xp])
    dx_sub = (cc[py, xm] - cc[py, xp]) / denom_x if abs(denom_x) > 1e-12 else 0.0
    dx_sub = float(np.clip(dx_sub, -0.5, 0.5))

    drow = float(py) + dy_sub
    dcol = float(px) + dx_sub
    if drow > 0.5 * h:
        drow -= h
    if dcol > 0.5 * w:
        dcol -= w

    return float(dcol), float(drow), peak_val


def measure_image_shift_with_map(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Like `measure_image_shift_px`, also returns the full normalized 2D correlation map `cc`.
    """
    ref_arr = np.asarray(ref_img)
    mov_arr = np.asarray(mov_img)
    if ref_arr.shape != mov_arr.shape:
        raise ValueError(
            f"Brightfield image shapes must match for registration ({ref_arr.shape} vs {mov_arr.shape})."
        )

    p1 = _prepare_registration_image(ref_arr)
    p2 = _prepare_registration_image(mov_arr)
    norm = float(np.sqrt(np.sum(p1 ** 2) * np.sum(p2 ** 2)))
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0, np.zeros(ref_arr.shape, dtype=np.float64)

    f1 = np.fft.fft2(p1)
    f2 = np.fft.fft2(p2)
    cc = np.real(np.fft.ifft2(f2 * np.conj(f1))) / norm
    dcol, drow, peak_val = _extract_subpixel_peak(cc)
    return dcol, drow, peak_val, cc


def compute_jog_correlations(
    img0: np.ndarray,
    img_x: np.ndarray,
    img_y: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes normalized autocorrelation (`img0 vs img0`) and 3-point stage jog
    cross-correlations (`img_x vs img0`, `img_y vs img0`) in one pass.
    """
    if img0.shape != img_x.shape or img0.shape != img_y.shape:
        raise ValueError(
            f"Jog images must share identical shape ({img0.shape}, {img_x.shape}, {img_y.shape})."
        )

    p0 = _prepare_registration_image(img0)
    px = _prepare_registration_image(img_x)
    py = _prepare_registration_image(img_y)

    e0 = float(np.sum(p0 ** 2))
    ex = float(np.sum(px ** 2))
    ey = float(np.sum(py ** 2))

    f0_conj = np.conj(np.fft.fft2(p0))
    f0 = np.conj(f0_conj)
    fx = np.fft.fft2(px)
    fy = np.fft.fft2(py)

    cc_auto = np.real(np.fft.ifft2(f0 * f0_conj)) / e0 if e0 > 1e-12 else np.zeros(img0.shape)
    norm_x = float(np.sqrt(e0 * ex))
    cc_x = np.real(np.fft.ifft2(fx * f0_conj)) / norm_x if norm_x > 1e-12 else np.zeros(img0.shape)
    norm_y = float(np.sqrt(e0 * ey))
    cc_y = np.real(np.fft.ifft2(fy * f0_conj)) / norm_y if norm_y > 1e-12 else np.zeros(img0.shape)

    dc_x, dr_x, pk_x = _extract_subpixel_peak(cc_x)
    dc_y, dr_y, pk_y = _extract_subpixel_peak(cc_y)

    return {
        "cc_auto": cc_auto,
        "cc_x": cc_x,
        "cc_y": cc_y,
        "dp_x": (dc_x, dr_x),
        "dp_y": (dc_y, dr_y),
        "peak_x": pk_x,
        "peak_y": pk_y,
    }


def measure_image_shift_px(ref_img: np.ndarray, mov_img: np.ndarray) -> Tuple[float, float, float]:
    """
    Measures the sub-pixel shift `(dcol_px, drow_px)` of features in `mov_img` relative to
    `ref_img` via flat-fielded normalized 2D FFT cross-correlation.

    Convention: if a feature at `(col, row)` in `ref_img` moves to `(col + dcol_px, row + drow_px)`
    in `mov_img`, this returns `(dcol_px, drow_px, peak_response)`.
    """
    dcol, drow, peak_val, _ = measure_image_shift_with_map(ref_img, mov_img)
    return dcol, drow, peak_val


def build_plate_calibration_from_jogs(
    dx_um: float,
    dy_um: float,
    dp_from_x_px: Tuple[float, float],
    dp_from_y_px: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Constructs the 2x2 pixel-to-stage affine Jacobian from 3-point stage jog shifts:
      - `dp_from_x_px = (dcol_x, drow_x)` measured after moving stage by `(+dx_um, 0)`
      - `dp_from_y_px = (dcol_y, drow_y)` measured after moving stage by `(0, +dy_um)`
    """
    if abs(dx_um) < 1e-6 or abs(dy_um) < 1e-6:
        raise ValueError("Jog step sizes (dx_um, dy_um) must be non-zero.")

    # Pixel velocity per micron: dp_px = P @ dS_um
    P = np.array(
        [
            [dp_from_x_px[0] / float(dx_um), dp_from_y_px[0] / float(dy_um)],
            [dp_from_x_px[1] / float(dx_um), dp_from_y_px[1] / float(dy_um)],
        ],
        dtype=np.float64,
    )
    det = float(np.linalg.det(P))
    if abs(det) < 1e-8:
        raise RuntimeError(
            "Brightfield stage auto-calibration failed: detected pixel shifts are collinear or zero. "
            "Make sure a textured sample feature is in view and try a larger jog step."
        )

    M_um_per_px = np.linalg.inv(P)
    px_per_um_x = float(np.hypot(P[0, 0], P[1, 0]))
    px_per_um_y = float(np.hypot(P[0, 1], P[1, 1]))
    um_per_px_x = 1.0 / px_per_um_x
    um_per_px_y = 1.0 / px_per_um_y
    um_per_px_mean = 0.5 * (um_per_px_x + um_per_px_y)
    rot_deg = float(np.rad2deg(np.arctan2(P[1, 0], P[0, 0])))

    return {
        "method": "auto_3pt_jog",
        "P_px_per_um": P,
        "M_um_per_px": M_um_per_px,
        "um_per_px": float(um_per_px_mean),
        "um_per_px_x": float(um_per_px_x),
        "um_per_px_y": float(um_per_px_y),
        "rotation_deg": rot_deg,
        "jog_dx_um": float(dx_um),
        "jog_dy_um": float(dy_um),
    }


def build_manual_plate_calibration(
    um_per_px: float,
    flip_x: bool = False,
    flip_y: bool = False,
    rotation_deg: float = 0.0,
) -> Dict[str, Any]:
    """Builds a 2x2 pixel-to-stage matrix manually from isotropic `um_per_px`, flips, and rotation."""
    scale = abs(float(um_per_px))
    if scale <= 0:
        raise ValueError("Pixel scale (um/px) must be positive.")

    th = np.deg2rad(float(rotation_deg))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float64)
    flips = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0])
    P = (1.0 / scale) * (R @ flips)
    M_um_per_px = np.linalg.inv(P)

    return {
        "method": "manual",
        "P_px_per_um": P,
        "M_um_per_px": M_um_per_px,
        "um_per_px": float(scale),
        "um_per_px_x": float(scale),
        "um_per_px_y": float(scale),
        "rotation_deg": float(rotation_deg),
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
    }


def pixel_shift_to_stage_delta_um(
    calib: Dict[str, Any],
    dcol_px: float,
    drow_px: float,
) -> Tuple[float, float]:
    """
    Converts an observed feature pixel shift `(dcol_px, drow_px)` in `mov_img` relative to `ref_img`
    into the ASI MS-2000 stage displacement `(dX_um, dY_um)` needed to re-center the sample
    feature at its reference location: `dS = -M_um_per_px @ [dcol_px, drow_px]`.
    """
    M = np.asarray(calib["M_um_per_px"], dtype=np.float64)
    dp = np.array([float(dcol_px), float(drow_px)], dtype=np.float64)
    dS = -(M @ dp)
    return float(dS[0]), float(dS[1])
