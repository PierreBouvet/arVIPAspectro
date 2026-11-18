"""
Acquisition Pipeline & HDF5_BLS Session Manager (Step 4).

Manages:
  1. HDF5 file creation/loading following the `HDF5_BLS` standard:
     - `Brillouin/Calibration/laser_reference/image` (`Calibration_spectrum` -> `Raw_data`)
     - `Brillouin/Calibration/Rb_absorption/image` (`Calibration_spectrum` -> `Raw_data`)
     - `Brillouin/Calibration/reference_sample/image` (`Calibration_spectrum` -> `Raw_data`, optional)
     - `Brillouin/<Map_name>` (`Measure`) with `X_um`, `Y_um` (`Abscissa`), `Treated ...` (`Treatment`),
       and optional 4D chunked `Raw_data` (`(Ny, Nx, H, W)`, `uint16`).
  2. 2D Spatial Map Grid calculation (`MapGridConfig`) supporting Start/Center + Step/Span
     or Corner A / Corner B teaching, Raster vs. Snake (Serpentine) trajectories,
     and memory/time estimation.
"""

import os
import threading
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import h5py
from HDF5_BLS import Wrapper

from . import config
from . import backend as bck
from . import laser_calib
from . import Rb_calib
from . import spectrum_fitting
from . import q_vector


@dataclass
class MapGridConfig:
    """
    Defines a 2D (or 1D / single-point) spatial scan grid in micrometers (µm).
    Grid indices are `(iy, ix)` with `0 <= iy < ny` and `0 <= ix < nx`.
    """

    x_start_um: float = 0.0
    y_start_um: float = 0.0
    dx_um: float = 10.0
    dy_um: float = 10.0
    nx: int = 5
    ny: int = 5
    scan_pattern: str = "snake"  # 'snake' (serpentine) or 'raster' (unidirectional X)

    @property
    def total_points(self) -> int:
        return max(1, int(self.nx)) * max(1, int(self.ny))

    @property
    def x_coords_um(self) -> np.ndarray:
        nx = max(1, int(self.nx))
        if nx == 1:
            return np.array([float(self.x_start_um)], dtype=np.float64)
        return float(self.x_start_um) + np.arange(nx, dtype=np.float64) * float(self.dx_um)

    @property
    def y_coords_um(self) -> np.ndarray:
        ny = max(1, int(self.ny))
        if ny == 1:
            return np.array([float(self.y_start_um)], dtype=np.float64)
        return float(self.y_start_um) + np.arange(ny, dtype=np.float64) * float(self.dy_um)

    @property
    def span_x_um(self) -> float:
        return max(0, int(self.nx) - 1) * abs(float(self.dx_um))

    @property
    def span_y_um(self) -> float:
        return max(0, int(self.ny) - 1) * abs(float(self.dy_um))

    @property
    def x_center_um(self) -> float:
        nx = max(1, int(self.nx))
        return float(self.x_start_um) + 0.5 * (nx - 1) * float(self.dx_um)

    @property
    def y_center_um(self) -> float:
        ny = max(1, int(self.ny))
        return float(self.y_start_um) + 0.5 * (ny - 1) * float(self.dy_um)

    @classmethod
    def from_center(
        cls,
        x_center_um: float,
        y_center_um: float,
        dx_um: float,
        dy_um: float,
        nx: int,
        ny: int,
        scan_pattern: str = "snake",
    ) -> "MapGridConfig":
        """
        Builds a `MapGridConfig` centered on `(x_center_um, y_center_um)` with
        step sizes `(dx_um, dy_um)` and `(nx, ny)` points.
        """
        nx_i = max(1, int(nx))
        ny_i = max(1, int(ny))
        x_start = float(x_center_um) - 0.5 * (nx_i - 1) * float(dx_um)
        y_start = float(y_center_um) - 0.5 * (ny_i - 1) * float(dy_um)
        return cls(
            x_start_um=x_start,
            y_start_um=y_start,
            dx_um=float(dx_um),
            dy_um=float(dy_um),
            nx=nx_i,
            ny=ny_i,
            scan_pattern=scan_pattern,
        )

    def build_trajectory(self) -> List[Tuple[int, int, float, float]]:
        """
        Returns the ordered list of scan points as `(iy, ix, y_um, x_um)`.
        """
        xs = self.x_coords_um
        ys = self.y_coords_um
        points: List[Tuple[int, int, float, float]] = []
        for iy, y_val in enumerate(ys):
            ix_order = range(len(xs))
            if self.scan_pattern == "snake" and (iy % 2 == 1):
                ix_order = reversed(range(len(xs)))
            for ix in ix_order:
                points.append((int(iy), int(ix), float(y_val), float(xs[ix])))
        return points

    def estimate_raw_size_bytes(self, frame_shape: Tuple[int, int]) -> int:
        """Returns the uncompressed size in bytes of a 4D `(Ny, Nx, H, W)` uint16 raw dataset."""
        h, w = int(frame_shape[0]), int(frame_shape[1])
        return self.total_points * h * w * 2

    def estimate_duration_seconds(
        self,
        exposure_s: float,
        n_avg: int = 1,
        settling_s: float = 0.05,
        move_overhead_s: float = 0.06,
    ) -> float:
        """Estimates total scan duration in seconds."""
        per_point = float(exposure_s) * max(1, int(n_avg)) + float(settling_s) + float(move_overhead_s)
        return self.total_points * per_point

    @classmethod
    def from_corners(
        cls,
        x_a_um: float,
        y_a_um: float,
        x_b_um: float,
        y_b_um: float,
        dx_um: float,
        dy_um: float,
        scan_pattern: str = "snake",
    ) -> "MapGridConfig":
        """
        Builds a `MapGridConfig` spanning from Corner A `(x_a_um, y_a_um)` to Corner B `(x_b_um, y_b_um)`
        with step magnitudes `(|dx_um|, |dy_um|)`.
        """
        step_x = max(0.01, abs(float(dx_um)))
        step_y = max(0.01, abs(float(dy_um)))
        span_x = float(x_b_um) - float(x_a_um)
        span_y = float(y_b_um) - float(y_a_um)

        nx = int(np.floor(abs(span_x) / step_x)) + 1 if abs(span_x) >= 1e-6 else 1
        ny = int(np.floor(abs(span_y) / step_y)) + 1 if abs(span_y) >= 1e-6 else 1

        signed_dx = step_x if span_x >= 0 else -step_x
        signed_dy = step_y if span_y >= 0 else -step_y

        return cls(
            x_start_um=float(x_a_um),
            y_start_um=float(y_a_um),
            dx_um=signed_dx,
            dy_um=signed_dy,
            nx=nx,
            ny=ny,
            scan_pattern=scan_pattern,
        )


class AcquisitionSession:
    """
    Manages the HDF5_BLS container, calibration spectra (`laser_reference`, `Rb_absorption`,
    optional `reference_sample`), and incremental spatial map storage.
    """

    CALIBRATION_GROUP = "Brillouin/Calibration"
    ROLE_LASER = "laser_reference"
    ROLE_RB = "Rb_absorption"
    ROLE_REF_SAMPLE = "reference_sample"

    def __init__(self):
        self._h5_lock = threading.RLock()
        self.h5_path: Optional[str] = None
        self.wrapper: Optional[Wrapper] = None

        # In-memory cached calibration frames (2D numpy arrays)
        self.laser_image: Optional[np.ndarray] = None
        self.rb_image: Optional[np.ndarray] = None
        self.ref_sample_image: Optional[np.ndarray] = None

        # Calibration metadata dictionaries
        self.laser_meta: Dict[str, Any] = {}
        self.rb_meta: Dict[str, Any] = {}
        self.ref_sample_meta: Dict[str, Any] = {}

        # Step 5.1: Laser Calibration results dictionary
        self.laser_calib_result: Optional[Dict[str, Any]] = None

        # Step 5.2: Rubidium Absorption & 2D Frequency Map Calibration results dictionary
        self.rb_calib_result: Optional[Dict[str, Any]] = None

        # Step 5.3: Reference / Test Sample DHO Fitting & Elliptical q-Vector Calibration results
        self.ref_sample_calib_result: Optional[Dict[str, Any]] = None

        # Multi-angle Reference Samples keyed by sanitized angle label
        self.multi_angle_ref_images: Dict[str, np.ndarray] = {}
        self.multi_angle_ref_meta: Dict[str, Dict[str, Any]] = {}
        self.multi_angle_ref_calibs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def sanitize_angle_label(label: str) -> str:
        clean = str(label).strip().replace("/", "_").replace("\\", "_")
        if not clean:
            raise ValueError("Angle label cannot be empty.")
        return clean

    @classmethod
    def ref_role_for_angle(cls, angle_label: Optional[str] = None) -> str:
        if angle_label is None or not str(angle_label).strip():
            return cls.ROLE_REF_SAMPLE
        return f"{cls.ROLE_REF_SAMPLE}_{cls.sanitize_angle_label(angle_label)}"

    @property
    def is_open(self) -> bool:
        return self.wrapper is not None and self.h5_path is not None

    @property
    def has_required_calibration_images(self) -> bool:
        return (self.laser_image is not None) and (self.rb_image is not None)

    def open_or_create_h5(self, filepath: str) -> Dict[str, Any]:
        """
        Opens an existing HDF5_BLS file or creates a new one at `filepath`.
        Ensures `Brillouin/Calibration` (`Brillouin_type = 'Calibration_spectrum'`) exists,
        and automatically loads any existing `laser_reference`, `Rb_absorption`, or
        `reference_sample` images present in the file.
        """
        filepath = os.path.abspath(filepath)
        if not filepath.endswith(".h5"):
            filepath += ".h5"
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        self.wrapper = Wrapper(filepath)
        self.h5_path = filepath

        # Ensure Brillouin/Calibration group exists with Brillouin_type = 'Calibration_spectrum'
        with h5py.File(self.h5_path, "a") as f:
            if "Brillouin" not in f:
                grp_root = f.create_group("Brillouin")
                grp_root.attrs["Brillouin_type"] = "Root"
            if self.CALIBRATION_GROUP not in f:
                grp_cal = f.create_group(self.CALIBRATION_GROUP)
                grp_cal.attrs["Brillouin_type"] = "Calibration_spectrum"

        # Load any calibration images already stored in this HDF5 file
        self.laser_image, self.laser_meta = self._try_load_calibration(self.ROLE_LASER)
        self.rb_image, self.rb_meta = self._try_load_calibration(self.ROLE_RB)
        self.ref_sample_image, self.ref_sample_meta = self._try_load_calibration(self.ROLE_REF_SAMPLE)
        self.laser_calib_result = self._try_load_laser_calibration_result()
        self.rb_calib_result = self._try_load_rb_calibration_result()

        self.multi_angle_ref_images.clear()
        self.multi_angle_ref_meta.clear()
        self.multi_angle_ref_calibs.clear()
        prefix = f"{self.ROLE_REF_SAMPLE}_"
        with h5py.File(self.h5_path, "r") as f:
            if self.CALIBRATION_GROUP in f:
                for k in f[self.CALIBRATION_GROUP].keys():
                    if k.startswith(prefix):
                        ang = k[len(prefix):]
                        img, meta = self._try_load_calibration(k)
                        if img is not None:
                            self.multi_angle_ref_images[ang] = img
                            self.multi_angle_ref_meta[ang] = meta

        return {
            "h5_path": self.h5_path,
            "has_laser": self.laser_image is not None,
            "has_laser_calib": self.laser_calib_result is not None,
            "has_rb": self.rb_image is not None,
            "has_rb_calib": self.rb_calib_result is not None,
            "has_ref_sample": self.ref_sample_image is not None,
            "existing_maps": self.list_map_groups(),
        }

    def _try_load_calibration(self, role: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        if not self.is_open:
            return None, {}
        ds_path = f"{self.CALIBRATION_GROUP}/{role}/image"
        grp_path = f"{self.CALIBRATION_GROUP}/{role}"
        with h5py.File(self.h5_path, "r") as f:
            if ds_path not in f:
                return None, {}
            img = np.asarray(f[ds_path][()])
            meta = dict(f[grp_path].attrs)
            return img, meta

    def save_calibration_image(
        self,
        role: str,
        image: np.ndarray,
        exposure_s: float,
        n_avg: int,
        roi: Tuple[int, int, int, int, int, int],
        stage_pos_um: Optional[Dict[str, Optional[float]]] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
    ):
        """
        Saves a 2D calibration image (`laser_reference`, `Rb_absorption`, or `reference_sample`)
        under `Brillouin/Calibration/<role>/image` following the `HDF5_BLS` standard.
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")

        grp_path = f"{self.CALIBRATION_GROUP}/{role}"
        attrs = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "exposure_s": float(exposure_s),
            "exposure_ms": float(exposure_s * 1e3),
            "frame_averaging": int(n_avg),
            "roi_hstart": int(roi[0]),
            "roi_hend": int(roi[1]),
            "roi_vstart": int(roi[2]),
            "roi_vend": int(roi[3]),
            "hardware_binning": int(roi[4]),
        }
        if stage_pos_um:
            if stage_pos_um.get("X") is not None:
                attrs["stage_x_um"] = float(stage_pos_um["X"])
            if stage_pos_um.get("Y") is not None:
                attrs["stage_y_um"] = float(stage_pos_um["Y"])
            if stage_pos_um.get("Z") is not None:
                attrs["stage_z_um"] = float(stage_pos_um["Z"])
        if extra_attrs:
            attrs.update(extra_attrs)

        self.wrapper.add_dictionary(
            {
                "Raw_data": {"Name": "image", "Data": np.asarray(image)},
                "Attributes": attrs,
            },
            parent_group=grp_path,
            create_group=True,
            Brillouin_type_parent_group="Calibration_spectrum",
            overwrite=True,
        )

        if role == self.ROLE_LASER:
            self.laser_image = np.asarray(image)
            self.laser_meta = attrs
            self.laser_calib_result = None  # Reset calibration state when a new raw laser image is saved
            self.rb_calib_result = None
        elif role == self.ROLE_RB:
            self.rb_image = np.asarray(image)
            self.rb_meta = attrs
            self.rb_calib_result = None
        elif role == self.ROLE_REF_SAMPLE:
            self.ref_sample_image = np.asarray(image)
            self.ref_sample_meta = attrs
            self.ref_sample_calib_result = None
        elif role.startswith(f"{self.ROLE_REF_SAMPLE}_"):
            ang = role[len(self.ROLE_REF_SAMPLE) + 1:]
            self.multi_angle_ref_images[ang] = np.asarray(image)
            self.multi_angle_ref_meta[ang] = attrs
            self.multi_angle_ref_calibs.pop(ang, None)

    # ==========================================================================
    # Step 5.1: Laser Reference Calibration (Steps 2 to 6 of arVIPAspectro)
    # ==========================================================================

    def _try_load_laser_calibration_result(self) -> Optional[Dict[str, Any]]:
        """
        Attempts to reconstruct previously saved Laser Calibration results from
        `Brillouin/Calibration/laser_reference` in the HDF5 file if present.
        """
        if not self.is_open or self.laser_image is None:
            return None

        path_laser = f"{self.CALIBRATION_GROUP}/{self.ROLE_LASER}"
        try:
            children = self.wrapper.get_children_elements(path_laser)
            attrs = self.wrapper.get_attributes(path_laser)
            if "image_corrected" not in children or "x0_fit_circular" not in attrs:
                return None

            surf = laser_calib.retrieve_anamorphic_surface_calibration(self.wrapper, path_laser)
            if surf is None:
                return None

            data_sub, base_2d = laser_calib.retrieve_denoised_data(self.wrapper, path_laser)
            h = data_sub.shape[0]
            y_center = min(int(config.Y_CENTER_LASER), h // 2)
            peaks, points = laser_calib.retrieve_fringes(self.wrapper, path_laser, y_center_laser=y_center)
            ref_fringes, ax, axs, ay, ak = laser_calib.retrieve_refined_fringes(self.wrapper, path_laser)
            if len(ax) == 0:
                return None
            x0_fit, y0_fit, xc_gnom, yc_gnom = laser_calib.retrieve_vipa_geometry(self.wrapper, path_laser)

            # Reconstruct diagnostic arrays using the persisted 2D anamorphic surface
            c_2d = surf["c_2d_order"]
            g_mean_ref = float(surf["g_mean_ref"])
            g_std_ref = float(surf["g_std_ref"])
            L_y = float(surf["L_y"])
            # Solve (deg+1)*(deg+2)/2 == len(c_2d) for polynomial degree (default 4)
            poly_deg = 4
            for d_cand in range(1, 10):
                if (d_cand + 1) * (d_cand + 2) // 2 == len(c_2d):
                    poly_deg = d_cand
                    break

            g_all_laser = bck.g_from_xy(
                ax, ay, x0_fit, y0_fit,
                f=config.F_PX_DEFAULT, phi=config.VIPA_TILT_ANGLE, n_r=config.N_FUSED_SILICA_780,
            )
            g_all_laser_smooth = bck.g_from_xy(
                axs, ay, x0_fit, y0_fit,
                f=config.F_PX_DEFAULT, phi=config.VIPA_TILT_ANGLE, n_r=config.N_FUSED_SILICA_780,
            )
            gn_laser = (g_all_laser - g_mean_ref) / g_std_ref
            yn_laser = (ay - y0_fit) / L_y
            q_pred_2d = bck.eval_phi_order_surface(
                g_all_laser, ay, c_2d, y0_fit, L_y, g_mean_ref, g_std_ref, deg=poly_deg
            )
            q_pred_2d_smooth = bck.eval_phi_order_surface(
                g_all_laser_smooth, ay, c_2d, y0_fit, L_y, g_mean_ref, g_std_ref, deg=poly_deg
            )
            err_2d_mhz = (q_pred_2d - ak) * config.FSR_VIPA * 1e3
            err_2d_smooth_mhz = (q_pred_2d_smooth - ak) * config.FSR_VIPA * 1e3

            full_cal = dict(surf)
            full_cal.update({
                "gn_laser": gn_laser,
                "yn_laser": yn_laser,
                "g_all_laser": g_all_laser,
                "q_pred_2d": q_pred_2d,
                "q_pred_2d_smooth": q_pred_2d_smooth,
                "err_2d_mhz": err_2d_mhz,
                "err_2d_smooth_mhz": err_2d_smooth_mhz,
            })
            y_min = max(0, int(y_center - config.DY_LASER))
            y_max = min(h, int(y_center + config.DY_LASER))
            avg_row = np.average(data_sub[y_min:y_max, :], axis=0)

            res = {
                "data_raw": self.laser_image.astype(np.float64),
                "data_subtracted": data_sub,
                "baseline_2d": base_2d,
                "peaks": peaks,
                "points": points,
                "average_row": avg_row,
                "y_center_laser": y_center,
                "dy_laser": config.DY_LASER,
                "refined_fringes": ref_fringes,
                "all_x_laser": ax,
                "all_x_laser_smooth": axs,
                "all_y_laser": ay,
                "all_k_laser": ak,
                "x0_fit": x0_fit,
                "y0_fit": y0_fit,
                "xc_gnom": xc_gnom,
                "yc_gnom": yc_gnom,
            }
            res.update(full_cal)
            return res
        except Exception:
            return None

    def run_laser_calibration(
        self,
        scale_laser: int = config.SCALE_LASER,
        ksize_se: int = config.KSIZE_STRUCTURING_ELEMENTS_LASER,
        ksize_blur: int = config.KSIZE_BLUR_LASER,
        y_center_laser: int = config.Y_CENTER_LASER,
        dy_laser: int = config.DY_LASER,
        nb_laser_modes: int = config.NB_LASER_MODES,
        distance_peaks: int = config.DISTANCE_PEAKS_LASER,
        poly_deg: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes Steps 2–6 of `arVIPAspectro.laser_calib` on `Brillouin/Calibration/laser_reference`:
          - Step 2: Morphological background extraction & denoising (`denoise_laser`)
          - Step 3: Seed peak detection & full-height vertical fringe tracing (`extract_laser_fringes`)
          - Step 4: Unsaturated quadratic-weighted sub-pixel centroiding & outlier rejection (`refine_laser_fringes`)
          - Step 5: Exact 2D gnomonic VIPA geometry fit (`fit_vipa_geometry`)
          - Step 6: 2D anamorphic polynomial order surface fit & persistence (`calibrate_anamorphic_order_surface`)
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")
        if self.laser_image is None:
            raise RuntimeError("No Laser Reference image has been acquired or loaded yet.")

        path_laser = f"{self.CALIBRATION_GROUP}/{self.ROLE_LASER}"
        data_raw = self.laser_image.astype(np.float64)
        h, w = data_raw.shape

        # Clamp y_center_laser and distance_peaks if a smaller ROI or binning is active
        y_center = int(y_center_laser)
        if y_center >= h - 10 or y_center < 10:
            y_center = h // 2
        dy_c = min(int(dy_laser), max(5, h // 6))
        dist_p = min(int(distance_peaks), max(10, w // (max(2, int(nb_laser_modes)) + 2)))

        # Step 2: Denoising
        data_sub, base_2d = laser_calib.denoise_laser(
            self.wrapper,
            path_laser,
            data_raw,
            scale=int(scale_laser),
            ksize_structuring_elements=int(ksize_se),
            ksize_blur=int(ksize_blur),
        )

        # Step 3: Fringe Detection & Vertical Tracing
        peaks, points, avg_row = laser_calib.extract_laser_fringes(
            self.wrapper,
            path_laser,
            data_sub,
            y_center_laser=y_center,
            dy_laser=dy_c,
            nb_laser_modes=int(nb_laser_modes),
            distance_peaks=dist_p,
            dy_trace=config.DY_TRACE,
            var_trace=config.VAR_TRACE,
        )
        if len(points) < 2:
            raise RuntimeError(
                f"Only {len(points)} laser fringe(s) were detected. "
                "Check laser exposure/contrast or adjust 'Symmetry Y' and 'Min Peak Distance'."
            )

        # Step 4: Sub-Pixel Refinement & Outlier Rejection
        ref_fringes, ax, axs, ay, ak = laser_calib.refine_laser_fringes(
            self.wrapper,
            path_laser,
            points,
            data_raw,
            data_sub,
            w_sub=config.W_SUB,
            saturation_limit=config.SATURATION_LIMIT,
            min_net_amp=config.MIN_NET_AMP,
        )
        if len(ref_fringes) < 2:
            raise RuntimeError(
                f"Only {len(ref_fringes)} fringe(s) survived sub-pixel refinement (check saturation or low signal)."
            )

        # Step 5: Exact 2D Gnomonic VIPA Geometry Fit
        x0_fit, y0_fit, xc_gnom, yc_gnom = laser_calib.fit_vipa_geometry(
            self.wrapper,
            path_laser,
            ref_fringes,
            ax,
            ay,
            ak,
            f_px=config.F_PX_DEFAULT,
            phi_rad=config.VIPA_TILT_ANGLE,
            n_r=config.N_FUSED_SILICA_780,
        )

        # Step 6: 2D Anamorphic Order Surface Calibration
        calib_res = laser_calib.calibrate_anamorphic_order_surface(
            self.wrapper,
            path_laser,
            ax,
            axs,
            ay,
            ak,
            x0_fit,
            y0_fit,
            f_px=config.F_PX_DEFAULT,
            phi_rad=config.VIPA_TILT_ANGLE,
            n_r=config.N_FUSED_SILICA_780,
            h_img=h,
            nb_laser_modes=len(ref_fringes),
            fsr_vipa=config.FSR_VIPA,
            deg=int(poly_deg),
        )
        laser_calib.save_anamorphic_surface_calibration(self.wrapper, path_laser, calib_res)

        result = {
            "data_raw": data_raw,
            "data_subtracted": data_sub,
            "baseline_2d": base_2d,
            "peaks": peaks,
            "points": points,
            "average_row": avg_row,
            "y_center_laser": y_center,
            "dy_laser": dy_c,
            "refined_fringes": ref_fringes,
            "all_x_laser": ax,
            "all_x_laser_smooth": axs,
            "all_y_laser": ay,
            "all_k_laser": ak,
            "x0_fit": x0_fit,
            "y0_fit": y0_fit,
            "xc_gnom": xc_gnom,
            "yc_gnom": yc_gnom,
        }
        result.update(calib_res)
        self.laser_calib_result = result
        return result

    # ==========================================================================
    # Step 5.2: Rubidium Absorption & 2D Frequency Map Calibration (Steps 7–10B)
    # ==========================================================================

    def _try_load_rb_calibration_result(self) -> Optional[Dict[str, Any]]:
        """
        Attempts to reconstruct previously saved Rubidium Calibration & 2D Frequency Map
        results from `Brillouin/Calibration/Rb_absorption` in the HDF5 file if present.
        """
        if not self.is_open or self.rb_image is None or self.laser_calib_result is None:
            return None

        path_rb = f"{self.CALIBRATION_GROUP}/{self.ROLE_RB}"
        try:
            children = self.wrapper.get_children_elements(path_rb)
            rb_attrs = self.wrapper.get_attributes(path_rb)
            if "image_corrected" not in children or "Frequency" not in children or "FSR" not in rb_attrs:
                return None

            lc = self.laser_calib_result
            t_cell = float(rb_attrs.get("T_cell_C", 70.0))
            l_cell = float(rb_attrs.get("L_cell_m", 0.07))
            fit_win = float(rb_attrs.get("fit_window_ghz", 10.0))
            laser_off = float(rb_attrs.get("delta_nu_laser_offset_ghz", 0.0))
            rb_lock = float(rb_attrs.get("rb_lock_offset_ghz", 1.323))

            det_table_MHz, trans_table, trans_fast = Rb_calib.create_rubidium_reference_table(
                T_cell=t_cell, L_cell=l_cell
            )
            data_abs, base_abs = Rb_calib.retrieve_denoised_rb(self.wrapper, path_rb)
            profiles_2d, radius_axis, avg_spec_r, phi_order_r = Rb_calib.extract_radial_absorption_profile(
                data_abs,
                xc_gnom=lc["xc_gnom"],
                yc_gnom=lc["yc_gnom"],
                y0_fit=lc["y0_fit"],
                c_2d_order=lc["c_2d_order"],
                g_mean_ref=lc["g_mean_ref"],
                g_std_ref=lc["g_std_ref"],
                L_y=lc["L_y"],
                f_px=config.F_PX_DEFAULT,
                phi_rad=config.VIPA_TILT_ANGLE,
                n_r=config.N_FUSED_SILICA_780,
            )
            fsr_res = Rb_calib.retrieve_fsr_calibration(
                self.wrapper,
                path_rb,
                lc["g_bar"],
                lc["q_orders"],
                phi_order_r,
                avg_spec_r,
                trans_fast,
                nu_0=config.nu_0,
                fit_window_ghz=fit_win,
                delta_nu_laser_offset=laser_off,
            )
            freq_map, _ = Rb_calib.retrieve_frequency_map(self.wrapper, path_rb)
            if freq_map is None:
                return None

            res = {
                "data_raw": self.rb_image.astype(np.float64),
                "data_absorption": data_abs,
                "baseline_absorption": base_abs,
                "det_table_MHz": det_table_MHz,
                "trans_table": trans_table,
                "trans_fast": trans_fast,
                "profiles_2d": profiles_2d,
                "radius_axis": radius_axis,
                "avg_spectrum_radius": avg_spec_r,
                "phi_order_r": phi_order_r,
                "freq_map": freq_map,
                "xi_elastic": float(rb_attrs.get("xi_elastic", (fsr_res["delta_nu_opt"] + rb_lock) / fsr_res["FSR_cal"])),
                "T_cell": t_cell,
                "L_cell": l_cell,
                "rb_lock_offset_ghz": rb_lock,
            }
            res.update(fsr_res)
            return res
        except Exception:
            return None

    def run_rb_calibration(
        self,
        t_cell_c: float = 70.0,
        l_cell_m: float = 0.07,
        scale_rb: int = 9,
        ksize_se_rb: int = 3,
        ksize_blur_rb: int = 11,
        delta_nu_laser_offset_ghz: float = 0.0,
        fit_window_ghz: float = 10.0,
        rb_lock_offset_ghz: float = 1.323,
    ) -> Dict[str, Any]:
        """
        Executes Steps 7–10B of `arVIPAspectro.Rb_calib` on `Brillouin/Calibration/Rb_absorption`:
          - Step 7: Theoretical natural Rubidium D2 transmission lookup table (`create_rubidium_reference_table`)
          - Step 8: Morphological baseline denoising of Rb absorption image (`denoise_rb`)
          - Step 9: Radial profile extraction around gnomonic center (`extract_radial_absorption_profile`)
          - Step 10: VIPA FSR & hyperfine offset calibration across folded orders (`calibrate_fsr_and_offset`)
          - Step 10B: Full 2D calibrated Frequency Map construction & HDF5 persistence (`compute_and_save_frequency_map`)
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")
        if self.laser_calib_result is None:
            raise RuntimeError(
                "Step 5.1 (Laser Reference Calibration) must be completed before running Step 5.2 (Rb Calibration)."
            )
        if self.rb_image is None:
            raise RuntimeError("No Rubidium Absorption image has been acquired or loaded yet.")

        lc = self.laser_calib_result
        data_raw = self.rb_image.astype(np.float64)
        if data_raw.shape != lc["data_raw"].shape:
            raise ValueError(
                f"Rb image shape {data_raw.shape} does not match calibrated Laser image shape {lc['data_raw'].shape}."
            )

        path_rb = f"{self.CALIBRATION_GROUP}/{self.ROLE_RB}"

        # Step 7: Theoretical Rubidium D2 Absorption Spectrum Model
        det_table_MHz, trans_table, trans_fast = Rb_calib.create_rubidium_reference_table(
            T_cell=float(t_cell_c),
            L_cell=float(l_cell_m),
        )

        # Step 8: Rubidium Absorption Image Denoising
        data_abs, base_abs = Rb_calib.denoise_rb(
            self.wrapper,
            path_rb,
            data_raw,
            scale=int(scale_rb),
            ksize_structuring_elements=int(ksize_se_rb),
            ksize_blur=int(ksize_blur_rb),
        )

        # Step 9: Radial Profile Extraction Around Gnomonic Center
        profiles_2d, radius_axis, avg_spec_r, phi_order_r = Rb_calib.extract_radial_absorption_profile(
            data_abs,
            xc_gnom=lc["xc_gnom"],
            yc_gnom=lc["yc_gnom"],
            y0_fit=lc["y0_fit"],
            c_2d_order=lc["c_2d_order"],
            g_mean_ref=lc["g_mean_ref"],
            g_std_ref=lc["g_std_ref"],
            L_y=lc["L_y"],
            f_px=config.F_PX_DEFAULT,
            phi_rad=config.VIPA_TILT_ANGLE,
            n_r=config.N_FUSED_SILICA_780,
            num_angles=1000,
        )

        # Step 10: VIPA Free Spectral Range (FSR) & Hyperfine Offset Calibration
        fsr_res = Rb_calib.calibrate_fsr_and_offset(
            self.wrapper,
            path_rb,
            lc["g_bar"],
            lc["q_orders"],
            phi_order_r,
            avg_spec_r,
            trans_fast,
            nu_0=config.nu_0,
            fit_window_ghz=float(fit_window_ghz),
            delta_nu_laser_offset=float(delta_nu_laser_offset_ghz),
        )

        # Step 10B: Full 2D Frequency Map Construction & Save
        freq_map, phi_order_map, xi_elastic = Rb_calib.compute_and_save_frequency_map(
            wrapper=self.wrapper,
            path_group=path_rb,
            img_shape=data_abs.shape,
            x0_fit=lc["x0_fit"],
            y0_fit=lc["y0_fit"],
            xc_gnom=lc["xc_gnom"],
            yc_gnom=lc["yc_gnom"],
            c_2d_order=lc["c_2d_order"],
            g_mean_ref=lc["g_mean_ref"],
            g_std_ref=lc["g_std_ref"],
            L_y=lc["L_y"],
            FSR_cal=fsr_res["FSR_cal"],
            delta_nu_opt=fsr_res["delta_nu_opt"],
            rb_lock_offset_ghz=float(rb_lock_offset_ghz),
            f_px=config.F_PX_DEFAULT,
            phi_rad=config.VIPA_TILT_ANGLE,
            n_r=config.N_FUSED_SILICA_780,
        )

        extra_rb_attrs = {
            "T_cell_C": float(t_cell_c),
            "L_cell_m": float(l_cell_m),
            "fit_window_ghz": float(fit_window_ghz),
            "delta_nu_laser_offset_ghz": float(delta_nu_laser_offset_ghz),
            "rb_lock_offset_ghz": float(rb_lock_offset_ghz),
        }
        self.wrapper.add_attributes(extra_rb_attrs, parent_group=path_rb, overwrite=True)

        result = {
            "data_raw": data_raw,
            "data_absorption": data_abs,
            "baseline_absorption": base_abs,
            "det_table_MHz": det_table_MHz,
            "trans_table": trans_table,
            "trans_fast": trans_fast,
            "profiles_2d": profiles_2d,
            "radius_axis": radius_axis,
            "avg_spectrum_radius": avg_spec_r,
            "phi_order_r": phi_order_r,
            "freq_map": freq_map,
            "phi_order_map": phi_order_map,
            "xi_elastic": float(xi_elastic),
            "T_cell": float(t_cell_c),
            "L_cell": float(l_cell_m),
            "rb_lock_offset_ghz": float(rb_lock_offset_ghz),
        }
        result.update(fsr_res)
        self.rb_calib_result = result
        return result

    # ==========================================================================
    # Step 5.3: Sample In-Situ Calibration, DHO Fitting & q-Vector (Steps 11–14)
    # ==========================================================================

    def run_sample_treatment_and_q_calibration(
        self,
        sample_image: Optional[np.ndarray] = None,
        bin_val_h: int = 4,
        bin_val_w: int = 2,
        expected_shift_ghz: float = 5.1,
        shift_tol_ghz: float = 2.0,
        expected_lw_ghz: float = 0.8,
        lw_bounds_ghz: Tuple[float, float] = (0.4, 1.3),
        run_q_calibration: bool = True,
        refractive_index: float = 1.33,
        shift_filter_bounds: Tuple[float, float] = (4.8, 5.3),
        lw_filter_bounds: Tuple[float, float] = (0.5, 1.2),
        fixed_ellipse_angle_deg: Optional[float] = 90.0,
        initial_m_theta: float = 3000.0,
        persist_to_ref_group: bool = True,
        ref_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes Steps 11–14 of `arVIPAspectro` (`spectrum_fitting` and `q_vector`) on the
        Reference Sample (`Brillouin/Calibration/<ref_role>`, defaulting to `reference_sample`)
        or a provided test sample frame:
          - Step 11 (`calibrate_sample_frequency_map`): Two-stage in-situ 2D frequency map
            self-calibration (`df_coarse`, `dz_map_2d`, `dinst_map_2d`, `freq_corrected`).
          - Step 12 (`fit_sample_dho`): Spatial binning (`bin_val_h x bin_val_w`) & vectorized
            DHO fitting across all Stokes/Anti-Stokes orders (`Shift`, `Linewidth`, `Errors`, `x/y`).
          - Step 14 (`extract_q_vector`, optional): 2D concentric coaxial ellipse & angular
            magnification `M_theta` calibration, computing `q_vector`, `theta`, `Shift_0`, and `V_s`.
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")
        if self.laser_calib_result is None or self.rb_calib_result is None:
            raise RuntimeError(
                "Steps 5.1 (Laser Calibration) and 5.2 (Rb Calibration) must be completed before Step 5.3."
            )

        target_role = ref_role.strip() if ref_role else self.ROLE_REF_SAMPLE
        ang_prefix = f"{self.ROLE_REF_SAMPLE}_"
        ang_label = target_role[len(ang_prefix):] if target_role.startswith(ang_prefix) else None

        if sample_image is None:
            if ang_label is not None:
                sample_image = self.multi_angle_ref_images.get(ang_label)
            else:
                sample_image = self.ref_sample_image
        if sample_image is None:
            raise RuntimeError(f"No Reference Sample image available for '{target_role}'.")

        data_sample = np.asarray(sample_image, dtype=np.float64)
        freq_map_base = self.rb_calib_result["freq_map"]
        fsr_cal = float(self.rb_calib_result["FSR_cal"])
        y0_fit = float(self.laser_calib_result["y0_fit"])

        if data_sample.shape != freq_map_base.shape:
            raise ValueError(
                f"Sample image shape {data_sample.shape} does not match calibrated frequency map {freq_map_base.shape}."
            )

        yy, xx = np.indices(data_sample.shape, dtype=np.float64)

        # Step 11: In-situ 2D Frequency Map Self-Calibration
        (
            freq_corrected,
            df_coarse,
            dz_map_2d,
            dinst_map_2d,
            bls_peaks,
            avg_data_sample,
            freq_avg_data_sample,
        ) = spectrum_fitting.calibrate_sample_frequency_map(
            freq_map_base=freq_map_base,
            data_clean_in=data_sample,
            xx_in=xx,
            yy_in=yy,
            fsr_val=fsr_cal,
            y0_center=y0_fit,
            bin_val_h=int(bin_val_h),
            bin_val_w=int(bin_val_w),
            expected_shift=float(expected_shift_ghz),
            shift_tol=float(shift_tol_ghz),
        )

        # Step 12: Spatial Binning & Vectorized DHO Peak Fitting
        dho_results = spectrum_fitting.fit_sample_dho(
            data_sample=data_sample,
            freq_corrected=freq_corrected,
            xx=xx,
            yy=yy,
            fsr_val=fsr_cal,
            bin_val_h=int(bin_val_h),
            bin_val_w=int(bin_val_w),
            ds_init=float(expected_shift_ghz),
            shift_tol=float(shift_tol_ghz),
            gamma_init=float(expected_lw_ghz),
            gamma_bounds=(float(lw_bounds_ghz[0]), float(lw_bounds_ghz[1])),
            verbose=False,
        )

        path_ref = f"{self.CALIBRATION_GROUP}/{target_role}"
        has_stored_ref = (
            (ang_label in self.multi_angle_ref_images)
            if ang_label is not None
            else (self.ref_sample_image is not None)
        )
        should_persist = bool(persist_to_ref_group and has_stored_ref)
        if should_persist:
            spectrum_fitting.save_treated_data(
                wrapper=self.wrapper,
                path_sample=path_ref,
                bin_val_h=int(bin_val_h),
                bin_val_w=int(bin_val_w),
                shift_rel=dho_results["shift_clipped"],
                shift_err=dho_results["shift_error"],
                lw_rel=dho_results["linewidth_clipped"],
                lw_err=dho_results["linewidth_error"],
                x_coords=dho_results["x_coords"],
                y_coords=dho_results["y_coords"],
            )

        # Step 14: Acoustic Wavevector (q-vector) & 2D Concentric Ellipse Extraction
        q_results = None
        if run_q_calibration:
            q_results = q_vector.extract_q_vector(
                data_source=dho_results,
                path_sample=path_ref if should_persist else None,
                wrapper=self.wrapper if should_persist else None,
                bin_val_h=int(bin_val_h),
                bin_val_w=int(bin_val_w),
                refractive_index=float(refractive_index),
                laser_wavelength=config.lambda_0,
                use_errors=True,
                error_percentile_floor=1.0,
                average_pairs=True,
                shift_bounds=(float(shift_filter_bounds[0]), float(shift_filter_bounds[1])),
                linewidth_bounds=(float(lw_filter_bounds[0]), float(lw_filter_bounds[1])),
                initial_M_theta=float(initial_m_theta),
                fixed_ellipse_angle_deg=fixed_ellipse_angle_deg,
                verbose=False,
            )
            if should_persist:
                q_vector.save_q_vector_results(
                    wrapper=self.wrapper,
                    path_sample=path_ref,
                    bin_val_h=int(bin_val_h),
                    bin_val_w=int(bin_val_w),
                    q_results=q_results,
                )

        result = {
            "ref_role": target_role,
            "angle_label": ang_label,
            "data_sample": data_sample,
            "freq_corrected": freq_corrected,
            "df_coarse": float(df_coarse),
            "dz_map_2d": dz_map_2d,
            "dinst_map_2d": dinst_map_2d,
            "bls_peaks": bls_peaks,
            "avg_data_sample": avg_data_sample,
            "freq_avg_data_sample": freq_avg_data_sample,
            "dho_results": dho_results,
            "q_results": q_results,
            "bin_val_h": int(bin_val_h),
            "bin_val_w": int(bin_val_w),
            "expected_shift_ghz": float(expected_shift_ghz),
            "shift_tol_ghz": float(shift_tol_ghz),
            "expected_lw_ghz": float(expected_lw_ghz),
            "lw_bounds_ghz": (float(lw_bounds_ghz[0]), float(lw_bounds_ghz[1])),
            "refractive_index": float(refractive_index),
        }
        if ang_label is not None:
            self.multi_angle_ref_calibs[ang_label] = result
        else:
            self.ref_sample_calib_result = result
        return result

    def determine_n_orders(self, bin_val_h: int = 4, bin_val_w: int = 2) -> int:
        """Determines the number of VIPA interference orders present on the calibrated frequency map."""
        if self.rb_calib_result is None:
            return 11
        freq_map = self.rb_calib_result["freq_map"]
        fsr_val = float(self.rb_calib_result["FSR_cal"])
        f_bin = bck.bin_2d_array(freq_map, int(bin_val_h), int(bin_val_w))
        n_rows = f_bin.shape[0]
        yc_idx = n_rows // 2
        f_mid = np.nanmean(f_bin[max(0, yc_idx - 25) : min(n_rows, yc_idx + 25), :], axis=0)
        return min(11, max(int(np.floor(np.nanmax(f_mid) / fsr_val)), 1))

    def treat_single_map_point(
        self,
        frame: np.ndarray,
        bin_val_h: int = 4,
        bin_val_w: int = 2,
        n_orders: Optional[int] = None,
        expected_shift_ghz: float = 5.1,
        shift_tol_ghz: float = 2.0,
        expected_lw_ghz: float = 0.8,
        lw_bounds_ghz: Tuple[float, float] = (0.4, 1.3),
        refractive_index: float = 1.33,
        fast_drift_only: bool = True,
        q_calib_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Step 5.4: On-the-fly treatment of a single spatial map frame `(iy, ix)`:
          1. Fast coarse elastic frequency drift tracking (`df_coarse`) on the calibrated
             2D frequency map (using the active Step 5.3 / angle-specific aberration-corrected map
             if available, or the Step 5.2 Rb frequency map).
          2. Spatial binning (`bin_val_h x bin_val_w`) & vectorized DHO peak fitting (`fit_sample_dho`).
          3. Fast normal backscattering shift `Shift_0` (GHz) and acoustic velocity `V_s` (m/s)
             deduction using the locked coaxial ellipse & `M_theta` geometry.
        """
        if self.laser_calib_result is None or self.rb_calib_result is None:
            raise RuntimeError("Laser and Rb calibrations are required for on-the-fly map treatment.")

        data_sample = np.asarray(frame, dtype=np.float64)
        h, w = data_sample.shape
        fsr_val = float(self.rb_calib_result["FSR_cal"])
        y0_fit = float(self.laser_calib_result["y0_fit"])

        active_ref = q_calib_result if q_calib_result is not None else self.ref_sample_calib_result
        if active_ref is not None and (
            active_ref["freq_corrected"].shape == (h, w)
        ):
            freq_base = active_ref["freq_corrected"]
        else:
            freq_base = self.rb_calib_result["freq_map"]

        yy, xx = np.indices((h, w), dtype=np.float64)

        if fast_drift_only:
            # Fast elastic drift tracking along central strip (~1-2 ms)
            dy_strip = min(100, h // 4)
            yc_int = int(round(min(max(y0_fit, dy_strip), h - dy_strip - 1)))
            avg_d = np.mean(data_sample[yc_int - dy_strip : yc_int + dy_strip, :], axis=0)
            avg_f = np.mean(freq_base[yc_int - dy_strip : yc_int + dy_strip, :], axis=0)

            q_cand = avg_f / fsr_val
            k_min = max(1, int(np.floor(np.nanmin(q_cand))))
            k_max = int(np.floor(np.nanmax(q_cand)))

            df_orders = []
            s_min_b = max(0.5, float(expected_shift_ghz) - 2.0 * float(shift_tol_ghz))
            s_max_b = float(expected_shift_ghz) + 2.0 * float(shift_tol_ghz)
            for k_ord in range(k_min, k_max + 1):
                f_el = k_ord * fsr_val
                m_as = (avg_f > f_el - s_max_b) & (avg_f < f_el - s_min_b)
                m_s = (avg_f > f_el + s_min_b) & (avg_f < f_el + s_max_b)
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
            freq_point = freq_base - df_coarse
        else:
            freq_point, df_coarse, _, _, _, _, _ = spectrum_fitting.calibrate_sample_frequency_map(
                freq_map_base=freq_base,
                data_clean_in=data_sample,
                xx_in=xx,
                yy_in=yy,
                fsr_val=fsr_val,
                y0_center=y0_fit,
                bin_val_h=int(bin_val_h),
                bin_val_w=int(bin_val_w),
                n_orders=n_orders,
                expected_shift=float(expected_shift_ghz),
                shift_tol=float(shift_tol_ghz),
            )

        if n_orders is None:
            n_orders = self.determine_n_orders(bin_val_h=bin_val_h, bin_val_w=bin_val_w)

        dho = spectrum_fitting.fit_sample_dho(
            data_sample=data_sample,
            freq_corrected=freq_point,
            xx=xx,
            yy=yy,
            fsr_val=fsr_val,
            bin_val_h=int(bin_val_h),
            bin_val_w=int(bin_val_w),
            n_orders=int(n_orders),
            ds_init=float(expected_shift_ghz),
            shift_tol=float(shift_tol_ghz),
            gamma_init=float(expected_lw_ghz),
            gamma_bounds=(float(lw_bounds_ghz[0]), float(lw_bounds_ghz[1])),
            verbose=False,
        )

        sa = dho["shift_clipped_averaged"]
        la = dho["linewidth_clipped_averaged"]
        xa = dho["x_coords_averaged"]
        ya = dho["y_coords_averaged"]
        va = dho["shift_err_averaged"]

        mean_shift = float(np.nanmean(sa)) if np.any(np.isfinite(sa)) else float("nan")
        mean_lw = float(np.nanmean(la)) if np.any(np.isfinite(la)) else float("nan")

        # Fast locked-geometry S(0) and V_s deduction
        q_ref = (
            active_ref.get("q_results")
            if active_ref is not None
            else None
        )
        if q_ref is not None:
            opt = q_ref["opt_params"]
            xc_e, yc_e, phi_e, gam_e = opt["xc"], opt["yc"], opt["phi_rad"], opt["gamma"]
            m_theta = float(q_ref["M_theta"])
        else:
            xc_e, yc_e, phi_e, gam_e = 0.5 * w, y0_fit, 0.5 * np.pi, 1.25
            m_theta = 3200.0

        valid = np.isfinite(sa) & np.isfinite(xa) & np.isfinite(ya)
        shift_0 = float("nan")
        v_s = float("nan")
        if np.sum(valid) >= 3:
            _, _, r_sph = bck.transform_to_spherical_coordinates(
                xa[valid], ya[valid], xc_e, yc_e, phi_e, gam_e
            )
            u_ang = np.clip(r_sph / max(m_theta, 1.0), -0.999, 0.999)
            cos_half_theta = np.cos(0.5 * np.arcsin(u_ang))
            v_w = va[valid]
            weights = np.where(np.isfinite(v_w) & (v_w > 1e-10), 1.0 / v_w, 1.0)
            denom = float(np.sum(weights * (cos_half_theta ** 2)))
            if denom > 0:
                shift_0 = float(np.sum(weights * sa[valid] * cos_half_theta) / denom)
                v_s = float((config.lambda_0 * (shift_0 * 1e9)) / (2.0 * float(refractive_index)))

        return {
            "dho_results": dho,
            "df_coarse": float(df_coarse),
            "mean_shift_ghz": mean_shift,
            "mean_lw_ghz": mean_lw,
            "Shift_0_ghz": shift_0,
            "V_s_ms": v_s,
        }

    def list_map_groups(self) -> List[str]:
        """Lists existing `Measure` map groups under `Brillouin/`."""
        if not self.is_open:
            return []
        maps = []
        with self._h5_lock:
            with h5py.File(self.h5_path, "r") as f:
                if "Brillouin" not in f:
                    return []
                for k, item in f["Brillouin"].items():
                    if isinstance(item, h5py.Group) and k != "Calibration":
                        maps.append(k)
        return sorted(maps)

    def initialize_map_group(
        self,
        map_name: str,
        grid: MapGridConfig,
        frame_shape: Tuple[int, int],
        exposure_s: float,
        n_avg: int,
        roi: Tuple[int, int, int, int, int, int],
        save_raw_data: bool,
        on_the_fly_treatment: bool = True,
        bin_val_h: int = 4,
        bin_val_w: int = 2,
        refractive_index: float = 1.33,
        overwrite: bool = True,
        extra_attrs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str], int]:
        """
        Creates `Brillouin/<map_name>` (`Brillouin_type = 'Measure'`) in the HDF5 file,
        stores `Y_um` (`Abscissa_0_1`) and `X_um` (`Abscissa_1_2`) coordinate vectors,
        allocates 2D measured stage coordinate grids (`stage_X_measured_um`, `stage_Y_measured_um`),
        optionally pre-allocates a chunked 4D `Raw_data` dataset `(Ny, Nx, H, W)` (`uint16`),
        and when `on_the_fly_treatment=True`, pre-allocates `Brillouin/<map_name>/Treated HxW bins`
        (`Brillouin_type = 'Treatment'`) with chunked 4D DHO parameter arrays `(Ny, Nx, n_rows, 2*n_orders)`
        and 2D spatial summary maps `(Ny, Nx)`.
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")

        clean_name = map_name.strip().replace("/", "_")
        if not clean_name:
            raise ValueError("Map name cannot be empty.")
        parent_group = f"Brillouin/{clean_name}"

        ny, nx = max(1, int(grid.ny)), max(1, int(grid.nx))
        h, w = int(frame_shape[0]), int(frame_shape[1])
        bin_h, bin_w = max(1, int(bin_val_h)), max(1, int(bin_val_w))
        n_rows = h // bin_h
        n_orders = self.determine_n_orders(bin_val_h=bin_h, bin_val_w=bin_w)
        n_cols_dho = 2 * n_orders

        attrs = {
            "timestamp_start": datetime.datetime.now().isoformat(timespec="seconds"),
            "MEASURE.Sampling_Matrix_Size_(Nx,Ny,Nz)_()": f"{nx},{ny},1",
            "MEASURE.Field_Of_View_(X,Y,Z)_(um)": f"{grid.span_x_um:.2f},{grid.span_y_um:.2f},0.0",
            "x_start_um": float(grid.x_start_um),
            "y_start_um": float(grid.y_start_um),
            "x_center_um": float(grid.x_center_um),
            "y_center_um": float(grid.y_center_um),
            "dx_um": float(grid.dx_um),
            "dy_um": float(grid.dy_um),
            "nx": int(nx),
            "ny": int(ny),
            "scan_pattern": str(grid.scan_pattern),
            "exposure_s": float(exposure_s),
            "exposure_ms": float(exposure_s * 1e3),
            "frame_averaging": int(n_avg),
            "roi_hstart": int(roi[0]),
            "roi_hend": int(roi[1]),
            "roi_vstart": int(roi[2]),
            "roi_vend": int(roi[3]),
            "hardware_binning": int(roi[4]),
            "save_raw_data": bool(save_raw_data),
            "on_the_fly_treatment": bool(on_the_fly_treatment),
        }
        if extra_attrs:
            attrs.update(extra_attrs)

        with self._h5_lock:
            self.wrapper.add_dictionary(
                {
                    "Abscissa_0_1": {
                        "Name": "Y_um",
                        "Data": grid.y_coords_um,
                        "Units": "um",
                        "Dim_start": 0,
                        "Dim_end": 1,
                    },
                    "Abscissa_1_2": {
                        "Name": "X_um",
                        "Data": grid.x_coords_um,
                        "Units": "um",
                        "Dim_start": 1,
                        "Dim_end": 2,
                    },
                    "Attributes": attrs,
                },
                parent_group=parent_group,
                create_group=True,
                Brillouin_type_parent_group="Measure",
                overwrite=overwrite,
            )

            treated_group_path: Optional[str] = None
            with h5py.File(self.h5_path, "a") as f:
                grp = f[parent_group]

                for ds_name in ("stage_X_measured_um", "stage_Y_measured_um", "Raw_data"):
                    if ds_name in grp:
                        del grp[ds_name]

                ds_x = grp.create_dataset(
                    "stage_X_measured_um",
                    shape=(ny, nx),
                    dtype=np.float64,
                    fillvalue=np.nan,
                )
                ds_x.attrs["Brillouin_type"] = "Other"
                ds_x.attrs["Units"] = "um"

                ds_y = grp.create_dataset(
                    "stage_Y_measured_um",
                    shape=(ny, nx),
                    dtype=np.float64,
                    fillvalue=np.nan,
                )
                ds_y.attrs["Brillouin_type"] = "Other"
                ds_y.attrs["Units"] = "um"

                if save_raw_data:
                    ds_raw = grp.create_dataset(
                        "Raw_data",
                        shape=(ny, nx, h, w),
                        dtype=np.uint16,
                        chunks=(1, 1, h, w),
                    )
                    ds_raw.attrs["Brillouin_type"] = "Raw_data"

                if on_the_fly_treatment and self.rb_calib_result is not None:
                    t_name = f"Treated {bin_h}x{bin_w} bins"
                    if t_name in grp:
                        del grp[t_name]
                    t_grp = grp.create_group(t_name)
                    t_grp.attrs["Brillouin_type"] = "Treatment"
                    t_grp.attrs["bin_val_h"] = int(bin_h)
                    t_grp.attrs["bin_val_w"] = int(bin_w)
                    t_grp.attrs["n_orders"] = int(n_orders)
                    t_grp.attrs["FSR_GHz"] = float(self.rb_calib_result["FSR_cal"])
                    t_grp.attrs["refractive_index"] = float(refractive_index)
                    treated_group_path = f"{parent_group}/{t_name}"

                    # 4D chunked DHO datasets: (ny, nx, n_rows, 2*n_orders)
                    dho_specs = [
                        ("Shift", "Shift"),
                        ("Shift error", "Shift_err"),
                        ("Linewidth", "Linewidth"),
                        ("Linewidth error", "Linewidth_err"),
                        ("x_coords", "Other"),
                        ("y_coords", "Other"),
                    ]
                    for ds_name, b_type in dho_specs:
                        ds = t_grp.create_dataset(
                            ds_name,
                            shape=(ny, nx, n_rows, n_cols_dho),
                            dtype=np.float64,
                            chunks=(1, 1, n_rows, n_cols_dho),
                            fillvalue=np.nan,
                        )
                        ds.attrs["Brillouin_type"] = b_type

                    # 2D spatial summary maps: (ny, nx)
                    for map_ds_name in (
                        "map_mean_shift_GHz",
                        "map_mean_linewidth_GHz",
                        "map_Shift_0_GHz",
                        "map_V_s_m_per_s",
                        "map_df_coarse_GHz",
                    ):
                        ds_m = t_grp.create_dataset(
                            map_ds_name,
                            shape=(ny, nx),
                            dtype=np.float64,
                            fillvalue=np.nan,
                        )
                        ds_m.attrs["Brillouin_type"] = "Other"

                f.flush()

        return parent_group, treated_group_path, n_orders

    def write_map_point(
        self,
        map_group_path: str,
        iy: int,
        ix: int,
        x_meas_um: Optional[float],
        y_meas_um: Optional[float],
        frame: Optional[np.ndarray],
        save_raw_data: bool,
        treated_group_path: Optional[str] = None,
        point_fit: Optional[Dict[str, Any]] = None,
    ):
        """
        Writes a single acquired (and optionally fitted) spatial point `(iy, ix)`
        to disk in `Brillouin/<map_name>` and flushes the HDF5 file.
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")

        with self._h5_lock:
            with h5py.File(self.h5_path, "a") as f:
                grp = f[map_group_path]
                if x_meas_um is not None and "stage_X_measured_um" in grp:
                    grp["stage_X_measured_um"][iy, ix] = float(x_meas_um)
                if y_meas_um is not None and "stage_Y_measured_um" in grp:
                    grp["stage_Y_measured_um"][iy, ix] = float(y_meas_um)
                if save_raw_data and frame is not None and "Raw_data" in grp:
                    grp["Raw_data"][iy, ix, :, :] = np.asarray(frame, dtype=np.uint16)

                if treated_group_path and point_fit is not None and treated_group_path in f:
                    t_grp = f[treated_group_path]
                    dho = point_fit["dho_results"]
                    t_grp["Shift"][iy, ix, :, :] = dho["shift_clipped"]
                    t_grp["Shift error"][iy, ix, :, :] = dho["shift_error"]
                    t_grp["Linewidth"][iy, ix, :, :] = dho["linewidth_clipped"]
                    t_grp["Linewidth error"][iy, ix, :, :] = dho["linewidth_error"]
                    t_grp["x_coords"][iy, ix, :, :] = dho["x_coords"]
                    t_grp["y_coords"][iy, ix, :, :] = dho["y_coords"]

                    t_grp["map_mean_shift_GHz"][iy, ix] = float(point_fit["mean_shift_ghz"])
                    t_grp["map_mean_linewidth_GHz"][iy, ix] = float(point_fit["mean_lw_ghz"])
                    t_grp["map_Shift_0_GHz"][iy, ix] = float(point_fit["Shift_0_ghz"])
                    t_grp["map_V_s_m_per_s"][iy, ix] = float(point_fit["V_s_ms"])
                    t_grp["map_df_coarse_GHz"][iy, ix] = float(point_fit["df_coarse"])

                f.flush()
