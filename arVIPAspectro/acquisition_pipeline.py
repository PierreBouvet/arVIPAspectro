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
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import h5py
from HDF5_BLS import Wrapper

from . import config
from . import laser_calib


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

        return {
            "h5_path": self.h5_path,
            "has_laser": self.laser_image is not None,
            "has_laser_calib": self.laser_calib_result is not None,
            "has_rb": self.rb_image is not None,
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
        elif role == self.ROLE_RB:
            self.rb_image = np.asarray(image)
            self.rb_meta = attrs
        elif role == self.ROLE_REF_SAMPLE:
            self.ref_sample_image = np.asarray(image)
            self.ref_sample_meta = attrs

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
            x0_fit, y0_fit, xc_gnom, yc_gnom = laser_calib.retrieve_vipa_geometry(self.wrapper, path_laser)

            # Recompute full calibration dict for diagnostic plotting
            full_cal = laser_calib.calibrate_anamorphic_order_surface(
                self.wrapper,
                path_laser,
                ax,
                axs,
                ay,
                ak,
                x0_fit,
                y0_fit,
                h_img=h,
                nb_laser_modes=len(ref_fringes),
                fsr_vipa=config.FSR_VIPA,
            )
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

    def list_map_groups(self) -> List[str]:
        """Lists existing `Measure` map groups under `Brillouin/`."""
        if not self.is_open:
            return []
        maps = []
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
        overwrite: bool = True,
    ) -> str:
        """
        Creates `Brillouin/<map_name>` (`Brillouin_type = 'Measure'`) in the HDF5 file,
        stores `Y_um` (`Abscissa_0_1`) and `X_um` (`Abscissa_1_2`) coordinate vectors,
        allocates 2D measured stage coordinate grids (`stage_X_measured_um`, `stage_Y_measured_um`),
        and optionally pre-allocates a chunked 4D `Raw_data` dataset `(Ny, Nx, H, W)` (`uint16`).
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")

        clean_name = map_name.strip().replace("/", "_")
        if not clean_name:
            raise ValueError("Map name cannot be empty.")
        parent_group = f"Brillouin/{clean_name}"

        ny, nx = max(1, int(grid.ny)), max(1, int(grid.nx))
        h, w = int(frame_shape[0]), int(frame_shape[1])

        attrs = {
            "timestamp_start": datetime.datetime.now().isoformat(timespec="seconds"),
            "MEASURE.Sampling_Matrix_Size_(Nx,Ny,Nz)_()": f"{nx},{ny},1",
            "MEASURE.Field_Of_View_(X,Y,Z)_(um)": f"{grid.span_x_um:.2f},{grid.span_y_um:.2f},0.0",
            "x_start_um": float(grid.x_start_um),
            "y_start_um": float(grid.y_start_um),
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
        }

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

        # Pre-allocate measured stage position arrays and optional 4D chunked Raw_data
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

            f.flush()

        return parent_group

    def write_map_point(
        self,
        map_group_path: str,
        iy: int,
        ix: int,
        x_meas_um: Optional[float],
        y_meas_um: Optional[float],
        frame: Optional[np.ndarray],
        save_raw_data: bool,
    ):
        """
        Writes a single acquired spatial point `(iy, ix)` to disk in `Brillouin/<map_name>`
        and flushes the HDF5 file so data is immediately persisted.
        """
        if not self.is_open:
            raise RuntimeError("No HDF5 session file is open.")

        with h5py.File(self.h5_path, "a") as f:
            grp = f[map_group_path]
            if x_meas_um is not None and "stage_X_measured_um" in grp:
                grp["stage_X_measured_um"][iy, ix] = float(x_meas_um)
            if y_meas_um is not None and "stage_Y_measured_um" in grp:
                grp["stage_Y_measured_um"][iy, ix] = float(y_meas_um)
            if save_raw_data and frame is not None and "Raw_data" in grp:
                grp["Raw_data"][iy, ix, :, :] = np.asarray(frame, dtype=np.uint16)
            f.flush()
