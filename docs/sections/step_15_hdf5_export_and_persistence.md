# Step 15 — HDF5 Data Storage & Results Persistence

## Summary and Input/Output
- **Purpose**: Persists all calibrated wavevector datasets, physical parameters, and statistical fit uncertainties into the standardized HDF5 container for downstream thermodynamic, viscoelastic, and multi-orientation analysis.
- **Inputs**: Extracted wavevector results dictionary (`q_vector`, `theta`, `r_sph`, `M_theta`, `Shift_0`, `V_s`, `V_s_err`, `opt_params`).
- **Outputs**: Formatted HDF5 group `Brillouin/<sample>/Treated <bin_h>x<bin_w> bins` containing datasets and attached metadata attributes.

---

## Algorithm
1. **Target Group Navigation**:
   Accesses or creates the group `Brillouin/<sample>/Treated 4x2 bins` within `Measures.h5`.
2. **Dataset Creation & Overwrite**:
   Writes 1D flattened arrays with double-precision floating-point format:
   - `q_vector`: Acoustic wavevector magnitude ($m^{-1}$).
   - `theta`: Internal scattering angle (radians).
   - `r_sph`: Spherically symmetric radius (pixels).
3. **Attribute Metadata Stamping**:
   Attaches scalar calibration and physical parameters directly to the HDF5 group:
   - `M_theta`, `M_theta_err`: Angular magnification factor and uncertainty (px).
   - `Shift_0`, `Shift_0_err`: Normal incidence Brillouin shift and uncertainty (GHz).
   - `V_s`, `V_s_err`: Longitudinal speed of sound and uncertainty (m/s).
   - `xc_ellipse`, `yc_ellipse`: Concentric ellipse center coordinates (px).
   - `phi_ellipse_deg`: Principal director axis angle (degrees).
   - `gamma_ellipse`: Collection aperture aspect ratio $\gamma$.
4. **Validation**:
   Re-reads the written datasets and attributes to confirm data integrity.

---

## Physical Rationale
Angle-resolved Brillouin experiments often span multiple samples, temperatures, and orientation angles (e.g. `plastic_0`, `plastic_60`, `plastic_120`, `plastic_180`). Saving the complete metadata — including the physical errors on $M_\theta$, $S_0$, and $V_s$ — ensures full experimental reproducibility and enables automated downstream scripts to load calibrated results without recomputing non-linear fits.

---

## Code Implementation
- **Function**: `save_q_vector_results(wrapper, path_sample, bin_val_h, bin_val_w, q_results)`
- **Function**: `retrieve_q_vector_results(wrapper, path_sample, bin_val_h, bin_val_w)`
- **Module**: [`arVIPAspectro/q_vector.py`](file:/arVIPAspectro/q_vector.py#L205-L270)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L565-L572)
