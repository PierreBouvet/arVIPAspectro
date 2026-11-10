# Step 11 — Sample In-Situ Frequency Map Self-Calibration

## Summary and Input/Output
- **Purpose**: Dynamically refines the master 2D frequency map using the in-situ Stokes/Anti-Stokes Brillouin peak pairs of the sample itself, correcting for experimental alignment drift, zero-frequency offset, and higher-order imaging lens aberrations.
- **Inputs**: Base frequency map $\nu_{\mathrm{map}}(x, y)$, sample image $I_{\mathrm{sample}}(x, y)$ (e.g. `water_5s` or `plastic_5s`), nominal shift and linewidth bounds.
- **Outputs**: Self-calibrated in-situ frequency map $\nu_{\mathrm{corr}}(x, y) \in \mathbb{R}^{2304 \times 2304}$ with Stokes/Anti-Stokes symmetry RMS reduced below $35\text{ MHz}$.

---

## Algorithm
The self-calibration operates in two sequential stages:

### Stage 1 — Elastic Zero-Offset Map $\delta\nu_{\mathrm{zero}}(y, k_{\mathrm{ord}})$
1. Any common-mode frequency error $\delta\nu_{\mathrm{zero}}$ in the nominal elastic position $\nu = k \cdot \mathrm{FSR}$ shifts the Stokes peak ($+1$) upward and the Anti-Stokes peak ($-1$) downward:
   $$\Delta\nu_S = \Delta\nu_{\mathrm{true}} + \delta\nu_{\mathrm{zero}}, \qquad \Delta\nu_{AS} = \Delta\nu_{\mathrm{true}} - \delta\nu_{\mathrm{zero}} \implies \delta\nu_{\mathrm{mid}} = \frac{\Delta\nu_S - \Delta\nu_{AS}}{2}$$
2. Runs a fast initial DHO fit on $4 \times 2$ binned data to measure the local midpoint $\delta\nu_{\mathrm{mid}}(y, k)$ across all visible orders $k \in \{1, \dots, 10\}$.
3. Fits a smooth 2D 3rd-order polynomial base plus Gaussian-filtered spline ($\sigma_y = 6\text{ bins}$) along vertical rows to model $\delta\nu_{\mathrm{zero}}(y, k)$.
4. Subtracts $\delta\nu_{\mathrm{zero}}(y, k)$ from each order $k_{\mathrm{ord}} = \operatorname{clip}(\operatorname{round}(\nu_{\mathrm{init}}/\mathrm{FSR}), 1, 10)$.

### Stage 2 — Higher-Order Dispersion Jacobian Scale $\alpha_{\mathrm{disp}}(y, k_{\mathrm{ord}})$
1. Decomposes the order-averaged Brillouin shift $\Delta\nu_{\mathrm{avg}}(y, k) = \frac{1}{2}(\Delta\nu_S + \Delta\nu_{AS})$ into:
   - A physical 6-parameter **2nd-order 2D elliptical paraboloid** $Z_{\mathrm{quad}}(x, y)$ fitted via 2-step Iteratively Reweighted Least Squares (IRLS) on the unclipped central region ($y \in [250, 1850]\text{ px}$, Fringes $0\dots 7$).
   - A stationary higher-order residual $\Delta\nu_{\mathrm{inst}}(y, k) = \mathcal{G}_{\sigma_y = 6}\big(\Delta\nu_{\mathrm{avg}} - Z_{\mathrm{quad}}\big)$ capturing 4th-order spherical aberration wing flares at sensor edges and central VIPA blaze distortion.
2. Scales the intra-order dispersion Jacobian:
   $$\alpha_{\mathrm{disp}}(y, k) = \frac{Z_{\mathrm{quad}}(y, k)}{Z_{\mathrm{quad}}(y, k) + \Delta\nu_{\mathrm{inst}}(y, k)}$$
3. Yields the final self-calibrated 2D frequency map:
   $$\nu_{\mathrm{corr}}(x, y) = k \cdot \mathrm{FSR} + \alpha_{\mathrm{disp}}(y, k) \cdot \Big(\nu_{\mathrm{map}}(x, y) - \mathrm{df}_{\mathrm{coarse}} - k \cdot \mathrm{FSR} - \delta\nu_{\mathrm{zero}}(y, k)\Big)$$

---

## Physical Rationale
Between calibrating the reference laser and mounting a scattering sample (e.g., placing a cuvette or plastic block), minor optical path perturbations occur:
- Mechanical drift and thermal expansion of the VIPA mount.
- Refractive index changes and beam displacement through sample cuvettes.
- Fine frequency drifts of the laser diode temperature controller ($\sim 10\text{--}50\text{ MHz}$).
Because Brillouin scattering generates Stokes and Anti-Stokes phonons of strictly identical energy ($\hbar\Omega$), any measured asymmetry between Stokes and Anti-Stokes shifts is an instrumental artifact. Using the sample's own spectral symmetry as an in-situ optical interferometer eliminates these drifts without requiring external recalibration.

---

## Physical Approximations
1. **Thermodynamic Stokes/Anti-Stokes Energy Equality**: In the classical thermal equilibrium regime ($k_B T \gg \hbar\Omega$, satisfied at room temperature where $k_B T \approx 6.2\text{ THz} \gg 10\text{ GHz}$), the acoustic phonon frequencies for creation (Stokes) and annihilation (Anti-Stokes) are rigorously identical: $\Omega_S = \Omega_{AS}$.

---

## Key Equations
- **Elastic Midpoint Splitting**:
  $$\delta\nu_{\mathrm{mid}}(y, k) = \frac{\Delta\nu_S(y, k) - \Delta\nu_{AS}(y, k)}{2}$$
- **Corrected Frequency Map**:
  $$\nu_{\mathrm{corr}}(x, y) = k\cdot\mathrm{FSR} + \alpha_{\mathrm{disp}}(y, k)\,\big[\nu_{\mathrm{map}}(x, y) - \mathrm{df} - k\cdot\mathrm{FSR} - \delta\nu_{\mathrm{zero}}(y, k)\big]$$

---

## Code Implementation
- **Function**: `calibrate_sample_frequency_map(freq_map_base, data_clean_in, ...)`
- **Module**: [`arVIPAspectro/spectrum_fitting.py`](file:/arVIPAspectro/spectrum_fitting.py#L18-L150)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L418-L445)
- **Diagnostic Plot**: Figure 11 (`11 - In-situ frequency map calibration.png`)
