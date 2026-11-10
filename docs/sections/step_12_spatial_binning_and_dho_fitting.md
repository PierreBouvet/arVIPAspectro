# Step 12 — Spatial Binning & Vectorized Batched DHO Peak Fitting

## Summary and Input/Output
- **Purpose**: Averages detector pixels into spatial bins to boost signal-to-noise ratio (SNR) and simultaneously fits the physical Damped Harmonic Oscillator (DHO) acoustic spectral response across all orders and spatial bins using a vectorized 3D batched Levenberg-Marquardt solver.
- **Inputs**: Net sample image $I_{\mathrm{net},\mathrm{sample}}(x, y)$, calibrated frequency map $\nu_{\mathrm{corr}}(x, y)$, binning factors ($H=4, W=2$).
- **Outputs**: 3D data cubes of Brillouin shift $\Delta\nu_B(y_b, k, \sigma)$, linewidth $\Gamma_B(y_b, k, \sigma)$, peak amplitude, background, fitting errors, and sub-pixel spatial coordinates $(x_b, y_b)$.

---

## Algorithm
1. **2D Block Binning**:
   Averages pixel blocks of height $H = 4\text{ px}$ and width $W = 2\text{ px}$ over both the baseline-subtracted sample image and the calibrated 2D frequency map, generating $N_y = 576$ vertical bins by $N_x = 1152$ horizontal bins.
2. **Local Spectral Window Extraction & Linear Chord Leveling**:
   For each spatial row bin $i \in \{0, \dots, 575\}$, order $k \in \{1, \dots, 10\}$, and branch $\sigma \in \{-1, +1\}$:
   - Extracts a 25-bin local frequency window centered on the expected peak position $k \cdot \mathrm{FSR} + \sigma \Delta\nu_{\mathrm{guess}}$.
   - Levels any local baseline slope between window boundary endpoints:
     $$\tilde{I}(j) = I(j) - \left( I_0 + \frac{I_{M-1} - I_0}{M - 1} j \right)$$
   - Maps the spectral axis to relative frequency shift: $\nu_{\mathrm{rel}} = |\nu - k \cdot \mathrm{FSR}|$.
3. **Vectorized Batched Levenberg-Marquardt Solver**:
   Stacks all $N_{\mathrm{peaks}} \approx 11,520$ windows into a 3D batch array of dimensions $(N_{\mathrm{peaks}}, M)$.
   - Evaluates the exact DHO model function:
     $$I_{\mathrm{DHO}}(\nu_{\mathrm{rel}}; b, a, \nu_0, \Gamma_B) = b + a \frac{(\Gamma_B \nu_0)^2}{(\nu_{\mathrm{rel}}^2 - \nu_0^2)^2 + (\Gamma_B \nu_{\mathrm{rel}})^2}$$
   - Evaluates the analytic Jacobian $\mathbf{J} \in \mathbb{R}^{N_{\mathrm{peaks}} \times M \times 4}$ containing exact partial derivatives $[\partial_b I, \partial_a I, \partial_a I, \partial_{\nu_0} I, \partial_{\Gamma_B} I]$.
   - Solves the damped normal equations $\big(\mathbf{J}^T \mathbf{J} + \lambda \text{diag}(\mathbf{J}^T \mathbf{J})\big) \Delta\mathbf{p} = \mathbf{J}^T \mathbf{r}$ across all peaks simultaneously using `np.einsum` and batched linear algebra.
   - Converges in $\sim 18$ iterations in **$< 0.5\text{ seconds}$** on standard multi-core CPUs.
4. **Sub-Pixel Coordinate Interpolation**:
   Interpolates the physical camera coordinates $(x, y)$ corresponding to the fitted peak frequency center $\nu_0$ using monotonic 1D cubic splines along each binned row.

---

## Physical Rationale
Brillouin scattering arises from optical diffraction off thermally excited acoustic sound waves (phonons) propagating in the material. In viscoelastic liquids and amorphous polymers, the dynamic structure factor $S(q, \omega)$ is governed by the damped acoustic harmonic oscillator response:
$$S(q, \omega) \propto \frac{\Gamma_B \omega_0^2}{(\omega^2 - \omega_0^2)^2 + (\Gamma_B \omega)^2}$$
Fitting an empirical Gaussian or symmetric Lorentzian systematically biases the linewidth because acoustic phonons have asymmetric tails vanishing as $\omega \to 0$ ($\propto \omega^2$). The exact DHO model accurately recovers the true phonon natural frequency $\nu_0$ and acoustic attenuation rate $\Gamma_B$.

---

## Physical Approximations
1. **Classical High-Temperature Phonon Occupation**: In the classical regime ($k_B T \gg \hbar\omega$), the Bose-Einstein thermal factor $[n(\omega) + 1] \approx k_B T / \hbar\omega$ cancels the pre-factor $\omega$, rendering the observed spectrum proportional to the imaginary susceptibility $\chi''(\omega)/\omega$, which yields the symmetric DHO intensity form.
2. **Single Longitudinal Acoustic Branch**: The sample is assumed isotropic (e.g. water or unoriented plastic), exhibiting a single longitudinal phonon branch with no transverse modes.

---

## Key Equations
- **Damped Harmonic Oscillator (DHO) Line Profile**:
  $$I_{\mathrm{DHO}}(\nu_{\mathrm{rel}}) = b + a \frac{(\Gamma_B \nu_0)^2}{(\nu_{\mathrm{rel}}^2 - \nu_0^2)^2 + (\Gamma_B \nu_{\mathrm{rel}})^2}$$
- **Phonon Damping Attenuation Relation**:
  $$\alpha_s = \frac{\pi \Gamma_B}{V_s}$$
- **Normal Equations for Vectorized LM**:
  $$(\mathbf{J}^T \mathbf{J} + \lambda \mathbf{D}) \Delta\mathbf{p} = \mathbf{J}^T (\mathbf{y} - \mathbf{f})$$

---

## Code Implementation
- **Function**: `fit_dho_batched(nu_cube, I_cube, p0_mat, ...)`
- **Function**: `fit_sample_dho_peaks(data_sample_clean, freq_corr, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L715-L870) and [`arVIPAspectro/spectrum_fitting.py`](file:/arVIPAspectro/spectrum_fitting.py#L152-L310)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L448-L485)
