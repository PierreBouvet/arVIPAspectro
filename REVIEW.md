# Physical Review: Potential Inconsistencies and Areas of Improvement in arVIPAspectro

> **Basis of Review**: This physical audit is based strictly on the theoretical documentation, optical derivations, and step specifications located in [`docs/`](docs/) and [`theory/`](theory/).
> **Objective**: Identify physical contradictions, approximations that break down under specific experimental conditions, mathematical inconsistencies in error propagation, and concrete areas of improvement.

---

## 1. VIPA Internal Snell's Law and Optical Path Difference Inconsistency (Step 5)

### Documentation Reference
- [`docs/sections/step_05_gnomonic_geometry_fit.md`](docs/sections/step_05_gnomonic_geometry_fit.md#L17-L46)
- [`docs/README.md`](docs/README.md#L296-L325)

### Physical & Mathematical Contradiction
In Step 5, the documentation defines the external directional cosine as $u(x, y; x_0, y_0) = \hat{\mathbf{k}} \cdot \hat{\mathbf{n}} = \cos\theta_{\mathrm{ext}}$. It then states:
$$\text{"By Snell's law, the cosine of the internal propagation angle } \theta_{\mathrm{int}} \text{ inside fused silica } (n_r) \text{ is:}$$
$$g(x, y; x_0, y_0) = \cos\theta_{\mathrm{int}} = \sqrt{u(x, y; x_0, y_0)^2 + n_r^2 - 1}"$$
And under "Key Equations", it writes:
$$2 n_r d \cdot g(x, y) = m \cdot \lambda_0 \implies g_m = \frac{m \lambda_0}{2 n_r d}$$

This contains a fundamental mathematical and physical contradiction:
1. **True Snell's Law**: 
   Since $n_{\mathrm{air}} \sin\theta_{\mathrm{ext}} = n_r \sin\theta_{\mathrm{int}}$ with $n_{\mathrm{air}} \approx 1$:
   $$\sin^2\theta_{\mathrm{int}} = \frac{\sin^2\theta_{\mathrm{ext}}}{n_r^2} = \frac{1 - u^2}{n_r^2}$$
   $$\cos^2\theta_{\mathrm{int}} = 1 - \sin^2\theta_{\mathrm{int}} = \frac{n_r^2 - 1 + u^2}{n_r^2} \implies \cos\theta_{\mathrm{int}} = \frac{\sqrt{u^2 + n_r^2 - 1}}{n_r}$$
2. **Incorrect Magnitude**:
   The quantity $g = \sqrt{u^2 + n_r^2 - 1}$ cannot be the cosine of an angle because for fused silica ($n_r \approx 1.45367$ at $780\text{ nm}$) and near-normal incidence ($u \approx 1$), $g \approx \sqrt{1 + 1.45367^2 - 1} = 1.45367 > 1$. The cosine of a real physical propagation angle must satisfy $|\cos\theta_{\mathrm{int}}| \le 1$.
3. **Double Counting of $n_r$**:
   If $g \equiv n_r \cos\theta_{\mathrm{int}}$, then the Fabry-Pérot optical path difference is:
   $$\Delta = 2 n_r d \cos\theta_{\mathrm{int}} = 2 d (n_r \cos\theta_{\mathrm{int}}) = 2 d \cdot g$$
   Writing $\Delta = 2 n_r d \cdot g$ as done in the documentation effectively multiplies by $n_r$ twice ($\Delta = 2 n_r^2 d \cos\theta_{\mathrm{int}}$), which is physically incorrect by a factor of $n_r \approx 1.454$.

### Areas of Improvement
- Clearly redefine $g(x, y)$ as the **reduced internal refractive factor**:
  $$g(x, y) \equiv n_r \cos\theta_{\mathrm{int}} = \sqrt{u(x, y)^2 + n_r^2 - 1}$$
- Correct the interference condition equation to:
  $$\Delta = 2 d \cdot g(x, y) = m \lambda_0 \implies g_m = \frac{m \lambda_0}{2 d}$$
  or explicitly divide by $n_r$ when defining $\cos\theta_{\mathrm{int}}$.

---

## 2. Gnomonic (Rectilinear / Tangent) vs. Aplanatic (Abbe Sine) Mapping Inconsistency (Step 5 vs. Step 14)

### Documentation Reference
- [`theory/abbe_sine_condition_and_pupil_mapping.md`](theory/abbe_sine_condition_and_pupil_mapping.md)
- [`docs/sections/step_05_gnomonic_geometry_fit.md`](docs/sections/step_05_gnomonic_geometry_fit.md#L11-L36)
- [`docs/sections/step_14_q_vector_and_elliptical_geometry.md`](docs/sections/step_14_q_vector_and_elliptical_geometry.md#L30-L55)

### Physical & Mathematical Contradiction
The documentation employs contradictory geometric mapping models for different optical stages without clarifying their boundary:
1. In **Step 5**, the lens following the VIPA is modeled as an ideal **rectilinear / gnomonic camera lens**:
   $$\hat{\mathbf{k}} = \frac{1}{\sqrt{f^2 + (x - x_0)^2 + (y - y_0)^2}} \begin{pmatrix} x - x_0 \\ y - y_0 \\ f \end{pmatrix}$$
   which assumes a flat principal plane where ray height is $r = f \tan\theta$.
2. In **Step 14** and `theory/abbe_sine_condition_and_pupil_mapping.md`, the documentation extensively argues that high-numerical-aperture lenses and Fourier collection systems must follow the **Abbe sine condition**:
   $$r_{\mathrm{pupil}} = F_{\mathrm{obj}} \sin\theta_{\mathrm{air}}$$
   and proves that $r = f \tan\theta$ introduces severe spherical aberration and coma at high NA.
3. In the collection train before the VIPA, the light from the sample passes through a high-NA objective ($r \propto \sin\theta$), but the VIPA output is imaged onto the detector by a $400\text{ mm}$ lens. The combination of these transformations creates a hybrid mapping that is neither purely gnomonic nor purely aplanatic.

### Areas of Improvement
- Explicitly state in the documentation that Step 5 uses the gnomonic tangent projection as a paraxial / geometric approximation for the VIPA output ray directions, and that any residual discrepancy between the tangent law and the actual camera lens projection is absorbed by the empirical 4th-degree polynomial surface in Step 6.
- Maintain a clear separation between the **collection objective mapping** ($r_{\mathrm{pupil}} = F_{\mathrm{obj}} \sin\theta_{\mathrm{air}}$) and the **spectrometer imaging lens mapping** ($x, y \leftrightarrow \theta_{\mathrm{VIPA}}$).

---

## 3. Rubidium "Saturated Absorption" Misnomer vs. Linear Doppler-Broadened Absorption & Optical Pumping (Step 7 & Step 10)

### Documentation Reference
- [`docs/sections/00_overview.md`](docs/sections/00_overview.md#L19)
- [`docs/sections/step_07_rubidium_absorption_model.md`](docs/sections/step_07_rubidium_absorption_model.md#L4-L47)
- [`docs/sections/step_10_fsr_and_frequency_offset_calibration.md`](docs/sections/step_10_fsr_and_frequency_offset_calibration.md)

### Physical & Mathematical Contradiction
1. **Misnomer ("Saturated Absorption")**:
   The documentation repeatedly describes Step 7 as modeling the *"theoretical saturated transmission spectrum $T_{\mathrm{Rb}}(\Delta\nu)$"*.
   In atomic physics, *Saturated Absorption Spectroscopy (SAS)* refers specifically to a Doppler-free pump-probe configuration producing sub-Doppler Lamb dips with natural linewidth ($\sim 6\text{ MHz}$).
   The equations in Step 7 evaluate a single-pass Beer-Lambert transmission through a vapor cell:
   $$T_{\mathrm{Rb}}(\Delta\nu) = \exp(-\alpha(\Delta\nu) L)$$
   where $\alpha(\Delta\nu)$ is the convolution of thermal Maxwell-Boltzmann Gaussian Doppler broadening ($\mathrm{FWHM}_D \approx 515\text{ MHz}$) and natural Lorentzian decay. This is a **linear Doppler-broadened absorption spectrum**, not a saturated absorption spectrum.
2. **Optical Pumping Distortion**:
   The theoretical weights $w_m$ assume statistical equilibrium ground-state populations:
   - $F=2: 5/12$ and $F=3: 7/12$ for $^{85}\text{Rb}$
   - $F=1: 3/8$ and $F=2: 5/8$ for $^{87}\text{Rb}$
   In reality, even low-intensity continuous-wave (CW) laser light tuned near the $D_2$ line induces strong optical hyperfine pumping (atoms get trapped in the uncoupled ground hyperfine level), which alters the relative depths of the absorption doublets.
3. **Cell Temperature Inconsistency**:
   `docs/sections/step_07_rubidium_absorption_model.md` specifies $T = 70.0^\circ\text{C}$, whereas earlier documentation (`algorithm_description.md`) used $T = 100^\circ\text{C}$. Between $70^\circ\text{C}$ and $100^\circ\text{C}$, the saturated vapor pressure of Rubidium increases by a factor of $\approx 8$ (from $\sim 1.1 \times 10^{-4}\text{ Torr}$ to $\sim 9.0 \times 10^{-4}\text{ Torr}$), drastically increasing the optical depth and broadening the absorption profiles.

### Areas of Improvement
- Rename references from "saturated transmission" to **"Doppler-broadened linear atomic absorption"**.
- Standardize the cell temperature across all documentation and configuration files, adding a note on how temperature uncertainty affects absorption depth versus line center positions.
- In Step 10, emphasize that the cross-correlation relies on the atomic hyperfine splittings (which are invariant) rather than absolute optical depth, making the FSR calibration robust against optical pumping and vapor pressure variations.

---

## 4. Step 11 Stage 2: Circular Reasoning & Erasure of Acoustic Dispersion, Anisotropy, and Viscoelastic Relaxation

### Documentation Reference
- [`docs/sections/step_11_insitu_frequency_self_calibration.md`](docs/sections/step_11_insitu_frequency_self_calibration.md#L20-L28)
- [`docs/sections/step_14_q_vector_and_elliptical_geometry.md`](docs/sections/step_14_q_vector_and_elliptical_geometry.md#L30-L42)

### Physical & Mathematical Contradiction
This is the most critical physical inconsistency in the data-processing pipeline:
1. **The Forcing Mechanism**:
   In Step 11 Stage 2, the pipeline takes the measured order-averaged Brillouin shift $\Delta\nu_{\mathrm{avg}}(x, y) = \frac{1}{2}(\Delta\nu_S + \Delta\nu_{AS})$, fits a 6-parameter **2nd-order 2D elliptical paraboloid** $Z_{\mathrm{quad}}(x, y)$ to it, defines any residual deviation as an "instrumental aberration":
   $$\Delta\nu_{\mathrm{inst}}(y, k) = \mathcal{G}_{\sigma}\big(\Delta\nu_{\mathrm{avg}} - Z_{\mathrm{quad}}\big)$$
   and then rescales the local frequency map Jacobian:
   $$\alpha_{\mathrm{disp}}(y, k) = \frac{Z_{\mathrm{quad}}(y, k)}{Z_{\mathrm{quad}}(y, k) + \Delta\nu_{\mathrm{inst}}(y, k)}$$
2. **Circular Reasoning with Step 14 Kinematics**:
   - In Step 12, the DHO peaks are refitted on this modified frequency map $\nu_{\mathrm{corr}}(x, y)$. By construction, the resulting Brillouin shifts are forced to follow $Z_{\mathrm{quad}}(x, y)$.
   - However, in Step 14, the pipeline attempts to fit the physical Abbe-sine kinematic dispersion:
     $$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right) \approx S_0 \left(1 - \frac{r_{\mathrm{sph}}^2}{8 M_\theta^2} - \frac{5 r_{\mathrm{sph}}^4}{128 M_\theta^4} - \dots\right)$$
   - Step 11 Stage 2 already stripped away all higher-order terms ($\propto r_{\mathrm{sph}}^4, r_{\mathrm{sph}}^6$), misidentifying them as "instrumental aberrations" $\Delta\nu_{\mathrm{inst}}$. Consequently, Step 14 is fitting kinematic parameters ($M_\theta, S_0$) to data artificially coerced into a 2nd-order paraboloid.
3. **Erasure of Sample Physics (Acoustic Relaxation & Anisotropy)**:
   - **Viscoelastic Relaxation**: For samples undergoing structural/viscoelastic relaxation in the GHz hypersonic regime (such as glycerol near its glass transition, polymers, or supercooled liquids), the speed of sound is frequency-dependent:
     $$V_s^2(\omega) = V_0^2 + (V_\infty^2 - V_0^2) \frac{\omega^2 \tau^2}{1 + \omega^2 \tau^2}$$
     As the scattering angle $\theta$ changes, $\omega$ changes, producing a non-linear acoustic dispersion $\Omega(q) \neq V_s \cdot q$. Step 11 Stage 2 misinterprets this genuine physical relaxation as an optical defect and flattens it out!
   - **Acoustic Anisotropy**: If the sample is anisotropic (e.g., solid crystals, oriented polymers, or liquid crystals), the acoustic velocity depends on the acoustic wavevector propagation direction relative to the sample's elastic stiffness tensor $c_{ijkl}$. Step 11 Stage 2 forces the shift into an isotropic/elliptical paraboloid, destroying anisotropic signatures.

### Areas of Improvement
- **Decouple Calibration from Unknown Samples**:
  Perform Step 11 Stage 2 (instrumental dispersion calibration) **only on a known, purely isotropic, non-relaxing reference sample** (such as pure water at $20^\circ\text{C}$). The resulting Jacobian correction map $\alpha_{\mathrm{disp}}(x, y)$ should then be stored as a fixed instrumental transfer function and applied to unknown samples without refitting $Z_{\mathrm{quad}}$.
- **Restrict In-Situ Self-Calibration on Unknown Samples to Stage 1**:
  On unknown or relaxing samples (e.g. glycerol, plastics), execute **only Stage 1** ($\delta\nu_{\mathrm{zero}}$ elastic zero locking via Stokes/Anti-Stokes symmetry), which relies strictly on the thermodynamic identity $\hbar\Omega_S = \hbar\Omega_{AS}$ and is completely independent of the sample's acoustic dispersion or anisotropy.
- **Implement a Generalized Dispersion Model**:
  If Stage 2 is ever refined using the sample itself, replace the empirical paraboloid with a physical model accounting for viscoelastic relaxation ($\tau$) or elastic stiffness tensor components.

---

## 5. Step 12: Direct DHO Fitting Without Instrumental Response Function (IRF) Deconvolution

### Documentation Reference
- [`docs/sections/step_12_spatial_binning_and_dho_fitting.md`](docs/sections/step_12_spatial_binning_and_dho_fitting.md#L20-L48)
- [`docs/README.md`](docs/README.md#L729-L747)

### Physical & Mathematical Contradiction
1. **Instrumental Linewidth vs. True Phonon Lifetime**:
   Step 12 fits the raw binned spectral profiles directly to an unconvoluted Damped Harmonic Oscillator (DHO) model:
   $$I_{\mathrm{DHO}}(\nu_{\mathrm{rel}}) = b + a \frac{(\Gamma_B \nu_0)^2}{(\nu_{\mathrm{rel}}^2 - \nu_0^2)^2 + (\Gamma_B \nu_{\mathrm{rel}})^2}$$
   and relates the fitted linewidth $\Gamma_B$ directly to the acoustic attenuation:
   $$\alpha_s = \frac{\pi \Gamma_B}{V_s}$$
2. **Neglect of the VIPA Instrumental Response**:
   A VIPA spectrometer has a finite instrumental resolution function (IRF), typically an Airy/Lorentzian profile with $\mathrm{FWHM}_{\mathrm{IRF}} \approx 500\text{--}800\text{ MHz}$.
   - For low-viscosity liquids like water, the intrinsic acoustic damping at $5\text{ GHz}$ is $\Gamma_{\mathrm{intrinsic}} \approx 50\text{--}150\text{ MHz}$.
   - The observed linewidth in the experiment is $\approx 800\text{ MHz}$, meaning **$> 80\%$ of the fitted linewidth is instrumental broadening, not sample acoustic damping!**
   - Directly calculating $\alpha_s = \frac{\pi \Gamma_B}{V_s}$ from the raw fitted $\Gamma_B$ overestimates the physical acoustic attenuation of water by an order of magnitude.
3. **Peak Position Pulling ($\nu_0$)**:
   The VIPA instrumental point-spread function is asymmetric along the dispersion axis $x$ (due to exponential decay of the beam leaking from the VIPA coating and beam clipping). Convolving an asymmetric IRF with a DHO profile shifts the apparent peak maximum, introducing a systematic bias into $\nu_0$ that varies with peak width.

### Areas of Improvement
- **Deconvolution Framework**:
  Extract the instrumental response function $I_{\mathrm{IRF}}(\nu)$ directly from the monochromatic laser reference fringes measured in Step 2.
- **Voigt-DHO Formulation**:
  Fit the observed line with a convolution model:
  $$I_{\mathrm{obs}}(\nu) = [I_{\mathrm{DHO}}(\Gamma_{\mathrm{intrinsic}}, \nu_0) * I_{\mathrm{IRF}}](\nu) + b$$
  or apply a pseudo-Voigt linewidth correction:
  $$\Gamma_{\mathrm{intrinsic}} \approx \Gamma_{\mathrm{fit}} - \frac{\Gamma_{\mathrm{IRF}}^2}{\Gamma_{\mathrm{fit}}}$$
- **Clarify Terminology**:
  In the documentation, explicitly distinguish between the **apparent linewidth** $\Gamma_{\mathrm{app}}$ and the **intrinsic acoustic phonon damping** $\Gamma_B$.

---

## 6. Step 12 & Step 14: Stokes/Anti-Stokes Coordinate Averaging and VIPA Dispersion-Angle Coupling

### Documentation Reference
- [`docs/sections/step_12_spatial_binning_and_dho_fitting.md`](docs/sections/step_12_spatial_binning_and_dho_fitting.md#L26)
- [`docs/sections/step_14_q_vector_and_elliptical_geometry.md`](docs/sections/step_14_q_vector_and_elliptical_geometry.md#L18)

### Physical & Mathematical Contradiction
1. **Spatial Separation on Detector**:
   Stokes ($\nu_0 - \nu_B$) and Anti-Stokes ($\nu_0 + \nu_B$) peaks from the same order $k$ are separated in optical frequency by $2\nu_B \approx 10\text{--}22\text{ GHz}$.
   Because the VIPA disperses frequencies horizontally along $x$ with a dispersion of $\sim 0.5\text{ GHz/px}$, the Stokes and Anti-Stokes peaks do NOT fall on the same detector pixels; they are physically separated along $x$ by $20\text{--}45\text{ pixels}$.
2. **Inconsistent Spatial Averaging**:
   Step 14 averages their coordinates:
   $$x_{\mathrm{avg}} = \frac{x_S + x_{AS}}{2}, \qquad y_{\mathrm{avg}} = \frac{y_S + y_{AS}}{2}$$
   and then evaluates the collection pupil radius $r_{\mathrm{sph}}(x_{\mathrm{avg}}, y_{\mathrm{avg}})$.
   - While $x_{\mathrm{avg}}$ approximates the unshifted Rayleigh position, the VIPA input collection angle $\theta_{\mathrm{in}, x}$ and the frequency shift $\Delta\nu$ are coupled along the horizontal axis $x$.
   - A ray entering the VIPA at an off-axis angle $\theta_x$ suffers an additional angular shift through the etalon that combines non-linearly with its frequency dispersion. Averaging $x_S$ and $x_{AS}$ assumes that the VIPA dispersion is strictly linear and symmetric around the elastic frequency, ignoring higher-order anamorphic distortion across the $2\nu_B$ gap.

### Areas of Improvement
- Document the VIPA spatio-spectral coupling relation along $x$:
  $$\theta_{\mathrm{out}, x}(\nu, \theta_{\mathrm{in}, x})$$
- Verify that the horizontal coordinate separation $\Delta x = x_S - x_{AS}$ scales linearly with $2\nu_B / (\partial\nu / \partial x)$, and quantify the residual distortion on $r_{\mathrm{sph}}$.

---

## 7. Step 14: Elliptical Geometry & Principal Axis Rotation ($\phi = 90^\circ$ vs. Free Fitting $\approx -68.4^\circ$)

### Documentation Reference
- [`docs/sections/step_14_q_vector_and_elliptical_geometry.md`](docs/sections/step_14_q_vector_and_elliptical_geometry.md#L20-L26)
- [`theory/refractive_index_magnification_derivation.md`](theory/refractive_index_magnification_derivation.md#L39-L50)

### Physical & Mathematical Contradiction
1. **Physical Origin of Anamorphism**:
   The circular collection cone from the objective transforms into an ellipse on the detector because of the anamorphic optics in the VIPA spectrometer:
   - Horizontal axis ($x$): cylindrical focusing into the VIPA input slit and angular dispersion inside the etalon.
   - Vertical axis ($y$): standard spherical imaging.
   If all optical components (VIPA etalon, cylindrical lenses, camera sensor) are aligned with the laboratory coordinate system, the principal axes of this anamorphic transformation **must be strictly parallel to the horizontal ($x$) and vertical ($y$) detector axes**, corresponding to $\phi = 0^\circ$ or $\phi = 90^\circ$.
2. **Unphysical Free Optimization Angle**:
   When $\phi$ is fitted freely without constraints, Nelder-Mead yields $\phi \approx -68.4^\circ$ to $-73^\circ$ (as documented in `docs/sections/step_14_q_vector_and_elliptical_geometry.md` and `algorithm_description.md`).
   - A rotation angle of $-68.4^\circ$ cannot arise from aligned cylindrical/VIPA optics.
   - It arises as an artifact caused by:
     - Asymmetric spatial sampling (interference orders are cropped unevenly by the rectangular sensor mask).
     - Residual baseline gradients.
     - Coupling between horizontal frequency shift and vertical spatial profile.
   - Enforcing `fixed_ellipse_angle_deg = 90.0` was introduced to prevent this unphysical rotation, but this exposes an underlying tension: the data does not naturally minimize variance at $90^\circ$ due to asymmetric sampling.

### Areas of Improvement
- Distinguish between **instrumental anamorphism** (which must be fixed at $\phi = 90^\circ$ based on hardware geometry) and **sample-induced acoustic anisotropy** (which can rotate the ellipse if the sample has anisotropic elastic stiffness tensors).
- When analyzing isotropic media (water, unaligned glycerol), fix $\phi = 90.0^\circ$ strictly and mask data symmetrically in $(X', Y')$ to prevent boundary clipping from biasing $(x_c, y_c)$ and $\gamma$.

---

## 8. Step 14: Mathematical Inconsistency in Sound Velocity ($V_s$) Error Propagation

### Documentation Reference
- [`docs/sections/step_14_q_vector_and_elliptical_geometry.md`](docs/sections/step_14_q_vector_and_elliptical_geometry.md#L68-L79)
- [`theory/refractive_index_magnification_derivation.md`](theory/refractive_index_magnification_derivation.md#L55-L75)

### Physical & Mathematical Contradiction
In Step 14, the longitudinal speed of sound $V_s$ is calculated using the normal backscattering shift $S_0$:
$$V_s = \frac{\lambda_0 \cdot S_0}{2 n}$$
However, look at the error propagation formula presented in Step 14:
$$\sigma_{V_s}^2 = J_{S_0}^2 \sigma_{S_0}^2 + J_{M_\theta}^2 \sigma_{M_\theta}^2 + 2 J_{S_0} J_{M_\theta} \mathrm{cov}(S_0, M_\theta)$$
where:
$$J_{S_0} = \frac{\partial V_s}{\partial S_0} = \frac{\lambda_0}{2 n}, \qquad J_{M_\theta} = \frac{\partial \widehat{V}_s}{\partial M_\theta} = 2\pi \frac{N' D - N D'}{D^2}$$
1. **Mathematical Inconsistency**:
   By definition, $V_s = \frac{\lambda_0 S_0}{2 n}$. The partial derivative of this expression with respect to $M_\theta$ is identically zero:
   $$\frac{\partial}{\partial M_\theta}\left( \frac{\lambda_0 S_0}{2 n} \right) \equiv 0$$
2. **Mixing Two Different Estimators**:
   The formula computes $J_{M_\theta}$ from an **entirely different linear regression estimator**:
   $$\widehat{V}_s(M_\theta) = 2\pi \frac{\sum_i w_i q_i(M_\theta) \cdot \mathrm{Shift}_i}{\sum_i w_i q_i(M_\theta)^2}$$
   Evaluating nominal $V_s$ from $S_0$ while evaluating its variance from a hybrid of $S_0$ and the dispersion slope estimator $\widehat{V}_s(M_\theta)$ violates the mathematical principles of Gaussian uncertainty propagation.

### Areas of Improvement
- Either:
  1. Define $V_s \equiv \frac{\lambda_0 S_0}{2 n}$, in which case the uncertainty is strictly:
     $$\sigma_{V_s} = \frac{\lambda_0}{2 n} \sigma_{S_0}$$
     (plus uncertainty on $n$ if $n$ is experimentally measured).
  2. Or define $V_s$ as the **global linear dispersion slope** $\widehat{V}_s$ over all $(q, \mathrm{Shift})$ points, and evaluate its full Jacobian covariance consistently:
     $$\widehat{V}_s = 2\pi \frac{\sum w_i q_i \mathrm{Shift}_i}{\sum w_i q_i^2}$$

---

## 9. Step 14: Refractive Index ($n$), Angular Magnification ($M_\theta$), and Velocity ($V_s$) Coupling

### Documentation Reference
- [`theory/refractive_index_magnification_derivation.md`](theory/refractive_index_magnification_derivation.md#L65-L120)
- [`docs/sections/step_14_q_vector_and_elliptical_geometry.md`](docs/sections/step_14_q_vector_and_elliptical_geometry.md#L30-L42)

### Physical & Mathematical Contradiction
1. **Exact Scaling Identity**:
   `theory/refractive_index_magnification_derivation.md` proves that under the Abbe sine condition:
   $$M_\theta = n \cdot M_0$$
   where $M_0 = \frac{F_{\mathrm{obj}} m_{\mathrm{relay}}}{p}$ is the intrinsic, sample-independent instrument constant.
2. **Current Pipeline Workflow**:
   In `arVIPAspectro/q_vector.py`, the user must supply $n$ as a known constant (`refractive_index`). The pipeline fits $M_\theta$ freely from the curve:
   $$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
   and then calculates $q = \frac{4\pi n}{\lambda_0} \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$ using the provided $n$.
3. **Physical Redundancy & Cross-Validation**:
   Because $M_\theta$ is fitted from the curvature:
   $$\frac{\Delta\nu(\theta)}{\Delta\nu(0)} \approx 1 - \frac{r_{\mathrm{sph}}^2}{8 M_\theta^2} = 1 - \frac{r_{\mathrm{sph}}^2}{8 (n M_0)^2}$$
   $M_\theta$ already contains the sample refractive index $n$!
   - If $M_0$ has been pre-calibrated using a reference liquid (e.g. water with $n=1.3330$), then for any subsequent unknown sample, **$n$ does NOT need to be assumed or looked up**:
     $$n_{\mathrm{measured}} = \frac{M_\theta}{M_0}$$
   - By requiring the user to input $n$ while fitting $M_\theta$ unconstrained, the pipeline introduces an unnecessary degree of freedom and misses the opportunity for an autonomous, simultaneous extraction of $n$ and $V_s$.

### Areas of Improvement
- **Establish a Two-Step Instrument Calibration**:
  1. Calibrate $M_0$ once using pure water ($n_{\mathrm{water}} = 1.3330$):
     $$M_0 = \frac{M_{\theta, \mathrm{water}}}{1.3330}$$
  2. For unknown or relaxing samples (e.g. glycerol or polymers), determine $n$ directly from the fitted $M_\theta$:
     $$n = \frac{M_\theta}{M_0}$$
  3. Deduce $V_s$ without requiring external refractometry:
     $$V_s = \frac{\lambda_0 S_0}{2 (M_\theta / M_0)} = \frac{\lambda_0 M_0}{2} \cdot \frac{S_0}{M_\theta}$$

---

## 10. Summary Matrix of Potential Inconsistencies & Actions

| Pipeline Stage | Physical Inconsistency / Approximation | Impact on Results | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Step 5** | $g = \sqrt{u^2 + n_r^2 - 1}$ termed $\cos\theta_{\mathrm{int}}$ and $\Delta = 2 n_r d g$ | Double counts $n_r$; mathematical error in docs | Redefine $g = n_r \cos\theta_{\mathrm{int}}$ and $\Delta = 2 d g$ |
| **Step 7 & 10** | Labeled "saturated absorption"; optical pumping ignored | Misnomer; possible dip ratio mismatch | Label "Doppler-broadened linear absorption"; note FSR robustness |
| **Step 11 (Stage 2)** | Forces Brillouin shift to match a 2D paraboloid $Z_{\mathrm{quad}}$ | Erases true acoustic relaxation ($\tau$), dispersion, and anisotropy | Restrict Stage 2 to water reference; apply only Stage 1 to unknown samples |
| **Step 12** | Direct DHO fit without IRF deconvolution | Linewidth dominated by VIPA IRF ($\sim 800\text{ MHz}$); $\alpha_s$ incorrect | Deconvolve VIPA IRF (from laser reference) or fit Voigt-DHO profile |
| **Step 12 & 14** | Stokes/Anti-Stokes coordinate averaging | Ignores horizontal VIPA spatio-spectral coupling | Characterize VIPA $\theta_x \leftrightarrow \nu$ dispersion-angle mapping |
| **Step 14** | Free ellipse rotation angle $\phi \approx -68.4^\circ$ | Unphysical tilt caused by asymmetric sensor masking | Fix $\phi = 90.0^\circ$ for isotropic samples; mask data symmetrically |
| **Step 14** | Error propagation mixes $S_0$ definition with $\widehat{V}_s(M_\theta)$ slope | Inconsistent covariance propagation for $\sigma_{V_s}$ | Use $\sigma_{V_s} = \frac{\lambda_0}{2n}\sigma_{S_0}$ or full slope covariance |
| **Step 14** | $M_\theta = n M_0$ relation is not leveraged | Misses direct extraction of $n$ without refractometer | Calibrate $M_0$ with water; autonomously deduce $n = M_\theta / M_0$ |
