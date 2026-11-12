# Step 14 — Acoustic Wavevector (q-vector) & Elliptical Geometry Extraction

## Summary and Input/Output
- **Purpose**: Extracts the 2D concentric coaxial elliptical geometry of the collection aperture, maps detector coordinates $(x, y)$ into spherically symmetric scattering coordinates $r_{\mathrm{sph}}$, fits the angular magnification factor $M_\theta$, and deduces the acoustic wavevector dispersion $\Omega(q)$ and longitudinal speed of sound $V_s \pm \sigma_{V_s}$.
- **Inputs**: Refined 2D coordinates $(x, y)$, Brillouin shift $\Delta\nu_B$ and linewidth $\Gamma_B$ with standard errors $\sigma_{\mathrm{Shift}}$ and $\sigma_{\mathrm{Linewidth}}$, refractive index $n_{\mathrm{sample}}$, laser wavelength $\lambda_0$.
- **Outputs**:
  - Ellipse parameters: Center $(x_c, y_c)$, director angle $\phi$, aspect ratio $\gamma$.
  - Kinematic parameters: Angular magnification $M_\theta \pm \sigma_{M_\theta}$, normal backscattering shift $S_0 \pm \sigma_{S_0}$.
  - Acoustic dispersion: Scattering angles $\theta$, wavevectors $q$, and longitudinal sound velocity $V_s \pm \sigma_{V_s}$.
  - Diagnostic Figures 14 and 14B.

---

## Algorithm
1. **Error-Weighted Data Preparation**:
   - Converts Levenberg-Marquardt fit variances to standard errors $\sigma_i = \sqrt{\mathrm{Var}_i}$.
   - Assigns point weights $w_i = 1 / \sigma_i$ with an empirical percentile floor ($1\%$) to regularize near-zero variances.
   - Averages paired Stokes and Anti-Stokes orders to eliminate residual zero-frequency offsets.
2. **2D Concentric Coaxial Ellipse Fitting with Error-Weighting & Variance Normalization**:
   - Rotates detector coordinates into the candidate ellipse director frame $(\phi)$:
     $$\begin{pmatrix} X' \\ Y' \end{pmatrix} = \begin{pmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} x - x_c \\ y - y_c \end{pmatrix}$$
   - Evaluates the joint quadratic surface model:
     $$Z_m(X', Y') = Z_{0,m} + a_m X'^2 + b_m Y'^2, \qquad m \in \{\mathrm{Shift}, \mathrm{Linewidth}\}$$
   - Solves for the shared center $(x_c, y_c)$ and orientation $\phi$ by minimizing the combined error-weighted objective normalized by dataset sample variance:
     $$\min_{x_c, y_c, \phi} \left[ \frac{\sum_i w_{S, i} (S_i - Z_S(X'_i, Y'_i))^2}{\mathrm{Var}(S) \sum_i w_{S, i}} + \frac{\sum_i w_{\Gamma, i} (\Gamma_i - Z_\Gamma(X'_i, Y'_i))^2}{\mathrm{Var}(\Gamma) \sum_i w_{\Gamma, i}} \right]$$
3. **Nelder-Mead Circular Variance Minimization for Aspect Ratio $\gamma$**:
   - Scales the minor axis by $\gamma$:
     $$x_{\mathrm{sph}} = X', \qquad y_{\mathrm{sph}} = \frac{Y'}{\gamma}, \qquad r_{\mathrm{sph}} = \sqrt{x_{\mathrm{sph}}^2 + y_{\mathrm{sph}}^2}$$
   - Optimizes $\gamma$ via `scipy.optimize.minimize(method='Nelder-Mead')` to minimize the radial variance of $\mathrm{Shift}$ around its moving average.
4. **Kinematic Fit of Angular Magnification $M_\theta$**:
   - Fits the non-linear kinematic curve via Levenberg-Marquardt (`curve_fit`):
     $$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cdot \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
   - Returns optimal parameters $S_0, M_\theta$ and their full $2 \times 2$ covariance matrix $\mathbf{\Sigma}_{\mathrm{kin}}$.
5. **Acoustic Wavevector Delineation & Longitudinal Sound Velocity $V_s$**:
   - Internal scattering angle: $\theta(r_{\mathrm{sph}}) = \arcsin(r_{\mathrm{sph}} / M_\theta)$.
   - Acoustic wavevector magnitude:
     $$q(r_{\mathrm{sph}}) = \frac{4\pi n}{\lambda_0} \cos\left(\frac{\theta(r_{\mathrm{sph}})}{2}\right)$$
   - Longitudinal sound velocity:
     $$V_s = \frac{\lambda_0 \cdot S_0}{2 n}$$
   - Full covariance error propagation from $(S_0, M_\theta)$ through the linear dispersion relation $\mathrm{Shift} = \frac{V_s}{2\pi} q(M_\theta)$:
     $$\sigma_{V_s} = \sqrt{\left(\frac{\partial V_s}{\partial S_0}\right)^2 \sigma_{S_0}^2 + \left(\frac{\partial V_s}{\partial M_\theta}\right)^2 \sigma_{M_\theta}^2 + 2 \left(\frac{\partial V_s}{\partial S_0}\right) \left(\frac{\partial V_s}{\partial M_\theta}\right) \mathrm{cov}(S_0, M_\theta)}$$

---

## Physical Rationale
In an angle-resolved backscattering geometry, the collection lens collects light scattered over a cone of angles $\theta \in [0, 45^\circ]$. Because the VIPA etalon only disperses along the horizontal axis ($x$) and introduces optical anamorphism between horizontal and vertical axes, the circular collection cone projects onto the detector as an elongated ellipse ($\gamma \approx 1.8\text{--}2.0$). 

Recovering the aspect ratio $\gamma$ restores isotropic spherical symmetry ($r_{\mathrm{sph}}$), which allows the exact physical scattering angle $\theta$ and acoustic wavevector $q$ to be deduced purely from geometry without relying on arbitrary polynomial scaling.

---

## Physical Approximations
1. **Affine Elliptical Pupil Deformation**: The anamorphic distortion of the collection cone is modeled as a 2D affine scaling along orthogonal principal axes.
2. **Abbe Sine Condition**: The aplanatic collection objective maps scattering angles to pupil coordinates via $r \propto \sin\theta$, modeled rigorously via the $\arcsin$ kinematic relation.
3. **Linear Acoustic Dispersion**: The phonon group velocity equals the phase velocity: $\Omega = V_s \cdot q$, which holds rigorously in the GHz frequency domain far below the Debye cutoff.
4. **Isotropic Medium**: The sample is assumed mechanically isotropic with scalar sound velocity $V_s$.

---

## Key Equations
- **Spherical Coordinate Transformation**:
  $$x_{\mathrm{sph}} = (x - x_c)\cos\phi + (y - y_c)\sin\phi, \qquad y_{\mathrm{sph}} = \frac{-(x - x_c)\sin\phi + (y - y_c)\cos\phi}{\gamma}$$
  $$r_{\mathrm{sph}} = \sqrt{x_{\mathrm{sph}}^2 + y_{\mathrm{sph}}^2}$$
- **Kinematic Shift Model**:
  $$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cdot \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
- **Acoustic Wavevector Formula**:
  $$q(r_{\mathrm{sph}}) = \frac{4\pi n}{\lambda_0} \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
- **Sound Velocity & Propagated Uncertainty**:
  $$V_s = \frac{\lambda_0 S_0}{2 n}$$
  $$\sigma_{V_s}^2 = J_{S_0}^2 \sigma_{S_0}^2 + J_{M_\theta}^2 \sigma_{M_\theta}^2 + 2 J_{S_0} J_{M_\theta} \mathrm{cov}(S_0, M_\theta)$$
  where:
  $$J_{S_0} = \frac{\lambda_0}{2 n}, \qquad J_{M_\theta} = \frac{\partial \widehat{V}_s}{\partial M_\theta} = 2\pi \frac{N' D - N D'}{D^2}$$

---

## Code Implementation
- **Theoretical Foundations**:
  - Exact Refractive Index Scaling & Magnification Derivation: [`theory/refractive_index_magnification_derivation.md`](file:/theory/refractive_index_magnification_derivation.md)
  - Abbe Sine Condition & Pupil Mapping: [`theory/abbe_sine_condition_and_pupil_mapping.md`](file:/theory/abbe_sine_condition_and_pupil_mapping.md)
- **Pipeline Function**: `extract_q_vector(...)` in [`arVIPAspectro/q_vector.py`](file:/arVIPAspectro/q_vector.py#L18-L177)
- **Core Solvers**:
  - `fit_concentric_ellipses_coaxial`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L900-L960)
  - `optimize_elliptical_symmetry_variance`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L962-L1010)
  - `fit_magnification_factor_M_theta`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L1012-L1050)
  - `compute_theta_and_q_vector`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L1052-L1145)
- **Diagnostic Plots**:
  - Figure 14: `plot_q_vector_dispersion` ([`arVIPAspectro/plotting.py`](file:/arVIPAspectro/plotting.py#L466-L587))
  - Figure 14B: `plot_complete_diagnostic_suite` ([`arVIPAspectro/plotting.py`](file:/arVIPAspectro/plotting.py#L589-L825))
