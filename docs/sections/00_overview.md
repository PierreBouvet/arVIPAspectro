# 00 — Optical Architecture & Pipeline Overview

## Summary and Input/Output
- **Purpose**: High-level overview of the physical principles, optical layout, and computational pipeline of Angle-Resolved Virtually Imaged Phased Array (ar-VIPA) Brillouin Light Scattering (BLS) spectroscopy.
- **Inputs**: Raw sCMOS camera acquisitions (`laser_reference.dat`, `Rb_absorption.dat`, `sample.dat`), optical spectrometer specifications, physical constants.
- **Outputs**: Calibrated 2D frequency maps, sub-pixel accurate spatial interference contours, DHO acoustic peak parameters (frequency shift $\Delta\nu_B$, linewidth $\Gamma_B$), acoustic wavevectors $q$, and longitudinal sound velocity $V_s \pm \sigma_{V_s}$.

---

## Algorithm
The complete data-processing and calibration workflow executes in four distinct phases:
1. **Laser Geometric Calibration (Steps 1–6)**:
   - Denoises the monochromatic laser reference interferogram.
   - Detects and traces 11 interference orders across the full sensor height ($2304\text{ px}$).
   - Computes squared-intensity sub-pixel centroids with polynomial outlier filtering.
   - Fits the exact 2D gnomonic projection center $(x_0, y_0)$ and VIPA tilt angle $\phi$.
   - Calibrates a 2D 4th-degree Anamorphic Polynomial Order Surface $\phi_{\mathrm{order}}(g, y)$ eliminating inter-order lens aberrations.
2. **Rubidium Atomic Absorption Frequency Calibration (Steps 7–10B)**:
   - Evaluates the theoretical saturated transmission spectrum $T_{\mathrm{Rb}}(\Delta\nu)$ of natural Rubidium ($^{85}\text{Rb} / ^{87}\text{Rb}$) across 12 hyperfine transitions.
   - Extracts radial absorption profiles along gnomonic iso-angle contours.
   - Cross-correlates experimental absorption dips with atomic theory to determine the exact Free Spectral Range ($\mathrm{FSR}_{\mathrm{cal}} \approx 31.02\text{ GHz}$) and laser detuning.
   - Synthesizes the master 2D frequency map $\nu_{\mathrm{map}}(x, y)$.
3. **Sample Spectrum In-Situ Calibration & DHO Peak Fitting (Steps 11–13)**:
   - Applies a two-stage in-situ self-calibration (elastic zero-offset map and higher-order dispersion Jacobian scaling) using Stokes/Anti-Stokes symmetry.
   - Bins the interferogram spatially ($4 \times 2$ pixel blocks).
   - Fits local spectral profiles with a vectorized 3D batched Levenberg-Marquardt Damped Harmonic Oscillator (DHO) model.
4. **2D Elliptical Geometry & Acoustic Wavevector Extraction (Steps 14–15)**:
   - Fits 2D concentric coaxial ellipses to the Brillouin shift and linewidth contours with statistical error-weighting ($w_i = 1/\sigma_i$).
   - Optimizes the aspect ratio $\gamma$ via circular variance minimization.
   - Transforms spatial detector coordinates into spherical scattering coordinates $r_{\mathrm{sph}}$.
   - Fits the angular magnification factor $M_\theta$ and normal shift $S_0$ via non-linear kinematic modeling.
   - Deduces scattering angles $\theta$, acoustic wavevectors $q$, and the longitudinal sound velocity $V_s$ with full covariance error propagation.

---

## Physical Rationale
Brillouin light scattering probes spontaneous acoustic phonons in condensed matter via inelastic photon-phonon scattering. However, extracting high-resolution acoustic dispersion relations $\Omega(q) = V_s \cdot q$ using traditional scanning Fabry-Pérot interferometers requires point-by-point angle-resolved measurements spanning hours.

The VIPA spectrometer enables single-shot angle-resolved spectroscopy by utilizing:
- **A tilted solid fused silica etalon**: Disperses optical frequencies horizontally along $x$ with high spectral resolution ($\mathcal{R} > 10^6$) and large angular dispersion ($\sim 0.5\text{ mrad/GHz}$).
- **Large-NA Collection Lens**: Collects light scattered over a continuum of angles $\theta \in [0, 45^\circ]$ and focuses it onto the vertical and horizontal axes of the VIPA input window.
- **Rubidium Vapor Notch Filter**: Suppresses intense elastic Rayleigh scattering by $> 40\text{ dB}$ while transmitting the Stokes and Anti-Stokes Brillouin peaks.

---

## Physical Approximations
1. **Quasi-Elastic Scattering**: The acoustic phonon frequency ($\sim 5\text{--}10\text{ GHz}$) is negligible compared to the optical frequency ($\nu_0 \approx 384\text{ THz}$), ensuring the incident and scattered optical wavevectors have identical magnitude: $k_i \approx k_s = \frac{2\pi n}{\lambda_0}$.
2. **Linear Acoustic Dispersion**: Acoustic phonons in the hydrodynamic regime follow a linear dispersion relation $\Omega = V_s \cdot q$, where $V_s$ is the longitudinal speed of sound in the sample.
3. **Paraxial Collection Projection**: While ray tracing inside the VIPA requires exact gnomonic geometry, the external imaging through the collection lens is well-described by an affine-deformed spherical coordinate projection.

---

## Key Equations
- **Brillouin Kinematic Scattering Condition**:
  $$q = 2 k_i \cos\left(\frac{\theta}{2}\right) = \frac{4\pi n}{\lambda_0} \cos\left(\frac{\theta}{2}\right)$$
- **Acoustic Dispersion Relation**:
  $$\Omega = 2\pi \cdot \mathrm{Shift} = V_s \cdot q \iff \mathrm{Shift}(\theta) = \left(\frac{2 n V_s}{\lambda_0}\right) \cos\left(\frac{\theta}{2}\right)$$
- **Normal Incidence Sound Velocity**:
  $$V_s = \frac{\lambda_0 \cdot S_0}{2 n}$$

---

## Code Implementation
- **Master Orchestrator**: [`arVIPAspectro/main.py`](file:/main.py)
- **Library Package**: [`arVIPAspectro/arVIPAspectro/`](file:/arVIPAspectro)
- **Configuration & Constants**: [`arVIPAspectro/arVIPAspectro/config.py`](file:/arVIPAspectro/config.py)
