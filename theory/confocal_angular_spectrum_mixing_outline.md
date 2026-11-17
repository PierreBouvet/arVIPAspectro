# Theoretical Framework: Angular Spectrum Mixing in Confocal Microscopy

## Overview
This document outlines the theoretical development to model how the angular spectrum of an optical field is mixed and filtered in a confocal microscope. The model explicitly accounts for the objective parameters, tube lens focal length, defocus of the pinhole relative to the tube lens image plane, and the three-dimensional geometry of the pinhole (radius and thickness).

---

## Phase 1: System parameterization and Coordinate Definitions
In this section, we will establish the coordinate systems and fundamental parameters governing the optical train from the sample to the detector.
*   **Optical Train Geometry:** Define the $4f$ (or general telecentric/non-telecentric) relay comprising the objective lens (focal length $f_{obj}$, numerical aperture $\text{NA}$) and the tube lens (focal length $f_{tl}$).
*   **Spatial and Angular Coordinates:** Define transverse spatial coordinates $\mathbf{r}_\perp = (x, y)$ and transverse wavevector components $\mathbf{k}_\perp = (k_x, k_y)$ in the sample plane, pupil plane, and detection plane.
*   **Defocus Parameter ($\Delta z$):** Define the axial displacement of the pinhole entrance face relative to the nominal paraxial image plane of the tube lens.
*   **Pinhole Geometry:** Define the cylindrical aperture of radius $a$ and finite axial thickness $d$.

---

## Phase 2: Field Propagation to the Pinhole Entrance
Here, we will relate the angular spectrum emitted or scattered by the sample to the spatial and angular field distribution incident upon the pinhole.
*   **Objective Pupil Mapping:** Apply the sine condition to map the angular spectrum of the sample $\tilde{E}_{sample}(\mathbf{k}_\perp)$ to the spatial field in the back focal plane (pupil) of the objective.
*   **Tube Lens Transformation:** Use Debye diffraction integrals or Fresnel propagation (depending on the magnification and numerical aperture at the image side) to propagate the pupil field through the tube lens to the nominal image plane.
*   **Defocus Propagation:** Apply the angular spectrum propagator $\exp(i k_z \Delta z)$, where $k_z = \sqrt{k^2 - |\mathbf{k}_\perp|^2}$, to shift the field from the nominal tube lens image plane to the entrance plane of the displaced pinhole ($z = z_{in}$).

---

## Phase 3: The Thick Pinhole as a Waveguide / Propagator
A pinhole of finite thickness $d$ cannot be treated simply as a multiplicative 2D transmission mask $T(\mathbf{r}_\perp)$. Instead, it acts as a short cylindrical waveguide or a volume diffraction element that couples and attenuates different spatial frequencies differently.
*   **Boundary Value Formulation:** Model the pinhole interior ($0 \le r \le a$, $z_{in} \le z \le z_{in} + d$) with appropriate wall boundary conditions (e.g., Perfectly Conducting Metal for reflective pinholes, or perfectly absorbing boundaries for black-coated pinholes).
*   **Modal Decomposition vs. Split-Step Propagation:** 
    *   *Approach A (Modal):* Expand the incident field at $z_{in}$ into the eigenmodes (Bessel modes $J_m$) of the cylindrical aperture, propagate each mode through thickness $d$ with its specific propagation constant $\beta_{mn}$, and project back onto the free-space angular spectrum at the exit plane $z_{out} = z_{in} + d$.
    *   *Approach B (Fabry-Pérot / Multiple Scattering):* Formulate a transmission operator $\mathcal{T}(\mathbf{k}'_\perp, \mathbf{k}_\perp; a, d)$ that directly describes the coupling from an incident plane wave $\mathbf{k}_\perp$ to an outgoing plane wave $\mathbf{k}'_\perp$.

---

## Phase 4: Angular Spectrum Mixing Kernel
This is the core analytical step where we synthesize the previous phases into a single mixing matrix or integral kernel.
*   **The Mixing Integral:** Express the outgoing angular spectrum after the pinhole $\tilde{E}_{out}(\mathbf{k}'_\perp)$ as a linear transformation of the incident angular spectrum before the pinhole:
    $$\tilde{E}_{out}(\mathbf{k}'_\perp) = \iint \mathcal{K}(\mathbf{k}'_\perp, \mathbf{k}_\perp; a, d, \Delta z, f_{tl}, f_{obj}) \, \tilde{E}_{pupil}(\mathbf{k}_\perp) \, d^2\mathbf{k}_\perp$$
*   **Thin-Pinhole Limit ($d \to 0$):** Verify that the kernel $\mathcal{K}$ reduces to the standard convolution with the Airy/Bessel spectrum of a circular aperture modulated by a defocus phase factor.
*   **Thick-Pinhole Regime ($d > 0$):** Analyze how finite thickness introduces angle-dependent vignetting (shadowing) and mode-selective attenuation, effectively breaking the shift-invariance of the angular convolution.

---

## Phase 5: Detection and Effective Optical Transfer Function (OTF)
Finally, we will evaluate how this angular mixing affects the measured confocal signal.
*   **Integrated Intensity:** Formulate the total power collected by a large-area bucket detector placed behind the pinhole using Parseval's theorem over $\tilde{E}_{out}(\mathbf{k}'_\perp)$.
*   **Cross-Spectral Density / Mutual Intensity:** Identify which pairs of incident wavevectors $(\mathbf{k}_{\perp 1}, \mathbf{k}_{\perp 2})$ are mixed to produce interference at the detector.
*   **Asymptotic Analysis:** Study specific physical regimes:
    1.  $a \to \infty$ (Widefield limit)
    2.  $a \to 0, d \to 0$ (Ideal point-detector confocal limit)
    3.  Finite $a$ and $d$ with varying $\Delta z$ (Axial sectioning degradation and aberration coupling).