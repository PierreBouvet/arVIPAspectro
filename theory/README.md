# Theoretical Derivations (`theory/`)

This directory contains the mathematical derivations and Fourier-optics models underpinning the optical collection, pupil-to-detector mapping, and confocal spatial filtering in the `arVIPAspectro` (angle-resolved VIPA Brillouin spectroscopy) pipeline.

---

## Quick Index

| Document | Topic Studied | Primary Result |
| :--- | :--- | :--- |
| [`abbe_sine_condition_and_pupil_mapping.md`](abbe_sine_condition_and_pupil_mapping.md) | Origin of the Abbe sine law in high-NA collection objectives | $r_{\mathrm{pupil}} = f_{\mathrm{obj}} \sin\theta_{\mathrm{air}} \propto k_\parallel$ |
| [`refractive_index_magnification_derivation.md`](refractive_index_magnification_derivation.md) | Relation between fitted angular magnification $M_\theta$ and sample refractive index $n$ | Exact arcsine law $M_\theta = n M_0$; legacy arctan correction $\mathcal{K}(\xi) \approx 1 - \frac{1}{3}(\mathrm{NA}/n)^2$ |
| [`confocal_angular_spectrum_mixing_outline.md`](confocal_angular_spectrum_mixing_outline.md) | 5-phase theoretical roadmap for angular spectrum mixing through a defocused thick confocal pinhole | Linear mixing kernel $\tilde{E}_{\mathrm{out}}(\mathbf{k}'_\perp) = \iint \mathcal{K}(\mathbf{k}'_\perp, \mathbf{k}_\perp)\, \tilde{E}_{\mathrm{pupil}}(\mathbf{k}_\perp)\, d^2\mathbf{k}_\perp$ |
| [`non_telecentric_angular_spectrum_invariance.md`](non_telecentric_angular_spectrum_invariance.md) | Effect of a non-telecentric tube lens distance ($d_1 \neq f_1$) on the field and angular spectrum at the pinhole | Spatial and angular intensity profiles $I_{\mathrm{im}}(\mathbf{r}_i)$ and $|\tilde{E}_{\mathrm{im}}(\mathbf{k}_\perp)|^2$ are invariant to $d_1$ |
| [`structured_illumination_angular_resolution_and_dispersion.md`](structured_illumination_angular_resolution_and_dispersion.md) | Pupil-plane intensity-only SLM illumination for angular resolution & minimal-frame dispersion recovery | Stationary-point broadening theorem, iso-$q$ / annular rings, radial S-matrices, and confocal kernel coupling |

---

## Detailed Summary of Documents & Derived Results

### 1. [`abbe_sine_condition_and_pupil_mapping.md`](abbe_sine_condition_and_pupil_mapping.md)
* **What it studies:** Why well-corrected high-NA microscope objectives map collection angles $\theta_{\mathrm{air}}$ to radial coordinates in the back focal plane (BFP) via a sine law rather than a paraxial tangent law ($r = f \tan\theta$). It presents three independent proofs: (i) Fermat's principle for stigmatic on-axis collimation, (ii) Abbe's thermodynamic/wave condition for coma-free off-axis aplanatic imaging, and (iii) transverse wavevector ($k$-space) linearity in Fourier optics.
* **Derived results:**
  * **Spherical Principal Surface:** Eliminating spherical aberration and coma requires the equivalent principal surface of an infinity-corrected objective to be a sphere of radius $f_{\mathrm{obj}}$ centered at the focal point, yielding the exact **Abbe sine projection**:
    $$r_{\mathrm{pupil}} = f_{\mathrm{obj}} \sin\theta_{\mathrm{air}}$$
  * **Linear Momentum ($k$-space) Mapping:** The BFP coordinate is strictly proportional to the transverse optical wavevector $k_\parallel = k_0 \sin\theta_{\mathrm{air}}$:
    $$\mathbf{r}_{\mathrm{pupil}} = f_{\mathrm{obj}} \frac{\mathbf{k}_\parallel}{k_0}$$
  * **Finite-NA Divergence between Sine and Tangent:** Series comparison ($\sin\theta \approx \theta - \theta^3/6$ vs. $\tan\theta \approx \theta + \theta^3/3$) shows a third-order discrepancy of $\Delta c_3 = \frac{1}{3} - \left(-\frac{1}{6}\right) = \frac{1}{2}$, reaching $15.5\%$ at $\mathrm{NA} = 0.5$ ($\theta = 30^\circ$) and $41.4\%$ at $\mathrm{NA} = 0.71$ ($\theta = 45^\circ$).

---

### 2. [`refractive_index_magnification_derivation.md`](refractive_index_magnification_derivation.md)
* **What it studies:** How the internal acoustic scattering angle $\theta_{\mathrm{mat}}$ in a sample of refractive index $n$ maps to the circularized radial detector coordinate $r_{\mathrm{sph}}$ (Step 14 / Step 14B), and how the fitted angular magnification parameter $M_\theta$ depends on $n$ and the collection numerical aperture $\mathrm{NA}$.
* **Derived results:**
  * **Fundamental Detector Mapping (Snell's Law + Abbe Sine Condition):** Combining $\sin\theta_{\mathrm{air}} = n \sin\theta_{\mathrm{mat}}$ with the relay magnification yields:
    $$r_{\mathrm{sph}} = M_0 \sin\theta_{\mathrm{air}} = n M_0 \sin\theta_{\mathrm{mat}}, \qquad M_0 \equiv \frac{F_{\mathrm{obj}}\, m_{\mathrm{relay}}}{p}$$
    where $M_0$ (in pixels) is a sample-independent hardware constant.
  * **Exact Congruence of the Arcsine Kinematic Model:** Using the modernized fitting model $\mathrm{Shift}_{\mathrm{fit}}(r_{\mathrm{sph}}) = S_0 \cos\!\left[\frac{1}{2}\arcsin\!\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right]$, the fitted magnification satisfies:
    $$M_\theta = n \cdot M_0$$
    **exactly** at all angles without higher-order geometric truncation error.
  * **Finite-Aperture Correction for Legacy Rectilinear ($\arctan$) Fits:** When fitting with $\arctan(r_{\mathrm{sph}}/M_\theta)$ over the pupil disk $r_{\mathrm{sph}} \in [0, M_0 \mathrm{NA}]$, the local magnification is $M_\theta(r_{\mathrm{sph}}) = n M_0 \sqrt{1 - (r_{\mathrm{sph}}/n M_0)^2}$, and the global least-squares optimal parameter $M_\theta^*$ obeys:
    $$M_\theta^* = n M_0 \cdot \mathcal{K}\!\left(\frac{\mathrm{NA}}{n}\right), \qquad \mathcal{K}(\xi) = 1 - \frac{1}{3}\xi^2 - \frac{1}{18}\xi^4 + \mathcal{O}(\xi^6)$$

---

### 3. [`confocal_angular_spectrum_mixing_outline.md`](confocal_angular_spectrum_mixing_outline.md)
* **What it studies:** A structured 5-phase theoretical program to model how the incident angular spectrum $\tilde{E}_{\mathrm{pupil}}(\mathbf{k}_\perp)$ is coupled, vignetted, and filtered by a 3D confocal pinhole of radius $a$, finite axial thickness $d$, and axial defocus $\Delta z$ relative to the tube lens image plane.
* **Key framework & formulation:**
  * **Phases 1–2 (Optical Train & Defocus):** Parameterizes the objective–tube-lens relay and propagates the pupil field to the defocused pinhole entrance via $\exp(i k_z \Delta z)$.
  * **Phase 3 (Thick Pinhole as a Cylindrical Waveguide):** Formulates propagation through the pinhole thickness $d$ via cylindrical Bessel eigenmode expansion ($J_m$) or a multiple-scattering Fabry–Pérot transmission operator $\mathcal{T}(\mathbf{k}'_\perp, \mathbf{k}_\perp; a, d)$.
  * **Phases 4–5 (Mixing Kernel & Detection):** Synthesizes the outgoing angular spectrum via the linear integral operator:
    $$\tilde{E}_{\mathrm{out}}(\mathbf{k}'_\perp) = \iint \mathcal{K}(\mathbf{k}'_\perp, \mathbf{k}_\perp; a, d, \Delta z, f_{\mathrm{tl}}, f_{\mathrm{obj}}) \, \tilde{E}_{\mathrm{pupil}}(\mathbf{k}_\perp) \, d^2\mathbf{k}_\perp$$
    recovering standard shift-invariant Airy convolution in the thin-pinhole limit ($d \to 0$) and quantifying angle-dependent shadowing for $d > 0$.

---

### 4. [`non_telecentric_angular_spectrum_invariance.md`](non_telecentric_angular_spectrum_invariance.md)
* **What it studies:** Whether operating the tube lens $L_1$ (focal length $f_1$) at a non-telecentric distance $d_1 \neq f_1$ from the objective BFP (e.g., $d_1 \approx 3f_1$ due to optomechanical constraints) distorts the spatial focal spot or the incident angular intensity profile $|\tilde{E}_{\mathrm{im}}(\mathbf{k}_\perp)|^2$ at the pinhole plane ($z = d_1 + f_1$). Includes a self-contained derivation of the back-focal-plane Fourier transform identity in Supplementary Section S1.
* **Derived results:**
  * **Exact Real-Space Field at the Pinhole Plane:** Propagating from the objective BFP ($z = 0$) over $d_1$ and through $L_1$ to $z = d_1 + f_1$ yields:
    $$E_{\mathrm{im}}(\mathbf{r}_i) = \frac{\exp[i k_0 (d_1 + f_1)]}{i \lambda f_1} \exp\!\left[ i \frac{k_0}{2 f_1} \left( 1 - \frac{d_1}{f_1} \right) |\mathbf{r}_i|^2 \right] \tilde{E}_{\mathrm{BFP}}\!\left( \frac{k_0}{f_1} \mathbf{r}_i \right) = \exp[i k_0(d_1 - f_1)] \exp\!\left[ i \frac{k_0}{2 f_1} \left( 1 - \frac{d_1}{f_1} \right) |\mathbf{r}_i|^2 \right] E_{\mathrm{im}}^{(\mathrm{tel})}(\mathbf{r}_i)$$
    Consequently, the real-space intensity profile is strictly unchanged: $I_{\mathrm{im}}(\mathbf{r}_i) = |E_{\mathrm{im}}(\mathbf{r}_i)|^2 = |E_{\mathrm{im}}^{(\mathrm{tel})}(\mathbf{r}_i)|^2$.
  * **Negligible Phase Chirp Across the Pinhole & Focal Spot:** Over a pinhole radius $a \sim 25\ \mu\mathrm{m}$ (with $f_1 = 150\ \mathrm{mm}$, $d_1 = 3f_1$, $\lambda = 532\ \mathrm{nm}$), the maximum quadratic phase variation is $\Delta \phi_{\max} = \frac{k_0}{2 f_1}|1 - d_1/f_1|a^2 \approx 0.049\ \mathrm{rad} \ll 1$ ($\lesssim 10^{-3}\ \mathrm{rad}$ across the PSF core).
  * **Invariance of the Angular Intensity Profile:** The pupil-to-angle mapping $\mathbf{k}_\perp = -\frac{k_0}{f_1}\mathbf{r}_p$ depends only on $f_1$, leaving the incident angular intensity profile completely invariant to $d_1$:
    $$I_{\mathrm{ang}}(\mathbf{k}_\perp) = |\tilde{E}_{\mathrm{im}}(\mathbf{k}_\perp)|^2 = \left|\tilde{E}_{\mathrm{im}}^{(\mathrm{tel})}(\mathbf{k}_\perp)\right|^2 = (\lambda f_1)^2 \left| E_{\mathrm{BFP}}\!\left( -\frac{f_1}{k_0} \mathbf{k}_\perp \right) \right|^2$$

---

## Supporting Assets

* [`schematical_configuration.svg`](schematical_configuration.svg): Vector schematic of the optical train geometry (objective BFP, tube lens $L_1$, non-telecentric distance $d_1$, and pinhole image plane).
* [`references.bib`](references.bib): BibTeX database containing literature references cited in the derivations (e.g., J. W. Goodman & M. E. Cox, *Introduction to Fourier Optics*, Ch. 5, Sec. 5.2).

