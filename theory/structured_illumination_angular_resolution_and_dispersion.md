# Structured Pupil-Plane Illumination for Angular Resolution Enhancement and Minimal-Measurement Dispersion Reconstruction in ar-VIPA Brillouin Spectroscopy

## Overview

This document develops the theoretical framework for **pupil-plane (back focal plane, BFP) intensity-only structured illumination** in an epi-detection, angle-resolved Virtually Imaged Phased Array (ar-VIPA) Brillouin Light Scattering (BLS) spectrometer. Using a Spatial Light Modulator (SLM) with finite pixel size placed conjugate to the illumination BFP, we address two central questions:

1. **Angular Resolution & Linewidth Broadening:** Can a specific illumination intensity structure eliminate—or minimize—the apparent Brillouin linewidth broadening $\Delta\Gamma_{\mathrm{ill}}$ caused by a finite illumination numerical aperture ($\mathrm{NA}_{\mathrm{ill}} > 0$) and finite SLM pixel size?
2. **Minimal-Measurement Dispersion Recovery:** What class of illumination functions $I_{\mathrm{ill}}^{(m)}(\mathbf{k}_{i,\perp})$ enables reconstruction of the full acoustic dispersion—both Brillouin frequency shift $\Omega_B(q)$ and intrinsic linewidth $\Gamma_B(q)$—at a single sample point in a minimal number of exposures $M$?

We formulate the problem in a **two-tier linear-operator framework**:
- **Tier 1 (Ideal Pupil-to-Detector Mapping):** Assumes thermal-phonon incoherence and ideal BFP-to-detector mapping via the Abbe sine condition, isolating the exact geometry of $\mathbf{q}$-broadening, iso-$q$ contours, SLM pixel limits, and multiplexed bases (annular rings, S-matrices, Zernike/Fourier modes).
- **Tier 2 (Confocal Pinhole Coupling):** Couples in the finite-pinhole angular spectrum mixing kernel $\mathcal{K}(\mathbf{k}'_{s,\perp}, \mathbf{k}_{s,\perp})$ from [`confocal_angular_spectrum_mixing_outline.md`](confocal_angular_spectrum_mixing_outline.md).

---

## 1. Wavevector Geometry and Accessible $q$-Range in Epi-Detection

Consider a sample of refractive index $n$ illuminated at vacuum wavelength $\lambda_0$ ($k_0 = 2\pi/\lambda_0$, medium wavenumber $k = n k_0$) through an aplanatic objective of numerical aperture $\mathrm{NA}$ and focal length $f_{\mathrm{obj}}$.

By the **Abbe sine condition** ([`abbe_sine_condition_and_pupil_mapping.md`](abbe_sine_condition_and_pupil_mapping.md)) and Snell's law at the sample interface ([`refractive_index_magnification_derivation.md`](refractive_index_magnification_derivation.md)), a transverse coordinate $\mathbf{r}_{i,\mathrm{BFP}}$ in the illumination BFP maps linearly to the transverse wavevector $\mathbf{k}_{i,\perp}$ inside the sample:

$$\mathbf{k}_{i,\perp} = \frac{k_0}{f_{\mathrm{obj}}}\,\mathbf{r}_{i,\mathrm{BFP}}, \qquad |\mathbf{k}_{i,\perp}| \le k_0\,\mathrm{NA} \equiv k_{\perp,\max}$$

Inside the sample, the incident wavevector (propagating in $+z$) and the scattered wavevector collected by the same objective (propagating in $-z$) are:

$$\mathbf{k}_i(\mathbf{k}_{i,\perp}) = \begin{pmatrix} \mathbf{k}_{i,\perp} \\ +k_{i,z} \end{pmatrix}, \qquad k_{i,z} = \sqrt{n^2 k_0^2 - |\mathbf{k}_{i,\perp}|^2}$$

$$\mathbf{k}_s(\mathbf{k}_{s,\perp}) = \begin{pmatrix} \mathbf{k}_{s,\perp} \\ -k_{s,z} \end{pmatrix}, \qquad k_{s,z} = \sqrt{n^2 k_0^2 - |\mathbf{k}_{s,\perp}|^2}$$

The acoustic phonon wavevector probed by the scattering pair $(\mathbf{k}_{i,\perp}, \mathbf{k}_{s,\perp})$ is $\mathbf{q} = \mathbf{k}_s - \mathbf{k}_i$:

$$\mathbf{q}(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp}) = \begin{pmatrix} \mathbf{k}_{s,\perp} - \mathbf{k}_{i,\perp} \\ -(k_{s,z} + k_{i,z}) \end{pmatrix}$$

Its exact scalar magnitude $q = |\mathbf{q}|$ is:

$$q^2(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp}) = 2n^2 k_0^2 + 2\left(k_{i,z}\, k_{s,z} - \mathbf{k}_{i,\perp}\cdot\mathbf{k}_{s,\perp}\right)$$

### Accessible Wavevector Span: On-Axis vs. Off-Axis Illumination

- **On-axis illumination ($\mathbf{k}_{i,\perp} = \mathbf{0}$):** $k_{i,z} = n k_0$, yielding $q(\mathbf{k}_{s,\perp}, \mathbf{0}) = 2 n k_0 \cos(\theta_s/2)$ where $\sin\theta_s = |\mathbf{k}_{s,\perp}|/(n k_0) \le \mathrm{NA}/n$. The accessible $q$-band is restricted to:
  $$q \in \left[\, \sqrt{2}\,n k_0 \sqrt{1 + \sqrt{1 - (\mathrm{NA}/n)^2}},\; 2 n k_0 \,\right]$$
- **Arbitrary pupil illumination ($|\mathbf{k}_{i,\perp}| \le k_0 \mathrm{NA}$):** Since $\mathbf{k}_{i,\perp}\cdot\mathbf{k}_{s,\perp} \le |\mathbf{k}_{i,\perp}||\mathbf{k}_{s,\perp}|$ and $k_{i,z}, k_{s,z} \ge k_0\sqrt{n^2 - \mathrm{NA}^2}$, $q$ is minimized when $\mathbf{k}_{i,\perp} = \mathbf{k}_{s,\perp}$ at the pupil rim ($|\mathbf{k}_{i,\perp}| = k_0 \mathrm{NA}$), and maximized at retro-reflection ($\mathbf{k}_{i,\perp} = \mathbf{k}_{s,\perp} = \mathbf{0}$, or more generally $\mathbf{k}_{s,\perp} = -\mathbf{k}_{i,\perp}$. 

*Note:* at $\mathbf{k}_{s,\perp} = -\mathbf{k}_{i,\perp}$, $k_{s,z}=k_{i,z}$, so $q^2 = 2n^2 k_0^2 + 2(k_{i,z}^2 + |\mathbf{k}_{i,\perp}|^2) = 4n^2 k_0^2$, meaning **every** antiparallel pair $\mathbf{k}_{s,\perp} = -\mathbf{k}_{i,\perp}$ probes $q_{\max} = 2n k_0$!). Thus the full accessible range is:
  $$q \in \left[\, 2 k_0 \sqrt{n^2 - \mathrm{NA}^2},\; 2 n k_0 \,\right]$$

For example, with $n = 1.33$ and $\mathrm{NA} = 1.0$, on-axis illumination spans only $q \in [2.42\,k_0,\, 2.66\,k_0]$ ($\sim 9\%$ of $q_{\max}$), whereas structured off-axis illumination spans $q \in [1.76\,k_0,\, 2.66\,k_0]$ ($\sim 34\%$ of $q_{\max}$, nearly $4\times$ wider).

---

## 2. Tier 1 Forward Model: Intensity-Only SLM with Finite Pixel Size

### 2.1 SLM Discretization in the Illumination BFP

An intensity-modulating SLM in the illumination BFP consists of $N_{\mathrm{pix}}$ pixels of pitch $\Delta p_{\mathrm{SLM}}$ (mapped to angular pitch $\Delta k_{\mathrm{pix}} = \frac{k_0}{M_{\mathrm{ill}} f_{\mathrm{obj}}}\,\Delta p_{\mathrm{SLM}}$). A programmed non-negative pattern $\mathbf{w} \in [0, 1]^{N_{\mathrm{pix}}}$ produces an incident angular intensity distribution over the pupil disk $\mathcal{P} = \{|\mathbf{k}_{i,\perp}| \le k_{\perp,\max}\}$:

$$I_{\mathrm{ill}}(\mathbf{k}_{i,\perp}) = \sum_{j=1}^{N_{\mathrm{pix}}} w_j \, \Pi_j(\mathbf{k}_{i,\perp}), \qquad w_j \in [0, 1]$$

where $\Pi_j(\mathbf{k}_{i,\perp})$ is the indicator (pixel aperture) function of the $j$-th SLM pixel.

### 2.2 Thermal-Phonon Incoherence and Spectral-Angular Mixing

In spontaneous Brillouin scattering, thermal acoustic fluctuations at distinct wavevectors $\mathbf{q}, \mathbf{q}'$ are delta-correlated:

$$\langle \delta\varepsilon(\mathbf{q}, \omega)\,\delta\varepsilon^*(\mathbf{q}', \omega') \rangle \propto S(\mathbf{q}, \omega)\,\delta(\mathbf{q}-\mathbf{q}')\,\delta(\omega-\omega')$$

Consequently, in the Tier 1 limit (large scattering volume relative to $1/\Delta k_{\mathrm{pix}}$ and ideal BFP-to-detector mapping), the scattered power spectral density measured at collection pupil coordinate $\mathbf{k}_{s,\perp}$ is an **incoherent linear integral** over the illumination pupil:

$$I_{\mathrm{meas}}(\mathbf{k}_{s,\perp}, \omega) = \iint_{\mathcal{P}} I_{\mathrm{ill}}(\mathbf{k}_{i,\perp})\, \mathcal{R}(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp})\, S\big(\mathbf{q}(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp}),\, \omega\big)\, d^2\mathbf{k}_{i,\perp}$$

where $\mathcal{R}(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp})$ captures polarization/dipole-radiation and Fresnel transmission factors, and $S(\mathbf{q}, \omega)$ is the damped harmonic oscillator (DHO) dynamic structure factor:

$$S(\mathbf{q}, \omega) \propto \frac{\Omega_B(\mathbf{q})^2\,\Gamma_B(\mathbf{q})}{\big(\omega^2 - \Omega_B(\mathbf{q})^2\big)^2 + \omega^2\,\Gamma_B(\mathbf{q})^2}$$

### 2.3 Reduction to a 1D $q$-Kernel for Isotropic Viscoelastic Media

For an **isotropic viscoelastic medium**, $S(\mathbf{q}, \omega) = S(q, \omega)$ depends only on $q = |\mathbf{q}|$. Changing variables from $\mathbf{k}_{i,\perp}$ to $(q, \psi)$ along the iso-$q$ contours $\mathcal{C}_q(\mathbf{k}_{s,\perp}) = \{\mathbf{k}_{i,\perp} \in \mathcal{P} : |\mathbf{k}_s - \mathbf{k}_i| = q\}$ collapses the 2D integral into a 1D integral operator:

$$I_{\mathrm{meas}}(\mathbf{k}_{s,\perp}, \omega) = \int_{q_{\min}}^{q_{\max}} W(\mathbf{k}_{s,\perp}, q\,;\, I_{\mathrm{ill}})\, S(q, \omega)\, dq$$

where the **$q$-mixing kernel** induced by $I_{\mathrm{ill}}$ is the weighted Radon-type line integral:

$$W(\mathbf{k}_{s,\perp}, q\,;\, I_{\mathrm{ill}}) = \int_{\mathcal{C}_q(\mathbf{k}_{s,\perp})} \frac{I_{\mathrm{ill}}(\mathbf{k}_{i,\perp})\, \mathcal{R}(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp})}{\left|\nabla_{\mathbf{k}_{i,\perp}} q(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp})\right|}\, ds$$

Here, the gradient of $q$ with respect to $\mathbf{k}_{i,\perp}$ is:

$$\nabla_{\mathbf{k}_{i,\perp}} q(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp}) = -\frac{1}{q}\left(\mathbf{k}_{s,\perp} + \frac{k_{s,z}}{k_{i,z}}\,\mathbf{k}_{i,\perp}\right)$$

---

## 3. Question 1: Can Structured Illumination Eliminate Linewidth Broadening?

### 3.1 Origin of Illumination-Induced Linewidth Broadening

Around a nominal illumination wavevector $\mathbf{k}_{i,0,\perp}$ with angular spread $\Delta k_i$, a first-order expansion gives:

$$\Delta q \approx \left|\nabla_{\mathbf{k}_{i,\perp}} q\right|_{\mathbf{k}_{i,0,\perp}} \Delta k_i + \frac{1}{2}\Delta k_i^2 \left\|\mathbf{H}_q\right\|$$

Because $\Omega_B(q) \approx V_s(q)\,q$, a spread $\Delta q$ in the kernel $W(\mathbf{k}_{s,\perp}, q)$ maps directly into a frequency spread $\Delta\Omega_{\mathrm{ill}} \approx V_s\,\Delta q$. Convolving the intrinsic Lorentzian-like DHO peak of width $\Gamma_B(q)$ with $W(\mathbf{k}_{s,\perp}, q)$ yields an apparent broadened linewidth $\Gamma_{\mathrm{app}} \approx \Gamma_B + \Delta\Gamma_{\mathrm{ill}}(\Delta q)$.

### 3.2 Stationary-Point Theorem (Where First-Order Broadening Vanishes)

Setting $\nabla_{\mathbf{k}_{i,\perp}} q = \mathbf{0}$ yields:

$$\mathbf{k}_{s,\perp} + \frac{k_{s,z}}{k_{i,z}}\,\mathbf{k}_{i,\perp} = \mathbf{0}$$

Since $k_{s,z}, k_{i,z} > 0$, $\mathbf{k}_{s,\perp}$ and $\mathbf{k}_{i,\perp}$ must be anti-parallel ($\mathbf{k}_{s,\perp} = -\alpha\,\mathbf{k}_{i,\perp}$, $\alpha > 0$). Taking norms and substituting $k_z = \sqrt{n^2 k_0^2 - k_\perp^2}$ forces $\alpha = 1$, i.e.:

$$\nabla_{\mathbf{k}_{i,\perp}} q = \mathbf{0} \iff \mathbf{k}_{s,\perp} = -\mathbf{k}_{i,\perp} \quad (\text{exact } 180^\circ \text{ retro-reflection, } q = 2n k_0)$$

**Consequence:**
- **No non-zero 2D area** in the illumination pupil can produce $\Delta q = 0$ simultaneously across all collection angles $\mathbf{k}_{s,\perp}$ at any $q < 2 n k_0$.
- For a single SLM pixel of width $\Delta k_{\mathrm{pix}}$, the irreducible single-pixel $q$-broadening away from retro-reflection is:
  $$\Delta q_{\mathrm{pix}}(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp}) = \frac{1}{q}\left|\mathbf{k}_{s,\perp} + \frac{k_{s,z}}{k_{i,z}}\,\mathbf{k}_{i,\perp}\right| \Delta k_{\mathrm{pix}}$$

### 3.3 Single-Shot Minimal-Broadening Structures

Although zero broadening across the entire 2D detector simultaneously is geometrically impossible in a single non-negative intensity exposure, **two classes of 1D structures** achieve pixel-limited broadening ($\Delta q = \Delta q_{\mathrm{pix}}$) while maximizing photon throughput ($N_{\mathrm{open}} \gg 1$ pixels):

1. **Concentric Annular Rings** ($I_{\mathrm{ill}}(\mathbf{k}_{i,\perp}) = \delta_{\mathrm{pix}}(|\mathbf{k}_{i,\perp}| - k_{i,r})$):
   - **Exact zero azimuthal broadening at the optical axis** ($\mathbf{k}_{s,\perp} = \mathbf{0}$): because $q(\mathbf{0}, \mathbf{k}_{i,\perp}) = \sqrt{2n^2 k_0^2 + 2 n k_0 \sqrt{n^2 k_0^2 - k_{i,r}^2}}$ depends *only* on the radius $k_{i,r}$, all $2\pi k_{i,r}/\Delta k_{\mathrm{pix}}$ pixels on the ring probe the **exact same $q$** at the detector center!
   - **Throughput gain:** $\sim 2\pi k_{i,r}/\Delta k_{\mathrm{pix}}$ times higher photon flux than a single pixel, with identical $\Delta q_{\mathrm{pix}}$.
   - *Limitation:* At off-axis detector points $\mathbf{k}_{s,\perp} \neq \mathbf{0}$, the $\mathbf{k}_{i,\perp}\cdot\mathbf{k}_{s,\perp} = k_{i,r} k_{s,r}\cos(\phi_i - \phi_s)$ term spreads $q$ across the ring's azimuth.

2. **Iso-$q$ Contours Matched to a Target $(\mathbf{k}_{s,0,\perp}, q_0)$:**
   - Illuminating along the 1-pixel-wide curve $\mathcal{C}_{q_0}(\mathbf{k}_{s,0,\perp})$ eliminates broadening *at the specific collection point $\mathbf{k}_{s,0,\perp}$* with maximal photon flux for that $q_0$.

### 3.4 Multi-Shot Elimination of Broadening (Virtual Kernel Sharpening & Deconvolution)

With intensity-only modulation ($w_j \ge 0$), single-shot kernels $W(\mathbf{k}_{s,\perp}, q)$ are strictly non-negative. However, across $M$ exposures $\{I_{\mathrm{ill}}^{(m)}\}_{m=1}^M$, linear combinations with signed weights $\alpha_m \in \mathbb{R}$ synthesize **effective virtual kernels**:

$$W_{\mathrm{virt}}(\mathbf{k}_{s,\perp}, q) = \sum_{m=1}^M \alpha_m\, W^{(m)}(\mathbf{k}_{s,\perp}, q)$$

- **Super-resolved / sub-pixel $q$-sharpening:** Subtracting adjacent overlapping patterns (e.g., an annular disk of radius $k_r + \delta k$ minus one of radius $k_r$, or S-matrix / Hadamard inversion) cancels the first-order broadening and isolates narrow $\Delta q$ bins—at the cost of noise amplification governed by the condition number $\kappa(\mathbf{W})$ of the forward operator.
- **Spectral-angular joint deconvolution:** Rather than sharpening $W$ before fitting, the cleanest way to recover true $(\Omega_B(q), \Gamma_B(q))$ without broadening bias is **joint forward-model fitting** of $I_{\mathrm{meas}}^{(m)}(\mathbf{k}_{s,\perp}, \omega)$ against $\int W^{(m)}(\mathbf{k}_{s,\perp}, q)\, S(q, \omega)\, dq$ parameterized directly by smooth dispersion curves $\Omega_B(q)$ and $\Gamma_B(q)$.

---

## 4. Question 2: Minimal-Measurement Bases for Full Dispersion Recovery

Discretizing $q \in [q_{\min}, q_{\max}]$ into $N_q$ bins and stacking all $N_s$ collection-pupil pixels across $M$ SLM exposures gives the linear system:

$$\mathbf{y}(\omega) = \mathbf{W}\,\mathbf{s}(\omega) + \boldsymbol{\eta}(\omega), \qquad \mathbf{y}(\omega) \in \mathbb{R}^{M N_s},\quad \mathbf{W} \in \mathbb{R}^{M N_s \times N_q},\quad \mathbf{s}(\omega) \in \mathbb{R}^{N_q}$$

where $s_j(\omega) = S(q_j, \omega)$. We seek illumination bases $\{I_{\mathrm{ill}}^{(m)}\}_{m=1}^M$ that minimize $M$ while keeping $\kappa(\mathbf{W})$ small and maximizing photon SNR.

### 4.1 Class 1: Concentric Annular Ring Basis (Minimal $M$ without Multiplexing)

Define $M$ thin concentric rings in the illumination BFP at radii $k_{i,r}^{(m)} \in (0, k_{\perp,\max}]$:

$$I_{\mathrm{ill}}^{(m)}(\mathbf{k}_{i,\perp}) = \mathbb{I}\!\left(\left||\mathbf{k}_{i,\perp}| - k_{i,r}^{(m)}\right| \le \frac{\Delta k_{\mathrm{pix}}}{2}\right)$$

**Why this is optimal for minimal $M$ in isotropic media:**
- Each single exposure $m$ simultaneously illuminates all azimuths $\phi_i \in [0, 2\pi)$ while the 2D detector records all $\mathbf{k}_{s,\perp} = (k_{s,r}, \phi_s)$.
- By rotational symmetry, $W^{(m)}(\mathbf{k}_{s,\perp}, q)$ depends only on $k_{s,r} = |\mathbf{k}_{s,\perp}|$, allowing azimuthal averaging on the detector to boost SNR by $\sqrt{N_\phi}$ without any $q$-broadening!
- For each ring $k_{i,r}^{(m)}$, varying $k_{s,r} \in [0, k_{\perp,\max}]$ sweeps $q$ over a broad sub-interval of $[q_{\min}, q_{\max}]$. Choosing $M \approx 3\text{--}6$ well-spaced rings (e.g., $k_{i,r}/k_{\perp,\max} \in \{0, 0.35, 0.7, 1.0\}$) covers the **entire** $[q_{\min}, q_{\max}]$ range with overlapping, well-conditioned support.

### 4.2 Class 2: Radial S-Matrix / Simplex-Hadamard Basis (SNR-Optimal under Photon Noise)

When photon shot noise dominates (the standard regime in spontaneous BLS), thin annular rings waste most of the laser power ($N_{\mathrm{open}}/N_{\mathrm{pix}} \ll 1$). Because the SLM is **intensity-only** ($w_j \in [0, 1]$), standard $\{-1, +1\}$ Hadamard matrices cannot be displayed directly in a single frame.

The optimal non-negative replacement is the **cyclic S-matrix** $\mathbf{S}_M \in \{0, 1\}^{M \times M}$ (obtained by deleting the first row and column of a normalized $(M+1)\times(M+1)$ Hadamard matrix and mapping $+1 \to 0, -1 \to 1$):

1. Partition the illumination pupil into $M$ equal-area concentric annular zones $\mathcal{Z}_1, \dots, \mathcal{Z}_M$ (or equal-$q_{\text{on-axis}}$ zones).
2. In exposure $m \in \{1, \dots, M\}$, turn ON zone $j$ if $(\mathbf{S}_M)_{m,j} = 1$ (opening $\frac{M+1}{2M} \approx 50\%$ of the pupil in every frame!).
3. Invert via $\mathbf{S}_M^{-1} = \frac{2}{M+1}\left(2\mathbf{S}_M^\top - \mathbf{1}\mathbf{1}^\top\right)$ to recover the **exact, unmixed response of every individual annular zone** $\mathcal{Z}_j$ with **Fellgett multiplex SNR gain**:

$$\frac{\mathrm{SNR}_{\text{S-matrix}}}{\mathrm{SNR}_{\text{single-ring}}} = \frac{\sqrt{M+1}}{2}$$

For $M = 15$ annular zones, this yields a $2\times$ SNR gain (equivalent to $4\times$ faster acquisition) while recovering $15$ narrow radial rings simultaneously, each still benefiting from full azimuthal detector averaging.

### 4.3 Class 3: Smooth Continuous Bases (Radial Zernike / Chebyshev / Cosine) for Few-Shot Parametric Dispersion

If the dispersion $\Omega_B(q), \Gamma_B(q)$ is smooth (e.g., viscoelastic Debye/Cole-Davidson relaxation over $[q_{\min}, q_{\max}]$) and can be represented by $P \ll N_q$ parameters, one can use $M \sim P$ **non-negative shifted radial Fourier/Zernike/Chebyshev modes**:

$$I_{\mathrm{ill}}^{(m)}(k_{i,r}) = \frac{1}{2}\left[1 + T_m\!\left(\frac{2 k_{i,r}^2}{k_{\perp,\max}^2} - 1\right)\right] \quad \text{or} \quad \frac{1}{2}\left[1 + \cos\!\left(\frac{m\pi\, k_{i,r}^2}{k_{\perp,\max}^2}\right)\right]$$

where $T_m$ are Chebyshev polynomials (or radial Zernike polynomials $R_{2m}^0(k_{i,r}/k_{\perp,\max})$). These maintain $\sim 50\%$ average pupil transmission and require only $M \approx 3\text{--}5$ exposures when inverted jointly with the 2D detector $k_{s,r}$-diversity.

### 4.4 Extension to 3D Anisotropic $\mathbf{q}$-Vector Dispersion

When the sample is mechanically anisotropic, $S(\mathbf{q}, \omega)$ depends on $(\theta_q, \phi_q, q)$ (the full 3D vector $\mathbf{q} = \mathbf{k}_s - \mathbf{k}_i$). Azimuthally symmetric rings then mix different crystal orientations $\phi_q$. To resolve 3D anisotropy in minimal measurements:
- Partition the illumination pupil into $M_r \times M_\phi$ **polar sectors** $(k_{i,r}, \phi_i)$ and multiplex them with a **2D Kronecker S-matrix** $\mathbf{S}_{M_r} \otimes \mathbf{S}_{M_\phi}$ (or non-negative shifted Zernike modes $Z_n^l(k_{i,r}, \phi_i)$).
- Inverting $\mathbf{S}_{M_r} \otimes \mathbf{S}_{M_\phi}$ recovers the full 4D dataset $I_{\mathrm{meas}}(\mathbf{k}_{s,\perp}, \mathbf{k}_{i,\perp}, \omega)$ at high photon throughput, directly enabling the 3D $\mathbf{q}$-tomography described in [`anisotropy_and_q_measures_probing_different_locations_pupil.md`](anisotropy_and_q_measures_probing_different_locations_pupil.md).

---

## 5. Tier 2 Coupling: Confocal Pinhole Angular Spectrum Mixing

In a real ar-VIPA spectrometer, a confocal pinhole of radius $a$, thickness $d$, and axial defocus $\Delta z$ sits in the detection path before the VIPA. As formulated in [`confocal_angular_spectrum_mixing_outline.md`](confocal_angular_spectrum_mixing_outline.md), the pinhole linearly mixes the scattered angular spectrum:

$$\tilde{E}_{\mathrm{out}}(\mathbf{k}'_{s,\perp}) = \iint_{\mathcal{P}} \mathcal{K}(\mathbf{k}'_{s,\perp}, \mathbf{k}_{s,\perp}; a, d, \Delta z, f_{\mathrm{tl}}, f_{\mathrm{obj}}) \, \tilde{E}_{\mathrm{pupil}}(\mathbf{k}_{s,\perp}) \, d^2\mathbf{k}_{s,\perp}$$

For thermally incoherent phonons, the detected intensity at $\mathbf{k}'_{s,\perp}$ after the pinhole averages over the incoherent source distribution, yielding an effective **intensity mixing kernel** $|\mathcal{K}(\mathbf{k}'_{s,\perp}, \mathbf{k}_{s,\perp})|^2$ (in the thin-pinhole limit $d \to 0$, this reduces to convolution with the Airy pattern $|\tilde{P}_{\mathrm{pinhole}}(\mathbf{k}'_{s,\perp} - \mathbf{k}_{s,\perp})|^2$).

Composing Tier 1 with the pinhole operator simply replaces the ideal kernel $W^{(m)}$ by the **effective confocal kernel**:

$$W_{\mathrm{eff}}^{(m)}(\mathbf{k}'_{s,\perp}, q) = \iint_{\mathcal{P}} \left|\mathcal{K}(\mathbf{k}'_{s,\perp}, \mathbf{k}_{s,\perp})\right|^2 W^{(m)}(\mathbf{k}_{s,\perp}, q)\, d^2\mathbf{k}_{s,\perp}$$

Thus:
- **Both** illumination broadening (from $I_{\mathrm{ill}}^{(m)}$) and detection pinhole broadening (from $\mathcal{K}$) combine into a single pre-computable linear operator $\mathbf{W}_{\mathrm{eff}}$.
- All S-matrix multiplexing, annular-ring, and joint forward-model inversion results from Sections 3–4 remain **formally identical** when $\mathbf{W}$ is replaced by $\mathbf{W}_{\mathrm{eff}}$, provided the pinhole angular width $\Delta k_{\mathrm{pin}} \sim 1/a$ does not overly smooth the singular values of $\mathbf{W}_{\mathrm{eff}}$.

---

## 6. Summary Comparison of Illumination Strategies

| Illumination Class | Exposures $M$ | Photon Throughput | Broadening $\Delta q$ | Best Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Single SLM Pixel** | $N_{\mathrm{pix}} \sim 10^3\text{--}10^4$ | $\sim 1/N_{\mathrm{pix}}$ (very low) | $\Delta q_{\mathrm{pix}}$ (irreducible) | Reference only; impractical SNR |
| **Concentric Annular Rings** | $3\text{--}8$ | $\sim 2\pi k_r \Delta k_{\mathrm{pix}} / \mathcal{A}_{\mathcal{P}}$ | $0$ azimuthal at $\mathbf{k}_{s,\perp}=\mathbf{0}$; $\Delta q_{\mathrm{pix}}$ radial | Fast isotropic dispersion $(\Omega_B(q), \Gamma_B(q))$ |
| **Iso-$q$ Contour** | $1$ per $(\mathbf{k}_{s,0,\perp}, q_0)$ | High along 1D contour | $\Delta q_{\mathrm{pix}}$ at $\mathbf{k}_{s,0,\perp}$ | Targeted high-resolution $q_0$ measurement |
| **Radial S-Matrix (Hadamard)** | $M = N_r$ (e.g. $7, 15$) | $\sim 50\%$ of pupil | $\Delta q_{\mathrm{zone}}$ (post-inversion) | **Photon-noise-limited** isotropic dispersion |
| **Shifted Radial Zernike / Cosine** | $3\text{--}5$ | $\sim 50\%$ of pupil | Parametric joint fit | Minimal-frame smooth viscoelastic dispersion |
| **2D Polar S-Matrix ($\mathbf{S}_{M_r}\!\otimes\!\mathbf{S}_{M_\phi}$)** | $M_r \times M_\phi$ | $\sim 25\%$–$50\%$ of pupil | $\Delta q_{\mathrm{sector}}$ (post-inversion) | **3D anisotropic** $\mathbf{q}$-vector tomography |
