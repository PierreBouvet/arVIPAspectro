# Exact Mathematical Development: Reluctance of Angular Magnification $M_\theta$ to Sample Refractive Index $n$

---

## 1. Physical Architecture and Coordinate Definitions

Consider an angle-resolved Brillouin light scattering (ar-BLS) experiment in backscattering geometry probing spontaneous longitudinal acoustic phonons in an isotropic medium of refractive index $n$.

### 1.1 Scattering Geometry inside the Sample
Let the incident laser beam propagate along $+\hat{\mathbf{z}}$ with vacuum wavelength $\lambda_0$ and angular frequency $\omega_0$.
- In the medium with refractive index $n$, the incident optical wavevector is:
  $$k_i = \frac{2\pi n}{\lambda_0}$$
- A collection lens collects backscattered light over an angular cone around the backward axis $-\hat{\mathbf{z}}$.
- Let $\theta \equiv \theta_{\mathrm{mat}} \in [0, \theta_{\mathrm{max}}]$ be the internal off-axis scattering angle relative to the exact backscattering direction $-\hat{\mathbf{z}}$. The total scattering angle is $\Theta = \pi - \theta_{\mathrm{mat}}$.
- The acoustic wavevector probed inside the medium has magnitude:
  $$q(\theta_{\mathrm{mat}}) = 2 k_i \sin\left(\frac{\Theta}{2}\right) = 2 k_i \cos\left(\frac{\theta_{\mathrm{mat}}}{2}\right) = \frac{4\pi n}{\lambda_0} \cos\left(\frac{\theta_{\mathrm{mat}}}{2}\right)$$
- Assuming linear acoustic dispersion in the hypersonic regime ($\Omega = V_s \cdot q$), the local Brillouin frequency shift is:
  $$\mathrm{Shift}(\theta_{\mathrm{mat}}) = \frac{\Omega}{2\pi} = \left(\frac{2 n V_s}{\lambda_0}\right) \cos\left(\frac{\theta_{\mathrm{mat}}}{2}\right) = S_0 \cos\left(\frac{\theta_{\mathrm{mat}}}{2}\right)$$
  where $S_0 \equiv \frac{2 n V_s}{\lambda_0}$ is the normal backscattering shift ($\theta_{\mathrm{mat}} = 0$).

---

## 2. Interface Refraction and Lens Optical Transfer

### 2.1 Refraction at the Planar Medium-Air Boundary
The scattered rays exit the medium through a planar interface perpendicular to the optical axis $\hat{\mathbf{z}}$ into air ($n_{\mathrm{air}} \approx 1$).
By Snell's law:
$$\sin\theta_{\mathrm{air}} = n \sin\theta_{\mathrm{mat}}$$

The maximum internal collection angle $\theta_{\mathrm{max}}$ is restricted either by the numerical aperture of the collection objective $\mathrm{NA} = \sin\theta_{\mathrm{air,max}}$ or by total internal reflection:
$$\sin\theta_{\mathrm{mat}} \le \frac{\mathrm{NA}}{n} \le \frac{1}{n}$$

### 2.2 Abbe Sine Condition (Aplanatic Collection Objective)
For a well-corrected, coma-free aplanatic collection lens (e.g., high-NA objective of focal length $F_{\mathrm{obj}}$), rays collected at angle $\theta_{\mathrm{air}}$ are mapped to the objective back focal plane (pupil plane) according to the Abbe sine condition:
$$r_{\mathrm{pupil}} = F_{\mathrm{obj}} \sin\theta_{\mathrm{air}} = F_{\mathrm{obj}} \cdot n \sin\theta_{\mathrm{mat}}$$

*(For a comprehensive geometric proof, wave-mechanical derivation, and comparison with tangent/rectilinear mappings, see [`theory/abbe_sine_condition_and_pupil_mapping.md`](file:///Users/pierrebouvet/Documents/Code/arVIPAspectro/theory/abbe_sine_condition_and_pupil_mapping.md).)*

### 2.3 Optical Relays and Detector Mapping
After the anamorphic VIPA spectrometer imaging train and the elliptical-to-circular transformation in Step 14 (which eliminates the VIPA anamorphic ratio $\gamma \approx 1.8\text{--}2.0$), the radial coordinate $r_{\mathrm{sph}}$ on the detector (in pixel units) is proportional to $r_{\mathrm{pupil}}$:
$$r_{\mathrm{sph}} = \frac{m_{\mathrm{relay}}}{p} r_{\mathrm{pupil}} = M_0 \sin\theta_{\mathrm{air}}$$
where:
- $p$ is the physical detector pixel pitch ($6.5\ \mu\text{m}$),
- $m_{\mathrm{relay}}$ is the effective transverse magnification between the objective pupil and the detector,
- $M_0 \equiv \frac{F_{\mathrm{obj}} m_{\mathrm{relay}}}{p}$ is the **intrinsic instrumental angular magnification constant** (expressed in pixels), which depends strictly on the optical hardware and is completely independent of the sample.

Substituting Snell's law yields the **fundamental mapping relation**:
$$r_{\mathrm{sph}} = M_0 \cdot n \sin\theta_{\mathrm{mat}}$$

---

## 3. The Step 14B Kinematic Model vs. Exact Optical Dispersion

### 3.1 The Modernized Arcsine Model in Step 14B
Following the Abbe sine condition, Step 14 and Step 14B directly model the radial variation of the Brillouin shift with the exact arcsine kinematic function:
$$\mathrm{Shift}_{\mathrm{fit}}(r_{\mathrm{sph}}) = S_0 \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
This formula defines the internal physical scattering angle directly:
$$\theta(r_{\mathrm{sph}}) = \arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right) \iff \sin\theta = \frac{r_{\mathrm{sph}}}{M_\theta}$$

### 3.2 Exact Brillouin Shift as a Function of Detector Coordinate $r_{\mathrm{sph}}$
From the physical relation:
$$\sin\theta_{\mathrm{mat}} = \frac{r_{\mathrm{sph}}}{n M_0}$$
we compute the internal scattering angle directly:
$$\theta_{\mathrm{mat}}(r_{\mathrm{sph}}) = \arcsin\left(\frac{r_{\mathrm{sph}}}{n M_0}\right)$$
The **exact physical shift** on the detector is therefore:
$$\mathrm{Shift}_{\mathrm{exact}}(r_{\mathrm{sph}}) = S_0 \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{n M_0}\right)\right)$$

### 3.3 Exact Congruence
Comparing $\mathrm{Shift}_{\mathrm{fit}}$ with $\mathrm{Shift}_{\mathrm{exact}}$:
$$\mathrm{Shift}_{\mathrm{fit}}(r_{\mathrm{sph}}) \equiv \mathrm{Shift}_{\mathrm{exact}}(r_{\mathrm{sph}}) \iff \mathbf{M_\theta = n \cdot M_0}$$
Because the model uses an arcsine function matching the Abbe sine condition, **$M_\theta = n \cdot M_0$ holds exactly across all angles and radial positions**, with zero higher-order geometric truncation error!

*(Note: In the historical rectilinear formulation using $\arctan(r_{\mathrm{sph}} / M_\theta)$, a fourth-order discrepancy $\mathcal{O}(\sin^4\theta)$ existed between the model and the physical dispersion, requiring an aperture correction factor $\mathcal{K}(\mathrm{NA}/n) \approx 1 - \frac{1}{3}(\mathrm{NA}/n)^2$. The modernized arcsine model eliminates this truncation error completely.)*

---

## 4. Exact Non-Approximated Formulation and Finite-Aperture Integration

### 4.1 Exact Local Relation
If we demand that the fitted angle matches the physical angle at any individual radius $r_{\mathrm{sph}}$, i.e., $\theta_{\mathrm{apparent}} = \theta_{\mathrm{mat}}$:
$$\tan\theta_{\mathrm{mat}} = \frac{r_{\mathrm{sph}}}{M_\theta(r_{\mathrm{sph}})}$$
Since $\tan\theta_{\mathrm{mat}} = \frac{\sin\theta_{\mathrm{mat}}}{\cos\theta_{\mathrm{mat}}} = \frac{\sin\theta_{\mathrm{mat}}}{\sqrt{1 - \sin^2\theta_{\mathrm{mat}}}}$ and $r_{\mathrm{sph}} = n M_0 \sin\theta_{\mathrm{mat}}$:
$$\frac{n M_0 \sin\theta_{\mathrm{mat}}}{\sqrt{1 - \sin^2\theta_{\mathrm{mat}}}} = \frac{n M_0 \sin\theta_{\mathrm{mat}}}{M_\theta(r_{\mathrm{sph}})}$$
which gives the exact local point-by-point relation:
$$M_\theta(r_{\mathrm{sph}}) = n M_0 \cos\theta_{\mathrm{mat}} = n M_0 \sqrt{1 - \left(\frac{r_{\mathrm{sph}}}{n M_0}\right)^2}$$

### 4.2 Global Least-Squares Relation over the Collection Pupil
In Step 14B, a single scalar parameter $M_\theta^*$ is extracted across the full circularized pupil disk $r_{\mathrm{sph}} \in [0, R_{\mathrm{max}}]$, where:
$$R_{\mathrm{max}} = M_0 \cdot \mathrm{NA}$$
The objective minimized by Levenberg-Marquardt is:
$$\mathcal{E}(M_\theta) = \int_0^{R_{\mathrm{max}}} \Big[ g_{\mathrm{exact}}(r) - g_{\mathrm{fit}}(r; M_\theta) \Big]^2 r \, \mathrm{d}r$$

Let $\xi \equiv \frac{R_{\mathrm{max}}}{n M_0} = \frac{\mathrm{NA}}{n} < 1$ denote the reduced collection numerical aperture inside the medium.
Expanding the difference $g_{\mathrm{exact}}(r) - g_{\mathrm{fit}}(r; M_\theta)$ up to fourth order:
$$\delta g(r) \approx \frac{1}{2} r^2 \left[ \frac{1}{M_\theta^2} - \frac{1}{(n M_0)^2} \right] - \frac{1}{8} r^4 \left[ \frac{1}{(n M_0)^4} + \frac{3}{M_\theta^4} \right]$$

Setting $\frac{\partial \mathcal{E}}{\partial M_\theta} = 0$, the optimal fitted parameter $M_\theta^*$ is related to $n$ and $M_0$ by:
$$M_\theta^* = n M_0 \cdot \mathcal{K}\left(\frac{\mathrm{NA}}{n}\right)$$
where the aperture shape factor $\mathcal{K}(\xi)$ is given by:
$$\mathcal{K}(\xi) = 1 - \frac{1}{3}\xi^2 - \frac{1}{18}\xi^4 + \mathcal{O}(\xi^6), \qquad \xi = \frac{\mathrm{NA}}{n}$$

### 4.3 Inversion Bounds and Physical Domain of Validity
For physical valid operation, the following constraints must strictly hold:

1. **Snell Refraction Limit**:
   $$\mathrm{NA} \le n \iff \xi \le 1$$
   Since the collection objective operates in air, $\mathrm{NA} = \sin\theta_{\mathrm{air,max}} < 1.0$. For condensed matter samples ($n_{\mathrm{water}} \approx 1.33$, $n_{\mathrm{plastic}} \approx 1.59$), $\xi = \frac{\mathrm{NA}}{n} \le \frac{0.85}{1.33} \approx 0.64$, ensuring rapid convergence of the series.

2. **Real-Valued Argument (Radicand Positivity)**:
   $$r_{\mathrm{sph}} \le R_{\mathrm{max}} = M_0 \mathrm{NA} < n M_0$$
   This guarantees that $1 - \left(\frac{r_{\mathrm{sph}}}{n M_0}\right)^2 > 0$ strictly holds everywhere on the detector.

3. **Inversion Formulation for Unknown $n$**:
   Given a calibrated instrument constant $M_0$ (determined using a known reference medium such as air $n=1$ or distilled water $n=1.333$):
   $$n = \frac{M_\theta^*}{M_0} \cdot \left[ 1 + \frac{1}{3}\left(\frac{\mathrm{NA}}{n}\right)^2 + \frac{1}{6}\left(\frac{\mathrm{NA}}{n}\right)^4 + \dots \right]$$
   Solving iteratively (fixed-point recursion):
   $$n^{(0)} = \frac{M_\theta^*}{M_0}$$
   $$n^{(k+1)} = \frac{M_\theta^*}{M_0} \left[ 1 + \frac{1}{3}\left(\frac{\mathrm{NA}}{n^{(k)}}\right)^2 \right]$$

---

## 5. Summary of Key Formulas

| Quantity | Formula | Remarks |
| :--- | :--- | :--- |
| **Intrinsic Instrument Constant** | $M_0 = \frac{F_{\mathrm{obj}} \cdot m_{\mathrm{relay}}}{p}$ | Hardware constant (pixels), sample-independent |
| **Leading Linear Scaling** | $M_\theta \approx n \cdot M_0$ | Paraxial approximation ($\mathrm{NA}/n \ll 1$) |
| **Exact Local Magnification** | $M_\theta(r_{\mathrm{sph}}) = n M_0 \sqrt{1 - \left(\frac{r_{\mathrm{sph}}}{n M_0}\right)^2}$ | Continuous angle-dependent mapping |
| **Global Pupil-Averaged Fit** | $M_\theta^* = n M_0 \left[ 1 - \frac{1}{3}\left(\frac{\mathrm{NA}}{n}\right)^2 + \mathcal{O}\left(\frac{\mathrm{NA}}{n}\right)^4 \right]$ | Levenberg-Marquardt fit over disk $[0, R_{\mathrm{max}}]$ |
| **Exact Shift Model** | $\mathrm{Shift}_{\mathrm{exact}}(r) = S_0 \sqrt{\frac{1 + \sqrt{1 - \left(\frac{r}{n M_0}\right)^2}}{2}}$ | Eliminates fourth-order truncation error |
