# The Abbe Sine Condition: Origin, Derivation, and Pupil Plane Mapping ($r_{\mathrm{pupil}} = f \sin\theta$)

---

## 1. Introduction and The Central Question

In elementary ray optics, a ray passing through the front focal point of a thin lens at angle $\theta$ is often drawn hitting a flat principal plane at height:
$$r = f \tan\theta \iff \theta = \arctan\left(\frac{r}{f}\right)$$

However, in well-corrected optical instruments—most notably microscope objectives, Fourier scatterometers, and angle-resolved spectroscopy collection systems—the radial position in the objective back focal plane (pupil plane) obeys a **sine law**, not a tangent law:
$$r_{\mathrm{pupil}} = f \sin\theta_{\mathrm{air}}$$

This fundamental result is known as the **Abbe Sine Condition** (or the aplanatic projection). This document provides the geometric, physical, and wave-mechanical proofs of why real high-numerical-aperture (high-NA) lenses map angles via the sine function.

---

## 2. Geometric Proof: The Spherical Principal Surface

To understand why a sine appears instead of a tangent, we examine how aberrations are eliminated in high-NA optics.

```
       Planar Principal Plane (Abberated)           Spherical Principal Surface (Aplanatic)
               r = f * tan(θ)                                 r = f * sin(θ)

             | Principal Plane                            ( Spherical Principal Surface
             |                                           (    (Radius = f)
             |--- r = f * tan(θ)                        (--- r = f * sin(θ)
            /|                                         / (
           / |                                        /   (
       θ  /  |                                    θ  /     (
       --•---|--------------------                --•-------|--------------------
         Focal Point (Sample)                       Focal Point (Sample)
```

### 2.1 Stigmatism and Fermat's Principle
Consider a point source at the front focal point $F$ emitting spherical wavefronts. For the lens to collimate these rays into a perfectly flat wavefront propagating parallel to the optical axis (stigmatic imaging of the focal point at infinity), **all rays must have the exact same optical path length (OPL)** between $F$ and the emerging flat wavefront.

Let $H$ be the point of refraction where the ray at angle $\theta$ is turned into a ray parallel to the optical axis:
1. The ray starts at $F$ and travels distance $L(\theta)$ to the equivalent refraction surface at $H$.
2. From $H$, the ray propagates along a straight line parallel to the optical axis to a reference wavefront $W$ perpendicular to the axis at distance $Z$ from $F$.

The total optical path length is:
$$\mathrm{OPL}(\theta) = L(\theta) + [Z - z_H(\theta)]$$

For this OPL to be strictly constant for all angles $\theta$, the equivalent refracting surface must satisfy:
$$L(\theta) = f \quad \text{for all } \theta$$
where $f$ is the focal length along the optical axis ($\theta = 0$).

### 2.2 Intersection Height on the Sphere
Because the distance from the focal point $F$ to the refracting surface is constant ($L(\theta) = f$), the equivalent principal surface of an aberration-free collimating lens is not a flat plane; it is a **sphere of radius $f$ centered on the front focal point $F$**.

Now, project the intersection point $H$ onto the optical axis to find the ray height $r$:
$$r = f \sin\theta$$

If the lens had a flat principal plane, rays at larger angles $\theta$ would travel a longer distance $f / \cos\theta$ before turning, introducing severe **spherical aberration** ($W_{040}$) and destroying the collimation of marginal rays.

---

## 3. Wave and Thermodynamic Derivation: Abbe's Sine Condition

The condition was originally derived by Ernst Abbe in 1873 by analyzing the imaging of off-axis object structures without coma.

### 3.1 Imaging an Off-Axis Element
Consider an object plane with refractive index $n_1$ and an image plane with refractive index $n_2$. Let an on-axis point $O$ be imaged to $O'$, and an adjacent off-axis point $A$ at height $y_1$ be imaged to $A'$ at height $y_2$.

```
     Object Space (n₁)                             Image Space (n₂)
          A                                              
          | y₁                                           
          O -------(ray at angle α₁)----------> O' -------(ray at angle α₂)-----> A'
                                                                                  | y₂
```

For $A'$ to be a sharp, diffraction-limited image of $A$, the optical path difference between any marginal ray leaving $A$ at angle $\alpha_1$ and the axial ray must be zero (Fermat's principle applied to the off-axis wavefront).

1. In object space, a ray emitted from $A$ at angle $\alpha_1$ has an optical path advance relative to the ray from $O$ given by:
   $$\Delta \mathrm{OPL}_1 = - n_1 y_1 \sin\alpha_1$$

2. In image space, the converging ray arrives at $A'$ at angle $\alpha_2$, with a corresponding optical path difference:
   $$\Delta \mathrm{OPL}_2 = - n_2 y_2 \sin\alpha_2$$

3. For constructive interference across the entire pupil (no coma or wavefront tilt distortion):
   $$\Delta \mathrm{OPL}_1 = \Delta \mathrm{OPL}_2 \implies n_1 y_1 \sin\alpha_1 = n_2 y_2 \sin\alpha_2$$

This yields the **Abbe Sine Condition for aplanatic imaging**:
$$\frac{\sin\alpha_1}{\sin\alpha_2} = \frac{n_2 y_2}{n_1 y_1} = \frac{n_2}{n_1} m$$
where $m \equiv y_2 / y_1$ is the lateral magnification. To eliminate coma across a finite field of view, $m$ must be **strictly constant for all ray angles $\alpha_1$**.

### 3.2 The Infinite-Conjugate Limit (Microscope Objective)
In modern infinity-corrected microscope objectives:
- Rays originating from the front focal point emerge collimated in image space (infinity conjugate).
- The objective is paired with a tube lens of focal length $f_{\mathrm{tube}}$ to form the final real image.

In image space, a collimated ray at pupil height $r_{\mathrm{pupil}}$ enters the tube lens and converges toward the intermediate image plane at angle $\alpha_2$. For a tube lens operating in the paraxial regime ($\alpha_2 \ll 1$):
$$\sin\alpha_2 \approx \tan\alpha_2 = \frac{r_{\mathrm{pupil}}}{f_{\mathrm{tube}}}$$

The lateral magnification of the complete microscope is:
$$m = -\frac{f_{\mathrm{tube}}}{f_{\mathrm{obj}}}$$

Substituting $m$ and $\sin\alpha_2$ into Abbe's sine condition (with object space in air, $n_1 = 1$, $n_2 = 1$):
$$\sin\alpha_1 = m \sin\alpha_2 = \left(\frac{f_{\mathrm{tube}}}{f_{\mathrm{obj}}}\right) \left(\frac{r_{\mathrm{pupil}}}{f_{\mathrm{tube}}}\right) = \frac{r_{\mathrm{pupil}}}{f_{\mathrm{obj}}}$$

Rearranging gives the exact pupil mapping relation:
$$\mathbf{r_{\mathrm{pupil}} = f_{\mathrm{obj}} \sin\theta_{\mathrm{air}}}$$

---

## 4. Fourier Optics & Momentum Conservation ($k$-space Linear Mapping)

A third, equally fundamental perspective comes from wave mechanics and Fourier optics.

### 4.1 Transverse Wavevector Conservation
In wave mechanics, a monochromatic plane wave propagating at angle $\theta_{\mathrm{air}}$ relative to the optical axis $+z$ has a transverse wavevector:
$$k_\parallel = k_x \hat{\mathbf{x}} + k_y \hat{\mathbf{y}}, \qquad |k_\parallel| = k_0 \sin\theta_{\mathrm{air}} = \frac{2\pi}{\lambda_0} \sin\theta_{\mathrm{air}}$$

Notice that momentum conservation across planar interfaces (Snell's law) is fundamentally an equality of transverse wavevectors:
$$k_{\parallel, \mathrm{mat}} = k_0 n \sin\theta_{\mathrm{mat}} = k_0 \sin\theta_{\mathrm{air}} = k_{\parallel, \mathrm{air}}$$

### 4.2 Fourier Transform Property of an Optical Lens
In Fourier optics, the back focal plane of an objective lens is the **spatial Fourier transform** of the front focal plane. The physical position $\mathbf{r}_{\mathrm{pupil}} = (x_p, y_p)$ in the back focal plane corresponds directly to the spatial frequency $\mathbf{f}_\perp = \frac{\mathbf{k}_\parallel}{2\pi}$:
$$\mathbf{r}_{\mathrm{pupil}} = \lambda_0 f_{\mathrm{obj}} \mathbf{f}_\perp = f_{\mathrm{obj}} \left(\frac{\mathbf{k}_\parallel}{k_0}\right)$$

Taking the magnitude:
$$r_{\mathrm{pupil}} = f_{\mathrm{obj}} \sin\theta_{\mathrm{air}}$$

**Physical Significance**:
Because the objective obeys the Abbe sine condition, **the pupil plane is a strictly linear representation of momentum ($k$-space)**:
$$r_{\mathrm{pupil}} \propto k_\parallel$$
If the lens followed a tangent projection ($r \propto \tan\theta$), the pupil would distort $k$-space non-linearly, destroying the exact spatial Fourier transform property at high angles.

---

## 5. Comparison: Abbe Sine vs. Rectilinear (Tangent) Mapping

| Projection Property | Aplanatic / Abbe Sine | Rectilinear / Gnomonic |
| :--- | :--- | :--- |
| **Pupil Formula** | $r = f \sin\theta$ | $r = f \tan\theta$ |
| **Inverse Angle** | $\theta = \arcsin(r / f)$ | $\theta = \arctan(r / f)$ |
| **Principal Surface** | Sphere of radius $f$ | Flat plane at distance $f$ |
| **Aberration Control** | Zero spherical aberration & zero coma | Sharp on-axis, but severe coma at high NA |
| **Primary Use Case** | Microscope objectives, high-NA imaging, Fourier optics | Photographic camera lenses (straight lines remain straight) |
| **$k$-space Mapping** | Strictly linear: $r \propto k_\parallel$ | Non-linear: $r \propto \frac{k_\parallel}{\sqrt{k_0^2 - k_\parallel^2}}$ |

### Taylor Series Divergence at Finite NA
Comparing the two projections in series expansion:
$$\sin\theta = \theta - \frac{\theta^3}{6} + \frac{\theta^5}{120} - \dots$$
$$\tan\theta = \theta + \frac{\theta^3}{3} + \frac{2\theta^5}{15} + \dots$$

- At small angles ($\theta < 10^\circ$, paraxial domain): $\sin\theta \approx \tan\theta \approx \theta$ within $2\%$.
- At $\theta = 30^\circ$ ($\mathrm{NA} = 0.5$): $\sin(30^\circ) = 0.5000$, whereas $\tan(30^\circ) = 0.5774$ (**$15.5\%$ difference**).
- At $\theta = 45^\circ$ ($\mathrm{NA} = 0.71$): $\sin(45^\circ) = 0.7071$, whereas $\tan(45^\circ) = 1.0000$ (**$41.4\%$ difference**).

---

## 6. Relevance to `arVIPAspectro` Pipeline

1. **Step 14B Fitting Model**:
   The empirical fitting function in Step 14B:
   $$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cos\left(\frac{1}{2}\arctan\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
   uses $\arctan$ (the inverse tangent), which assumes a flat-plane rectilinear mapping.

2. **Theoretical Correction Factor**:
   Because the collection objective actually operates via the Abbe sine condition ($r_{\mathrm{sph}} \propto \sin\theta_{\mathrm{air}}$), the fitted parameter $M_\theta^*$ extracted over a finite numerical aperture $\mathrm{NA}$ absorbs this difference:
   $$M_\theta^* = n M_0 \left[ 1 - \frac{1}{3}\left(\frac{\mathrm{NA}}{n}\right)^2 + \mathcal{O}\left(\frac{\mathrm{NA}}{n}\right)^4 \right]$$
   The $-\frac{1}{3}(\mathrm{NA}/n)^2$ coefficient originates directly from the difference between the third-order series coefficients of tangent ($+1/3$) and sine ($-1/6$):
   $$\frac{1}{3} - \left(-\frac{1}{6}\right) = \frac{1}{2}$$
   integrated over the circular aperture disk.

---

## 7. Cross-References

- Detailed derivation relating $M_\theta$ to sample refractive index $n$: [`refractive_index_magnification_derivation.md`](file:///Users/pierrebouvet/Documents/Code/arVIPAspectro/theory/refractive_index_magnification_derivation.md)
- Step 14 Documentation: [`docs/sections/step_14_q_vector_and_elliptical_geometry.md`](file:///Users/pierrebouvet/Documents/Code/arVIPAspectro/docs/sections/step_14_q_vector_and_elliptical_geometry.md)
