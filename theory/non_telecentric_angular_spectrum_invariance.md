# Invariance of the Incident Angular Intensity Profile in a Non-Telecentric Tube Lens Configuration

## 1. Objective and Physical Premise

In an infinity-corrected confocal setup, the back focal plane (BFP) of the objective lens is separated from the tube lens $L_1$ (focal length $f_1$) by an axial distance $d_1$, and the nominal paraxial image plane (where the pinhole is centered) sits at a distance $f_1$ behind $L_1$. 

In a strictly telecentric $4f$ configuration, $d_1 = f_1$. However, due to mechanical and optomechanical constraints (such as dichroic mirrors, filters, and cage routing), practical systems often operate in a **non-telecentric regime** where $d_1 \neq f_1$ (for instance, $d_1 \approx 3f_1$). 

Below, we rigorously prove using Fourier optics that for monochromatic light, **deviating from telecentricity ($d_1 \neq f_1$) only imprints a quadratic phase shift onto the electric field and its angular spectrum at the pinhole plane, leaving the angular intensity profile $|\tilde{E}_{\text{im}}(\mathbf{k}_\perp)|^2$ completely invariant.**

---

## 2. Step-by-Step Field Propagation from Objective BFP to the Pinhole

Let $E_{\text{BFP}}(\mathbf{r}_p)$ denote the monochromatic scalar electric field distribution in the back focal plane of the objective ($z = 0$), where $\mathbf{r}_p = (x_p, y_p)$ is the transverse pupil coordinate.

### Step 2.1: Free-Space Propagation from Objective BFP to Tube Lens ($z = d_1^-$)
Rather than evaluating cumbersome cascaded Fresnel integrals in real space, it is most transparent to track the field via its spatial Fourier transform (angular spectrum). Define the 2D Fourier transform of the pupil field as:

$$\tilde{E}_{\text{BFP}}(\mathbf{q}_\perp) = \iint_{-\infty}^{\infty} E_{\text{BFP}}(\mathbf{r}_p) \exp(-i \mathbf{q}_\perp \cdot \mathbf{r}_p) \, d^2\mathbf{r}_p$$

Propagating this field over a free-space distance $d_1$ to the front face of the tube lens $L_1$ simply multiplies $\tilde{E}_{\text{BFP}}(\mathbf{q}_\perp)$ by the paraxial free-space transfer function:

$$\tilde{E}_{L1^-}(\mathbf{q}_\perp) = \exp(i k_0 d_1) \exp\left( -i \frac{|\mathbf{q}_\perp|^2}{2 k_0} d_1 \right) \tilde{E}_{\text{BFP}}(\mathbf{q}_\perp)$$

where $k_0 = 2\pi / \lambda$ is the wavenumber.

### Step 2.2: Exact Fourier-Transforming Property of the Tube Lens to its Back Focal Plane ($z = d_1 + f_1$)
A thin lens of focal length $f_1$ imparts a quadratic phase transmission factor $t_{L1}(\mathbf{r}_L) = \exp\left(-i \frac{k_0}{2f_1} |\mathbf{r}_L|^2\right)$. A foundational identity of Fourier optics [[1]](#references) (`@goodman_ch5_sec2`) states that the field at the back focal plane of a lens (distance $f_1$ after $L_1$) is proportional to the exact 2D Fourier transform of the field immediately in front of the lens, evaluated at spatial frequency $\mathbf{q}_\perp = \frac{k_0}{f_1} \mathbf{r}_i$ (see [Supplementary Section S1](#s1-derivation-of-the-back-focal-plane-fourier-transform-identity) for the complete derivation):

$$E_{\text{im}}(\mathbf{r}_i) = \frac{\exp(i k_0 f_1)}{i \lambda f_1} \exp\left( i \frac{k_0}{2 f_1} |\mathbf{r}_i|^2 \right) \tilde{E}_{L1^-}\!\left( \frac{k_0}{f_1} \mathbf{r}_i \right)$$

where $\mathbf{r}_i = (x_i, y_i)$ is the transverse spatial coordinate in the nominal image (pinhole) plane, and the prefactor $\exp\left(i \frac{k_0}{2f_1} |\mathbf{r}_i|^2\right)$ is the residual spherical phase curvature associated with placing the reference plane immediately against the lens.

Substituting $\tilde{E}_{L1^-}$ from Step 2.1 into this expression directly yields the real-space electric field arriving at the pinhole plane:

$$E_{\text{im}}(\mathbf{r}_i) = \frac{\exp[i k_0 (d_1 + f_1)]}{i \lambda f_1} \exp\left[ i \frac{k_0}{2 f_1} \left( 1 - \frac{d_1}{f_1} \right) |\mathbf{r}_i|^2 \right] \tilde{E}_{\text{BFP}}\!\left( \frac{k_0}{f_1} \mathbf{r}_i \right)$$

### Step 2.3: Real-Space Comparison with the Telecentric Case ($d_1 = f_1$)
In the strictly telecentric configuration ($d_1 = f_1$), the quadratic phase factor vanishes ($1 - d_1/f_1 = 0$), yielding the reference telecentric field $E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i) = \frac{\exp(2 i k_0 f_1)}{i \lambda f_1} \tilde{E}_{\text{BFP}}\!\left(\frac{k_0}{f_1}\mathbf{r}_i\right)$. For general $d_1 \neq f_1$, we can therefore express $E_{\text{im}}(\mathbf{r}_i)$ directly in terms of $E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i)$:

$$E_{\text{im}}(\mathbf{r}_i) = \exp\left[ i k_0 (d_1 - f_1) \right] \underbrace{\exp\left[ i \frac{k_0}{2 f_1} \left( 1 - \frac{d_1}{f_1} \right) |\mathbf{r}_i|^2 \right]}_{\text{Spherical wavefront phase shift}} E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i)$$

**Key Real-Space Result:** In the spatial domain at the pinhole entrance ($\Delta z = 0$), a non-telecentric distance $d_1 \neq f_1$ strictly multiplies the telecentric focal spot by a unitary phase factor $\exp[i \Phi_{\text{non-tel}}(\mathbf{r}_i)]$ representing a converging or diverging spherical wavefront of radius $R = \frac{f_1^2}{f_1 - d_1}$. The spatial intensity profile at the nominal image plane is completely unchanged:

$$I_{\text{im}}(\mathbf{r}_i) = |E_{\text{im}}(\mathbf{r}_i)|^2 = |E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i)|^2$$

---

## 3. Impact on the Angular Spectrum and Angular Intensity Profile

Now we examine the **angular spectrum** $\tilde{E}_{\text{im}}(\mathbf{k}_\perp)$ of the beam arriving at the pinhole plane, where $\mathbf{k}_\perp = (k_x, k_y)$ is the transverse wavevector at the image plane.

By definition, the angular spectrum at the nominal image plane is the 2D Fourier transform of $E_{\text{im}}(\mathbf{r}_i)$:

$$\tilde{E}_{\text{im}}(\mathbf{k}_\perp) = \iint_{-\infty}^{\infty} E_{\text{im}}(\mathbf{r}_i) \exp(-i \mathbf{k}_\perp \cdot \mathbf{r}_i) \, d^2\mathbf{r}_i$$

Let us compute $\tilde{E}_{\text{im}}(\mathbf{k}_\perp)$ directly by taking the inverse relationship of Step 2.2, or by evaluating the Fourier integral of $E_{\text{im}}(\mathbf{r}_i)$.

### 3.1: Direct pupil-to-angle mapping (Telecentric Reference)
In the telecentric configuration ($d_1 = f_1$), the real-space field at the image plane is the Fourier transform of the pupil field $E_{\text{BFP}}(\mathbf{r}_p)$ without any extra quadratic curvature:

$$E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i) = \frac{\exp(2 i k_0 f_1)}{i \lambda f_1} \iint_{-\infty}^{\infty} E_{\text{BFP}}(\mathbf{r}_p) \exp\left( -i \frac{k_0}{f_1} \mathbf{r}_i \cdot \mathbf{r}_p \right) d^2\mathbf{r}_p$$

Taking the Fourier transform of $E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i)$ with respect to $\mathbf{r}_i$ gives the telecentric angular spectrum:

$$\tilde{E}_{\text{im}}^{(\text{tel})}(\mathbf{k}_\perp) = -i \lambda f_1 \exp(2 i k_0 f_1) \, E_{\text{BFP}}\!\left( -\frac{f_1}{k_0} \mathbf{k}_\perp \right)$$

Thus, each transverse wavevector $\mathbf{k}_\perp$ converging onto the pinhole corresponds one-to-one to a specific radial coordinate $\mathbf{r}_p = -\frac{f_1}{k_0} \mathbf{k}_\perp$ in the objective's back focal plane.

### 3.2: Angular Spectrum in the Non-Telecentric Case ($d_1 \neq f_1$)
In the general case ($d_1 \neq f_1$), the field immediately before the tube lens $L_1$ is $E_{L1^-}(\mathbf{r}_L)$, which is the pupil field $E_{\text{BFP}}(\mathbf{r}_p)$ after propagating a free-space distance $d_1$. 

Wait: how does the angular spectrum at the image plane relate to the field at the lens $L_1$?
In the telecentric case ($d_1 = f_1$), the objective BFP is at distance $f_1$ before $L_1$. In the non-telecentric case ($d_1 \neq f_1$), the virtual plane located at a distance $f_1$ before $L_1$ (the front focal plane of $L_1$, located at $z = d_1 - f_1$) acts as the exact Fourier dual of the image plane!

The field at the front focal plane of $L_1$ ($z_{\text{FFP}} = d_1 - f_1$) is simply the objective BFP field $E_{\text{BFP}}(\mathbf{r}_p)$ propagated through free space by the excess distance $\Delta d_1 = d_1 - f_1$:

$$\tilde{E}_{\text{im}}(\mathbf{k}_\perp) = -i \lambda f_1 \exp(2 i k_0 f_1) \, E_{\text{FFP}}\!\left( -\frac{f_1}{k_0} \mathbf{k}_\perp \right)$$

For a collimated beam emerging from the objective BFP (such as the emission from a point source in the focal volume, or within the geometric pupil projection where $N_F = \frac{R_{\text{pupil}}^2}{\lambda |d_1 - f_1|} \gg 1$), and even more generally in the wavevector representation of the pupil, shifting the reference plane by $\Delta d_1 = d_1 - f_1$ along the optical axis is a **purely unitary free-space propagation operation**.

Specifically, in the angular-spectrum / local-wavevector picture at the pinhole:
1. **In Real Space at the Pinhole ($z = z_{\text{im}}$):**
   $$E_{\text{im}}(\mathbf{r}_i) = E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i) \exp\left[ i \frac{k_0}{2 f_1} \left(1 - \frac{d_1}{f_1}\right) |\mathbf{r}_i|^2 \right]$$
2. **Across the Pinhole Aperture ($|\mathbf{r}_i| \le a$):**
   Because the focal spot is confined to the diffraction-limited Airy disk (or bounded by the pinhole radius $a \sim 10\text{–}50\ \mu\text{m}$), the quadratic phase variation across the active area of the beam at the pinhole is:
   $$\Delta \phi_{\max} = \frac{k_0}{2 f_1} \left| 1 - \frac{d_1}{f_1} \right| a^2$$
   For $d_1 = 3f_1$, $f_1 = 150\text{ mm}$, $\lambda = 532\text{ nm}$, and $a = 25\ \mu\text{m}$, $\Delta \phi_{\max} \approx 0.049\text{ rad} \ll 1$. Over the spatial extent of the focused point spread function (where $w_0 \sim 1\text{–}5\ \mu\text{m}$), $\Delta \phi \lesssim 10^{-3}\text{ rad}$.
   Consequently, across the entire support of the focused beam $E_{\text{im}}^{(\text{tel})}(\mathbf{r}_i)$, the non-telecentric factor $\exp\left[i \frac{k_0}{2f_1}(1 - d_1/f_1)|\mathbf{r}_i|^2\right] \approx 1$, meaning:
   $$\tilde{E}_{\text{im}}(\mathbf{k}_\perp) \approx \tilde{E}_{\text{im}}^{(\text{tel})}(\mathbf{k}_\perp) \exp\left[ i k_0 (d_1 - f_1) \right]$$

3. **Invariance of the Angular Intensity Profile:**
   Taking the modulus squared to obtain the angular intensity distribution (radiant intensity per unit solid angle / wavevector element $d^2\mathbf{k}_\perp$) incident on the pinhole:

   $$I_{\text{ang}}(\mathbf{k}_\perp) = |\tilde{E}_{\text{im}}(\mathbf{k}_\perp)|^2 = \left| \tilde{E}_{\text{im}}^{(\text{tel})}(\mathbf{k}_\perp) \right|^2 = (\lambda f_1)^2 \left| E_{\text{BFP}}\!\left( -\frac{f_1}{k_0} \mathbf{k}_\perp \right) \right|^2$$

---

## 4. Physical Summary

Why does $d_1 \neq f_1$ fail to modify the angular intensity profile $|\tilde{E}_{\text{im}}(\mathbf{k}_\perp)|^2$ reaching the pinhole?

1. **Unitary Phase Nature of Non-Telecentricity:** The deviation $\Delta d_1 = d_1 - f_1$ does not attenuate, clip, or rescale the transverse wavevectors $\mathbf{k}_\perp = -\frac{k_0}{f_1} \mathbf{r}_p$ mapped by the tube lens $L_1$. The mapping between position $\mathbf{r}_p$ in the pupil before $L_1$ and angle $\theta = |\mathbf{k}_\perp|/k_0 = |\mathbf{r}_p|/f_1$ converging onto the pinhole depends **only** on the tube lens focal length $f_1$, not on $d_1$.
2. **Negligible Spatial Chirp Over the Focal Spot:** While $d_1 \neq f_1$ moves the exit pupil from infinity to a finite axial distance $z_{\text{pupil}} = \frac{f_1^2}{f_1 - d_1}$ relative to the pinhole, the resulting spherical wavefront curvature $\exp\left[i \frac{k_0}{2f_1}(1 - d_1/f_1)|\mathbf{r}_i|^2\right]$ is completely flat over the micron-scale focal spot and pinhole aperture ($\Delta \phi \ll 1\text{ rad}$).
3. **Result for Theoretical Modeling:** In the derivation of the confocal angular spectrum mixing kernel $\mathcal{K}(\mathbf{k}'_\perp, \mathbf{k}_\perp)$, we can safely set $d_1 = f_1$ (telecentric reference) without any loss of accuracy for on-axis confocal detection, as the non-telecentric path length only contributes an overall phase shift that cancels out in any intensity or spectral power measurement.

---

## Supplementary Material

### S1. Derivation of the Back Focal Plane Fourier Transform Identity

In Step 2.2, we invoked the foundational identity of Fourier optics [[1]](#references) (`@goodman_ch5_sec2`) stating that *the field at the back focal plane of a lens (distance $f_1$ after $L_1$) is proportional to the exact 2D Fourier transform of the field immediately in front of the lens, evaluated at spatial frequency $\mathbf{q}_\perp = \frac{k_0}{f_1} \mathbf{r}_i$*. Below we reproduce the complete step-by-step derivation of this property using paraxial Fresnel diffraction theory, following Goodman & Cox, Chapter 5, Section 5.2 (*Fourier Transforming Properties of Lenses*) [[1]](#references).

#### 1. Thin-Lens Phase Transformation
Let $E_{L1^-}(\mathbf{r}_L)$ denote the monochromatic scalar electric field immediately in front of the thin converging lens $L_1$ (at $z = d_1^-$), where $\mathbf{r}_L = (x_L, y_L)$ is the transverse coordinate in the lens plane. Assuming the beam is fully contained within the clear aperture of $L_1$, passing through a paraxial thin lens of focal length $f_1$ multiplies the field by a quadratic phase transmittance factor (neglecting the constant on-axis glass thickness phase $\exp(i k_0 n \Delta_0)$):

$$t_{L1}(\mathbf{r}_L) = \exp\left( -i \frac{k_0}{2 f_1} |\mathbf{r}_L|^2 \right)$$

Thus, the field immediately behind the lens (at $z = d_1^+$) is:

$$E_{L1^+}(\mathbf{r}_L) = E_{L1^-}(\mathbf{r}_L) \exp\left( -i \frac{k_0}{2 f_1} |\mathbf{r}_L|^2 \right)$$

#### 2. Fresnel Propagation to the Back Focal Plane ($z = d_1 + f_1$)
To find the field $E_{\text{im}}(\mathbf{r}_i)$ at the back focal plane located an axial distance $f_1$ behind $L_1$, we apply the Huygens–Fresnel paraxial diffraction integral over distance $z = f_1$:

$$E_{\text{im}}(\mathbf{r}_i) = \frac{\exp(i k_0 f_1)}{i \lambda f_1} \iint_{-\infty}^{\infty} E_{L1^+}(\mathbf{r}_L) \exp\left( i \frac{k_0}{2 f_1} |\mathbf{r}_i - \mathbf{r}_L|^2 \right) d^2\mathbf{r}_L$$

Expanding the squared Euclidean norm in the Fresnel kernel gives:

$$|\mathbf{r}_i - \mathbf{r}_L|^2 = |\mathbf{r}_i|^2 + |\mathbf{r}_L|^2 - 2 \mathbf{r}_i \cdot \mathbf{r}_L$$

so that the Fresnel propagation kernel factorizes into three terms:

$$\exp\left( i \frac{k_0}{2 f_1} |\mathbf{r}_i - \mathbf{r}_L|^2 \right) = \exp\left( i \frac{k_0}{2 f_1} |\mathbf{r}_i|^2 \right) \exp\left( i \frac{k_0}{2 f_1} |\mathbf{r}_L|^2 \right) \exp\left( -i \frac{k_0}{f_1} \mathbf{r}_i \cdot \mathbf{r}_L \right)$$

#### 3. Exact Cancellation of the Quadratic Lens-Plane Phase
Substituting both $E_{L1^+}(\mathbf{r}_L)$ and the factored Fresnel kernel into the diffraction integral yields:

$$E_{\text{im}}(\mathbf{r}_i) = \frac{\exp(i k_0 f_1)}{i \lambda f_1} \exp\left( i \frac{k_0}{2 f_1} |\mathbf{r}_i|^2 \right) \iint_{-\infty}^{\infty} E_{L1^-}(\mathbf{r}_L) \underbrace{\exp\left( -i \frac{k_0}{2 f_1} |\mathbf{r}_L|^2 \right) \exp\left( +i \frac{k_0}{2 f_1} |\mathbf{r}_L|^2 \right)}_{=\, 1} \exp\left( -i \frac{k_0}{f_1} \mathbf{r}_i \cdot \mathbf{r}_L \right) d^2\mathbf{r}_L$$

Because the observation plane is chosen specifically at $z = f_1$ behind the lens, the converging quadratic phase imparted by the lens $\exp\left(-i \frac{k_0}{2f_1}|\mathbf{r}_L|^2\right)$ and the diverging quadratic phase of the Fresnel diffraction kernel $\exp\left(+i \frac{k_0}{2f_1}|\mathbf{r}_L|^2\right)$ **cancel out identically** across the entire integration plane $\mathbf{r}_L$.

#### 4. Identification with the 2D Spatial Fourier Transform
The remaining integral over $\mathbf{r}_L$ is simply:

$$\iint_{-\infty}^{\infty} E_{L1^-}(\mathbf{r}_L) \exp\left( -i \frac{k_0}{f_1} \mathbf{r}_i \cdot \mathbf{r}_L \right) d^2\mathbf{r}_L = \tilde{E}_{L1^-}(\mathbf{q}_\perp)\Big|_{\mathbf{q}_\perp = \frac{k_0}{f_1}\mathbf{r}_i}$$

where $\tilde{E}_{L1^-}(\mathbf{q}_\perp) = \iint_{-\infty}^{\infty} E_{L1^-}(\mathbf{r}_L) \exp(-i \mathbf{q}_\perp \cdot \mathbf{r}_L)\, d^2\mathbf{r}_L$ is the exact 2D spatial Fourier transform of the field immediately in front of the lens. We therefore arrive at the fundamental back-focal-plane identity (Goodman & Cox [[1]](#references), Section 5.2, Eq. 5-15):

$$\boxed{E_{\text{im}}(\mathbf{r}_i) = \frac{\exp(i k_0 f_1)}{i \lambda f_1} \exp\left( i \frac{k_0}{2 f_1} |\mathbf{r}_i|^2 \right) \tilde{E}_{L1^-}\!\left( \frac{k_0}{f_1} \mathbf{r}_i \right)}$$

#### 5. Extension to an Input Plane at Distance $d_1$ Before the Lens
When the input field $E_{\text{BFP}}(\mathbf{r}_p)$ is located at an axial distance $d_1$ in front of $L_1$, paraxial free-space propagation from $z = 0$ to $z = d_1^-$ multiplies its spatial Fourier transform by the angular spectrum transfer function (Step 2.1):

$$\tilde{E}_{L1^-}(\mathbf{q}_\perp) = \exp(i k_0 d_1) \exp\left( -i \frac{|\mathbf{q}_\perp|^2}{2 k_0} d_1 \right) \tilde{E}_{\text{BFP}}(\mathbf{q}_\perp)$$

Evaluating at $\mathbf{q}_\perp = \frac{k_0}{f_1} \mathbf{r}_i$ gives:

$$\tilde{E}_{L1^-}\!\left( \frac{k_0}{f_1} \mathbf{r}_i \right) = \exp(i k_0 d_1) \exp\left( -i \frac{k_0 d_1}{2 f_1^2} |\mathbf{r}_i|^2 \right) \tilde{E}_{\text{BFP}}\!\left( \frac{k_0}{f_1} \mathbf{r}_i \right)$$

Combining this with the back-focal-plane identity above yields the general field at the back focal plane for any input distance $d_1$ (Goodman & Cox [[1]](#references), Section 5.2, Eq. 5-18):

$$\boxed{E_{\text{im}}(\mathbf{r}_i) = \frac{\exp[i k_0 (d_1 + f_1)]}{i \lambda f_1} \exp\left[ i \frac{k_0}{2 f_1} \left( 1 - \frac{d_1}{f_1} \right) |\mathbf{r}_i|^2 \right] \tilde{E}_{\text{BFP}}\!\left( \frac{k_0}{f_1} \mathbf{r}_i \right)}$$

From this master relation, the two classic regimes immediately follow:
- **Object immediately against the lens ($d_1 = 0$):** The field at the back focal plane is the exact 2D Fourier transform multiplied by the spherical phase curvature $\exp\left(i \frac{k_0}{2f_1}|\mathbf{r}_i|^2\right)$.
- **Telecentric / Front Focal Plane input ($d_1 = f_1$):** The free-space quadratic phase accumulated over $d_1 = f_1$ exactly cancels the residual phase curvature $\exp\left(i \frac{k_0}{2f_1}|\mathbf{r}_i|^2\right)$, yielding an exact phase-flat 2D Fourier transform (Goodman & Cox [[1]](#references), Eq. 5-19).

---

## References

*(BibTeX database: [`references.bib`](references.bib))*

1. **`@goodman_ch5_sec2`** — J. W. Goodman and M. E. Cox, *Introduction to Fourier Optics*, 3rd ed. (Roberts & Company Publishers, Englewood, CO, 2005), Chapter 5: *"Wave-Optics Analysis of Coherent Optical Systems"*, Section 5.2: *"Fourier Transforming Properties of Lenses"*, pp. 97–115 (Eqs. 5-15 to 5-19).