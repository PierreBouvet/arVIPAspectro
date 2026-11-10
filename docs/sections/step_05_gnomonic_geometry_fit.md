# Step 05 — Exact 2D Gnomonic VIPA Geometry Fit

## Summary and Input/Output
- **Purpose**: Fits the physical optical axis coordinates $(x_0, y_0)$ and VIPA etalon tilt angle $\phi$ on the detector plane using exact 3D gnomonic ray tracing and Snell's law inside the fused silica substrate.
- **Inputs**: Refined sub-pixel laser fringe coordinates $\{(x_i, y_i)\}_{i \in \mathcal{F}_q}$, focal length $f = 400\text{ mm}$, pixel size $p = 6.5\ \mu\text{m}$, refractive index $n_r = 1.45367$.
- **Outputs**: Optical center $(x_0, y_0)$, gnomonic ring center $(x_{c,\mathrm{gnom}}, y_{c,\mathrm{gnom}})$, and internal incidence cosine values $g(x, y)$ for each fringe.

---

## Algorithm
1. **Gnomonic Direction Cosine Modeling**:
   For any detector pixel $(x, y)$, a ray passing through the center of an imaging lens of focal length $f$ (in pixel units $f_{\mathrm{px}} = f / p \approx 61,538.46\text{ px}$) has normalized direction vector:
   $$\hat{\mathbf{k}} = \frac{1}{\sqrt{f^2 + (x - x_0)^2 + (y - y_0)^2}} \begin{pmatrix} x - x_0 \\ y - y_0 \\ f \end{pmatrix}$$
2. **VIPA Tilt Rotation & Internal Refraction**:
   With the VIPA normal $\hat{\mathbf{n}}$ tilted by angle $\phi \approx 2.6^\circ$ in the horizontal $xz$-plane ($\hat{\mathbf{n}} = (-\sin\phi, 0, \cos\phi)^T$), the external directional cosine is:
   $$u(x, y; x_0, y_0) = \hat{\mathbf{k}} \cdot \hat{\mathbf{n}} = \frac{f \cos\phi - (x - x_0)\sin\phi}{\sqrt{f^2 + (x - x_0)^2 + (y - y_0)^2}}$$
   By Snell's law, the cosine of the internal propagation angle $\theta_{\mathrm{int}}$ inside fused silica ($n_r$) is:
   $$g(x, y; x_0, y_0) = \cos\theta_{\mathrm{int}} = \sqrt{u(x, y; x_0, y_0)^2 + n_r^2 - 1}$$
3. **Levenberg-Marquardt Optimization**:
   Because the laser is strictly monochromatic, all points on a given fringe order $q$ must correspond to the exact same internal interference angle $\cos\theta_{\mathrm{int}} = \bar{g}_q$. The optical axis $(x_0, y_0)$ is obtained by minimizing the intra-fringe residual variance across all 11 orders simultaneously:
   $$\min_{x_0, y_0} \sum_{q=0}^{10} \sum_{i \in \mathcal{F}_q} \Big[ 10^5 \cdot \big(g(x_i, y_i; x_0, y_0) - \bar{g}_q\big) \Big]^2$$
   Converging to:
   $$x_0 \approx 689.9\text{ px}, \qquad y_0 \approx 948.5\text{ px}$$
   with gnomonic apex $(x_{c,\mathrm{gnom}}, y_{c,\mathrm{gnom}}) = (x_0 - f \tan\phi, y_0) \approx (-2104.5, 948.5)\text{ px}$.

---

## Physical Rationale
The interference condition of a Fabry-Pérot or VIPA etalon depends on the internal optical path difference $\Delta = 2 n_r d \cos\theta_{\mathrm{int}}$. Standard paraxial approximations ($u \approx 1 - \theta^2/2$) break down over large sensors ($2304 \times 2304$ pixels spanning $> 2^\circ$). Exact gnomonic geometry accurately accounts for the 3D spherical-to-planar projection geometry, ensuring constant optical path difference along true physical contours.

---

## Physical Approximations
1. **Planar Etalon Geometry**: The fused silica VIPA etalon is assumed to have perfectly plane-parallel internal reflecting faces.
2. **Thin-Lens Gnomonic Mapping**: The imaging lens is modeled as an ideal f-tan($\theta$) gnomonic mapping; higher-order lens aberrations (distortion, astigmatism) are absorbed by the subsequent Step 6 anamorphic surface.

---

## Key Equations
- **External Directional Cosine**:
  $$u(x, y) = \frac{f \cos\phi - (x - x_0)\sin\phi}{\sqrt{f^2 + (x - x_0)^2 + (y - y_0)^2}}$$
- **Internal VIPA Cosine**:
  $$g(x, y) = \sqrt{u(x, y)^2 + n_r^2 - 1}$$
- **Fabry-Pérot Interference Condition**:
  $$2 n_r d \cdot g(x, y) = m \cdot \lambda_0 \implies g_m = \frac{m \lambda_0}{2 n_r d}$$

---

## Code Implementation
- **Function**: `_geom_residuals_px(params, data_fringes, f_px, ...)`
- **Function**: `fit_gnomonic_geometry(data_fringes, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/backend.py#L300-L380)
- **Pipeline Call**: [`arVIPAspectro/laser_calib.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/laser_calib.py#L102-L175), invoked in [`arVIPAspectro/main.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/main.py#L211-L245)
- **Diagnostic Plot**: Figure 5 (`5 - Fit laser reference geometry.png`)
