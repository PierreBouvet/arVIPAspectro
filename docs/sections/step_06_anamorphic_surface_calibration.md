# Step 06 — 2D Anamorphic Polynomial Order Surface Calibration

## Summary and Input/Output
- **Purpose**: Calibrates a 2D 4th-degree polynomial surface mapping local optical coordinates $(g(x, y), y)$ directly to the continuous interference order coordinate $\phi_{\mathrm{order}}(x, y)$, correcting for spherical aberration, lens astigmatism, and camera roll tilt.
- **Inputs**: Calculated $g(x_i, y_i)$ values for all $17,212$ calibrated sub-pixel laser points across all 11 orders $q \in \{0, \dots, 10\}$.
- **Outputs**: 2D polynomial surface coefficients $c_{m,n}$ and the 2D fractional order map $\phi_{\mathrm{order}}(x, y) \in \mathbb{R}^{2304 \times 2304}$.

---

## Algorithm
1. **Coordinate Normalization**: Normalizes coordinates to avoid ill-conditioned matrix inversion:
   $$\tilde{g} = \frac{g - \bar{g}}{\sigma_g}, \qquad \tilde{y} = \frac{y - y_0}{1000}$$
2. **2D Polynomial Basis Expansion**: Constructs a 2D Vandermonde design matrix containing all cross-terms up to total degree $M \le 4$:
   $$\phi_{\mathrm{order}}(x, y) = \sum_{m=0}^4 \sum_{n=0}^{4 - m} c_{m,n} \, \tilde{g}^m \, \tilde{y}^n$$
3. **Linear Least-Squares Regression**: Solves for $\mathbf{c} = (c_{0,0}, c_{1,0}, c_{0,1}, \dots)^T$ by minimizing the squared error against known integer orders $q \in \{0, \dots, 10\}$:
   $$\mathbf{c} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{q}$$
4. **Grid Evaluation**: Evaluates $\phi_{\mathrm{order}}(x, y)$ over the full $2304 \times 2304$ detector grid.
5. **Quality Assessment**: Evaluates the per-order mean residual $\overline{\Delta\phi}_q = \langle \phi_{\mathrm{order}} - q \rangle_{\mathcal{F}_q}$ and root-mean-square residual $\mathrm{RMS}_{\phi}$.

---

## Physical Rationale
A purely 1D dispersion model $\phi(g) = a g + b$ or $\phi(g) = a g^2 + b g + c$ assumes that the imaging lens has identical focal lengths along $x$ and $y$ and zero field aberrations. In reality:
- Spherical lenses exhibit astigmatism ($f_x \neq f_y$) across a wide $2304 \times 2304$ field.
- 4th-order Seidel spherical aberration flares the outer orders ($q \ge 8$ and $q \le 1$).
- A 1D quadratic model leaves a parabolic inter-order error of up to $\pm 70\text{ MHz}$.
The 2D Anamorphic Order Surface achieves **$0.00\text{ MHz}$ inter-order bias** and **$< 7\text{ MHz}$ de-rippled trajectory RMS**, unlocking genuine sub-10 MHz calibration.

---

## Physical Approximations
1. **Polynomial Lens Aberration Field**: Field curvature, tangential distortion, and astigmatism are well-modeled by 4th-degree polynomials in spatial coordinates.
2. **Newton-Ring Etalon Micro-Ripples**: High-frequency spatial ripples ($\sim 16\text{ px}$ wavelength, $\approx 23\text{ MHz}$ peak-to-peak) caused by parasitic reflections within the etalon are treated as stationary zero-mean noise filtered out by the smooth polynomial surface.

---

## Key Equations
- **Normalized Coordinates**:
  $$\tilde{g} = \frac{g(x, y) - \mu_g}{\sigma_g}, \qquad \tilde{y} = \frac{y - y_0}{1000}$$
- **2D Anamorphic Polynomial Surface**:
  $$\phi_{\mathrm{order}}(\tilde{g}, \tilde{y}) = \sum_{m+n \le 4} c_{m,n} \tilde{g}^m \tilde{y}^n$$
- **Linear Least-Squares Normal Equation**:
  $$\mathbf{X}^T \mathbf{X} \mathbf{c} = \mathbf{X}^T \mathbf{q}$$

---

## Code Implementation
- **Function**: `fit_anamorphic_order_surface(g_vals, y_vals, orders, order_deg=4)`
- **Function**: `eval_anamorphic_order_surface(g_grid, y_grid, c_matrix, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L385-L450)
- **Pipeline Call**: [`arVIPAspectro/laser_calib.py`](file:/arVIPAspectro/laser_calib.py#L177-L235), invoked in [`arVIPAspectro/main.py`](file:/main.py#L248-L280)
- **Diagnostic Plot**: Figure 6 (`6 - Order vs g laser reference.png`)
