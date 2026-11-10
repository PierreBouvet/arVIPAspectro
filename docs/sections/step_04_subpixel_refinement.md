# Step 04 — Unsaturated Sub-Pixel Centroid Refinement & Outlier Filtering

## Summary and Input/Output
- **Purpose**: Refines the integer-pixel coordinates of all traced laser fringes to sub-pixel accuracy ($\sim 0.05\text{ px}$) using squared-intensity weighted centroids while discarding saturated ADC pixels, edge flares, and statistical outliers.
- **Inputs**: Traced integer coordinates $(x_{q, i}, y_{q, i})$, net image $I_{\mathrm{net}}$, raw image $I_{\mathrm{raw}}$.
- **Outputs**: Arrays of refined sub-pixel coordinates $(x_c, y)$, 4th-order polynomial trajectory fits $x_q(y)$, and residual standard deviations for each order.

---

## Algorithm
1. **Window Extraction & Local Chord Subtraction**: For each traced point on fringe $q$, extracts a horizontal window of half-width $W = 10\text{ px}$ centered at $x_0$. Computes a local chord baseline:
   $$b_{\mathrm{loc}} = \frac{I_{\mathrm{net}}(x_0 - W, y_0) + I_{\mathrm{net}}(x_0 + W, y_0)}{2}$$
2. **Quality & Saturation Gating**:
   - Rejects the point if $\max_x I_{\mathrm{raw}}(x, y_0) \ge 65,000$ counts (ADC clipping).
   - Rejects the point if peak amplitude above baseline $\max_x I_{\mathrm{net}}(x) - b_{\mathrm{loc}} < 200$ counts.
   - Rejects edge reflection flares (e.g. $y_0 > 1750\text{ px}$ on Fringe 0, $y_0 < 420\text{ px}$ on Fringe 10).
3. **Squared-Intensity Centroid Calculation**:
   $$x_c(y_0) = \frac{\sum_{x = x_0 - W}^{x_0 + W} x \cdot \big[\max(I_{\mathrm{net}}(x, y_0) - b_{\mathrm{loc}}, 0)\big]^2}{\sum_{x = x_0 - W}^{x_0 + W} \big[\max(I_{\mathrm{net}}(x, y_0) - b_{\mathrm{loc}}, 0)\big]^2}$$
   Squaring the net intensity sharpens the effective weighting profile, reducing sensitivity to noise in the line wings.
4. **Iterative $2.5\sigma$ Polynomial Outlier Rejection**:
   - Fits a 4th-order polynomial $x_q(y) = \sum_{p=0}^4 a_p y^p$ to each fringe.
   - Computes residuals $\delta x = x_c - x_q(y)$ and discards points with $|\delta x| > 2.5\,\sigma_{\delta x}$.
   - Iterates 3 times until the set of retained points stabilizes ($> 17,200$ clean points across 11 orders).

---

## Physical Rationale
At the VIPA dispersion scale ($\sim 0.5\text{ GHz/px}$), an uncertainty of $0.5\text{ px}$ in fringe center position corresponds to a frequency error of $250\text{ MHz}$ — unacceptably high for sub-10 MHz precision. Sub-pixel centroiding reduces positional uncertainty below $0.05\text{ px}$ ($\sim 25\text{ MHz}$ raw, and $< 7\text{ MHz}$ once filtered), establishing an ultra-precise optical coordinate grid.

---

## Physical Approximations
1. **Locally Symmetric Diffraction Wing Profile**: The instrumental point-spread function of the VIPA in the dispersion direction is modeled as locally symmetric near its central peak after subtracting the chord baseline.
2. **Smooth Boundary Field**: Optical aberrations distorting the fringes along $y$ are smooth and well-captured by 4th-degree polynomials.

---

## Key Equations
- **Squared-Intensity Sub-Pixel Centroid**:
  $$x_c = \frac{\sum_i x_i \, w_i^2}{\sum_i w_i^2}, \qquad w_i = \max\big(I_{\mathrm{net}}(x_i) - b_{\mathrm{loc}},\, 0\big)$$
- **Fitted Fringe Trajectory**:
  $$x_q(y) = a_4 y^4 + a_3 y^3 + a_2 y^2 + a_1 y + a_0$$

---

## Code Implementation
- **Function**: `subpixel_center_coordinates_and_interpolate_zeros(data_fringes, image, image_raw, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/backend.py#L210-L295)
- **Pipeline Call**: [`arVIPAspectro/laser_calib.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/laser_calib.py#L72-L100), invoked in [`arVIPAspectro/main.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/main.py#L183-L208)
- **Diagnostic Plot**: Figure 4 (`4 - Refine fringe laser reference.png`)
