# Step 09 — Radial Profile Extraction of Rubidium Absorption

## Summary and Input/Output
- **Purpose**: Extracts the 1D spectral absorption profile of Rubidium by integrating image intensity along 2D concentric circular arcs centered at the gnomonic ring center $(x_{c,\mathrm{gnom}}, y_{c,\mathrm{gnom}})$, preventing horizontal curvature smearing.
- **Inputs**: Net Rubidium frame $I_{\mathrm{net},\mathrm{Rb}}(x, y)$ and gnomonic ring apex $(x_{c,\mathrm{gnom}}, y_{c,\mathrm{gnom}}) = (-2104.5, 948.5)\text{ px}$.
- **Outputs**: 1D radial absorption profile $\bar{I}_{\mathrm{Rb}}(R)$ as a function of radial distance $R \in [R_{\min}, R_{\max}]$.

---

## Algorithm
1. **Radial Coordinate Mapping**: For every detector pixel $(x, y)$, computes its Euclidean distance from the gnomonic apex:
   $$R(x, y) = \sqrt{(x - x_{c,\mathrm{gnom}})^2 + (y - y_{c,\mathrm{gnom}})^2}$$
2. **Angular Arc Binning**: Bins pixels into fine radial intervals $[R_k, R_k + \Delta R]$ ($\Delta R = 1.0\text{ px}$) within an angular acceptance sector matching the beam footprint.
3. **Intensity Averaging**: Computes the mean intensity within each radial annulus:
   $$\bar{I}_{\mathrm{Rb}}(R_k) = \frac{1}{N_k} \sum_{(x, y) \in \mathcal{A}_k} I_{\mathrm{net},\mathrm{Rb}}(x, y)$$
4. **HDF5 Caching**: Stores the extracted 1D profile `profile` under `Rb_absorption` in `Measures.h5`.

---

## Physical Rationale
Because the VIPA interference fringes are curved in 2D space, integrating vertically along detector columns or horizontally along rows causes severe instrumental smearing: the sharp $\sim 500\text{ MHz}$ Rubidium hyperfine dips would blur together and lose their depth. Integrating along exact gnomonic concentric circular arcs collapses thousands of camera pixels onto true iso-frequency contours, maximizing signal-to-noise without sacrificing spectral resolution.

---

## Physical Approximations
1. **Concentric Gnomonic Iso-Frequency Contours**: Fringes are approximated as circular arcs around the gnomonic center $(x_{c,\mathrm{gnom}}, y_{c,\mathrm{gnom}})$ to first order (with fine higher-order corrections handled during FSR optimization).

---

## Key Equations
- **Gnomonic Radial Distance**:
  $$R(x, y) = \sqrt{(x - x_0 + f \tan\phi)^2 + (y - y_0)^2}$$
- **Iso-Frequency Arc Integration**:
  $$\bar{I}_{\mathrm{Rb}}(R) = \int_{\theta_1}^{\theta_2} I_{\mathrm{net},\mathrm{Rb}}\big(x(R, \theta), y(R, \theta)\big) \, \mathrm{d}\theta$$

---

## Code Implementation
- **Function**: `extract_profiles_2d(image, xc, yc, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/backend.py#L565-L620)
- **Pipeline Call**: [`arVIPAspectro/Rb_calib.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/Rb_calib.py#L142-L195), invoked in [`arVIPAspectro/main.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/main.py#L333-L355)
- **Diagnostic Plot**: Figure 9 (`9 - Radial profile rubidium absorption.png`)
