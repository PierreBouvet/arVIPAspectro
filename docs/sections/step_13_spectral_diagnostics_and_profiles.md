# Step 13 — Spectral Profiles, Shift & Linewidth Diagnostics

## Summary and Input/Output
- **Purpose**: Generates publication-grade diagnostic plots of the fitted DHO parameters along individual interference orders and across the 2D detector face to verify Stokes/Anti-Stokes convergence and spatial continuity.
- **Inputs**: Fitted 3D arrays of Brillouin shift $\Delta\nu_B$, linewidth $\Gamma_B$, and sub-pixel spatial coordinates $(x, y)$.
- **Outputs**: High-resolution diagnostic figures (Figures 13 and 13B) validating the quality and physical consistency of the spectral fits.

---

## Algorithm
1. **Convergence Filtering**:
   Discards fits where parameters hit the optimizer boundary limits (e.g. $\nu_0 < 0$, $\Gamma_B < 0.2\text{ GHz}$, or $\Gamma_B > 2.5\text{ GHz}$), where the standard error exceeds user-defined thresholds, or where edge truncation occurred.
2. **Branch Symmetry & Order Averaging**:
   For each spatial row bin $i$ and interference order $k$, evaluates the symmetry between Stokes ($\sigma = +1$) and Anti-Stokes ($\sigma = -1$) components:
   $$\Delta\nu_{\mathrm{avg}}(i, k) = \frac{\Delta\nu_S(i, k) + \Delta\nu_{AS}(i, k)}{2}$$
   $$\Gamma_{\mathrm{avg}}(i, k) = \frac{\Gamma_S(i, k) + \Gamma_{AS}(i, k)}{2}$$
3. **Multi-Panel Plotting Suite**:
   - **Figure 13**: 4-panel overview showing 2D scatter plots of $\Delta\nu_B(x, y)$ and $\Gamma_B(x, y)$ overlaid on the interferogram, alongside 1D profiles along the vertical detector coordinate $y$.
   - **Figure 13B**: 10-panel per-fringe breakdown grid showing individual Stokes (red), Anti-Stokes (blue), and order-averaged (black) profiles for Fringes $0$ through $9$.

---

## Physical Rationale
In an ideal angle-resolved BLS spectrometer:
- The Brillouin shift $\Delta\nu_B(y)$ along a fringe must form a smooth, concave-down parabolic arc reflecting the collection lens numerical aperture angle profile $\theta(y)$.
- Stokes and Anti-Stokes branches must overlay each other within experimental noise ($\le 35\text{ MHz}$).
- Discontinuities, sharp jumps, or divergences indicate localized optical defects, dust, or detector boundary reflections that must be screened before extracting acoustic wavevectors.

---

## Key Equations
- **Order-Averaged Brillouin Frequency Shift**:
  $$\Delta\nu_{\mathrm{avg}}(y, k) = \frac{\Delta\nu_S(y, k) + \Delta\nu_{AS}(y, k)}{2}$$
- **Stokes/Anti-Stokes Splitting Residual**:
  $$\Delta\nu_{\mathrm{split}}(y, k) = |\Delta\nu_S(y, k) - \Delta\nu_{AS}(y, k)|$$

---

## Code Implementation
- **Function**: `plot_fringe_shift_and_linewidth_overview(...)`
- **Function**: `plot_per_fringe_breakdown(...)`
- **Module**: [`arVIPAspectro/plotting.py`](file:/arVIPAspectro/plotting.py#L380-L465)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L488-L515)
- **Diagnostic Plots**: Figure 13 (`13 - <sample> shift and linewidth values.png`) and Figure 13B (`13B - <sample> per-fringe breakdown.png`)
