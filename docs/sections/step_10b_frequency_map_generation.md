# Step 10B — Calibrated 2D Frequency Map Generation

## Summary and Input/Output
- **Purpose**: Generates the master $2304 \times 2304$ 2D optical frequency map $\nu_{\mathrm{map}}(x, y)$ in gigahertz referenced directly to the atomic Rubidium absorption notch filter lock line.
- **Inputs**: 2D order surface $\phi_{\mathrm{order}}(x, y)$, calibrated $\mathrm{FSR}_{\mathrm{cal}} \approx 31.0221\text{ GHz}$, atomic lock offset $\xi_{\mathrm{elastic}} = 0.52584$.
- **Outputs**: Absolute 2D frequency array $\nu_{\mathrm{map}}(x, y) \in \mathbb{R}^{2304 \times 2304}$ saved to `Measures.h5`.

---

## Algorithm
1. **Atomic Lock Frequency Referencing**:
   During sample measurements, the excitation laser is locked inside the deep $^{85}\text{Rb}\ (5^2S_{1/2}, F=2 \to 5^2P_{3/2})$ hyperfine absorption line located $+1.323\text{ GHz}$ above the Rubidium $D_2$ line center of gravity. In fractional order units relative to the `laser_reference` acquisition, this corresponds to:
   $$\xi_{\mathrm{elastic}} = \frac{14.980 + 1.323}{\mathrm{FSR}_{\mathrm{cal}}} \approx 0.52584$$
2. **Frequency Map Synthesis**:
   Transforms the continuous 2D fractional order surface $\phi_{\mathrm{order}}(x, y)$ into physical gigahertz frequencies:
   $$\nu_{\mathrm{map}}(x, y) = \big(\phi_{\mathrm{order}}(x, y) - \xi_{\mathrm{elastic}}\big) \cdot \mathrm{FSR}_{\mathrm{cal}}$$
   Under this mapping, pixels where $\nu_{\mathrm{map}}(x, y) = 0$ (or exact multiples of $\mathrm{FSR}$) correspond to the unshifted elastic excitation frequency.
3. **HDF5 Persistence**:
   Stores `Frequency` in `Brillouin/Rb_absorption` along with calibration attributes `FSR`, `x0_fit`, `y0_fit`, `xc_opt`, `yc_opt`, and `xi_elastic`.

---

## Physical Rationale
To fit Brillouin peaks across the entire detector image without spatial interpolation artifacts or bin-by-bin calibration, every physical sCMOS camera pixel $(x, y)$ must be assigned a unique optical frequency. Mapping continuous fractional orders directly to frequency using the atomic lock offset creates an absolute, monotonic coordinate grid spanning all interference orders.

---

## Physical Approximations
1. **Periodic Dispersion Continuity**: Frequency increases monotonically with order coordinate $\phi_{\mathrm{order}}$, with identical FSR spacing between all adjacent orders across the detector.

---

## Key Equations
- **Calibrated 2D Frequency Map**:
  $$\nu_{\mathrm{map}}(x, y) = \big(\phi_{\mathrm{order}}(x, y) - \xi_{\mathrm{elastic}}\big) \cdot \mathrm{FSR}_{\mathrm{cal}}$$
- **Order Boundary Definition**:
  $$k_{\mathrm{ord}}(x, y) = \operatorname{round}\left( \frac{\nu_{\mathrm{map}}(x, y)}{\mathrm{FSR}_{\mathrm{cal}}} \right)$$

---

## Code Implementation
- **Function**: `eval_anamorphic_order_surface(...)`
- **Module**: [`arVIPAspectro/backend.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/backend.py#L420-L460)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/main.py#L383-L415)
- **Diagnostic Plot**: Figure 10B (`10B - Frequency map.png`)
