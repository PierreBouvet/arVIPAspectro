# Step 07 — Theoretical Rubidium Spectrum Model

## Summary and Input/Output
- **Purpose**: Computes the high-accuracy quantum transmission spectrum $T_{\mathrm{Rb}}(\Delta\nu)$ of natural Rubidium ($^{85}\text{Rb} / ^{87}\text{Rb}$) vapor near the $D_2$ line ($780.241\text{ nm}$) as an absolute primary atomic frequency standard.
- **Inputs**: Vapor cell temperature $T = 70.0^\circ\text{C}$, cell length $L = 7.0\text{ cm}$, Steck/CODATA atomic transition constants.
- **Outputs**: High-resolution reference lookup table $(\Delta\nu, T_{\mathrm{Rb}}(\Delta\nu))$ over a $70\text{ GHz}$ span (70,001 points, $1\text{ MHz}$ spacing).

---

## Algorithm
1. **Rubidium Vapor Pressure & Number Density**:
   Calculates total saturated vapor pressure $P_{\mathrm{Torr}}$ using the Alcock et al. empirical equation:
   $$\log_{10} P_{\mathrm{Torr}} = 2.881 + 4.312 - \frac{4040}{T_{\mathrm{Kelvin}}}$$
   Total atomic density $n_{\mathrm{Rb}} = P / (k_B T)$. Isotopic abundances: $^{85}\text{Rb}$ ($72.17\%$), $^{87}\text{Rb}$ ($27.83\%$).
2. **Hyperfine Transition Multiplet Summation**:
   Identifies all 12 allowed electric-dipole transitions $5^2S_{1/2} (F) \to 5^2P_{3/2} (F')$:
   - 6 transitions for $^{85}\text{Rb}$ ($F=2, 3$).
   - 6 transitions for $^{87}\text{Rb}$ ($F=1, 2$).
   Each transition $m$ has detuning $\Delta\nu_m$ from the center-of-gravity and relative line strength $w_m$.
3. **Voigt Line Profile Evaluation**:
   Convolves the thermal Maxwell-Boltzmann Gaussian Doppler distribution ($\mathrm{FWHM}_D \approx 515\text{ MHz}$) with the Lorentzian natural linewidth ($\Gamma_{\mathrm{nat}} = 6.0666\text{ MHz}$) plus self-broadening ($\gamma_{\mathrm{self}} n_{\mathrm{Rb}}$):
   $$V(\nu - \nu_m; \sigma_D, \gamma_L) = \frac{\mathrm{Re}\big[w(z)\big]}{\sqrt{2\pi}\sigma_D}, \qquad z = \frac{(\nu - \nu_m) + i \gamma_L}{\sqrt{2}\sigma_D}$$
   evaluated using the complex Faddeeva function `scipy.special.wofz`.
4. **Beer-Lambert Law Transmission**:
   $$\alpha(\Delta\nu) = \sum_{m=1}^{12} n_i \, w_m \, \sigma_{\mathrm{int}} \, V(\Delta\nu - \Delta\nu_m)$$
   $$T_{\mathrm{Rb}}(\Delta\nu) = \exp\big(-\alpha(\Delta\nu) \cdot L\big)$$

---

## Physical Rationale
Unlike optical cavity lengths that drift with ambient temperature, humidity, and barometric pressure, atomic energy levels are fundamental invariants. Natural Rubidium provides two deep absorption doublets separated by $\sim 3\text{--}7\text{ GHz}$ that act as an unyielding frequency ruler across the VIPA free spectral range.

---

## Physical Approximations
1. **Linear Absorption Regime**: Assumes laser intensity inside the cell is well below the saturation intensity ($I \ll I_{\mathrm{sat}} \approx 1.67\text{ mW/cm}^2$), valid for diffuse reference illumination.
2. **Pure Isotopic Vapor**: Buffer gas broadening and shift (e.g. from Argon or Nitrogen) are zero for evacuated pure-metal quartz cells.

---

## Key Equations
- **Alcock Vapor Pressure Equation**:
  $$\log_{10} P_{\mathrm{Torr}} = 7.193 - \frac{4040}{T}$$
- **Integrated Absorption Cross-Section**:
  $$\sigma_{\mathrm{int}} = \frac{3 \lambda_0^2}{2\pi} \cdot \left(\frac{\pi \Gamma_{\mathrm{nat}}}{2}\right)$$
- **Beer-Lambert Saturated Transmission**:
  $$T_{\mathrm{Rb}}(\Delta\nu) = \exp\left( -L \sum_{m=1}^{12} n_m \sigma_{\mathrm{int}} V(\Delta\nu - \Delta\nu_m; \sigma_D, \Gamma_{\mathrm{eff}}/2) \right)$$

---

## Code Implementation
- **Function**: `rubidium_absorption_spectrum(T_C, L_cm, ...)`
- **Function**: `create_rubidium_reference_table(...)`
- **Module**: [`arVIPAspectro/backend.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/backend.py#L480-L560)
- **Pipeline Call**: [`arVIPAspectro/Rb_calib.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/Rb_calib.py#L18-L90), invoked in [`arVIPAspectro/main.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/main.py#L283-L305)
- **Diagnostic Plot**: Figure 7 (`7 - Theoretical Rubidium Spectrum.png`)
