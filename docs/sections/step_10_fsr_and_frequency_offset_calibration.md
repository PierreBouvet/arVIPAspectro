# Step 10 — VIPA FSR & Rubidium Hyperfine Offset Calibration

## Summary and Input/Output
- **Purpose**: Calibrates the precise Free Spectral Range ($\mathrm{FSR}_{\mathrm{cal}} \approx 31.02\text{ GHz}$) of the VIPA etalon and absolute excitation laser detuning $\delta\nu_{\mathrm{opt}}$ by cross-correlating multi-order folded absorption profiles with the theoretical Rubidium atomic standard.
- **Inputs**: 1D radial Rubidium absorption profile $\bar{I}_{\mathrm{Rb}}(R)$, theoretical transmission table $T_{\mathrm{Rb}}(\Delta\nu)$, reduced dispersion slope $s_{\mathrm{cal}}$ from laser fringes.
- **Outputs**: Calibrated Free Spectral Range $\mathrm{FSR}_{\mathrm{cal}}$, optimal frequency offset $\delta\nu_{\mathrm{opt}}$, fine detuning residual $\delta\nu_{\mathrm{fine}}$, and peak correlation coefficient $r$.

---

## Algorithm
1. **Reduced Dispersion Constant**:
   From the linear regression of mean internal cosines $\bar{g}_q$ versus order $q$ obtained in Step 5:
   $$\bar{g}_q = A_0 (1 - q \cdot s_{\mathrm{cal}}) \implies s_{\mathrm{cal}} \approx 8.0735 \times 10^{-5}$$
   The exact VIPA FSR is related to absolute laser frequency $\nu_0 + \delta\nu$ by:
   $$\mathrm{FSR}(\delta\nu) = (\nu_{\mathrm{Rb}} + \delta\nu) \cdot s_{\mathrm{cal}}$$
2. **Coherent Multi-Order Folding**:
   Translates radial coordinates $R$ into fractional order coordinates $\xi \in [-0.5, +0.5]$ using the VIPA dispersion relation and coherently sums all visible orders into a single periodic cycle.
3. **Transmission Envelope Leveling**:
   Fits a smooth convex illumination envelope $E(R)$ above the absorption dips and calculates the normalized experimental absorption:
   $$A_{\mathrm{exp}}(R) = 1 - \frac{\bar{I}_{\mathrm{Rb}}(R)}{E(R)}$$
4. **Cross-Correlation Optimization**:
   Performs a 1D bounded optimization (`scipy.optimize.minimize_scalar`) over laser detuning $\delta\nu \in [-30, +30]\text{ GHz}$ maximizing the normalized Pearson correlation coefficient $r(\delta\nu)$ between $A_{\mathrm{exp}}$ and $1 - T_{\mathrm{Rb}}(\Delta\nu)$:
   $$r(\delta\nu) = \frac{\langle (A_{\mathrm{exp}} - \mu_A)(A_{\mathrm{theo}}(\delta\nu) - \mu_T) \rangle}{\sigma_A \, \sigma_T}$$
   Reaching peak correlation $r = 0.9550$ at:
   $$\mathrm{FSR}_{\mathrm{cal}} = 31.0221\text{ GHz}, \qquad \delta\nu_{\mathrm{opt}} = +14.9426\text{ GHz}$$

---

## Physical Rationale
The manufacturer's nominal FSR specification ($\sim 30\text{ GHz}$) is only approximate ($\pm 1\text{ GHz}$) and sensitive to temperature. Because the absolute Brillouin shift $\Delta\nu_B$ scales directly with the FSR, an error of $1\text{ GHz}$ in FSR would create a $\sim 300\text{ MHz}$ systematic error in acoustic velocity. Anchoring the dispersion scale to atomic Rubidium hyperfine transitions ensures frequency calibration accuracy better than $10\text{ MHz}$.

---

## Physical Approximations
1. **Linear Order-FSR Relation**: Over the optical bandwidth probed by the laser reference ($\sim 300\text{ GHz}$ across 10 orders), the VIPA dispersion parameter $s$ is treated as constant.

---

## Key Equations
- **VIPA Free Spectral Range Relation**:
  $$\mathrm{FSR} = \frac{c}{2 n_r d \cos\theta_{\mathrm{int}}} = (\nu_0 + \delta\nu) \cdot s_{\mathrm{cal}}$$
- **Pearson Cross-Correlation Objective**:
  $$\max_{\delta\nu} r(\delta\nu) = \frac{\sum_i (A_{\mathrm{exp}, i} - \bar{A})(A_{\mathrm{theo}, i}(\delta\nu) - \bar{A}_{\mathrm{theo}})}{\sqrt{\sum_i (A_{\mathrm{exp}, i} - \bar{A})^2 \sum_i (A_{\mathrm{theo}, i}(\delta\nu) - \bar{A}_{\mathrm{theo}})^2}}$$

---

## Code Implementation
- **Function**: `calibrate_vipa_fsr_with_rubidium(profile_1d, T_Rb, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L625-L710)
- **Pipeline Call**: [`arVIPAspectro/Rb_calib.py`](file:/arVIPAspectro/Rb_calib.py#L197-L265), invoked in [`arVIPAspectro/main.py`](file:/main.py#L358-L380)
- **Diagnostic Plot**: Figure 10 (`10 - Calibrate FSR and offset.png`)
