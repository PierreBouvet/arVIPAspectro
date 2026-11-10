# Step 08 — Rubidium Absorption Preprocessing & Denoising

## Summary and Input/Output
- **Purpose**: Preprocesses the raw Rubidium absorption camera image by removing ambient stray light, cell window reflection gradients, and slow illumination variations to isolate pure atomic transmission dips.
- **Inputs**: Raw Rubidium absorption frame $I_{\mathrm{raw},\mathrm{Rb}}(x, y) \in \mathbb{R}^{2304 \times 2304}$.
- **Outputs**: 2D morphological baseline $B_{\mathrm{Rb}}(x, y)$ and net transmission frame $I_{\mathrm{net},\mathrm{Rb}}(x, y)$.

---

## Algorithm
1. **Multi-Scale Decimation**: Decimates $I_{\mathrm{raw},\mathrm{Rb}}$ by scaling factor $S = 9$ using area interpolation.
2. **Morphological Opening**: Executes grayscale erosion followed by dilation with an elliptical structuring element ($K = 9\text{ px}$) and Gaussian smoothing ($11 \times 11\text{ px}$):
   $$B_{\mathrm{Rb}}(x, y) = \mathcal{G}_{\sigma}\Big(\mathcal{U}_S\big(\mathcal{D}_S(I_{\mathrm{raw},\mathrm{Rb}}) \circ \mathcal{B}_K\big)\Big)$$
3. **Subtraction & Caching**: Subtracts the baseline to eliminate background gradients and stores `image_corrected` and `baseline` under `Rb_absorption` in `Measures.h5`.

---

## Physical Rationale
When the broad laser beam passes through the heated Rubidium vapor cell ($70^\circ\text{C}$), Fresnel reflections from the input and exit optical windows, cell wall scattering, and spatial beam non-uniformity introduce slow intensity gradients. Removing this envelope is essential to ensure that radial integration in Step 9 accurately reflects absorption dip depths without distortion from the beam profile.

---

## Physical Approximations
1. **Smooth Window Transmission**: Cell window reflection losses are spatially smooth over the field of view.

---

## Key Equations
- **Net Absorption Interferogram**:
  $$I_{\mathrm{net},\mathrm{Rb}}(x, y) = \max\big(I_{\mathrm{raw},\mathrm{Rb}}(x, y) - B_{\mathrm{Rb}}(x, y),\, 0\big)$$

---

## Code Implementation
- **Function**: `denoise_image(wrapper, 'Rb_absorption', ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L90-L108)
- **Pipeline Call**: [`arVIPAspectro/Rb_calib.py`](file:/arVIPAspectro/Rb_calib.py#L92-L140), invoked in [`arVIPAspectro/main.py`](file:/main.py#L308-L330)
- **Diagnostic Plot**: Figure 8 (`8 - Denoise rubidium absorption.png`)
