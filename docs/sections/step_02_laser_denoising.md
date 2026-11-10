# Step 02 — Laser Reference Denoising & Background Removal

## Summary and Input/Output
- **Purpose**: Removes the diffuse 2D background halo, secondary etalon ghost reflections, and detector dark offset from the monochromatic laser reference image while strictly preserving sharp interference fringes.
- **Inputs**: Raw laser reference frame $I_{\mathrm{raw}}(x, y) \in \mathbb{R}^{2304 \times 2304}$.
- **Outputs**: Smooth 2D morphological baseline $B_{\mathrm{morph}}(x, y)$ and net fringe interferogram $I_{\mathrm{net}}(x, y) = I_{\mathrm{raw}}(x, y) - B_{\mathrm{morph}}(x, y)$.

---

## Algorithm
1. **Downsampling**: Decimates the raw $2304 \times 2304$ image by a scaling factor $S = 9$ (`cv2.resize` with area interpolation) to accelerate morphological computation and ensure the structuring element spans entire fringe widths.
2. **Morphological Opening**: Applies a grayscale morphological opening (erosion followed by dilation) using an elliptical structuring element of kernel size $K = 9\text{ px}$:
   $$\mathcal{O}(I) = (I \ominus \mathcal{B}) \oplus \mathcal{B}$$
   Because the interference fringes have widths $< 15\text{ px}$ in full resolution (corresponding to $\sim 1.6\text{ px}$ in downsampled space), the opening completely erases the fringes while retaining the slowly varying background envelope.
3. **Gaussian Smoothing**: Convolves the opened image with a 2D Gaussian kernel ($11 \times 11\text{ px}$) to eliminate discretization artifacts.
4. **Bilinear Upsampling**: Interpolates the background back to the full sensor dimensions ($2304 \times 2304$).
5. **Subtraction & Non-Negativity Clipping**:
   $$I_{\mathrm{net}}(x, y) = \max\big(I_{\mathrm{raw}}(x, y) - B_{\mathrm{morph}}(x, y),\, 0\big)$$
6. **HDF5 Caching**: Saves `image_corrected` and `baseline` to `laser_reference` in `Measures.h5`.

---

## Physical Rationale
The detector captures light coming from different phenomena: the laser light being spatio-spectrally dispersed by the spectrometer of course but also noise arising from autofluorescence from the Rubidium cell, parasistic light being reflected on the anti-reflection coated surfaces, ... The role of this step is to capture how the spectrometer disperses spectrally a signal in a qualitative manner, by observing its response to a monochromatic light. This beam being extremely spectrally tight (lower than 1 MHz), we expect (and see) a response of the spectrometer that is essentially composed of narrow lines. These these lines being much narrower than the noise, to improve their identification, we remove any artifacts larger than a few hundreds of MHz. This filtering drastically the quality of the obtained frequency map and sometimes even allows it altogether.

---

## Physical Approximations
1. **Scale Separation**: The spatial frequency spectrum of the background envelope is assumed to be strictly separated from the high-spatial-frequency interference fringes ($\Delta x_{\mathrm{fringe}} \ll \Delta x_{\mathrm{background}}$).
2. **Additive Incoherent Stray Light**: Stray light and parasitic reflections are modeled as an additive intensity pedestal, neglecting coherent speckle interference with the main beam.

---

## Key Equations
- **Grayscale Morphological Opening**:
  $$B_{\mathrm{morph}}(x, y) = \mathcal{G}_{\sigma} \left[ \mathcal{U}_S \Big( \big(\mathcal{D}_S(I_{\mathrm{raw}}) \ominus \mathcal{B}_K\big) \oplus \mathcal{B}_K \Big) \right]$$
- **Net Laser Image**:
  $$I_{\mathrm{net}}(x, y) = I_{\mathrm{raw}}(x, y) - B_{\mathrm{morph}}(x, y)$$

---

## Code Implementation
- **Function**: `extract_baseline_image_morphology(image, scale=9, ksize=9, ksize_blur=11)`
- **Function**: `denoise_image(wrapper, path_group, name_dataset, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L58-L108)
- **Pipeline Call**: [`arVIPAspectro/laser_calib.py`](file:/arVIPAspectro/laser_calib.py#L18-L36), invoked in [`arVIPAspectro/main.py`](file:/main.py#L130-L155)
- **Diagnostic Plot**: Figure 2 (`2 - Denoise laser reference.png`)
