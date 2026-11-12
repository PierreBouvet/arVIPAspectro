<p align="center">
  <img src="../_static/icon.jpg" alt="arVIPAspectro Icon" width="160">
</p>

# arVIPAspectro — Complete Pipeline Documentation & Theory Guide

> **Automated Build**: Generated on `2026-11-11 22:36:40` (Commit `ad6bf97`).
> This document is automatically compiled from individual step modules in [`sections/`](sections/).
> To regenerate, run `python build_docs.py`.

---

## Table of Contents

- [00 — Optical Architecture & Pipeline Overview](#00-optical-architecture-pipeline-overview)
- [Step 01 — Environment Setup & HDF5 Wrapper Initialization](#step-01-environment-setup-hdf5-wrapper-initialization)
- [Step 02 — Laser Reference Denoising & Background Removal](#step-02-laser-reference-denoising-background-removal)
- [Step 03 — Saturation-Safe Fringe Detection & Full-Height Tracing](#step-03-saturation-safe-fringe-detection-full-height-tracing)
- [Step 04 — Unsaturated Sub-Pixel Centroid Refinement & Outlier Filtering](#step-04-unsaturated-sub-pixel-centroid-refinement-outlier-filtering)
- [Step 05 — Exact 2D Gnomonic VIPA Geometry Fit](#step-05-exact-2d-gnomonic-vipa-geometry-fit)
- [Step 06 — 2D Anamorphic Polynomial Order Surface Calibration](#step-06-2d-anamorphic-polynomial-order-surface-calibration)
- [Step 07 — Theoretical Rubidium Spectrum Model](#step-07-theoretical-rubidium-spectrum-model)
- [Step 08 — Rubidium Absorption Preprocessing & Denoising](#step-08-rubidium-absorption-preprocessing-denoising)
- [Step 09 — Radial Profile Extraction of Rubidium Absorption](#step-09-radial-profile-extraction-of-rubidium-absorption)
- [Step 10 — VIPA FSR & Rubidium Hyperfine Offset Calibration](#step-10-vipa-fsr-rubidium-hyperfine-offset-calibration)
- [Step 10B — Calibrated 2D Frequency Map Generation](#step-10b-calibrated-2d-frequency-map-generation)
- [Step 11 — Sample In-Situ Frequency Map Self-Calibration](#step-11-sample-in-situ-frequency-map-self-calibration)
- [Step 12 — Spatial Binning & Vectorized Batched DHO Peak Fitting](#step-12-spatial-binning-vectorized-batched-dho-peak-fitting)
- [Step 13 — Spectral Profiles, Shift & Linewidth Diagnostics](#step-13-spectral-profiles-shift-linewidth-diagnostics)
- [Step 14 — Acoustic Wavevector (q-vector) & Elliptical Geometry Extraction](#step-14-acoustic-wavevector-q-vector-elliptical-geometry-extraction)
- [Step 15 — HDF5 Data Storage & Results Persistence](#step-15-hdf5-data-storage-results-persistence)

---

# 00 — Optical Architecture & Pipeline Overview

## Summary and Input/Output
- **Purpose**: High-level overview of the physical principles, optical layout, and computational pipeline of Angle-Resolved Virtually Imaged Phased Array (ar-VIPA) Brillouin Light Scattering (BLS) spectroscopy.
- **Inputs**: Raw sCMOS camera acquisitions (`laser_reference.dat`, `Rb_absorption.dat`, `sample.dat`), optical spectrometer specifications, physical constants.
- **Outputs**: Calibrated 2D frequency maps, sub-pixel accurate spatial interference contours, DHO acoustic peak parameters (frequency shift $\Delta\nu_B$, linewidth $\Gamma_B$), acoustic wavevectors $q$, and longitudinal sound velocity $V_s \pm \sigma_{V_s}$.

---

## Algorithm
The complete data-processing and calibration workflow executes in four distinct phases:
1. **Laser Geometric Calibration (Steps 1–6)**:
   - Denoises the monochromatic laser reference interferogram.
   - Detects and traces 11 interference orders across the full sensor height ($2304\text{ px}$).
   - Computes squared-intensity sub-pixel centroids with polynomial outlier filtering.
   - Fits the exact 2D gnomonic projection center $(x_0, y_0)$ and VIPA tilt angle $\phi$.
   - Calibrates a 2D 4th-degree Anamorphic Polynomial Order Surface $\phi_{\mathrm{order}}(g, y)$ eliminating inter-order lens aberrations.
2. **Rubidium Atomic Absorption Frequency Calibration (Steps 7–10B)**:
   - Evaluates the theoretical saturated transmission spectrum $T_{\mathrm{Rb}}(\Delta\nu)$ of natural Rubidium ($^{85}\text{Rb} / ^{87}\text{Rb}$) across 12 hyperfine transitions.
   - Extracts radial absorption profiles along gnomonic iso-angle contours.
   - Cross-correlates experimental absorption dips with atomic theory to determine the exact Free Spectral Range ($\mathrm{FSR}_{\mathrm{cal}} \approx 31.02\text{ GHz}$) and laser detuning.
   - Synthesizes the master 2D frequency map $\nu_{\mathrm{map}}(x, y)$.
3. **Sample Spectrum In-Situ Calibration & DHO Peak Fitting (Steps 11–13)**:
   - Applies a two-stage in-situ self-calibration (elastic zero-offset map and higher-order dispersion Jacobian scaling) using Stokes/Anti-Stokes symmetry.
   - Bins the interferogram spatially ($4 \times 2$ pixel blocks).
   - Fits local spectral profiles with a vectorized 3D batched Levenberg-Marquardt Damped Harmonic Oscillator (DHO) model.
4. **2D Elliptical Geometry & Acoustic Wavevector Extraction (Steps 14–15)**:
   - Fits 2D concentric coaxial ellipses to the Brillouin shift and linewidth contours with statistical error-weighting ($w_i = 1/\sigma_i$).
   - Optimizes the aspect ratio $\gamma$ via circular variance minimization.
   - Transforms spatial detector coordinates into spherical scattering coordinates $r_{\mathrm{sph}}$.
   - Fits the angular magnification factor $M_\theta$ and normal shift $S_0$ via non-linear kinematic modeling.
   - Deduces scattering angles $\theta$, acoustic wavevectors $q$, and the longitudinal sound velocity $V_s$ with full covariance error propagation.

---

## Physical Rationale
Brillouin light scattering probes spontaneous acoustic phonons in condensed matter via inelastic photon-phonon scattering. However, extracting high-resolution acoustic dispersion relations $\Omega(q) = V_s \cdot q$ using traditional scanning Fabry-Pérot interferometers requires point-by-point angle-resolved measurements spanning hours.

The VIPA spectrometer enables single-shot angle-resolved spectroscopy by utilizing:
- **A tilted solid fused silica etalon**: Disperses optical frequencies horizontally along $x$ with high spectral resolution ($\mathcal{R} > 10^6$) and large angular dispersion ($\sim 0.5\text{ mrad/GHz}$).
- **Large-NA Collection Lens**: Collects light scattered over a continuum of angles $\theta \in [0, 45^\circ]$ and focuses it onto the vertical and horizontal axes of the VIPA input window.
- **Rubidium Vapor Notch Filter**: Suppresses intense elastic Rayleigh scattering by $> 40\text{ dB}$ while transmitting the Stokes and Anti-Stokes Brillouin peaks.

---

## Physical Approximations
1. **Quasi-Elastic Scattering**: The acoustic phonon frequency ($\sim 5\text{--}10\text{ GHz}$) is negligible compared to the optical frequency ($\nu_0 \approx 384\text{ THz}$), ensuring the incident and scattered optical wavevectors have identical magnitude: $k_i \approx k_s = \frac{2\pi n}{\lambda_0}$.
2. **Linear Acoustic Dispersion**: Acoustic phonons in the hydrodynamic regime follow a linear dispersion relation $\Omega = V_s \cdot q$, where $V_s$ is the longitudinal speed of sound in the sample.
3. **Paraxial Collection Projection**: While ray tracing inside the VIPA requires exact gnomonic geometry, the external imaging through the collection lens is well-described by an affine-deformed spherical coordinate projection.

---

## Key Equations
- **Brillouin Kinematic Scattering Condition**:
  $$q = 2 k_i \cos\left(\frac{\theta}{2}\right) = \frac{4\pi n}{\lambda_0} \cos\left(\frac{\theta}{2}\right)$$
- **Acoustic Dispersion Relation**:
  $$\Omega = 2\pi \cdot \mathrm{Shift} = V_s \cdot q \iff \mathrm{Shift}(\theta) = \left(\frac{2 n V_s}{\lambda_0}\right) \cos\left(\frac{\theta}{2}\right)$$
- **Normal Incidence Sound Velocity**:
  $$V_s = \frac{\lambda_0 \cdot S_0}{2 n}$$

---

## Code Implementation
- **Master Orchestrator**: [`arVIPAspectro/main.py`](file:/main.py)
- **Library Package**: [`arVIPAspectro/arVIPAspectro/`](file:/arVIPAspectro)
- **Configuration & Constants**: [`arVIPAspectro/arVIPAspectro/config.py`](file:/arVIPAspectro/config.py)

---

# Step 01 — Environment Setup & HDF5 Wrapper Initialization

## Summary and Input/Output
- **Purpose**: Initializes the execution environment, verifies directory structures, and opens or creates the master HDF5 container (`Measures.h5`) for unified data access and persistence across all pipeline steps.
- **Inputs**: File system paths (`DATA_DIR`, `TARGET_FILE`), spectrometer configuration constants.
- **Outputs**: Instantiated `HDF5_BLS.Wrapper` file handle providing cached read/write access to raw and processed arrays.

---

## Algorithm
1. Inspects the specified `DATA_DIR` for `Measures.h5`.
2. Initializes the `HDF5_BLS.Wrapper` object with SWMR (Single Writer Multiple Reader) mode and error handling.
3. Automatically scans for legacy binary data files (`laser_reference.dat`, `Rb_absorption.dat`, `sample_*.dat`) if not already populated in the HDF5 group hierarchy.
4. Verifies the existence of primary groups (`laser_reference`, `Rb_absorption`, `Brillouin/<sample>`).

---

## Code Implementation
- **Function**: `create_wrapper(directory, name='Measures.h5', exclude_dir=None)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L22-L55)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L115-L125)
- **Relation to Algorithm**: Step 1 establishes the root wrapper object passed sequentially to all subsequent calibration and fitting steps.

---

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

---

# Step 03 — Saturation-Safe Fringe Detection & Full-Height Tracing

## Summary and Input/Output
- **Purpose**: Detects the discrete spatial interference orders of the monochromatic laser reference beam and traces their continuous curved trajectories from top to bottom across the entire $2304\text{ px}$ height of the detector.
- **Inputs**: Baseline-subtracted laser image $I_{\mathrm{net}}(x, y)$ and raw image $I_{\mathrm{raw}}(x, y)$.
- **Outputs**: Seed peak positions $\{x_{q, \mathrm{seed}}\}_{q=0}^{10}$ and initial integer-pixel coordinate arrays $(x_{q, i}, y_{q, i})$ for all 11 orders.

---

## Algorithm
1. **Unsaturated Row Band Selection**: Rather than seeding at the detector center ($y = 1000\text{ px}$), where intense orders 3 and 4 saturate the 16-bit ADC ($65,535\text{ counts}$), the algorithm samples an unsaturated band centered at $y_c = 750\text{ px}$ with vertical averaging window $\Delta y = \pm 40\text{ px}$:
   $$\bar{I}_{\mathrm{seed}}(x) = \frac{1}{2\Delta y} \sum_{y = y_c - \Delta y}^{y_c + \Delta y} I_{\mathrm{net}}(x, y)$$
2. **Order Peak Identification**: Calls `scipy.signal.find_peaks` on $\bar{I}_{\mathrm{seed}}(x)$ with a minimum peak separation `distance = 100 px` and prominence filter to identify all $N_q = 11$ interference orders ($q = 0 \dots 10$).
3. **Bidirectional Continuous Tracking**: For each detected order seed $(x_{q, \mathrm{seed}}, y_c)$:
   - Traces upwards from $y = y_c$ to $y = 0$.
   - Traces downwards from $y = y_c$ to $y = 2303$.
   - At each step $y \to y \pm 1$, searches for the local maximum within a horizontal neighborhood $[x_{k-1} - \mathrm{var}, x_{k-1} + \mathrm{var}]$ with step tolerance $\mathrm{var} = 5\text{ px}$.
   - Retains only traces where the local maximum is continuous and well-defined.

---

## Physical Rationale
The monochromatic laser beam transmits through the VIPA etalon at discrete angles satisfying the Fabry-Pérot interference condition $2 n_r d \cos\theta_m = m \lambda_0$. When imaged by a spherical lens, these angular orders project onto the detector as slightly curved vertical fringes. Due to Gaussian beam profiling, the central rows ($y \approx 950\text{--}1100\text{ px}$) on low-order fringes exceed the sensor full-well capacity. Selecting an unsaturated seeding zone prevents the fringe tracker from aborting early or locking onto flat saturation plateaus.

---

## Physical Approximations
1. **Spatial Smoothness**: Fringe curvature is gradual ($|\partial x / \partial y| \ll 1$), allowing a small horizontal search radius ($\pm 5\text{ px}$) per vertical row step.
2. **Monochromatic Singlet**: The reference laser is assumed single-frequency with negligible side-modes, ensuring exactly one fringe per interference order.

---

## Key Equations
- **Averaged 1D Seeding Profile**:
  $$\bar{I}_{\mathrm{seed}}(x) = \frac{1}{81} \sum_{y = 710}^{790} I_{\mathrm{net}}(x, y)$$
- **Fringe Tracking Update**:
  $$x_q(y \pm 1) = \arg\max_{x \in [x_q(y) - 5, x_q(y) + 5]} I_{\mathrm{net}}(x, y \pm 1)$$

---

## Code Implementation
- **Function**: `extract_fringes(image, image_raw, y_center=750, distance=100, var=5, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L110-L208)
- **Pipeline Call**: [`arVIPAspectro/laser_calib.py`](file:/arVIPAspectro/laser_calib.py#L38-L70), invoked in [`arVIPAspectro/main.py`](file:/main.py#L158-L180)
- **Diagnostic Plot**: Figure 3 (`3 - Extract fringe laser reference.png`)

---

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
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L210-L295)
- **Pipeline Call**: [`arVIPAspectro/laser_calib.py`](file:/arVIPAspectro/laser_calib.py#L72-L100), invoked in [`arVIPAspectro/main.py`](file:/main.py#L183-L208)
- **Diagnostic Plot**: Figure 4 (`4 - Refine fringe laser reference.png`)

---

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
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L300-L380)
- **Pipeline Call**: [`arVIPAspectro/laser_calib.py`](file:/arVIPAspectro/laser_calib.py#L102-L175), invoked in [`arVIPAspectro/main.py`](file:/main.py#L211-L245)
- **Diagnostic Plot**: Figure 5 (`5 - Fit laser reference geometry.png`)

---

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

---

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
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L480-L560)
- **Pipeline Call**: [`arVIPAspectro/Rb_calib.py`](file:/arVIPAspectro/Rb_calib.py#L18-L90), invoked in [`arVIPAspectro/main.py`](file:/main.py#L283-L305)
- **Diagnostic Plot**: Figure 7 (`7 - Theoretical Rubidium Spectrum.png`)

---

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

---

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
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L565-L620)
- **Pipeline Call**: [`arVIPAspectro/Rb_calib.py`](file:/arVIPAspectro/Rb_calib.py#L142-L195), invoked in [`arVIPAspectro/main.py`](file:/main.py#L333-L355)
- **Diagnostic Plot**: Figure 9 (`9 - Radial profile rubidium absorption.png`)

---

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

---

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
  $$k_{\mathrm{ord}}(x, y) = \text{round}\left( \frac{\nu_{\mathrm{map}}(x, y)}{\mathrm{FSR}_{\mathrm{cal}}} \right)$$

---

## Code Implementation
- **Function**: `eval_anamorphic_order_surface(...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L420-L460)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L383-L415)
- **Diagnostic Plot**: Figure 10B (`10B - Frequency map.png`)

---

# Step 11 — Sample In-Situ Frequency Map Self-Calibration

## Summary and Input/Output
- **Purpose**: Dynamically refines the master 2D frequency map using the in-situ Stokes/Anti-Stokes Brillouin peak pairs of the sample itself, correcting for experimental alignment drift, zero-frequency offset, and higher-order imaging lens aberrations.
- **Inputs**: Base frequency map $\nu_{\mathrm{map}}(x, y)$, sample image $I_{\mathrm{sample}}(x, y)$ (e.g. `water_5s` or `plastic_5s`), nominal shift and linewidth bounds.
- **Outputs**: Self-calibrated in-situ frequency map $\nu_{\mathrm{corr}}(x, y) \in \mathbb{R}^{2304 \times 2304}$ with Stokes/Anti-Stokes symmetry RMS reduced below $35\text{ MHz}$.

---

## Algorithm
The self-calibration operates in two sequential stages:

### Stage 1 — Elastic Zero-Offset Map $\delta\nu_{\mathrm{zero}}(y, k_{\mathrm{ord}})$
1. Any common-mode frequency error $\delta\nu_{\mathrm{zero}}$ in the nominal elastic position $\nu = k \cdot \mathrm{FSR}$ shifts the Stokes peak ($+1$) upward and the Anti-Stokes peak ($-1$) downward:
   $$\Delta\nu_S = \Delta\nu_{\mathrm{true}} + \delta\nu_{\mathrm{zero}}, \qquad \Delta\nu_{AS} = \Delta\nu_{\mathrm{true}} - \delta\nu_{\mathrm{zero}} \implies \delta\nu_{\mathrm{mid}} = \frac{\Delta\nu_S - \Delta\nu_{AS}}{2}$$
2. Runs a fast initial DHO fit on $4 \times 2$ binned data to measure the local midpoint $\delta\nu_{\mathrm{mid}}(y, k)$ across all visible orders $k \in \{1, \dots, 10\}$.
3. Fits a smooth 2D 3rd-order polynomial base plus Gaussian-filtered spline ($\sigma_y = 6\text{ bins}$) along vertical rows to model $\delta\nu_{\mathrm{zero}}(y, k)$.
4. Subtracts $\delta\nu_{\mathrm{zero}}(y, k)$ from each order $k_{\mathrm{ord}} = \text{clip}(\text{round}(\nu_{\mathrm{init}}/\mathrm{FSR}), 1, 10)$.

### Stage 2 — Higher-Order Dispersion Jacobian Scale $\alpha_{\mathrm{disp}}(y, k_{\mathrm{ord}})$
1. Decomposes the order-averaged Brillouin shift $\Delta\nu_{\mathrm{avg}}(y, k) = \frac{1}{2}(\Delta\nu_S + \Delta\nu_{AS})$ into:
   - A physical 6-parameter **2nd-order 2D elliptical paraboloid** $Z_{\mathrm{quad}}(x, y)$ fitted via 2-step Iteratively Reweighted Least Squares (IRLS) on the unclipped central region ($y \in [250, 1850]\text{ px}$, Fringes $0\dots 7$).
   - A stationary higher-order residual $\Delta\nu_{\mathrm{inst}}(y, k) = \mathcal{G}_{\sigma_y = 6}\big(\Delta\nu_{\mathrm{avg}} - Z_{\mathrm{quad}}\big)$ capturing 4th-order spherical aberration wing flares at sensor edges and central VIPA blaze distortion.
2. Scales the intra-order dispersion Jacobian:
   $$\alpha_{\mathrm{disp}}(y, k) = \frac{Z_{\mathrm{quad}}(y, k)}{Z_{\mathrm{quad}}(y, k) + \Delta\nu_{\mathrm{inst}}(y, k)}$$
3. Yields the final self-calibrated 2D frequency map:
   $$\nu_{\mathrm{corr}}(x, y) = k \cdot \mathrm{FSR} + \alpha_{\mathrm{disp}}(y, k) \cdot \Big(\nu_{\mathrm{map}}(x, y) - \mathrm{df}_{\mathrm{coarse}} - k \cdot \mathrm{FSR} - \delta\nu_{\mathrm{zero}}(y, k)\Big)$$

---

## Physical Rationale
Between calibrating the reference laser and mounting a scattering sample (e.g., placing a cuvette or plastic block), minor optical path perturbations occur:
- Mechanical drift and thermal expansion of the VIPA mount.
- Refractive index changes and beam displacement through sample cuvettes.
- Fine frequency drifts of the laser diode temperature controller ($\sim 10\text{--}50\text{ MHz}$).
Because Brillouin scattering generates Stokes and Anti-Stokes phonons of strictly identical energy ($\hbar\Omega$), any measured asymmetry between Stokes and Anti-Stokes shifts is an instrumental artifact. Using the sample's own spectral symmetry as an in-situ optical interferometer eliminates these drifts without requiring external recalibration.

---

## Physical Approximations
1. **Thermodynamic Stokes/Anti-Stokes Energy Equality**: In the classical thermal equilibrium regime ($k_B T \gg \hbar\Omega$, satisfied at room temperature where $k_B T \approx 6.2\text{ THz} \gg 10\text{ GHz}$), the acoustic phonon frequencies for creation (Stokes) and annihilation (Anti-Stokes) are rigorously identical: $\Omega_S = \Omega_{AS}$.

---

## Key Equations
- **Elastic Midpoint Splitting**:
  $$\delta\nu_{\mathrm{mid}}(y, k) = \frac{\Delta\nu_S(y, k) - \Delta\nu_{AS}(y, k)}{2}$$
- **Corrected Frequency Map**:
  $$\nu_{\mathrm{corr}}(x, y) = k\cdot\mathrm{FSR} + \alpha_{\mathrm{disp}}(y, k)\,\big[\nu_{\mathrm{map}}(x, y) - \mathrm{df} - k\cdot\mathrm{FSR} - \delta\nu_{\mathrm{zero}}(y, k)\big]$$

---

## Code Implementation
- **Function**: `calibrate_sample_frequency_map(freq_map_base, data_clean_in, ...)`
- **Module**: [`arVIPAspectro/spectrum_fitting.py`](file:/arVIPAspectro/spectrum_fitting.py#L18-L150)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L418-L445)
- **Diagnostic Plot**: Figure 11 (`11 - In-situ frequency map calibration.png`)

---

# Step 12 — Spatial Binning & Vectorized Batched DHO Peak Fitting

## Summary and Input/Output
- **Purpose**: Averages detector pixels into spatial bins to boost signal-to-noise ratio (SNR) and simultaneously fits the physical Damped Harmonic Oscillator (DHO) acoustic spectral response across all orders and spatial bins using a vectorized 3D batched Levenberg-Marquardt solver.
- **Inputs**: Net sample image $I_{\mathrm{net},\mathrm{sample}}(x, y)$, calibrated frequency map $\nu_{\mathrm{corr}}(x, y)$, binning factors ($H=4, W=2$).
- **Outputs**: 3D data cubes of Brillouin shift $\Delta\nu_B(y_b, k, \sigma)$, linewidth $\Gamma_B(y_b, k, \sigma)$, peak amplitude, background, fitting errors, and sub-pixel spatial coordinates $(x_b, y_b)$.

---

## Algorithm
1. **2D Block Binning**:
   Averages pixel blocks of height $H = 4\text{ px}$ and width $W = 2\text{ px}$ over both the baseline-subtracted sample image and the calibrated 2D frequency map, generating $N_y = 576$ vertical bins by $N_x = 1152$ horizontal bins.
2. **Local Spectral Window Extraction & Linear Chord Leveling**:
   For each spatial row bin $i \in \{0, \dots, 575\}$, order $k \in \{1, \dots, 10\}$, and branch $\sigma \in \{-1, +1\}$:
   - Extracts a 25-bin local frequency window centered on the expected peak position $k \cdot \mathrm{FSR} + \sigma \Delta\nu_{\mathrm{guess}}$.
   - Levels any local baseline slope between window boundary endpoints:
     $$\tilde{I}(j) = I(j) - \left( I_0 + \frac{I_{M-1} - I_0}{M - 1} j \right)$$
   - Maps the spectral axis to relative frequency shift: $\nu_{\mathrm{rel}} = |\nu - k \cdot \mathrm{FSR}|$.
3. **Vectorized Batched Levenberg-Marquardt Solver**:
   Stacks all $N_{\mathrm{peaks}} \approx 11,520$ windows into a 3D batch array of dimensions $(N_{\mathrm{peaks}}, M)$.
   - Evaluates the exact DHO model function:
     $$I_{\mathrm{DHO}}(\nu_{\mathrm{rel}}; b, a, \nu_0, \Gamma_B) = b + a \frac{(\Gamma_B \nu_0)^2}{(\nu_{\mathrm{rel}}^2 - \nu_0^2)^2 + (\Gamma_B \nu_{\mathrm{rel}})^2}$$
   - Evaluates the analytic Jacobian $\mathbf{J} \in \mathbb{R}^{N_{\mathrm{peaks}} \times M \times 4}$ containing exact partial derivatives $[\partial_b I, \partial_a I, \partial_a I, \partial_{\nu_0} I, \partial_{\Gamma_B} I]$.
   - Solves the damped normal equations $\big(\mathbf{J}^T \mathbf{J} + \lambda \text{diag}(\mathbf{J}^T \mathbf{J})\big) \Delta\mathbf{p} = \mathbf{J}^T \mathbf{r}$ across all peaks simultaneously using `np.einsum` and batched linear algebra.
   - Converges in $\sim 18$ iterations in **$< 0.5\text{ seconds}$** on standard multi-core CPUs.
4. **Sub-Pixel Coordinate Interpolation**:
   Interpolates the physical camera coordinates $(x, y)$ corresponding to the fitted peak frequency center $\nu_0$ using monotonic 1D cubic splines along each binned row.

---

## Physical Rationale
Brillouin scattering arises from optical diffraction off thermally excited acoustic sound waves (phonons) propagating in the material. In viscoelastic liquids and amorphous polymers, the dynamic structure factor $S(q, \omega)$ is governed by the damped acoustic harmonic oscillator response:
$$S(q, \omega) \propto \frac{\Gamma_B \omega_0^2}{(\omega^2 - \omega_0^2)^2 + (\Gamma_B \omega)^2}$$
Fitting an empirical Gaussian or symmetric Lorentzian systematically biases the linewidth because acoustic phonons have asymmetric tails vanishing as $\omega \to 0$ ($\propto \omega^2$). The exact DHO model accurately recovers the true phonon natural frequency $\nu_0$ and acoustic attenuation rate $\Gamma_B$.

---

## Physical Approximations
1. **Classical High-Temperature Phonon Occupation**: In the classical regime ($k_B T \gg \hbar\omega$), the Bose-Einstein thermal factor $[n(\omega) + 1] \approx k_B T / \hbar\omega$ cancels the pre-factor $\omega$, rendering the observed spectrum proportional to the imaginary susceptibility $\chi''(\omega)/\omega$, which yields the symmetric DHO intensity form.
2. **Single Longitudinal Acoustic Branch**: The sample is assumed isotropic (e.g. water or unoriented plastic), exhibiting a single longitudinal phonon branch with no transverse modes.

---

## Key Equations
- **Damped Harmonic Oscillator (DHO) Line Profile**:
  $$I_{\mathrm{DHO}}(\nu_{\mathrm{rel}}) = b + a \frac{(\Gamma_B \nu_0)^2}{(\nu_{\mathrm{rel}}^2 - \nu_0^2)^2 + (\Gamma_B \nu_{\mathrm{rel}})^2}$$
- **Phonon Damping Attenuation Relation**:
  $$\alpha_s = \frac{\pi \Gamma_B}{V_s}$$
- **Normal Equations for Vectorized LM**:
  $$(\mathbf{J}^T \mathbf{J} + \lambda \mathbf{D}) \Delta\mathbf{p} = \mathbf{J}^T (\mathbf{y} - \mathbf{f})$$

---

## Code Implementation
- **Function**: `fit_dho_batched(nu_cube, I_cube, p0_mat, ...)`
- **Function**: `fit_sample_dho_peaks(data_sample_clean, freq_corr, ...)`
- **Module**: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L715-L870) and [`arVIPAspectro/spectrum_fitting.py`](file:/arVIPAspectro/spectrum_fitting.py#L152-L310)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L448-L485)

---

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

---

# Step 14 — Acoustic Wavevector (q-vector) & Elliptical Geometry Extraction

## Summary and Input/Output
- **Purpose**: Extracts the 2D concentric coaxial elliptical geometry of the collection aperture, maps detector coordinates $(x, y)$ into spherically symmetric scattering coordinates $r_{\mathrm{sph}}$, fits the angular magnification factor $M_\theta$, and deduces the acoustic wavevector dispersion $\Omega(q)$ and longitudinal speed of sound $V_s \pm \sigma_{V_s}$.
- **Inputs**: Refined 2D coordinates $(x, y)$, Brillouin shift $\Delta\nu_B$ and linewidth $\Gamma_B$ with standard errors $\sigma_{\mathrm{Shift}}$ and $\sigma_{\mathrm{Linewidth}}$, refractive index $n_{\mathrm{sample}}$, laser wavelength $\lambda_0$.
- **Outputs**:
  - Ellipse parameters: Center $(x_c, y_c)$, director angle $\phi$, aspect ratio $\gamma$.
  - Kinematic parameters: Angular magnification $M_\theta \pm \sigma_{M_\theta}$, normal backscattering shift $S_0 \pm \sigma_{S_0}$.
  - Acoustic dispersion: Scattering angles $\theta$, wavevectors $q$, and longitudinal sound velocity $V_s \pm \sigma_{V_s}$.
  - Diagnostic Figures 14 and 14B.

---

## Algorithm
1. **Error-Weighted Data Preparation**:
   - Converts Levenberg-Marquardt fit variances to standard errors $\sigma_i = \sqrt{\mathrm{Var}_i}$.
   - Assigns point weights $w_i = 1 / \sigma_i$ with an empirical percentile floor ($1\%$) to regularize near-zero variances.
   - Averages paired Stokes and Anti-Stokes orders to eliminate residual zero-frequency offsets.
2. **2D Concentric Coaxial Ellipse Fitting with Error-Weighting & Variance Normalization**:
   - Rotates detector coordinates into the candidate ellipse director frame $(\phi)$:
     $$\begin{pmatrix} X' \\ Y' \end{pmatrix} = \begin{pmatrix} \cos\phi & \sin\phi \\ -\sin\phi & \cos\phi \end{pmatrix} \begin{pmatrix} x - x_c \\ y - y_c \end{pmatrix}$$
   - Evaluates the joint quadratic surface model:
     $$Z_m(X', Y') = Z_{0,m} + a_m X'^2 + b_m Y'^2, \qquad m \in \{\mathrm{Shift}, \mathrm{Linewidth}\}$$
   - Solves for the shared center $(x_c, y_c)$ and orientation $\phi$ by minimizing the combined error-weighted objective normalized by dataset sample variance:
     $$\min_{x_c, y_c, \phi} \left[ \frac{\sum_i w_{S, i} (S_i - Z_S(X'_i, Y'_i))^2}{\mathrm{Var}(S) \sum_i w_{S, i}} + \frac{\sum_i w_{\Gamma, i} (\Gamma_i - Z_\Gamma(X'_i, Y'_i))^2}{\mathrm{Var}(\Gamma) \sum_i w_{\Gamma, i}} \right]$$
3. **Nelder-Mead Circular Variance Minimization for Aspect Ratio $\gamma$**:
   - Scales the minor axis by $\gamma$:
     $$x_{\mathrm{sph}} = X', \qquad y_{\mathrm{sph}} = \frac{Y'}{\gamma}, \qquad r_{\mathrm{sph}} = \sqrt{x_{\mathrm{sph}}^2 + y_{\mathrm{sph}}^2}$$
   - Optimizes $\gamma$ via `scipy.optimize.minimize(method='Nelder-Mead')` to minimize the radial variance of $\mathrm{Shift}$ around its moving average.
4. **Kinematic Fit of Angular Magnification $M_\theta$**:
   - Fits the non-linear kinematic curve via Levenberg-Marquardt (`curve_fit`):
     $$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cdot \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
   - Returns optimal parameters $S_0, M_\theta$ and their full $2 \times 2$ covariance matrix $\mathbf{\Sigma}_{\mathrm{kin}}$.
5. **Acoustic Wavevector Delineation & Longitudinal Sound Velocity $V_s$**:
   - Internal scattering angle: $\theta(r_{\mathrm{sph}}) = \arcsin(r_{\mathrm{sph}} / M_\theta)$.
   - Acoustic wavevector magnitude:
     $$q(r_{\mathrm{sph}}) = \frac{4\pi n}{\lambda_0} \cos\left(\frac{\theta(r_{\mathrm{sph}})}{2}\right)$$
   - Longitudinal sound velocity:
     $$V_s = \frac{\lambda_0 \cdot S_0}{2 n}$$
   - Full covariance error propagation from $(S_0, M_\theta)$ through the linear dispersion relation $\mathrm{Shift} = \frac{V_s}{2\pi} q(M_\theta)$:
     $$\sigma_{V_s} = \sqrt{\left(\frac{\partial V_s}{\partial S_0}\right)^2 \sigma_{S_0}^2 + \left(\frac{\partial V_s}{\partial M_\theta}\right)^2 \sigma_{M_\theta}^2 + 2 \left(\frac{\partial V_s}{\partial S_0}\right) \left(\frac{\partial V_s}{\partial M_\theta}\right) \mathrm{cov}(S_0, M_\theta)}$$

---

## Physical Rationale
In an angle-resolved backscattering geometry, the collection lens collects light scattered over a cone of angles $\theta \in [0, 45^\circ]$. Because the VIPA etalon only disperses along the horizontal axis ($x$) and introduces optical anamorphism between horizontal and vertical axes, the circular collection cone projects onto the detector as an elongated ellipse ($\gamma \approx 1.8\text{--}2.0$). 

Recovering the aspect ratio $\gamma$ restores isotropic spherical symmetry ($r_{\mathrm{sph}}$), which allows the exact physical scattering angle $\theta$ and acoustic wavevector $q$ to be deduced purely from geometry without relying on arbitrary polynomial scaling.

---

## Physical Approximations
1. **Affine Elliptical Pupil Deformation**: The anamorphic distortion of the collection cone is modeled as a 2D affine scaling along orthogonal principal axes.
2. **Abbe Sine Condition**: The aplanatic collection objective maps scattering angles to pupil coordinates via $r \propto \sin\theta$, modeled rigorously via the $\arcsin$ kinematic relation.
3. **Linear Acoustic Dispersion**: The phonon group velocity equals the phase velocity: $\Omega = V_s \cdot q$, which holds rigorously in the GHz frequency domain far below the Debye cutoff.
4. **Isotropic Medium**: The sample is assumed mechanically isotropic with scalar sound velocity $V_s$.

---

## Key Equations
- **Spherical Coordinate Transformation**:
  $$x_{\mathrm{sph}} = (x - x_c)\cos\phi + (y - y_c)\sin\phi, \qquad y_{\mathrm{sph}} = \frac{-(x - x_c)\sin\phi + (y - y_c)\cos\phi}{\gamma}$$
  $$r_{\mathrm{sph}} = \sqrt{x_{\mathrm{sph}}^2 + y_{\mathrm{sph}}^2}$$
- **Kinematic Shift Model**:
  $$\mathrm{Shift}(r_{\mathrm{sph}}) = S_0 \cdot \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
- **Acoustic Wavevector Formula**:
  $$q(r_{\mathrm{sph}}) = \frac{4\pi n}{\lambda_0} \cos\left(\frac{1}{2}\arcsin\left(\frac{r_{\mathrm{sph}}}{M_\theta}\right)\right)$$
- **Sound Velocity & Propagated Uncertainty**:
  $$V_s = \frac{\lambda_0 S_0}{2 n}$$
  $$\sigma_{V_s}^2 = J_{S_0}^2 \sigma_{S_0}^2 + J_{M_\theta}^2 \sigma_{M_\theta}^2 + 2 J_{S_0} J_{M_\theta} \mathrm{cov}(S_0, M_\theta)$$
  where:
  $$J_{S_0} = \frac{\lambda_0}{2 n}, \qquad J_{M_\theta} = \frac{\partial \widehat{V}_s}{\partial M_\theta} = 2\pi \frac{N' D - N D'}{D^2}$$

---

## Code Implementation
- **Theoretical Foundations**:
  - Exact Refractive Index Scaling & Magnification Derivation: [`theory/refractive_index_magnification_derivation.md`](file:/theory/refractive_index_magnification_derivation.md)
  - Abbe Sine Condition & Pupil Mapping: [`theory/abbe_sine_condition_and_pupil_mapping.md`](file:/theory/abbe_sine_condition_and_pupil_mapping.md)
- **Pipeline Function**: `extract_q_vector(...)` in [`arVIPAspectro/q_vector.py`](file:/arVIPAspectro/q_vector.py#L18-L177)
- **Core Solvers**:
  - `fit_concentric_ellipses_coaxial`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L900-L960)
  - `optimize_elliptical_symmetry_variance`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L962-L1010)
  - `fit_magnification_factor_M_theta`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L1012-L1050)
  - `compute_theta_and_q_vector`: [`arVIPAspectro/backend.py`](file:/arVIPAspectro/backend.py#L1052-L1145)
- **Diagnostic Plots**:
  - Figure 14: `plot_q_vector_dispersion` ([`arVIPAspectro/plotting.py`](file:/arVIPAspectro/plotting.py#L466-L587))
  - Figure 14B: `plot_complete_diagnostic_suite` ([`arVIPAspectro/plotting.py`](file:/arVIPAspectro/plotting.py#L589-L825))

---

# Step 15 — HDF5 Data Storage & Results Persistence

## Summary and Input/Output
- **Purpose**: Persists all calibrated wavevector datasets, physical parameters, and statistical fit uncertainties into the standardized HDF5 container for downstream thermodynamic, viscoelastic, and multi-orientation analysis.
- **Inputs**: Extracted wavevector results dictionary (`q_vector`, `theta`, `r_sph`, `M_theta`, `Shift_0`, `V_s`, `V_s_err`, `opt_params`).
- **Outputs**: Formatted HDF5 group `Brillouin/<sample>/Treated <bin_h>x<bin_w> bins` containing datasets and attached metadata attributes.

---

## Algorithm
1. **Target Group Navigation**:
   Accesses or creates the group `Brillouin/<sample>/Treated 4x2 bins` within `Measures.h5`.
2. **Dataset Creation & Overwrite**:
   Writes 1D flattened arrays with double-precision floating-point format:
   - `q_vector`: Acoustic wavevector magnitude ($m^{-1}$).
   - `theta`: Internal scattering angle (radians).
   - `r_sph`: Spherically symmetric radius (pixels).
3. **Attribute Metadata Stamping**:
   Attaches scalar calibration and physical parameters directly to the HDF5 group:
   - `M_theta`, `M_theta_err`: Angular magnification factor and uncertainty (px).
   - `Shift_0`, `Shift_0_err`: Normal incidence Brillouin shift and uncertainty (GHz).
   - `V_s`, `V_s_err`: Longitudinal speed of sound and uncertainty (m/s).
   - `xc_ellipse`, `yc_ellipse`: Concentric ellipse center coordinates (px).
   - `phi_ellipse_deg`: Principal director axis angle (degrees).
   - `gamma_ellipse`: Collection aperture aspect ratio $\gamma$.
4. **Validation**:
   Re-reads the written datasets and attributes to confirm data integrity.

---

## Physical Rationale
Angle-resolved Brillouin experiments often span multiple samples, temperatures, and orientation angles (e.g. `plastic_0`, `plastic_60`, `plastic_120`, `plastic_180`). Saving the complete metadata — including the physical errors on $M_\theta$, $S_0$, and $V_s$ — ensures full experimental reproducibility and enables automated downstream scripts to load calibrated results without recomputing non-linear fits.

---

## Code Implementation
- **Function**: `save_q_vector_results(wrapper, path_sample, bin_val_h, bin_val_w, q_results)`
- **Function**: `retrieve_q_vector_results(wrapper, path_sample, bin_val_h, bin_val_w)`
- **Module**: [`arVIPAspectro/q_vector.py`](file:/arVIPAspectro/q_vector.py#L205-L270)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:/main.py#L565-L572)

---

