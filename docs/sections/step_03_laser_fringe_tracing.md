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
