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
- **Module**: [`arVIPAspectro/backend.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/arVIPAspectro/backend.py#L22-L55)
- **Pipeline Call**: [`arVIPAspectro/main.py`](file:///Users/pierrebouvet/Documents/Code/Treatment/ar_BLS/AI%20codes/arVIPAspectro/main.py#L115-L125)
- **Relation to Algorithm**: Step 1 establishes the root wrapper object passed sequentially to all subsequent calibration and fitting steps.
