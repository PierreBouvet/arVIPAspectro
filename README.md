<p align="center">
  <img src="_static/icon.jpg" alt="arVIPAspectro Icon" width="180">
</p>

# arVIPAspectro

**arVIPAspectro** is a high-precision Python package for calibrating, analyzing, and extracting physical acoustic parameters from Angle-Resolved Virtually Imaged Phased Array (ar-VIPA) Brillouin Light Scattering (BLS) spectrometers.

---

## Architecture & Layout

```
arVIPAspectro/
├── arVIPAspectro/                     # Core Python Library Package
│   ├── __init__.py                   # Package exports and version
│   ├── config.py                     # Physical constants (Rb D2, Steck/CODATA) & VIPA specs
│   ├── backend.py                    # Pure mathematical, physical, and fitting solvers
│   ├── laser_calib.py                # Steps 2–6: Laser reference calibration & 2D surface fit
│   ├── Rb_calib.py                   # Steps 7–10: Rubidium absorption model, FSR & 2D freq map
│   ├── spectrum_fitting.py           # Steps 11–13: In-situ self-calibration & vectorized DHO fits
│   ├── q_vector.py                   # Step 14: 2D concentric coaxial ellipse & q-dispersion
│   ├── q_vector_extraction.py        # Step 14: Standalone wrapper & CLI runner
│   ├── plotting.py                   # Comprehensive diagnostic plotting suite (Figs 2 to 14B)
│   └── gui/                          # Interactive PySide6 GUI Application Suite
│       ├── __init__.py               # GUI launcher function
│       ├── app.py                    # Main window & layout orchestrator
│       ├── canvas.py                 # Matplotlib canvas & multi-view visualizer
│       ├── h5_explorer.py            # HDF5 hierarchy browser & spectrum selector
│       ├── step_widgets.py           # Collapsible parameter sections & step cards
│       └── workers.py                # Asynchronous background execution threads
├── _static/                          # Project assets and branding icon
├── docs/                             # Complete pipeline documentation and mathematical theory
├── local_codes/                      # Modular partial processing scripts (git-ignored)
│   ├── calib_laser.py                # Steps 1–6 (laser reference calibration)
│   ├── calib_Rb.py                   # Steps 7–10B (Rubidium absorption & 2D frequency map)
│   ├── peak_fit.py                   # Steps 11–13 (autonomous DHO peak fitting)
│   └── analysis_results.py           # Steps 14–15 (q-vector & speed of sound extraction)
├── main.py                           # Top-level pipeline orchestrator script
├── gui.py                            # Standalone GUI launcher script
├── run_gui.sh                        # Shell launcher activating virtual environment
├── pyproject.toml                    # Modern PEP 517/621 package metadata
└── README.md
```

---

## Installation

To install in editable (development) mode:
```bash
pip install -e .
```

---

## Quick Usage

### 1. Interactive Graphical User Interface (GUI)
Launch the 2-frame interactive GUI application:
```bash
./run_gui.sh
# or
python gui.py [path/to/Measures.h5]
```
- **Left Frame**: Interactive Matplotlib visualizer with figure switcher dropdown, colormaps, cursor readout, and navigation toolbar.
- **Right Frame (Dual Tab)**:
  - **📁 HDF5 Data Explorer**: Inspect HDF5 hierarchy, preview 1D/2D datasets directly, convert `.dat` directories, and assign dataset roles.
  - **⚙ Processing Steps**: Collapsible accordion cards for Laser calibration, Rb calibration, DHO peak fitting (with single and batch mode), and acoustic dispersion analysis.
- **Bottom Frame**: Collapsible execution log console with live output, progress bar, and status indicators.

### 2. Running the Full Pipeline via CLI
```bash
python main.py
```

### 3. Importing as a Python Module
```python
import arVIPAspectro as ar
from arVIPAspectro import backend as bck
from arVIPAspectro import config
from arVIPAspectro import laser_calib
from arVIPAspectro import Rb_calib
from arVIPAspectro import spectrum_fitting
from arVIPAspectro import q_vector
from arVIPAspectro import plotting
```
