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
│   └── plotting.py                   # Comprehensive diagnostic plotting suite (Figs 2 to 14B)
├── main.py                           # Top-level pipeline orchestrator script
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

### 1. Running the Full Pipeline
```bash
python main.py
```

### 2. Importing as a Python Module
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
