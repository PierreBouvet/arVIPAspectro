"""
arVIPAspectro: Angle-Resolved VIPA Brillouin Spectroscopy Calibration and Analysis Package.
"""

from . import config
from . import backend
from . import laser_calib
from . import Rb_calib
from . import spectrum_fitting
from . import q_vector
from . import q_vector_extraction
from . import plotting

__version__ = "0.1.0"

__all__ = [
    'config',
    'backend',
    'laser_calib',
    'Rb_calib',
    'spectrum_fitting',
    'q_vector',
    'q_vector_extraction',
    'plotting',
]
