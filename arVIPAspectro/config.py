"""
Configuration parameters, physical constants, and spectrometer specifications for VIPA calibration.
"""

import numpy as np

# ==============================================================================
# VIPA Spectrometer Parameters
# ==============================================================================
FSR_VIPA = 30.0                             # Estimated Free Spectral Range of the VIPA (GHz)
F_DETECTOR = 400e-3                         # Focal length of lens after VIPA (m)
PX_SIZE = 6.5e-6                            # Pixel pitch on the sCMOS detector (m)
N_FUSED_SILICA_780 = 1.45367                # Refractive index of fused silica VIPA at 780 nm
VIPA_TILT_ANGLE_OUT = np.arctan(50.0 / 470.0)  # VIPA tilt angle external to the medium (rad)
VIPA_TILT_ANGLE = np.arcsin(np.sin(VIPA_TILT_ANGLE_OUT) / N_FUSED_SILICA_780)  # Internal angle (rad)
F_PX_DEFAULT = F_DETECTOR / PX_SIZE          # Focal length in pixels (~61538.46 px)

# ==============================================================================
# Fundamental Physical Constants & Rubidium D2 Line Parameters
# ==============================================================================
c_light = 2.99792458e8                      # Speed of light (m/s)
kB = 1.3806504e-23                          # Boltzmann constant (J/K)
amu = 1.660538782e-27                       # Atomic mass unit (kg)
m_rb85 = 84.911789732 * amu                 # 85Rb atomic mass (kg)
m_rb87 = 86.909180520 * amu                 # 87Rb atomic mass (kg)
lambda_0 = 780.241368e-9                    # Rb D2 transition wavelength (m)
nu_0 = c_light / lambda_0                   # Rb D2 transition center of gravity frequency (Hz)
Gamma_nat_rad = 2.0 * np.pi * 6.0666e6      # Natural linewidth (rad/s, FWHM = 6.0666 MHz)
Gamma_nat_Hz = Gamma_nat_rad / (2.0 * np.pi)
sigma_int = (3.0 * lambda_0**2 / (2.0 * np.pi)) * (np.pi * Gamma_nat_Hz / 2.0)
gamma_self_rate = 1.0e-13                   # Self-broadening rate coefficient (Hz * m^3)

# Steck / CODATA Hyperfine Transitions: (detuning_MHz from center of gravity, relative_weight)
transitions_85 = [
    (-1164.68, 0.7217 * (7 / 12) * (9 / 14)),   # 85Rb F=3 -> F'=4
    (-1285.33, 0.7217 * (7 / 12) * (5 / 18)),   # 85Rb F=3 -> F'=3
    (-1348.73, 0.7217 * (7 / 12) * (5 / 63)),   # 85Rb F=3 -> F'=2
    ( 1750.40, 0.7217 * (5 / 12) * (14 / 45)),  # 85Rb F=2 -> F'=3
    ( 1687.00, 0.7217 * (5 / 12) * (7 / 18)),   # 85Rb F=2 -> F'=2
    ( 1657.63, 0.7217 * (5 / 12) * (3 / 10)),   # 85Rb F=2 -> F'=1
]

transitions_87 = [
    (-2291.17, 0.2783 * (5 / 8) * (7 / 10)),    # 87Rb F=2 -> F'=3
    (-2557.82, 0.2783 * (5 / 8) * (1 / 4)),     # 87Rb F=2 -> F'=2
    (-2714.76, 0.2783 * (5 / 8) * (1 / 20)),    # 87Rb F=2 -> F'=1
    ( 4276.87, 0.2783 * (3 / 8) * (5 / 12)),    # 87Rb F=1 -> F'=2
    ( 4119.93, 0.2783 * (3 / 8) * (5 / 12)),    # 87Rb F=1 -> F'=1
    ( 4047.71, 0.2783 * (3 / 8) * (1 / 6)),     # 87Rb F=1 -> F'=0
]

# ==============================================================================
# Default Laser Calibration Algorithm Parameters
# ==============================================================================
SCALE_LASER = 9                             # Downsampling factor for morphological baseline
KSIZE_STRUCTURING_ELEMENTS_LASER = 3        # Morphological opening structuring element size
KSIZE_BLUR_LASER = 11                       # Gaussian blur kernel size for baseline smoothing

Y_CENTER_LASER = 1200                       # Approximate vertical axis of symmetry (px)
DY_LASER = 50                               # Half-window height for row-averaging around symmetry axis (px)
NB_LASER_MODES = 14                         # Expected number of laser reference fringes
DISTANCE_PEAKS_LASER = 100                  # Minimum horizontal distance between peaks (px)
DY_TRACE = 20                               # Window height for vertical fringe tracing
VAR_TRACE = 2                               # Max horizontal step variation between consecutive rows (px)

W_SUB = 10                                  # Half-width for sub-pixel centroiding window (px)
SATURATION_LIMIT = 65000.0                  # Pixel saturation threshold (counts)
MIN_NET_AMP = 200.0                         # Minimum net peak amplitude above baseline (counts)

# ==============================================================================
# Reference Sample Physical Parameters
# (Tuned directly in main.py and local_codes scripts)
# ==============================================================================

# --- Polystyrene (Plastic) Reference ---
# N_SAMPLE = 1.59                             # Refractive index of plastic (polystyrene) at 780 nm
# EXPECTED_BRILLOUIN_SHIFT = 9.7              # Expected Brillouin shift nu_B in GHz
# BRILLOUIN_SHIFT_TOLERANCE = 2.0             # Search half-window around expected shift in GHz
# EXPECTED_LINEWIDTH = 0.8                    # Expected Brillouin linewidth Gamma_B in GHz
# LINEWIDTH_BOUNDS = (0.4, 1.3)               # Acceptable linewidth range (min, max) in GHz
# SHIFT_RANGE_SAMPLE = (9.3, 9.9)             # Filtering range for Brillouin shift (GHz)
# LINEWIDTH_RANGE_SAMPLE = (0.5, 1.2)         # Filtering range for Brillouin linewidth (GHz)

# --- Water Reference ---
# N_SAMPLE = 1.33                              # Refractive index of water at 780 nm
# EXPECTED_BRILLOUIN_SHIFT = 5.1              # Expected Brillouin shift nu_B in GHz
# BRILLOUIN_SHIFT_TOLERANCE = 2.0             # Search half-window around expected shift in GHz
# EXPECTED_LINEWIDTH = 0.8                    # Expected Brillouin linewidth Gamma_B in GHz
# LINEWIDTH_BOUNDS = (0.4, 1.3)               # Acceptable linewidth range (min, max) in GHz
# SHIFT_RANGE_SAMPLE = (4.8, 5.3)             # Filtering range for Brillouin shift (GHz)
# LINEWIDTH_RANGE_SAMPLE = (0.5, 1.2)         # Filtering range for Brillouin linewidth (GHz)

