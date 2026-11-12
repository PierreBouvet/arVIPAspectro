from HDF5_BLS import Wrapper
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = '/Volumes/LAUDATE/Data/260916 - arBLS/261110'
TARGET_FILE = 'Measures.h5'
FIGURE_DIRECTORY_NAME = 'figures'
OUT_DIR = os.path.join(DATA_DIR, FIGURE_DIRECTORY_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

name = []
M_theta = []
M_theta_err = []
Shift_0 = []
Shift_0_err = []
V_s = []
V_s_err = []
phi = []
gamma = []
xc = []
yc = []

wrapper = Wrapper(os.path.join(DATA_DIR, TARGET_FILE))

for i in range(1, 12):
    if i<10:
        sample = f'Brillouin/0{i}_glycerol_{24+i}/Treated 4x2 bins'
        name.append(29+i)
    elif i<12:
        sample = f'Brillouin/{i}_glycerol_{24+i}/Treated 4x2 bins'
        name.append(29+i)
    else:
        sample = f'Brillouin/{i}_water_{52-i}/Treated 4x2 bins'
        name.append(52-i)

    attributes = wrapper.get_attributes(sample)
    M_theta.append(float(attributes['M_theta']))
    M_theta_err.append(float(attributes['M_theta_err']))
    Shift_0.append(float(attributes['Shift_0']))
    Shift_0_err.append(float(attributes['Shift_0_err']))
    V_s.append(float(attributes['V_s']))
    V_s_err.append(float(attributes['V_s_err']))
    phi.append(float(attributes['phi_ellipse_deg']))
    gamma.append(float(attributes['gamma_ellipse']))
    xc.append(float(attributes['xc_ellipse']))
    yc.append(float(attributes['yc_ellipse']))

M_theta = np.array(M_theta)
M_theta_err = np.array(M_theta_err)
Shift_0 = np.array(Shift_0)
Shift_0_err = np.array(Shift_0_err)
V_s = np.array(V_s)
V_s_err = np.array(V_s_err)
phi = np.array(phi)
gamma = np.array(gamma)
xc = np.array(xc)
yc = np.array(yc)


plt.figure()
plt.subplot(131)
plt.errorbar(name,M_theta,M_theta_err)
plt.xlabel("Repetition")
plt.ylabel(r"$M_\theta$")
plt.subplot(132)
plt.errorbar(name,Shift_0,Shift_0_err)
plt.xlabel("Repetition")
plt.ylabel(r"Backscattering shift $S_0$ (GHz)")
plt.subplot(133)
plt.errorbar(name,V_s,V_s_err)
plt.xlabel("Repetition")
plt.ylabel(r"Speed of sound $V_s$ (m/s)")
plt.tight_layout()
plt.show()