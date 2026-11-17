# Naive angle resolved BLS measurements with the ar-BLS VIPA spectrometer

## Abstract
This document presents a naive approach to measuring the Brillouin signal at different angles using a movement of the illumination laser. Using these measures, we present a theoretical framework to measure in tissues a high-frequency longitudinal dispersion and a local high-frequency longitudinal anisotropy. 

## Theoretical framework

Following the Abbe sine relation, the radial position $r$ of a ray on the Back Focal Plane (BFP) of an objective is related to its angle $\theta_{air}$ in air with respect to the optical axis by:

$$r = f \sin \theta_{air}$$

where $f$ is the focal length of the objective. Accounting for refraction at the planar interface between the sample (of refractive index $n$) and air ($n_{air} \approx 1$) via Snell's law, $\sin \theta_{air} = n \sin \theta$, the position on the BFP is related to the internal scattering polar angle $\theta$ inside the sample by:

$$r = n f \sin \theta$$

Following this relation, assuming the detector is an image of the BFP of the objective, each position on the detector can be related to a position on the back focal plane of the objective and therefore a polar ($\theta$) and azimuthal ($\phi$) angle of collection in the sample, following the relation:

$$x_{obj} = n f \sin \theta \cos \phi$$
$$y_{obj} = n f \sin \theta \sin \phi$$

assuming zero astigmatism, we can then establish:

$$x_{det} = M x_{obj} = n M f \sin \theta \cos \phi$$
$$y_{det} = M y_{obj} = n M f \sin \theta \sin \phi$$

In Brillouin scattering, when measuring an isotropic sample, we find the relation between the internal angle of collection (which corresponds to the polar angle) $\theta$ and the measured Brillouin shift $\Delta \nu$ to be:

$$\Delta \nu(\theta) = \frac{2n}{\lambda} \cos(\theta/2) V_s$$

where $\lambda$ is the wavelength of the incident light, $n$ is the refractive index of the sample and $V_s$ is the velocity of sound in the sample. 

This leads to a relation between the shift $\Delta \nu$ and the position on the detector $(x_{det}, y_{det})$ for an isotropic sample:

$$\Delta \nu(x_{det}, y_{det}) = \frac{2n}{\lambda} \cos\left(\frac{1}{2}\arcsin \left(\frac{\sqrt{x_{obj}^2 + y_{obj}^2}}{n f}\right)\right) V_s= \frac{2n}{\lambda} \cos\left(\frac{1}{2}\arcsin \left(\frac{\sqrt{x_{det}^2 + y_{det}^2}}{n M f}\right)\right)V_s$$

We can use the identity $\cos(\frac{1}{2}\arcsin(x)) = \sqrt{\frac{1+\sqrt{1-x^2}}{2}}$ to obtain:

$$\Delta \nu(x_{det}, y_{det}) = \frac{2n}{\lambda} \sqrt{\frac{1+\sqrt{1-\left(\frac{\sqrt{x_{det}^2 + y_{det}^2}}{n M f}\right)^2}}{2}}V_s$$

which develops to:

$$\Delta \nu(x_{det}, y_{det}) = \frac{n}{\lambda}  \sqrt{2 + \frac{2\sqrt{(nMf)^2-x_{det}^2 - y_{det}^2}}{nMf}} V_s$$

This relation hides the geometry of scattering in $x_{det}$ and $y_{det}$ by assuming that the position of the optical axis falls on the origin of this system of coordinates. By defining $x_{det, 0}$ and $y_{det, 0}$ as the position of the optical axis on the detector, we can write the more general relation:

$$\Delta \nu_{x_{det,0}, y_{det,0}}(x_{det}, y_{det}) = \frac{n}{\lambda}  \sqrt{2 + \frac{2\sqrt{(nMf)^2-(x_{det}-x_{det,0})^2 - (y_{det}-y_{det,0})^2}}{nMf}} V_s$$

We can rewrite this expression with a geometric factor $C_{n,M,f,x_{det,0},y_{det,0}}$ imposed by the architecture of the spectrometer, the geometry of the measurement, and the refractive index $n$ of the sample (via Snell's law):

$$\Delta \nu_{x_{det,0}, y_{det,0}}(x_{det}, y_{det}) = C_{n,M,f,x_{det,0},y_{det,0}}(x_{det}, y_{det}) \frac{n V_s}{\lambda}$$

### Calibration of Mf in samples of known refractive index.

We can calibrate $C_{n,M,f,x_{det,0},y_{det,0}}(x_{det}, y_{det})$ in known samples by preforming a fit of this function to the observed results. 

In the ar-BLS setup, this can be done with arbitrary angles of tilt of the VIPA, meaning that we can in theory determine the value of $n M f$ by performing an arbitrary number of acquisitions with different angles of tilt of the VIPA. 


### Multiple illumination geometries in arbitrary samples

Considering two illumination geometries defined by $\{x_{det,0}^{(1)}, y_{det,0}^{(1)}\}$ and $\{x_{det,0}^{(2)}, y_{det,0}^{(2)}\}$, we get in an isotropic sample:

$$\frac{\Delta \nu_{x_{det,0}^{(1)}, y_{det,0}^{(1)}}(x_{det}, y_{det})}{\Delta \nu_{x_{det,0}^{(2)}, y_{det,0}^{(2)}}(x_{det}, y_{det})} = \sqrt{\frac{2 + \frac{2\sqrt{(nMf)^2-(x_{det}-x_{det,0}^{(1)})^2 - (y_{det}-y_{det,0}^{(1)})^2}}{nMf}}{2 + \frac{2\sqrt{(nMf)^2-(x_{det}-x_{det,0}^{(2)})^2 - (y_{det}-y_{det,0}^{(2)})^2}}{nMf}}} $$

$$ \frac{\Delta \nu_{x_{det,0}^{(1)}, y_{det,0}^{(1)}}(x_{det}, y_{det})}{\Delta \nu_{x_{det,0}^{(2)}, y_{det,0}^{(2)}}(x_{det}, y_{det})} = \frac{C_{n,M,f,x_{det,0}^{(1)},y_{det,0}^{(1)}}(x_{det}, y_{det})}{C_{n,M,f,x_{det,0}^{(2)},y_{det,0}^{(2)}}(x_{det}, y_{det})}
$$

This ratio is independent of the wavelength of illumination $\lambda$ and the speed of sound $V_s$ in the sample, and depends only on the refractive index $n$ of the sample (for a calibrated instrument $M f$)! 

This means that for an isotropic sample, this ratio directly constrains the refractive index $n$ independently of $V_s$, and any deviation from this relation indicates mechanical or optical anisotropy. 

It is practically difficult to measure optical anisotropy in confocal volumes due to the generally amorphous nature of the samples, so with this coherent but strong hypothesis in mind, we can establish that any deviation from this relation indicates mechanical anisotropy (specifically a difference in the longitudinal speed of sound depending on the angle of illumination).

### Same illumination geometry in arbitrary samples

Considering now the same illumination geometry defined by $\{x_{det,0}, y_{det,0}\}$ across two isotropic samples, we get:

$$\frac{\Delta \nu_{x_{det,0}, y_{det,0}}^{sample1}(x_{det}, y_{det})}{\Delta \nu_{x_{det,0}, y_{det,0}}^{sample2}(x_{det}, y_{det})} = \frac{C_{n_1,M,f,x_{det,0},y_{det,0}}(x_{det}, y_{det})}{C_{n_2,M,f,x_{det,0},y_{det,0}}(x_{det}, y_{det})} \frac{n_1}{n_2} \frac{V_{s,1}}{V_{s,2}}$$

Because of Snell's law, the geometric factor $C$ depends on the refractive index of each sample; the ratio across the detector $(x_{det}, y_{det})$ is therefore no longer spatially uniform when $n_1 \neq n_2$, reducing strictly to $\frac{n_1 V_{s,1}}{n_2 V_{s,2}}$ only on the optical axis $(x_{det}, y_{det}) = (x_{det,0}, y_{det,0})$ where $C = 2$. 

###
