# Detector response and finite pixels

The real detector is a plane. Its reciprocal-space acceptance is a curved
patch of the elastic Ewald sphere. The unchanged `physics.pixel_q` normalizes
the ray to **each subpixel**, then forms Q = k s_f − k_i. Pixel-centre Q/HKL
coordinates and analytic rod markers are unchanged by intensity options.

## Two quantities, one scattering model

For sample differential intensity I(Q), each square pixel of area A gives:

```
intrinsic_intensity = (1/A) integral_pixel I(Q(u,v)) du dv
detector_signal = integral_pixel I(Q(u,v)) P(u,v) dOmega / Omega_ref
dOmega = D du dv / (D^2 + u^2 + v^2)^(3/2)
Omega_ref = 1e-9 sr (fixed across all geometries and images)
```

The fixed reference makes detector signal convenient to display while retaining
pixel-area and inverse-square distance dependence. It is a **relative signal**,
not calibrated photon counts. No incident flux, efficiency, sensor absorption,
beam footprint, divergence, bandwidth, point-spread function or dark counts are
assumed. Unit efficiency is used. `solid_angle_sr` is exported separately.

The default is 2 × 2 equal-area midpoint samples per pixel. UI choices are
1, 2, 4 or 8 samples per axis; convergence checks can additionally evaluate 16.
One sample is a centre-sampling diagnostic, not an exact area integral.
The teaching model's former isotropic Gaussian pixel broadening was removed.
Gaussian widths now depend only on coherence, so changing detector pitch does
not change the intrinsic model.

`intrinsic_intensity` includes the optional column factor; the exported
`morphology_intensity` always excludes it. Both are unweighted pixel averages.
Both image modes and their row profiles switch between intrinsic intensity and
detector signal. Exports contain all three arrays and the response settings.
Geometry-only JSON remains backward compatible and does not save the response;
the linked-image NPZ and surface-settings JSON do save it.

No image is divided by its own maximum. Color limits can adjust by integer
decades to show a column enhancement, and all comparison cases share one scale.
Changing display mode does not require rerunning a completed surface calculation.

## Polarization

With incident beam along lab +x, horizontal electric field is y and vertical
electric field is z. The optional scalar Thomson factor is

```
P = f_horizontal*(1 - s_f,y^2) + (1-f_horizontal)*(1 - s_f,z^2).
```

Values 1, 0 and 0.5 represent horizontal, vertical and unpolarized incidence.
Disabled means P = 1, not an assumption about beamline polarization. No outgoing
polarization analyzer, resonant tensor or magnetic scattering is modeled.

## Optional common column factor

The morphology solver computes the normalized lateral/height envelope M(Q).
The optional column adds I(Q) = M(Q) |Fs + Fb z/(1-z)|², where

```
z = exp(-1/N - i*qz*a).
```

Fs and Fb are user-specified constant complex amplitudes of the surface and
bulk unit cells. Their defaults are 1. N is an **amplitude** attenuation length
in unit cells, not film thickness. This is a damped semi-infinite column with
a surface cell and a bulk tail. The attenuation makes the otherwise divergent
kinematic Bragg peak finite. It is phenomenological, not an absorption or
dynamical-diffraction calculation. `expm1` keeps the denominator stable.

This factor is applied at each quadrature point before pixel integration. For
the explicit height maps, the whole column termination shifts with height;
islands and substrate share its lattice/material model. Heteroepitaxial films
such as LNO on STO require separate amplitudes and film/substrate interference.
Constant F values are only a local approximation; no atomic form-factor
database, systematic extinction, termination chemistry or Q-dependent F is
supplied. Morphology-only remains the default until users choose this model.

## Rod crossing and integration conventions

The finite-width intrinsic intensity is evaluated on the Ewald sphere. The
crossing angle therefore already changes the spot shape and integrated weight.
Do **not** add a universal rod-interception or rocking-scan Lorentz multiplier
to this static forward image: this can count the geometry twice. An experimental
ROI/rocking-scan reduction needs its own Jacobian and acquisition definition.
The displayed row sums are sums in detector coordinates, not corrected rod
structure factors. The Gaussian teaching model still has an illustrative
coherence normalization rather than an absolute cross section.

## Coherence, missing pixels and convergence

For each subpixel and coherent patch the solver sums complex amplitudes before
squaring. It then averages **intensities** over subpixels and independent
patches. It never averages amplitudes between pixels. Subpixels are streamed
to limit memory; computational cost grows approximately with sample count.

Nonphysical outgoing rays contribute zero. If any physical subpixel lies
outside the FFT sampling range, the entire pixel is NaN/missing for all cases.
Partially valid pixels are not silently treated as fully collected pixels.
Exported `physical` means any sampled ray is physical; `sampling_valid` means
all physical subpoints are resolved. Row sums exclude missing pixels; they
remain partial sums if the mask removes part of a rod.

Convergence checks compare n × n with 2n × 2n using the same seed, surface grid,
FFT padding and physical parameters. They report L1 image change normalized
by finer-image L1 intensity, total-signal change, and common valid fraction.
Only common finite pixels are compared. Zero/no-signal images are reported as
unassessable. Less than 1% change with full coverage is a useful diagnostic,
not a certified accuracy bound. Extremely narrow peaks can be missed at both
orders; repeat refinement and independently check surface/FFT resolution,
especially for large N near Bragg peaks. The check does not replace the
displayed result with the finer result.

## Validation

`test_detector_response.py` checks the rectangular-detector analytic solid
angle, quadrature convergence, inverse-square distance/pixel-area scaling,
signal conservation under exact pixel subdivision, a narrow Gaussian against
its erf integral, polarization limits, a direct complex layer sum, missing
subpixels, and a real-space coherent sum independent of the FFT implementation.
Existing geometry/HKL, surface and plot regression tests remain in use.

The default 512 × 800 / 55 µm / 1700 mm geometry has a center-to-corner
solid-angle-density reduction of 0.0352974%. This small smooth effect should
not be confused with the much stronger possible rod-crossing effects.

## References

- Schlepütz et al., *Angle calculations for a (2+3)-type diffractometer: focus
  on area detectors*, J. Appl. Cryst. 44, 73–83 (2011),
  https://doi.org/10.1107/S0021889810048922 (pixel mapping and flat-detector factors).
- Gasser et al., *Intensity corrections for grazing-incidence X-ray diffraction
  of thin films using static area detectors*, J. Appl. Cryst. 58, 96–106 (2025),
  https://doi.org/10.1107/S1600576724010628 (polarization and solid angle).
- Petach et al., *Crystal truncation rods from miscut surfaces*,
  Phys. Rev. B 95, 184104 (2017), https://arxiv.org/abs/1706.00484
  (column/rod model and the distinction between model and corrected data).
- Harlow et al., *HAT: a high-energy surface X-ray diffraction analysis toolkit*,
  J. Appl. Cryst. 56, 312–321 (2023), https://doi.org/10.1107/S1600576723000092
  (dependence of correction factors on the integration procedure).
