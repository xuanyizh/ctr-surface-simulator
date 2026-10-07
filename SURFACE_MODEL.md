# Explicit nonideal surfaces

Open **Surface forward simulation**, choose the three feature switches, then
click **Run surface simulation**. The sidebar still controls the exact geometry,
mean miscut, lattice constant and coherent-patch dimensions. The existing
Linked views image remains the fast teaching model.

The new tab compares ideal steps, each enabled feature alone, and the selected
combination. Every case has the same intensity normalization. Independent
random-number streams preserve a feature when another feature is toggled.
Surface height maps, cross-sections, detector images, row sums, realized
statistics and reproducible JSON/NPZ exports are provided. A warning marks old
results if the geometry changes before another run.

## What comes from the paper

Petach et al., *Crystal truncation rods from miscut surfaces*,
[Phys. Rev. B 95, 184104 (2017)](https://doi.org/10.1103/PhysRevB.95.184104),
[open manuscript](https://arxiv.org/pdf/1706.00484).

The paper uses Gaussian step-position deviations and the particular covariance
in Eq. (10). Under those assumptions the ensemble intensity separates into
sharp sub-rod peaks `Ip`, a width-disorder component `Iw`, and an edge-disorder
component `Is` (Eqs. 11, 12, 22, 26). Both disorder parameters attenuate the
sharp peaks; width disorder also gives a broad component beneath them. The
paper neglects correlations along a jagged edge in this simplified model.
Its Fig. 3 is an ensemble decomposition, not three single-surface images to add.

**The new numerical solver is an extension, not a reproduction of those closed
forms.** Independent positive terrace widths accumulate into a renewal process:
step-position variance grows with step separation. This is different from the
bounded different-step variance specified in Eq. (10). Do not identify our
input width CV with the paper's fitted `sigma_w / mean_width` without matching
their correlation definitions. Islands are an additional explicit morphology.

## Surface construction

Coordinates are crystal x (across descending steps), y (along steps), and z.
One step is one cubic cell of height `a`; the mean width is `w = a/tan(mu)`.
Zero miscut generates a flat base and bypasses step disorder, but allows islands.

### Terrace-width variation

For requested coefficient of variation `c = std(width)/mean(width)`:

- Gamma: shape `1/c²`, scale `w*c²` before conditioning.
- Lognormal: log variance `log(1+c²)` and log mean `log(w)-log(1+c²)/2`.
- Truncated normal: draw `N(w, (c*w)²)`, rejecting nonpositive widths.

Zero CV gives identical widths. Positive draws have a tiny numerical floor and
are rescaled together to fix the total span and preserve the nominal mean
miscut. Thus the finite set is conditioned and no longer strictly independent;
its measured CV is reported. For the truncated normal, `c` controls the
**untruncated** Gaussian, so especially at large CV the resulting CV differs.
The central step is anchored at x=0 to avoid a random translation of the entire
patch. These choices are explicit modeling assumptions, not a step-interaction
equilibrium model. Widths may correlate in a real sample; that is not fitted here.

### Jagged edges

Each boundary has a zero-mean Gaussian displacement field `u_m(y)` added to its
base x position. Before finite-domain demeaning, the along-edge covariance is
proportional to `exp(-|dy|/ell)`, with user-controlled RMS and correlation length
`ell`. A shared Gaussian field mixed with independent fields gives a nominal
common correlation `rho` between different edges. At rho=1 all boundaries
meander together and preserve their local mutual spacing. Demeaning and finite
sampling slightly change the requested covariance; realized RMS is reported.

The random paths are generated on a fixed 2049-point y grid over the patch and
interpolated onto the chosen surface grid. Refinement therefore samples the
same realization. For ell=0 the reference samples are independent; this is a
grid-cutoff white-noise limit, not resolution-independent continuum roughness.

Boundaries cannot cross. If necessary, a single scale factor reduces all edge
displacements so at least 5% of each base gap remains on the reference grid.
Interpolation preserves that ordering. The app reports the scale factor and
actual RMS; it does not sort or exchange step identities. Sub-cell meandering,
short correlations and narrow terraces are flagged as unresolved.

### Islands

Islands are circular patches with radius R and integer height b above the local
stepped surface. A homogeneous Poisson distribution of centres has density

`density = -log(1-theta)/(pi*R²)`.

The union of the disks has expected coverage theta. Overlaps merge and do not
stack; requested coverage is not an exact constraint on a finite patch.
Both area fraction and Gaussian-amplitude-weighted fraction are reported.
Centres are generated outside the patch as well to avoid losing crossing disks
at the boundary. Centre positions are independent of surface grid resolution.
Theta=0 gives no islands; theta=1 adds a complete uniform layer of height b.

Patches crossing a step follow its local height; they are not single flat-top
mesas spanning multiple terraces. No elastic relaxation, terrace-dependent
nucleation, size distribution or growth kinetics is assumed. Those require
additional morphology rules. Substrate and islands have the same lattice and
scattering factor in this initial implementation.

## Coherent scattering

For an ideal semi-infinite same-material column with top at `a*n`, the
kinematic vertical amplitude away from a bulk Bragg singularity is

`F(q) * exp(i*qz*a*n) / (1-exp(-i*qz*a))`.

The common column factor is omitted, leaving the normalized lateral morphology
amplitude

```text
S(q) = sum_j W_j exp{i[(qx-Gx)*x_j + (qy-Gy)*y_j + qz*a*n_j]} / sum_j W_j
M(q) = |S(q)|²
```

`Gx=2*pi*h_family/a`, `Gy=2*pi*k_family/a`. The lateral grid is a coarse
envelope near this selected reciprocal-lattice family, not an atomic lattice
summation valid throughout reciprocal space. Constant-height rectangular tiles
add `sinc((qx-Gx)*dx/2)*sinc((qy-Gy)*dy/2)`, using the unnormalized sinc here.
Code uses NumPy's normalized sinc with the appropriate 2*pi denominator.

Height and lateral phases are both retained. For combined features, the solver
first constructs one height map, then sums its amplitudes and squares. It does
not add separately simulated intensities. If a structure is decomposed as
`A_base + DeltaA`, its intensity includes `2*Re(A_base*conj(DeltaA))`.
The isolated-case images each contain a base crystal, so summing them would
also count that base repeatedly. Turning features on/off is a controlled
comparison, not a unique additive assignment of measured intensity.

### Patch / coherence convention

`W(x,y) = exp[-(x/xi_x)²-(y/xi_y)²]` is an **amplitude** window over approximately
`[-3*xi_x, 3*xi_x] × [-3*xi_y, 3*xi_y]`. A flat surface has intensity proportional
to `exp[-((kx*xi_x)²+(ky*xi_y)²)/2]`, consistent with the teaching model's
inverse-coherence width convention. The domain is not periodically tiled in
real space. This single-window patch approximation does not independently
describe the full beam footprint and transverse mutual coherence.

One realization retains coherent speckle. Multiple independent surface patches
produce `mean(|S_r|²)`, never `|mean(S_r)|²`. Patch averaging is an illustrative
incoherent ensemble; no temporal evolution or XPCS correlation is calculated.

### Detector and numerical resolution

Every detector pixel uses the existing `pixel_q` mapping. The solver keeps the
pixel's individual qx, qy **and qz**, rather than projecting onto one fixed-qz
plane. For each integer height, its weighted lateral mask is Fourier transformed
with zero padding; complex bilinear interpolation evaluates off-grid lateral q,
then the height phase is applied. The sign convention uses a positive Fourier
exponent and matches descending +x steps and the existing rod geometry.

Points outside the lateral FFT range are returned as NaN and shown blank,
with an exported sampling-valid mask. They are not zero signal. Row sums omit
these missing pixels and use the same mask for every case. Detector maps sample
pixel centres: there is **no** pixel-area integration or PSF convolution.
The teaching model's approximate pixel blur is not applied to this solver.

Before interpreting weak diffuse scattering or speckle quantitatively:

1. Increase surface grid resolution at fixed physical parameters and seed.
2. Increase FFT padding (2, 4, 8) to check complex interpolation convergence.
3. Check detector sampling relative to peak/speckle widths. A smaller detector
   pitch changes the sampled field of view unless its dimensions also change.
4. Check finite-patch and ensemble-size dependence. More patches smooth speckle;
   changing coherence changes the physical model, not just numerical accuracy.

The UI limits FFT memory and height-level counts to keep browser runs bounded.
CPU use scales with patches × cases × height levels × padded FFT size.

## Interpretation and remaining scope

The common scale makes attenuation and intensity redistribution visible without
normalizing each image to its own maximum. Different q values probe different
Fourier components, so their intensities need not respond identically to one
morphology change. A detector ROI does not select a unique real-space region.

This is a forward **morphology** simulator. It does not predict absolute counts,
film/substrate interference for different materials, atomic structure factors,
Bragg enhancement, refraction, dynamical diffraction, absorption, polarization,
detector noise or temporal growth/XPCS dynamics. In particular, LNO islands on
STO need separate material amplitudes before quantitative experimental fitting.
The exact geometry engine is unchanged.

## Reproduce or extend in Python

```python
from physics import preset
from surface_model import SurfaceParameters, simulate_detector

g = preset('Oblique steps')
p = SurfaceParameters(width_enabled=True, width_distribution='gamma',
                      width_cv=0.15, jagged_enabled=True, edge_rms_nm=20,
                      edge_correlation_nm=50, islands_enabled=True,
                      island_coverage=0.15, island_radius_nm=60, seed=2026)
result = simulate_detector(g, p, compare=True)
image = result['cases']['Selected combination']['intensity']
```

`surface_settings.json` contains geometry, parameters, comparison mode and all
realization statistics. Restore with `Geometry(**data['geometry'])` and
`SurfaceParameters(**data['surface'])`. The NPZ contains case names, intensity
maps, actual q maps, masks, first-realization surfaces/edges/widths, the amplitude
window and the same JSON metadata. It needs no pickle. All other realizations
can be regenerated with `generate_surface(g, p, realization_index)`, using
`comparison_cases` for individual-feature settings. Settings files are restored
through Python; the existing geometry-only uploader does not load this bundle.

Run `python -m unittest -v test_physics test_surface_model` and
`python test_app.py`. The tests compare FFT amplitudes against an independent
direct tile sum, recover ideal rod weights and spacing, check anti-Bragg island
cancellation and integer-l contrast loss, verify intensity ensemble averaging,
and exercise resolution masks, feature independence and no-crossing constraints.
