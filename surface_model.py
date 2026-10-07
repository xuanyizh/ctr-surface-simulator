"""Seeded stepped surfaces and coherent morphology scattering (NumPy only).

Lengths internal to this module are Angstrom. Heights are integer cubic cells.
This is a coarse lateral, kinematic, same-material column model; see
SURFACE_MODEL.md. It is not an implementation of Petach Eqs. 12/22/26.
"""
from dataclasses import dataclass, replace
import numpy as np
from physics import pixel_q


@dataclass(frozen=True)
class SurfaceParameters:
    width_enabled: bool = True
    width_distribution: str = 'gamma'
    width_cv: float = 0.15
    jagged_enabled: bool = True
    edge_rms_nm: float = 20.
    edge_correlation_nm: float = 50.
    edge_neighbor_correlation: float = 0.
    islands_enabled: bool = True
    island_coverage: float = 0.15
    island_radius_nm: float = 60.
    island_height_uc: int = 1
    seed: int = 2026
    nx: int = 512
    ny: int = 256
    padding: int = 4
    realizations: int = 1

    def validate(self):
        if self.width_distribution not in ('gamma', 'lognormal', 'truncated_normal'):
            raise ValueError('Unknown terrace width distribution.')
        for name in ('width_cv', 'edge_rms_nm', 'edge_correlation_nm',
                     'edge_neighbor_correlation', 'island_coverage', 'island_radius_nm'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        if self.width_cv > 1 or self.edge_neighbor_correlation > 1 or self.island_coverage > 1:
            raise ValueError('Width CV, neighbor correlation and coverage must be at most 1.')
        if self.island_radius_nm <= 0:
            raise ValueError('Island radius must be positive.')
        for name in ('nx', 'ny', 'padding', 'realizations', 'island_height_uc', 'seed'):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f'{name} must be an integer.')
        if self.seed < 0 or not 1 <= self.island_height_uc <= 20:
            raise ValueError('Seed must be nonnegative; island height must be 1–20 cells.')
        if not (16 <= self.nx <= 1024 and 16 <= self.ny <= 512):
            raise ValueError('Surface grid must be 16–1024 by 16–512.')
        if self.nx % 2 or self.ny % 2:
            raise ValueError('Use even surface grid dimensions.')
        if self.padding not in (2, 4, 8) or not 1 <= self.realizations <= 16:
            raise ValueError('Padding must be 2, 4 or 8; realizations must be 1–16.')
        if self.nx * self.ny * self.padding**2 > 8_388_608:
            raise ValueError('FFT too large. Reduce grid size or padding.')


@dataclass
class Surface:
    x: np.ndarray
    y: np.ndarray
    height_uc: np.ndarray
    island_mask: np.ndarray
    edges: np.ndarray                 # (y, step index), Angstrom
    base_widths: np.ndarray
    illumination: np.ndarray          # Gaussian amplitude, not intensity
    stats: dict


def _widths(rng, count, mean, cv, distribution):
    """Positive widths conditioned on a fixed total span / mean miscut."""
    if cv == 0:
        return np.full(count, mean)
    if distribution == 'gamma':
        values = rng.gamma(1 / cv**2, cv**2, count)
    elif distribution == 'lognormal':
        variance = np.log1p(cv**2)
        values = rng.lognormal(-variance / 2, np.sqrt(variance), count)
    else:
        values = rng.normal(1, cv, count)
        bad = values <= 0
        while np.any(bad):
            values[bad] = rng.normal(1, cv, np.count_nonzero(bad))
            bad = values <= 0
    # Positive distributions can contain tiny values. A negligible floor also
    # leaves a nonzero geometric separation for the no-crossing constraint.
    values = np.maximum(values, 1e-6)
    return values * (count * mean / values.sum())


def generate_surface(g, p=SurfaceParameters(), realization=0):
    p.validate()
    if not np.isfinite([g.a, g.coherence_x, g.coherence_y, g.miscut]).all():
        raise ValueError('Lattice, coherence and miscut must be finite.')
    if min(g.a, g.coherence_x, g.coherence_y) <= 0 or not 0 <= g.miscut < 45:
        raise ValueError('Use positive lattice/coherence and 0 ≤ miscut < 45°.')
    if int(realization) != realization or realization < 0:
        raise ValueError('Realization index must be a nonnegative integer.')
    # Independent streams keep each feature identical when another is toggled.
    streams = np.random.SeedSequence([p.seed, int(realization)]).spawn(3)
    rw, re, ri = [np.random.default_rng(s) for s in streams]
    xi_x, xi_y = g.coherence_x * 1e4, g.coherence_y * 1e4
    dx, dy = 6 * xi_x / p.nx, 6 * xi_y / p.ny
    x = (np.arange(p.nx) - p.nx // 2) * dx
    y = (np.arange(p.ny) - p.ny // 2) * dy
    X, Y = np.meshgrid(x, y)
    window = np.exp(-(X / xi_x)**2 - (Y / xi_y)**2)
    warnings = []
    edge_scale = 1.
    if g.miscut == 0:
        edges = np.empty((p.ny, 0))
        widths = np.array([])
        heights = np.zeros((p.ny, p.nx), dtype=np.int32)
        if p.width_enabled or p.jagged_enabled:
            warnings.append('At zero miscut there are no steps; width and edge controls have no effect.')
    else:
        w = g.terrace_width
        half = int(np.ceil(3 * xi_x / w)) + 2
        if 2 * half + p.island_height_uc > 160:
            raise ValueError('Too many terrace height levels for the browser solver. Reduce miscut or across-step coherence.')
        widths = _widths(rw, 2 * half, w, p.width_cv if p.width_enabled else 0., p.width_distribution)
        base = np.r_[-half * w, -half * w + np.cumsum(widths)]
        # All realizations share the same central step reference (no random
        # translation of the whole surface); total span and mean width stay fixed.
        base -= base[half]
        displacement = np.zeros((p.ny, len(base)))
        if p.jagged_enabled and p.edge_rms_nm > 0:
            # A fixed reference path lets grid-refinement runs sample the SAME
            # stochastic edges. The no-crossing limit also uses this fixed path.
            reference_y = np.linspace(-3 * xi_y, 3 * xi_y, 2049)
            reference = np.zeros((len(reference_y), len(base)))
            rho = p.edge_neighbor_correlation
            noise = (np.sqrt(1 - rho) * re.normal(size=reference.shape)
                     + np.sqrt(rho) * re.normal(size=(len(reference_y), 1)))
            corr = np.exp(-(reference_y[1] - reference_y[0]) / (10 * p.edge_correlation_nm)) if p.edge_correlation_nm else 0.
            reference[0] = noise[0]
            for j in range(1, len(reference_y)):
                reference[j] = corr * reference[j - 1] + np.sqrt(1 - corr**2) * noise[j]
            reference -= reference.mean(axis=0, keepdims=True)
            rms = np.sqrt(np.mean(reference**2))
            if rms > 0:
                reference *= p.edge_rms_nm * 10 / rms
            # Scale the entire displacement field, never sort crossing steps.
            # Sorting would silently change their identity and correlations.
            delta = np.diff(reference, axis=1)
            closing = np.maximum(-delta, 0).max(axis=0)
            use = closing > 0
            if np.any(use):
                edge_scale = min(1., float(np.min(.95 * widths[use] / closing[use])))
            reference *= edge_scale
            displacement = np.stack([np.interp(y, reference_y, column) for column in reference.T], axis=1)
            if edge_scale < .999:
                warnings.append(f'No-crossing constraint scaled edge RMS to {edge_scale:.1%} of the request; inspect realized RMS.')
        edges = base[None, :] + displacement
        if np.any(edges[:, 0] > x[0] - dx / 2) or np.any(edges[:, -1] < x[-1] + dx / 2):
            raise ValueError('Disordered steps do not cover the full patch. Try another seed or a smaller width CV / edge RMS.')
        heights = np.array([half + 1 - np.searchsorted(row, x, side='right') for row in edges], dtype=np.int32)
        if w < 8 * dx:
            warnings.append('Fewer than 8 x samples per mean terrace: refine the grid or shorten across-step coherence.')
        if np.min(widths) < 2 * dx:
            warnings.append('Some terraces are narrower than 2 x cells; their scattering is not spatially resolved.')
        if p.jagged_enabled and 0 < p.edge_correlation_nm * 10 < 2 * dy:
            warnings.append('Edge correlation length is below 2 y cells; refine the grid.')
        if p.jagged_enabled and 0 < p.edge_rms_nm * 10 * edge_scale < dx:
            warnings.append('Edge RMS is below one x cell; refine the grid to resolve the meandering.')
    mask = np.zeros_like(heights, dtype=bool)
    if p.islands_enabled and p.island_coverage > 0:
        if p.island_coverage == 1:
            mask[:] = True
        else:
            radius = p.island_radius_nm * 10
            # Poisson Boolean disks: theta = 1-exp(-density*pi*R^2).
            # Padding the centre domain removes island loss at patch boundaries.
            lx, ly = p.nx * dx, p.ny * dy
            density = -np.log1p(-p.island_coverage) / (np.pi * radius**2)
            expected = density * (lx + 2 * radius) * (ly + 2 * radius)
            if expected > 5000:
                raise ValueError('Too many islands. Increase radius or reduce coverage/coherence.')
            count = ri.poisson(expected)
            cx = ri.uniform(-lx / 2 - radius, lx / 2 + radius, count)
            cy = ri.uniform(-ly / 2 - radius, ly / 2 + radius, count)
            for xx, yy in zip(cx, cy):
                mask |= (X - xx)**2 + (Y - yy)**2 <= radius**2
            if radius < 3 * max(dx, dy):
                warnings.append('Island radius is below 3 grid cells; refine the grid to resolve island shape.')
    heights += p.island_height_uc * mask
    levels = np.unique(heights)
    if len(levels) > 160:
        raise ValueError('Too many height levels; reduce miscut, coherence or island height.')
    edge_rms = float(np.sqrt(np.mean((edges - edges.mean(axis=0))**2))) if edges.size else 0.
    stats = dict(dx_nm=dx / 10, dy_nm=dy / 10,
                 mean_width_nm=float(widths.mean() / 10) if widths.size else None,
                 realized_width_cv=float(widths.std() / widths.mean()) if widths.size else 0.,
                 realized_edge_rms_nm=edge_rms / 10, edge_scale=edge_scale,
                 island_area_fraction=float(mask.mean()),
                 illuminated_island_fraction=float(np.sum(window * mask) / window.sum()),
                 height_levels=int(len(levels)), warnings=warnings)
    return Surface(x, y, heights, mask, edges, widths, window, stats)


def _interpolation_indices(axis, values):
    position = (values - axis[0]) / (axis[1] - axis[0])
    idx = np.clip(np.floor(position).astype(int), 0, len(axis) - 2)
    fraction = np.clip(position - idx, 0, 1)
    valid = (values >= axis[0]) & (values <= axis[-1])
    return idx, fraction, valid


def _consume_steps(steps, progress=None):
    """Run a cooperative calculation synchronously, retaining its return value."""
    while True:
        try:
            update = next(steps)
        except StopIteration as finished:
            return finished.value
        if progress is not None:
            progress(update)


def morphology_amplitude(surface, qc, g, padding=4):
    """Synchronous interface to the same cooperative Fourier calculation."""
    return _consume_steps(_morphology_amplitude_steps(surface, qc, g, padding))


def _morphology_amplitude_steps(surface, qc, g, padding=4):
    """Complex normalized amplitude at arbitrary crystal q, including varying qz.

    Sum separately transformed integer-height masks with exp(i*qz*a*n).
    Off-grid FFT interpolation is approximate; check grid/padding convergence.
    Pixels beyond the lateral grid Nyquist range are NaN, never wrapped.
    """
    qc = np.asarray(qc, dtype=float)
    if qc.shape[-1] != 3 or not np.isfinite(qc).all():
        raise ValueError('q must have a finite final dimension of size 3.')
    if padding not in (2, 4, 8):
        raise ValueError('FFT padding must be 2, 4 or 8.')
    shape = qc.shape[:-1]
    q = qc.reshape(-1, 3)
    kx, ky = q[:, 0] - 2 * np.pi * g.h / g.a, q[:, 1] - 2 * np.pi * g.k / g.a
    ny, nx = surface.height_uc.shape
    dx, dy = surface.x[1] - surface.x[0], surface.y[1] - surface.y[0]
    sx, sy = nx * padding, ny * padding
    fx = np.fft.fftshift(np.fft.fftfreq(sx, dx)) * 2 * np.pi
    fy = np.fft.fftshift(np.fft.fftfreq(sy, dy)) * 2 * np.pi
    ix, tx, vx = _interpolation_indices(fx, kx)
    iy, ty, vy = _interpolation_indices(fy, ky)
    amplitude = np.zeros(len(q), dtype=complex)
    window = surface.illumination / surface.illumination.sum()
    pad = ((sy // 2 - ny // 2, sy // 2 - ny // 2),
           (sx // 2 - nx // 2, sx // 2 - nx // 2))
    levels = np.unique(surface.height_uc)
    for level_index, n in enumerate(levels):
        field = np.pad(window * (surface.height_uc == n), pad)
        ft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field))) * sx * sy
        value = ((1 - ty) * ((1 - tx) * ft[iy, ix] + tx * ft[iy, ix + 1])
                 + ty * ((1 - tx) * ft[iy + 1, ix] + tx * ft[iy + 1, ix + 1]))
        amplitude += value * np.exp(1j * q[:, 2] * g.a * n)
        yield level_index + 1, len(levels)
    # Integrate each constant-height rectangular lateral tile, rather than
    # silently treating a coarse grid cell as an atomic point scatterer.
    amplitude *= np.sinc(kx * dx / (2 * np.pi)) * np.sinc(ky * dy / (2 * np.pi))
    valid = vx & vy
    amplitude[~valid] = np.nan + 1j * np.nan
    return amplitude.reshape(shape), valid.reshape(shape)


def comparison_cases(p, compare=True):
    ideal = replace(p, width_enabled=False, jagged_enabled=False, islands_enabled=False)
    if not compare:
        return {'Selected combination': p}
    cases = {'Ideal steps': ideal}
    if p.width_enabled:
        cases['Width only'] = replace(ideal, width_enabled=True)
    if p.jagged_enabled:
        cases['Jagged edges only'] = replace(ideal, jagged_enabled=True)
    if p.islands_enabled:
        cases['Islands only'] = replace(ideal, islands_enabled=True)
    cases['Selected combination'] = p
    return cases


def simulate_detector(g, p=SurfaceParameters(), compare=True, progress=None):
    """Synchronous API; progress receives a fraction between zero and one."""
    callback = (lambda update: progress(update['fraction'])) if progress is not None else None
    return _consume_steps(simulate_detector_steps(g, p, compare), callback)


def simulate_detector_steps(g, p=SurfaceParameters(), compare=True):
    """Coherent patches; independent realizations are averaged as intensities.

    Pixel centres use the existing exact elastic geometry. No pixel blur or
    pixel-area integration is implicit. No independently normalized maps.
    """
    p.validate()
    u = (np.arange(g.nx) - (g.nx - 1) / 2) * g.pitch
    v = (np.arange(g.ny) - (g.ny - 1) / 2) * g.pitch
    U, V = np.meshgrid(u, v)
    qlab, qc, out = pixel_q(g, U, V)
    physical = (out @ g.normal > 0) & (g.incidence > 0)
    results = {}
    cases = comparison_cases(p, compare)
    completed = 0
    total = len(cases) * p.realizations
    for name, params in cases.items():
        intensity = np.zeros(U.shape)
        first = None
        statistics = []
        for j in range(p.realizations):
            stage = f'{name} · patch {j + 1}/{p.realizations}'
            yield dict(fraction=completed / total, text=f'{stage} · building surface')
            surface = generate_surface(g, params, j)
            if first is None:
                first = surface
            steps = _morphology_amplitude_steps(surface, qc, g, p.padding)
            while True:
                try:
                    level, levels = next(steps)
                except StopIteration as finished:
                    amplitude, valid = finished.value
                    break
                yield dict(fraction=(completed + level / levels) / total,
                           text=f'{stage} · height {level}/{levels}')
            intensity += np.where(physical, np.abs(amplitude)**2, 0.) / p.realizations
            statistics.append(surface.stats)
            completed += 1
        results[name] = dict(intensity=intensity, surface=first, statistics=statistics)
    return dict(u_mm=u, v_mm=v, q_lab=qlab, q_crystal=qc,
                sampling_valid=valid, physical=physical, cases=results)
