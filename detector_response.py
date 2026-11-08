"""Pixel integration and instrument factors, independent of the sample model.

Signal is in fixed relative units: integral(I * P * dOmega) / 1e-9 sr.
No flux, efficiency, absorption, refraction, or scan Lorentz factor is assumed.
"""
from dataclasses import dataclass
import numpy as np
from physics import pixel_q

SIGNAL_REFERENCE_SR = 1e-9


@dataclass(frozen=True)
class DetectorResponse:
    subpixels: int = 2
    horizontal_fraction: float | None = None
    column_factor: bool = False
    attenuation_cells: float = 1000.
    bulk_real: float = 1.
    bulk_imag: float = 0.
    surface_real: float = 1.
    surface_imag: float = 0.

    def validate(self):
        if self.subpixels not in (1, 2, 4, 8, 16):
            raise ValueError('Pixel integration must use 1, 2, 4, 8 or 16 samples per axis.')
        if self.horizontal_fraction is not None and not 0 <= self.horizontal_fraction <= 1:
            raise ValueError('Horizontal polarization fraction must be between 0 and 1.')
        values = [self.attenuation_cells, self.bulk_real, self.bulk_imag,
                  self.surface_real, self.surface_imag]
        if not np.isfinite(values).all() or self.attenuation_cells <= 0:
            raise ValueError('Column parameters must be finite with positive attenuation length.')


def pixel_offsets(response):
    """Equal-area midpoint quadrature, including one-point diagnostic mode."""
    response.validate()
    offsets = (np.arange(response.subpixels) + .5) / response.subpixels - .5
    return [(x, y) for y in offsets for x in offsets]


def solid_angle_density(g, u, v):
    """dOmega/dA in sr/mm^2; D is the perpendicular sample-plane distance."""
    return g.distance / (g.distance**2 + np.asarray(u)**2 + np.asarray(v)**2)**1.5


def polarization_factor(out, response):
    # ki is lab +x; horizontal electric field is lab y, vertical is lab z.
    f = response.horizontal_fraction
    if f is None:
        return np.ones(np.asarray(out).shape[:-1])
    return f * (1 - out[..., 1]**2) + (1 - f) * (1 - out[..., 2]**2)


def column_intensity(g, qc, response):
    """Constant-F, damped semi-infinite column: Fs + Fb*z/(1-z).

    z = exp(-1/N - i*qz*a). N is an amplitude attenuation length in cells,
    a phenomenological regularizer, not an X-ray absorption calculation.
    expm1 preserves precision near integer L. No per-image normalization.
    """
    if not response.column_factor:
        return np.ones(np.asarray(qc).shape[:-1])
    exponent = -1 / response.attenuation_cells - 1j * qc[..., 2] * g.a
    z = np.exp(exponent)
    fb = complex(response.bulk_real, response.bulk_imag)
    fs = complex(response.surface_real, response.surface_imag)
    return np.abs(fs + fb * z / (-np.expm1(exponent)))**2


def sample_factors(g, response, u, v, qc, out):
    physical = (out @ g.normal > 0) & (g.incidence > 0)
    column = column_intensity(g, qc, response)
    omega = g.pitch**2 * solid_angle_density(g, u, v)
    return physical, column, omega * polarization_factor(out, response) / SIGNAL_REFERENCE_SR


def integrate_pixels(g, response, evaluate, u=None, v=None):
    """Integrate an arbitrary differential intensity; never average amplitudes.

    Nonphysical rays contribute zero. Any unresolved physical subpoint marks
    its entire pixel missing, preventing partial-pixel flux underestimates.
    Coordinates returned by callers remain pixel-centre coordinates.
    """
    if u is None:
        u = (np.arange(g.nx) - (g.nx - 1) / 2) * g.pitch
    if v is None:
        v = (np.arange(g.ny) - (g.ny - 1) / 2) * g.pitch
    U, V = np.meshgrid(u, v)
    morphology = np.zeros(U.shape)
    intrinsic = np.zeros(U.shape)
    signal = np.zeros(U.shape)
    omega = np.zeros(U.shape)
    valid = np.ones(U.shape, dtype=bool)
    physical_any = np.zeros(U.shape, dtype=bool)
    offsets = pixel_offsets(response)
    for du, dv in offsets:
        us, vs = U + du * g.pitch, V + dv * g.pitch
        _, qc, out = pixel_q(g, us, vs)
        physical, column, weight = sample_factors(g, response, us, vs, qc, out)
        value = np.asarray(evaluate(qc))
        valid &= np.isfinite(value) | ~physical
        physical_any |= physical
        value = np.where(physical, value, 0.)
        morphology += value / len(offsets)
        intrinsic += value * column / len(offsets)
        signal += value * column * weight / len(offsets)
        omega += g.pitch**2 * solid_angle_density(g, us, vs) / len(offsets)
    for values in (morphology, intrinsic, signal):
        values[~valid] = np.nan
    return dict(morphology_intensity=morphology, intrinsic_intensity=intrinsic,
                detector_signal=signal, solid_angle_sr=omega,
                sampling_valid=valid, physical=physical_any)


def convergence_metrics(coarse, fine):
    """Flux and L1 image changes on common finite pixels, not a proof of accuracy."""
    valid = np.isfinite(coarse) & np.isfinite(fine)
    if not valid.any():
        return dict(l1_relative=None, flux_relative=None, valid_fraction=0.)
    a, b = np.asarray(coarse)[valid], np.asarray(fine)[valid]
    denominator = np.sum(np.abs(b))
    if denominator == 0:
        return dict(l1_relative=None, flux_relative=None, valid_fraction=float(valid.mean()))
    return dict(l1_relative=float(np.sum(np.abs(a - b)) / denominator),
                flux_relative=float(abs(np.sum(a) - np.sum(b)) / denominator),
                valid_fraction=float(valid.mean()))


def log_upper_limit(arrays):
    maximum = max((float(np.nanmax(a)) for a in arrays if np.isfinite(a).any()), default=0.)
    return max(0, int(np.ceil(np.log10(max(maximum, 1e-300)))))
