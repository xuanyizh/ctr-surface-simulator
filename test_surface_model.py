"""Physical limits and independent Fourier checks for the surface solver."""
import io
import unittest
from dataclasses import replace
import numpy as np
from physics import preset
from surface_model import SurfaceParameters, generate_surface, morphology_amplitude, simulate_detector


def direct_amplitude(surface, qc, g):
    """Independent O(pixels*cells) reference, with exact rectangular tile integral."""
    X, Y = np.meshgrid(surface.x, surface.y)
    dx, dy = np.diff(surface.x)[0], np.diff(surface.y)[0]
    values = []
    for q in np.asarray(qc).reshape(-1, 3):
        kx, ky = q[:2] - 2 * np.pi / g.a * np.array([g.h, g.k])
        phase = kx * X + ky * Y + q[2] * g.a * surface.height_uc
        total = np.sum(surface.illumination * np.exp(1j * phase)) / surface.illumination.sum()
        values.append(total * np.sinc(kx * dx / (2 * np.pi)) * np.sinc(ky * dy / (2 * np.pi)))
    return np.array(values).reshape(np.asarray(qc).shape[:-1])


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.g = replace(preset('Oblique steps'), nx=24, ny=20)
        self.p = SurfaceParameters(nx=64, ny=32, padding=4)

    def test_seed_and_independent_feature_streams(self):
        s = generate_surface(self.g, self.p)
        same = generate_surface(self.g, self.p)
        np.testing.assert_array_equal(s.height_uc, same.height_uc)
        no_islands = generate_surface(self.g, replace(self.p, islands_enabled=False))
        np.testing.assert_array_equal(s.edges, no_islands.edges)
        np.testing.assert_array_equal(s.height_uc - no_islands.height_uc, s.island_mask * self.p.island_height_uc)
        straight = generate_surface(self.g, replace(self.p, jagged_enabled=False))
        np.testing.assert_array_equal(s.base_widths, straight.base_widths)
        np.testing.assert_array_equal(s.island_mask, straight.island_mask)

    def test_positive_widths_fixed_mean_all_distributions(self):
        for dist in ['gamma', 'lognormal', 'truncated_normal']:
            s = generate_surface(self.g, replace(self.p, width_distribution=dist, width_cv=.4))
            self.assertTrue(np.all(s.base_widths > 0))
            self.assertAlmostEqual(s.base_widths.mean(), self.g.terrace_width)
            self.assertGreater(s.base_widths.std(), 0)

    def test_grid_refinement_resamples_the_same_realization(self):
        coarse = generate_surface(self.g, self.p)
        fine = generate_surface(self.g, replace(self.p, nx=2*self.p.nx, ny=2*self.p.ny))
        np.testing.assert_allclose(fine.edges[::2], coarse.edges, atol=1e-10)
        np.testing.assert_array_equal(fine.height_uc[::2, ::2], coarse.height_uc)
        np.testing.assert_array_equal(fine.island_mask[::2, ::2], coarse.island_mask)

    def test_noncrossing_even_for_extreme_jaggedness(self):
        s = generate_surface(self.g, replace(self.p, edge_rms_nm=300., edge_correlation_nm=50.))
        self.assertTrue(np.all(np.diff(s.edges, axis=1) > 0))
        self.assertLess(s.stats['edge_scale'], 1.)
        self.assertLess(s.stats['realized_edge_rms_nm'], 300.)

    def test_shared_meandering_preserves_local_widths(self):
        s = generate_surface(self.g, replace(self.p, edge_neighbor_correlation=1.))
        np.testing.assert_allclose(np.diff(s.edges, axis=1), np.broadcast_to(s.base_widths, (self.p.ny, len(s.base_widths))), atol=1e-10)

    def test_zero_amplitude_settings_restore_ideal(self):
        ideal = generate_surface(self.g, replace(self.p, width_enabled=False, jagged_enabled=False, islands_enabled=False))
        zero = generate_surface(self.g, replace(self.p, width_cv=0., edge_rms_nm=0., island_coverage=0.))
        np.testing.assert_array_equal(ideal.height_uc, zero.height_uc)
        expected = -np.floor(ideal.x / self.g.terrace_width).astype(int)
        np.testing.assert_array_equal(ideal.height_uc, np.broadcast_to(expected, ideal.height_uc.shape))

    def test_flat_surface_gaussian_transform(self):
        g = replace(self.g, miscut=0.)
        s = generate_surface(g, replace(self.p, islands_enabled=False, nx=256, ny=128))
        qc = np.array([[0., 0., .73], [1 / (g.coherence_x * 1e4), 0., .8],
                       [0., 1 / (g.coherence_y * 1e4), 1.2]])
        amplitude, valid = morphology_amplitude(s, qc, g, 8)
        self.assertTrue(valid.all())
        expected = np.exp(-.25 * ((qc[:, 0] * g.coherence_x * 1e4)**2 + (qc[:, 1] * g.coherence_y * 1e4)**2))
        np.testing.assert_allclose(amplitude, expected, atol=.001)

    def test_fft_matches_independent_direct_sum_at_fft_nodes(self):
        s = generate_surface(self.g, self.p)
        dx, dy = np.diff(s.x)[0], np.diff(s.y)[0]
        qx = 2 * np.pi / (self.p.nx * self.p.padding * dx)
        qy = 2 * np.pi / (self.p.ny * self.p.padding * dy)
        qc = np.array([[0., 0., .71], [3 * qx, -2 * qy, .82], [-11 * qx, 5 * qy, 1.11]])
        actual, valid = morphology_amplitude(s, qc, self.g, self.p.padding)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(actual, direct_amplitude(s, qc, self.g), atol=1e-12)

    def test_off_grid_interpolation_converges_for_variable_qz(self):
        s = generate_surface(self.g, self.p)
        rng = np.random.default_rng(456)
        qc = rng.uniform([-8e-4, -2e-4, .73], [8e-4, 2e-4, .88], (30, 3))
        ref = direct_amplitude(s, qc, self.g)
        a2, _ = morphology_amplitude(s, qc, self.g, 2)
        a8, _ = morphology_amplitude(s, qc, self.g, 8)
        self.assertLess(np.linalg.norm(a8 - ref), np.linalg.norm(a2 - ref) / 4)
        self.assertLess(np.max(np.abs(a8 - ref)), .003)

    def test_height_translation_changes_only_phase(self):
        s = generate_surface(self.g, self.p)
        moved = replace(s, height_uc=s.height_uc + 3)
        qc = np.array([[.0007, .0001, .72], [-.0003, .0002, .85]])
        a, _ = morphology_amplitude(s, qc, self.g)
        b, _ = morphology_amplitude(moved, qc, self.g)
        np.testing.assert_allclose(b, a * np.exp(1j * qc[:, 2] * self.g.a * 3), atol=1e-12)

    def test_island_antibragg_cancellation_and_full_layer(self):
        g = replace(self.g, miscut=0.)
        p = replace(self.p, island_coverage=.5, island_radius_nm=150.)
        s = generate_surface(g, p)
        q = np.array([0., 0., np.pi / g.a])
        a, _ = morphology_amplitude(s, q, g)
        theta = s.stats['illuminated_island_fraction']
        self.assertAlmostEqual(float(abs(a)**2), (1 - 2 * theta)**2, places=12)
        full = generate_surface(g, replace(p, island_coverage=1.))
        b, _ = morphology_amplitude(full, q, g)
        self.assertAlmostEqual(float(abs(b)**2), 1., places=12)
        self.assertTrue(np.all(full.height_uc == 1))

    def test_integer_l_removes_same_material_height_contrast(self):
        s = generate_surface(self.g, self.p)
        flat = replace(s, height_uc=np.zeros_like(s.height_uc))
        q = np.array([[.0006, .0001, 2 * np.pi / self.g.a]])
        a, _ = morphology_amplitude(s, q, self.g)
        b, _ = morphology_amplitude(flat, q, self.g)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_ideal_rod_sign_spacing_and_relative_weights(self):
        g = replace(self.g, coherence_x=2.)
        p = replace(self.p, nx=1024, width_enabled=False, jagged_enabled=False, islands_enabled=False)
        s = generate_surface(g, p)
        L = np.array([-1, 0, 1, 2])
        ell = .48
        qx = 2 * np.pi * (ell - L) / g.terrace_width
        qc = np.stack([qx, np.zeros(len(L)), np.full(len(L), 2 * np.pi * ell / g.a)], axis=1)
        a, _ = morphology_amplitude(s, qc, g, 8)
        np.testing.assert_allclose(abs(a)**2, np.sinc(ell - L)**2, atol=.01)

    def test_alias_range_is_marked_missing(self):
        s = generate_surface(self.g, self.p)
        a, valid = morphology_amplitude(s, np.array([[10., 0., .8], [0., 10., .8]]), self.g)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(a).all())

    def test_ensemble_averages_intensity_not_amplitude(self):
        p = replace(self.p, realizations=2)
        result = simulate_detector(self.g, p, compare=False)
        qc = result['q_crystal']
        amplitudes = [morphology_amplitude(generate_surface(self.g, p, j), qc, self.g, p.padding)[0] for j in range(2)]
        expected = np.where(result['physical'], np.mean(np.abs(amplitudes)**2, axis=0), 0.)
        actual = result['cases']['Selected combination']['intensity']
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        self.assertGreater(np.nanmax(np.abs(expected - abs(np.mean(amplitudes, axis=0))**2)), 1e-8)

    def test_invalid_incidence_is_blank(self):
        result = simulate_detector(replace(self.g, alpha=-10.), self.p, compare=False)
        self.assertFalse(result['physical'].any())
        self.assertTrue(np.all(result['cases']['Selected combination']['intensity'] == 0))

    def test_exports_are_pickle_free_and_reproduce_the_surface(self):
        from surface_ui import export_npz
        from dataclasses import asdict
        p = replace(self.p, realizations=1)
        result = simulate_detector(self.g, p, compare=False)
        content = export_npz(result, dict(geometry=asdict(self.g), surface=asdict(p)))
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            for key in data.files:
                self.assertNotEqual(data[key].dtype, object)
            np.testing.assert_array_equal(data['height_uc_first_realization'][0], generate_surface(self.g, p).height_uc)
            self.assertEqual(data['morphology_intensity'].shape, (1, self.g.ny, self.g.nx))

    def test_invalid_parameters_and_browser_memory_guard(self):
        for p in [replace(self.p, width_cv=-.1), replace(self.p, island_coverage=1.1),
                  replace(self.p, edge_rms_nm=float('nan')), replace(self.p, nx=65),
                  replace(self.p, nx=1024, ny=512, padding=8)]:
            with self.assertRaises(ValueError):
                generate_surface(self.g, p)


if __name__ == '__main__':
    unittest.main()
