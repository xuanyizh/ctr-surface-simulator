"""Physical conservation, analytic integrals and independent surface checks."""
import math
import unittest
from dataclasses import replace
import numpy as np
from physics import Geometry, intensity_at, pixel_q, detector_calculation
from detector_response import (DetectorResponse, SIGNAL_REFERENCE_SR, integrate_pixels,
                               solid_angle_density, polarization_factor, column_intensity,
                               convergence_metrics)


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.g = replace(Geometry(), nx=8, ny=6, alpha=20., elevation=40.)

    def test_uniform_scattering_matches_exact_rectangle_solid_angle(self):
        g = replace(self.g, nx=40, ny=30, distance=20., pitch=1.)
        r = integrate_pixels(g, DetectorResponse(subpixels=16), lambda q: np.ones(q.shape[:-1]))
        a,b = g.nx*g.pitch/2, g.ny*g.pitch/2
        exact = 4*np.arctan(a*b/(g.distance*np.sqrt(g.distance**2+a*a+b*b)))
        # Use geometric solid angle: some extreme pixels may be below the surface.
        error=abs(float(r['solid_angle_sr'].sum())/exact-1.)
        self.assertLess(error,1e-6)
        coarse=integrate_pixels(g,DetectorResponse(subpixels=8),lambda q:np.ones(q.shape[:-1]))
        self.assertLess(error,abs(float(coarse['solid_angle_sr'].sum())/exact-1.)/3.9)

    def test_default_flat_detector_variation(self):
        g=Geometry()
        u=(g.nx-1)*g.pitch/2; v=(g.ny-1)*g.pitch/2
        ratio=solid_angle_density(g,u,v)/solid_angle_density(g,0.,0.)
        self.assertAlmostEqual(100*(1-ratio),.0352974089488,places=10)

    def test_distance_and_pixel_area_scale_collected_signal(self):
        g=replace(self.g,nx=1,ny=1)
        r=DetectorResponse(subpixels=1)
        uniform=lambda q:np.ones(q.shape[:-1])
        base=integrate_pixels(g,r,uniform)['detector_signal'][0,0]
        self.assertAlmostEqual(integrate_pixels(replace(g,distance=2*g.distance),r,uniform)['detector_signal'][0,0]/base,.25)
        self.assertAlmostEqual(integrate_pixels(replace(g,pitch=2*g.pitch),r,uniform)['detector_signal'][0,0]/base,4.)

    def test_pixel_subdivision_conserves_signal_and_average(self):
        g=replace(self.g,pitch=.4)
        r=DetectorResponse(subpixels=4,horizontal_fraction=.73,column_factor=True,attenuation_cells=30.)
        def model(q):
            return np.exp(-((q[...,0]-.001)/.003)**2)*(.2+np.cos(q[...,2]*5)**2)
        coarse=integrate_pixels(g,r,model)
        fine=integrate_pixels(replace(g,nx=g.nx*2,ny=g.ny*2,pitch=g.pitch/2),replace(r,subpixels=2),model)
        for key, divisor in [('detector_signal',1),('solid_angle_sr',1),('intrinsic_intensity',4)]:
            binned=fine[key].reshape(g.ny,2,g.nx,2).sum(axis=(1,3))/divisor
            np.testing.assert_allclose(coarse[key],binned,rtol=2e-12,atol=1e-18)

    def test_narrow_peak_integral_converges_to_erf(self):
        g=replace(self.g,nx=1,ny=1,pitch=1.,distance=100.)
        n,eu,ev=g.basis
        sigma=.12
        def model(qc):
            outgoing=qc@g.R.T+g.ki
            u=g.distance*(outgoing@eu)/(outgoing@n)
            return np.exp(-u*u/(2*sigma*sigma))
        exact=math.sqrt(2*math.pi)*sigma*math.erf(.5/(math.sqrt(2)*sigma))
        a=integrate_pixels(g,DetectorResponse(subpixels=1),model)['intrinsic_intensity'][0,0]
        b=integrate_pixels(g,DetectorResponse(subpixels=16),model)['intrinsic_intensity'][0,0]
        self.assertGreater(abs(a-exact),.5)
        self.assertLess(abs(b-exact),1e-5)

    def test_sample_model_is_independent_of_pixel_pitch(self):
        q=pixel_q(self.g,np.array([0.,.1]),np.array([0.,.2]))[1]
        np.testing.assert_array_equal(intensity_at(self.g,q),intensity_at(replace(self.g,pitch=.8),q))

    def test_polarization_matches_thomson_unpolarized(self):
        out=pixel_q(self.g,np.array([0.,100.,-50.]),np.array([2.,-80.,70.]))[2]
        np.testing.assert_allclose(polarization_factor(out,DetectorResponse(horizontal_fraction=.5)),.5*(1+out[...,0]**2))
        self.assertEqual(float(polarization_factor(np.array([0.,1.,0.]),DetectorResponse(horizontal_fraction=1.))),0.)
        np.testing.assert_array_equal(polarization_factor(out,DetectorResponse()),np.ones(3))

    def test_column_matches_direct_complex_layer_sum(self):
        r=DetectorResponse(column_factor=True,attenuation_cells=7.,bulk_real=2.,bulk_imag=.3,surface_real=.7,surface_imag=-.2)
        l=np.array([0.,.5,1.,1.123])
        qc=np.stack([l*0,l*0,l*2*np.pi/self.g.a],axis=-1)
        z=np.exp(-1/r.attenuation_cells-2j*np.pi*l)
        direct=complex(r.surface_real,r.surface_imag)+complex(r.bulk_real,r.bulk_imag)*np.sum(z[:,None]**np.arange(1,1001),axis=1)
        np.testing.assert_allclose(column_intensity(self.g,qc,r),abs(direct)**2,rtol=1e-12)

    def test_missing_subpixel_is_not_silently_zero_or_renormalized(self):
        g=replace(self.g,nx=1,ny=1)
        center=pixel_q(g,0.,0.)[1][1]
        r=integrate_pixels(g,DetectorResponse(),lambda q:np.where(q[...,1]>center,np.nan,1.))
        self.assertFalse(r['sampling_valid'][0,0])
        self.assertTrue(np.isnan(r['detector_signal'][0,0]))
        below=integrate_pixels(replace(g,alpha=-20.),DetectorResponse(),lambda q:np.full(q.shape[:-1],np.nan))
        self.assertEqual(below['detector_signal'][0,0],0.)

    def test_convergence_does_not_claim_success_for_zero_or_missing_signal(self):
        for array in (np.zeros((2,2)),np.full((2,2),np.nan)):
            self.assertIsNone(convergence_metrics(array,array)['l1_relative'])

    def test_surface_subpixels_match_independent_direct_scattering(self):
        from surface_model import SurfaceParameters, generate_surface, simulate_detector
        from test_surface_model import direct_amplitude
        g=replace(Geometry(),nx=4,ny=4)
        p=SurfaceParameters(nx=64,ny=32,padding=8,realizations=2)
        response=DetectorResponse(subpixels=2,horizontal_fraction=.8,column_factor=True,attenuation_cells=25.)
        surfaces=[generate_surface(g,p,j) for j in range(2)]
        expected=integrate_pixels(g,response,lambda q:np.mean([abs(direct_amplitude(s,q,g))**2 for s in surfaces],axis=0))
        actual=simulate_detector(g,p,compare=False,response=response)['cases']['Selected combination']
        for key in ('morphology_intensity','intrinsic_intensity','detector_signal'):
            np.testing.assert_allclose(actual[key],expected[key],rtol=.02,atol=2e-4)

    def test_instrument_factors_do_not_move_hkl_coordinates(self):
        a=detector_calculation(self.g,DetectorResponse(subpixels=1))
        b=detector_calculation(self.g,DetectorResponse(subpixels=4,column_factor=True,horizontal_fraction=.5))
        np.testing.assert_array_equal(a['q_crystal'],b['q_crystal'])
        np.testing.assert_array_equal(a['q_lab'],b['q_lab'])


if __name__=='__main__':
    unittest.main()
