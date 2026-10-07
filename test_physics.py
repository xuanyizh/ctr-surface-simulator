"""Independent geometric invariants and limiting-case checks. python -m unittest -v"""
import unittest
from dataclasses import replace
import numpy as np
from physics import *

class GeometryTests(unittest.TestCase):
    def test_orthonormal_frames(self):
        g=replace(preset('Oblique steps'),azimuth=17.,roll=29.)
        np.testing.assert_allclose(g.R.T@g.R,np.eye(3),atol=1e-14)
        B=np.array(g.basis)
        np.testing.assert_allclose(B@B.T,np.eye(3),atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(B),1.)
    def test_specular_bragg_law(self):
        for E,L in [(10.16,.5),(15.5,1.6),(10.16,2.)]:
            theta=np.rad2deg(np.arcsin(HC/E*L/(2*3.905)))
            g=Geometry(energy=E,alpha=theta,elevation=2*theta,phi=0.,miscut=0.)
            _,qc,_=pixel_q(g,0.,0.)
            np.testing.assert_allclose(qc*g.a/(2*np.pi),[0,0,L],atol=1e-12)
    def test_all_pixels_elastic(self):
        g=replace(preset('Oblique steps'),azimuth=13.,roll=57.)
        U,V=np.meshgrid(np.linspace(-10,10,13),np.linspace(-10,10,11))
        q,qc,out=pixel_q(g,U,V)
        np.testing.assert_allclose(np.linalg.norm(q+g.ki,axis=-1),g.wavevector,atol=1e-12)
        np.testing.assert_allclose(qc@g.R.T,q,atol=1e-12)
    def test_projection_round_trip(self):
        g=replace(preset('Oblique steps'),azimuth=-6.,roll=75.)
        for u,v in [(-5.,4.),(0.,0.),(9.,-8.)]:
            q,_,out=pixel_q(g,u,v)
            up,vp,front=project_direction(g,out)
            self.assertTrue(front)
            np.testing.assert_allclose([up,vp],[u,v],atol=1e-11)
    def test_analytic_rod_intersections(self):
        for name in ['Oblique steps','Across steps','Along steps','Flat surface','Paper-like L=1.6']:
            g=preset(name)
            rs=rod_intersections(g)
            self.assertGreater(sum(r['status']=='Captured' for r in rs),0)
            for r in rs:
                if r['q'] is None: continue
                q=r['q']; qc=q@g.R; hkl=qc*g.a/(2*np.pi)
                self.assertAlmostEqual(np.linalg.norm(q+g.ki),g.wavevector,places=11)
                self.assertAlmostEqual(hkl[0]-g.h,np.tan(np.deg2rad(g.miscut))*(hkl[2]-r['L']),places=11)
                self.assertAlmostEqual(hkl[1],g.k,places=11)
                if r['status']=='Captured':
                    q2,_,_=pixel_q(g,r['u'],r['v'])
                    np.testing.assert_allclose(q2,q,atol=1e-11)
    def test_flat_limit_deduplicated(self):
        g=preset('Flat surface')
        cap=[r for r in rod_intersections(g) if r['status']=='Captured']
        self.assertEqual(len(cap),1)
        np.testing.assert_allclose([cap[0]['u'],cap[0]['v']],[0,0],atol=1e-10)
    def test_step_spacing(self):
        g=preset('Oblique steps'); l=.5
        q1=2*np.pi/g.a*np.tan(np.deg2rad(g.miscut))*(l-0)
        q2=2*np.pi/g.a*np.tan(np.deg2rad(g.miscut))*(l-1)
        self.assertAlmostEqual(q1-q2,2*np.pi/g.terrace_width,places=12)
    def test_roll_rotates_image_coordinates(self):
        g=preset('Oblique steps')
        out=pixel_q(g,3.,7.)[2]
        u,v,_=project_direction(replace(g,roll=90.),out)
        np.testing.assert_allclose([u,v],[7,-3],atol=1e-12)
    def test_distance_scales_spot_offsets(self):
        g=preset('Oblique steps'); g2=replace(g,distance=2*g.distance)
        r1=[r for r in rod_intersections(g) if r['status']=='Captured']
        r2=rod_intersections(g2)
        for r in r1:
            match=min([s for s in r2 if s['L']==r['L']],key=lambda s:np.linalg.norm(s['q']-r['q']))
            np.testing.assert_allclose([match['u'],match['v']],2*np.array([r['u'],r['v']]),atol=1e-10)
    def test_detector_intensity_and_jacobian(self):
        g=replace(preset('Oblique steps'),nx=128,ny=128)
        u,v,I,q,qc=detector_image(g)
        self.assertTrue(np.isfinite(I).all());self.assertTrue((I>=0).all());self.assertGreater(I.max(),.01)
        J=pixel_jacobian(g,0,0)
        self.assertEqual(J.shape,(3,2));self.assertEqual(np.linalg.matrix_rank(J),2)
        self.assertAlmostEqual(float((q[64,64]+g.ki)@(g.R@J[:,0])),0,places=5)
    def test_invalid_incidence_blanks_image(self):
        g=replace(preset('Oblique steps'),alpha=-10.,nx=128,ny=128)
        self.assertFalse(any(r['status']=='Captured' for r in rod_intersections(g)))
        self.assertEqual(detector_image(g)[2].max(),0)

if __name__=='__main__': unittest.main()
