"""Detector capture filtering and crystal-coordinate labels."""
import unittest
from dataclasses import replace
import numpy as np
from physics import preset, rod_intersections, detector_image, pixel_q
from plots import reciprocal, detector, project_2d


class DetectorPlotTests(unittest.TestCase):
    def test_filter_keeps_only_captured_rods_and_preserves_colours(self):
        g=replace(preset('Oblique steps'),nx=128,ny=128)
        all_rods=reciprocal(g,0.,0.,zoom=True)
        filtered=reciprocal(g,0.,0.,zoom=True,captured_only=True)
        expected={f"Sub-rod L={r['L']}" for r in rod_intersections(g) if r['status']=='Captured'}
        colours={t.name:t.line.color for t in all_rods.data if t.name.startswith('Sub-rod')}
        actual={t.name:t.line.color for t in filtered.data if t.name.startswith('Sub-rod')}
        self.assertGreater(len(expected),0)
        self.assertLess(len(expected),len(colours))
        self.assertEqual(actual,{name:colours[name] for name in expected})
        projected=project_2d(filtered)
        self.assertEqual({t.name for t in projected.data if t.name.startswith('Sub-rod')},expected)

    def test_empty_capture_and_flat_surface(self):
        for name in ['Oblique steps','Flat surface']:
            g=replace(preset(name),alpha=-10.)
            fig=reciprocal(g,0.,0.,zoom=True,captured_only=True)
            self.assertFalse(any(t.name.startswith(('Sub-rod','Flat CTR','Captured')) for t in fig.data))
            self.assertTrue(any(t.name=='Detector acceptance on Ewald sphere' for t in fig.data))
        fig=reciprocal(preset('Flat surface'),0.,0.,zoom=True,captured_only=True)
        self.assertEqual(sum(t.name=='Flat CTR' for t in fig.data),1)

    def test_q_labels_follow_local_directions_and_roll(self):
        for roll,u0,v0 in [(0.,0.,0.),(90.,0.,0.),(37.,6.,-8.)]:
            g=replace(preset('Oblique steps'),nx=16,ny=16,roll=roll)
            u,v,I,_,qc=detector_image(g)
            fig=detector(g,u,v,I,qc,u0,v0,rod_intersections(g))
            texts=[a.text for a in fig.layout.annotations]
            self.assertIn('Crystal (Qx, Qy, Qz)',texts[0])
            # Independently differentiate normalized outgoing rays analytically.
            n,eu,ev=g.basis
            r=g.distance*n+u0*eu+v0*ev
            s=r/np.linalg.norm(r)
            for i,e in enumerate([eu,ev]):
                direction=(e-s*np.dot(s,e))@g.R
                direction/=np.linalg.norm(direction)
                displayed=np.fromstring(texts[i+1].split('(')[1].rstrip(')'),sep=',')
                np.testing.assert_allclose(displayed,direction,atol=5.1e-5)
            np.testing.assert_allclose(np.asarray(fig.data[0].customdata)[...,3:],qc)
            np.testing.assert_allclose(np.asarray(fig.data[0].customdata)[...,:3],qc*g.a/(2*np.pi))


if __name__=='__main__':
    unittest.main()
