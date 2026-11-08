"""Detector capture filtering and crystal-coordinate labels."""
import unittest
from dataclasses import replace
import numpy as np
from physics import Geometry, preset, rod_intersections, detector_image, pixel_q
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
            self.assertIn('Crystal (Qx, Qy, Qz)',fig.layout.annotations[0].text)
            titles=[fig.layout.xaxis.title.text,fig.layout.yaxis.title.text]
            # Independently differentiate normalized outgoing rays analytically.
            n,eu,ev=g.basis
            r=g.distance*n+u0*eu+v0*ev
            s=r/np.linalg.norm(r)
            for i,e in enumerate([eu,ev]):
                direction=(e-s*np.dot(s,e))@g.R
                direction/=np.linalg.norm(direction)
                self.assertIn('(mm)',titles[i])
                displayed=np.fromstring(titles[i].split('Q direction: (')[1].rstrip(')'),sep=',')
                np.testing.assert_allclose(displayed,direction,atol=5.1e-5)
            np.testing.assert_allclose(np.asarray(fig.data[0].customdata)[...,3:],qc)
            np.testing.assert_allclose(np.asarray(fig.data[0].customdata)[...,:3],qc*g.a/(2*np.pi))

    def test_peak_and_cursor_readouts_use_fractional_intersection_coordinates(self):
        g=Geometry()
        u,v,I,_,qc=detector_image(g)
        records=rod_intersections(g)
        captured=[r for r in records if r['status']=='Captured']
        fig=detector(g,u,v,I,qc,u[10],v[20],records)
        peaks=next(t for t in fig.data if t.name=='Peaks (H, K, L)')
        self.assertEqual(len(peaks.x),3)
        np.testing.assert_allclose(np.asarray(peaks.customdata)[:,:3],[r['hkl'] for r in captured])
        np.testing.assert_allclose(np.asarray(peaks.customdata)[:,6],[r['L'] for r in captured])
        for text,r in zip(peaks.text,captured):
            displayed=np.fromstring(text.strip('()'),sep=',')
            np.testing.assert_allclose(displayed,r['hkl'],atol=5.1e-5)
            self.assertGreater(abs(displayed[2]-r['L']),.1)
        selected=next(t for t in fig.data if t.name=='Selected pixel')
        # Slider-selected point and image cell must agree, even under the cross.
        np.testing.assert_allclose(selected.customdata[0],fig.data[0].customdata[20,10])
        for trace in fig.data:
            if trace.name=='Y intensity profile':
                continue
            self.assertIn('(H, K, L)',trace.hovertemplate)
            for i in range(6):
                self.assertIn(f'customdata[{i}]',trace.hovertemplate)

    def test_label_toggles_preserve_hover_and_empty_detector(self):
        for name,alpha in [('Oblique steps',None),('Flat surface',None),('Oblique steps',-10.)]:
            g=preset(name)
            if alpha is not None: g=replace(g,alpha=alpha)
            u,v,I,_,qc=detector_image(g)
            fig=detector(g,u,v,I,qc,0.,0.,rod_intersections(g),layer='l',
                         show_q_directions=False,label_peaks=False)
            self.assertNotIn('Q direction',fig.layout.xaxis.title.text)
            self.assertNotIn('Q direction',fig.layout.yaxis.title.text)
            self.assertFalse(any('Qx' in a.text for a in fig.layout.annotations))
            np.testing.assert_allclose(fig.data[0].z,qc[...,2]*g.a/(2*np.pi))
            for trace in fig.data[1:]:
                if trace.name=='Y intensity profile':
                    continue
                self.assertEqual(trace.mode,'markers')
                self.assertFalse(trace.text)
                self.assertIn('(H, K, L)',trace.hovertemplate)

    def test_y_profile_sums_linear_columns_and_shares_image_rows(self):
        g=replace(Geometry(),nx=3,ny=4)
        u,v,_,_,qc=detector_image(g)
        # Unequal dimensions, an empty row and very weak values distinguish a
        # row sum from a column sum, mean, log sum or display-floor clipping.
        intensity=np.array([[1.,2.,4.],[0.,0.,0.],[1e-12,2e-12,3e-12],[.1,.2,.4]])
        expected=[7.,0.,6e-12,.7]
        for layer in ['Intensity','h','k','l']:
            fig=detector(g,u,v,intensity,qc,0.,0.,[],layer=layer)
            profile=next(t for t in fig.data if t.name=='Y intensity profile')
            np.testing.assert_allclose(profile.x,expected,rtol=1e-14,atol=0.)
            np.testing.assert_array_equal(profile.y,fig.data[0].y)
            np.testing.assert_array_equal(profile.customdata,np.arange(g.ny))
            self.assertEqual(profile.xaxis,'x2')
            self.assertEqual(profile.yaxis,fig.data[0].yaxis or 'y')
            self.assertGreater(fig.layout.xaxis2.domain[0],fig.layout.xaxis.domain[1])
            self.assertEqual(fig.layout.yaxis.range,(-g.ny*g.pitch/2,g.ny*g.pitch/2))

    def test_y_profile_empty_detector_stays_zero(self):
        g=replace(Geometry(),nx=16,ny=24,alpha=-10.)
        u,v,intensity,_,qc=detector_image(g)
        fig=detector(g,u,v,intensity,qc,0.,0.,rod_intersections(g))
        profile=next(t for t in fig.data if t.name=='Y intensity profile')
        np.testing.assert_array_equal(profile.x,np.zeros(g.ny))
        self.assertTrue(np.isfinite(profile.x).all())


if __name__=='__main__':
    unittest.main()
