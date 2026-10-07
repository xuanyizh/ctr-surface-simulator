"""Smoke test widgets, example loading, target alignment and map selection."""
from streamlit.testing.v1 import AppTest
from physics import preset

at=AppTest.from_file('app.py',default_timeout=30).run()
assert not at.exception
for name in ['Across steps','Along steps','Flat surface','Paper-like L=1.6','Oblique steps']:
    at.selectbox(key='preset_name').select(name).run()
    [b for b in at.button if b.label=='Load example'][0].click().run()
    assert not at.exception, at.exception
    g=preset(name)
    assert at.session_state.phi==g.phi
    assert at.session_state.miscut==g.miscut
    assert at.session_state.energy==g.energy
    print('PASS example:',name)
for layer in ['h','k','l','Intensity']:
    [s for s in at.selectbox if s.label=='Detector colour map'][0].select(layer).run()
    assert not at.exception
print('PASS all detector map modes')
at.number_input(key='target_l').set_value(1.).run()
[b for b in at.button if b.label=='Align sample and detector centre to target'][0].click().run()
assert not at.exception
assert abs(at.session_state.elevation-2*at.session_state.alpha)<1e-10
assert any('captured geometrically' in s.value for s in at.success)
print('PASS target alignment and geometric access')
at.number_input(key='alpha').set_value(-10.).run()
assert not at.exception
assert any('not incident' in s.value for s in at.warning)
print('PASS invalid incidence message')

# Nonideal forward controls run only on explicit form submission.
[b for b in at.button if b.label=='Load example'][0].click().run()
at.selectbox(key='surf_grid').select('256 × 128')
[b for b in at.button if b.label=='Run surface simulation'][0].click().run(timeout=90)
assert not at.exception, at.exception
assert len(at.session_state.surface_result['cases']) == 5
assert any('Simulation complete' in s.value for s in at.success)
assert len(at.get('download_button')) == 4
print('PASS surface comparison maps and exports')
at.number_input(key='miscut').set_value(.15).run()
assert not at.exception
assert any('Settings have changed' in w.value for w in at.warning)
print('PASS stale forward-result warning')
at.checkbox(key='surf_width').uncheck()
at.checkbox(key='surf_jagged').uncheck()
at.checkbox(key='surf_islands').uncheck()
[b for b in at.button if b.label=='Run surface simulation'][0].click().run(timeout=90)
assert not at.exception
import numpy as np
cases=at.session_state.surface_result['cases']
assert list(cases)==['Ideal steps','Selected combination']
np.testing.assert_array_equal(cases['Ideal steps']['intensity'],cases['Selected combination']['intensity'])
print('PASS independent feature switches and ideal fallback')

# A rejected run must show an error and must not report the old result as new.
previous_signature = at.session_state.surface_signature
at.selectbox(key='surf_grid').select('1024 × 512')
at.selectbox(key='surf_padding').select(8)
[b for b in at.button if b.label=='Run surface simulation'][0].click().run(timeout=30)
assert not at.exception, at.exception
assert any('FFT too large' in e.value for e in at.error)
assert not any('Simulation complete' in s.value for s in at.success)
assert at.session_state.surface_signature == previous_signature
print('PASS invalid-run feedback and preservation of previous result')
