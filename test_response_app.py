"""Exercise new response choices, convergence, rerun signatures and exports."""
from streamlit.testing.v1 import AppTest

at=AppTest.from_file('app.py',default_timeout=60).run()
assert not at.exception,at.exception
assert at.selectbox(key='response_samples').value==2
at.number_input(key='nx').set_value(24)
at.number_input(key='ny').set_value(20)
at.checkbox(key='response_column').check()
at.checkbox(key='response_polarization').check().run()
assert not at.exception,at.exception
at.number_input(key='response_bulk_real').set_value(2.)
at.number_input(key='response_surface_imag').set_value(.2)
at.slider(key='response_horizontal').set_value(.5)
at.button(key='linked_convergence').click().run()
assert not at.exception,at.exception
assert 'linked_convergence_result' in at.session_state
at.selectbox(key='surf_grid').select('256 × 128')
at.checkbox(key='surf_compare').uncheck()
at.checkbox(key='surf_check_pixels').check()
[b for b in at.button if b.label=='Run surface simulation'][0].click().run(timeout=120)
assert not at.exception,at.exception
result=at.session_state.surface_result
assert 'convergence' in result
assert len(result['cases'])==1
assert len(at.get('download_button'))==4
assert at.session_state.surface_signature['response']['column_factor']
print('PASS new response controls, linked and surface convergence, exports')
signature=at.session_state.surface_signature
at.selectbox(key='response_view').select('Intrinsic intensity').run()
assert not at.exception,at.exception
assert at.session_state.surface_signature==signature
assert not any('Settings have changed' in w.value for w in at.warning)
at.selectbox(key='response_samples').select(4).run()
assert not at.exception,at.exception
assert any('Settings have changed' in w.value for w in at.warning)
print('PASS display-only switching and response-change stale-result warning')
