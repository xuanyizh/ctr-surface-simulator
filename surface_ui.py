"""Optional, explicitly run surface-forward simulation inside the explorer."""
import io
import json
import asyncio
from time import perf_counter
from dataclasses import asdict
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from surface_model import SurfaceParameters, simulate_detector_steps
from surface_plots import surface_comparison



def export_npz(result, metadata):
    arrays = dict(metadata_json=np.array(json.dumps(metadata)),
                  u_mm=result['u_mm'], v_mm=result['v_mm'],
                  q_lab_Ainv=result['q_lab'], q_crystal_Ainv=result['q_crystal'],
                  sampling_valid=result['sampling_valid'], physical=result['physical'])
    names = list(result['cases'])
    arrays['case_names'] = np.array(names)
    arrays['morphology_intensity'] = np.stack([result['cases'][n]['intensity'] for n in names])
    arrays['height_uc_first_realization'] = np.stack([result['cases'][n]['surface'].height_uc for n in names])
    arrays['island_mask_first_realization'] = np.stack([result['cases'][n]['surface'].island_mask for n in names])
    for j, name in enumerate(names):
        arrays[f'edges_A_case_{j}'] = result['cases'][name]['surface'].edges
        arrays[f'base_widths_A_case_{j}'] = result['cases'][name]['surface'].base_widths
    surface = result['cases'][names[0]]['surface']
    arrays.update(x_A=surface.x, y_A=surface.y, amplitude_window=surface.illumination)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


async def render_surface_simulation(g, export_button):
    st.subheader('Build a surface and calculate its coherent scattering')
    st.write('Switch terrace-width variation, jagged edges and islands on independently. '
             'The comparison evaluates each enabled feature alone and their selected combination, '
             'using the same geometry, random seed and intensity scale.')
    st.caption('Uses the geometry, lattice, miscut and coherence controls in the sidebar. '
               'The teaching roughness slider and parent-L display range do not enter this solver. '
               'One realization is one coherent Gaussian patch; multiple independent patches are averaged as intensities.')
    with st.form('surface_form'):
        a, b, c = st.columns(3)
        with a:
            width = st.checkbox('Terrace-width variation', value=True, key='surf_width')
            distribution = st.selectbox('Positive width distribution',
                                       ['gamma', 'lognormal', 'truncated_normal'], key='surf_distribution')
            cv = st.slider('Width standard deviation / mean', 0., 1., .15, .01, key='surf_cv')
            st.caption('Mean width = a / tan μ. Widths are sampled, then rescaled to preserve the mean. '
                       'The normal distribution is truncated at zero. Realized statistics are shown below.')
        with b:
            jagged = st.checkbox('Jagged step edges', value=True, key='surf_jagged')
            rms = st.number_input('Edge displacement RMS (nm)', 0., 1000., 20., 1., key='surf_rms')
            correlation = st.number_input('Along-edge correlation length (nm)', 0., 10000., 50., 5., key='surf_correlation')
            neighbor = st.slider('Shared meandering between steps', 0., 1., 0., .05, key='surf_neighbor')
            st.caption('0: independent edges; 1: all edges meander together. '
                       'Crossing steps are prevented; any RMS reduction is reported.')
        with c:
            islands = st.checkbox('Islands', value=True, key='surf_islands')
            coverage = st.slider('Expected island coverage', 0., 1., .15, .01, key='surf_coverage')
            radius = st.number_input('Island radius (nm)', 1., 2000., 60., 5., key='surf_radius')
            height = st.number_input('Island height (unit cells)', 1, 20, 1, key='surf_height')
            st.caption('Random circular patches of the same material, above the local terrace. '
                       'Overlaps merge at the selected height; they do not stack. Coverage is an ensemble expectation.')
        a, b, c, d = st.columns(4)
        seed = a.number_input('Random seed', 0, 2**31 - 1, 2026, key='surf_seed')
        quality = b.selectbox('Surface grid', ['256 × 128', '512 × 256', '1024 × 512'], index=1, key='surf_grid')
        padding = c.selectbox('FFT padding', [2, 4, 8], index=1, key='surf_padding')
        realizations = d.selectbox('Independent patches', [1, 2, 4, 8, 16], key='surf_realizations')
        compare = st.checkbox('Compare ideal, individual features and combination', value=True, key='surf_compare')
        st.caption('Progress and results appear directly below this button. Keep the tab open while it runs. '
                   'Start with one patch; larger grids and more patches can take several minutes. '
                   'Refine grid and padding to check convergence; more patches reduce speckle.')
        run = st.form_submit_button('Run surface simulation', type='primary')
    nx, ny = [int(s.strip()) for s in quality.split('×')]
    p = SurfaceParameters(width_enabled=width, width_distribution=distribution, width_cv=cv,
                          jagged_enabled=jagged, edge_rms_nm=rms, edge_correlation_nm=correlation,
                          edge_neighbor_correlation=neighbor, islands_enabled=islands,
                          island_coverage=coverage, island_radius_nm=radius, island_height_uc=height,
                          seed=seed, nx=nx, ny=ny, padding=padding, realizations=realizations)
    signature = dict(geometry=asdict(g), surface=asdict(p), compare=compare)
    if run:
        started = perf_counter()
        notice = st.empty()
        notice.info('Running surface simulation… Progress updates below; results will appear here.')
        bar = st.progress(0., text='Starting the calculation…')
        # Stlite shares one Python event loop. Yield before and between FFTs
        # so queued UI updates actually reach the page during the calculation.
        await asyncio.sleep(.1)
        steps = simulate_detector_steps(g, p, compare)
        try:
            while True:
                try:
                    update = next(steps)
                except StopIteration as finished:
                    result = finished.value
                    break
                elapsed = perf_counter() - started
                bar.progress(update['fraction'], text=f"{update['fraction']:.0%} · {update['text']} · {elapsed:.0f} s elapsed")
                await asyncio.sleep(.01)
            st.session_state.surface_result = result
            st.session_state.surface_signature = signature
            st.session_state.surface_elapsed = perf_counter() - started
            notice.empty()
        except (ValueError, MemoryError) as exc:
            notice.error(f'Cannot run these settings: {exc}. Try one patch and reduce grid size or FFT padding.')
            return
        except Exception as exc:
            notice.error('The simulation failed. Try one patch and a smaller grid; the error details are below.')
            with st.expander('Error details', expanded=True):
                st.exception(exc)
            return
        finally:
            steps.close()
            bar.empty()
    if 'surface_result' not in st.session_state:
        st.info('Choose the features above, then click Run surface simulation.')
        return
    saved = st.session_state.surface_signature
    if saved != signature:
        st.warning('Settings have changed since this result. Click Run surface simulation to update the maps and exports.')
    result = st.session_state.surface_result
    cases = result['cases']
    elapsed = st.session_state.get('surface_elapsed', 0.)
    st.success(f'Simulation complete — {len(cases)} case(s) calculated in {elapsed:.1f} s. Results are below.')
    # Let the completion notice paint before preparing the figures and exports.
    if run:
        await asyncio.sleep(.1)
    for message in sorted({w for case in cases.values() for stats in case['statistics'] for w in stats['warnings']}):
        st.warning(message)
    if not np.any(result['physical']):
        st.warning('No outgoing detector rays satisfy top-surface reflection in this geometry. The image is blank.')
    fraction = np.mean(result['sampling_valid'])
    if fraction < 1:
        st.warning(f'{1 - fraction:.1%} of detector pixels lie outside this surface grid’s Fourier range. '
                   'They are blank / NaN, not zero intensity. Use a finer grid, a smaller patch (coherence), '
                   'or a narrower angular field of view.')
    st.subheader('Surface and detector comparison')
    view_a, view_b, view_c = st.columns(3)
    surface_view = view_a.selectbox('Real-space extent',
                                    ['Central crop (preview)', 'Full simulated patch'], key='surf_view_real')
    detector_view = view_b.selectbox('Detector extent',
                                     ['Rod close-up (preview)', 'Full detector'], key='surf_view_detector')
    intensity_view = view_c.selectbox('Intensity colour range',
                                      ['10⁻⁶ to 1 (preview)', '10⁻⁸ to 1'], key='surf_view_intensity')
    log_floor = -6 if intensity_view.startswith('10⁻⁶') else -8
    comparison = surface_comparison(result, saved['geometry'],
                                    central_surface=surface_view.startswith('Central'),
                                    detector_closeup=detector_view.startswith('Rod'), log_floor=log_floor)
    st.caption('Top: real-space height for the first coherent patch, with x across steps and y along steps. '
               'Bottom: coherent detector intensity. Each column is the same case in both spaces. '
               'Every height map shares one scale; every detector map shares one logarithmic intensity scale.')
    st.plotly_chart(comparison, width='stretch', theme=None, key='surface_comparison_chart',
                    config={'toImageButtonOptions': {'filename': 'surface_detector_comparison',
                                                     'width': 1800, 'height': 850, 'scale': 2}})
    crop_note = ('Central surface crop: |x| ≤ 700 nm, |y| ≤ 500 nm. '
                 if surface_view.startswith('Central') else 'Full simulated surface patch. ')
    detector_note = ('Detector close-up: |u| ≤ 1.4 mm, |v| ≤ 8.5 mm, limited to the detector bounds. '
                     if detector_view.startswith('Rod') else 'Full detector field of view. ')
    st.caption(crop_note + detector_note + 'Axes have unequal scales in this compact view. '
               'Use the chart fullscreen button for a larger view. Changing these display controls reuses the calculated result.')
    st.caption('Grey detector pixels are outside the resolved Fourier range; dark pixels are low intensity. '
               'Values below the colour range use its darkest colour. Cyan circles are ideal reference rod centres. '
               'M is a normalized same-material morphology factor, with the common material CTR factor omitted. '
               'Individual-case intensities must not be added together.')
    with st.expander('Surface cross-section at y = 0'):
        fig = go.Figure()
        for name, case in cases.items():
            s = case['surface']
            fig.add_trace(go.Scatter(x=s.x / 10, y=s.height_uc[len(s.y) // 2], mode='lines',
                                     name=name, line_shape='hv'))
        fig.update_layout(title='Surface cross-section at y = 0', height=390, template='plotly_white',
                          xaxis_title='Crystal x (nm)', yaxis_title='Height (uc)',
                          legend=dict(orientation='h'), margin=dict(l=35, r=15, t=50, b=45))
        st.plotly_chart(fig, width='stretch', theme=None)
    labels = dict(dx_nm='Cell Δx (nm)', dy_nm='Cell Δy (nm)', mean_width_nm='Mean terrace (nm)',
                  realized_width_cv='Width CV', realized_edge_rms_nm='Edge RMS (nm)',
                  edge_scale='Edge scale', island_area_fraction='Island area fraction',
                  illuminated_island_fraction='Illuminated island fraction', height_levels='Height levels')
    st.dataframe([dict(Case=name, **{labels[k]: v for k, v in case['surface'].stats.items() if k != 'warnings'})
                  for name, case in cases.items()], hide_index=True, width='stretch')
    st.caption('Statistics and height maps above describe the first realization. The export records statistics for every patch. '
               'Width CV measures the mean straight-edge spacings before jaggedness; edge RMS is measured after preventing crossings.')
    profile = go.Figure()
    valid_rows = np.any(result['sampling_valid'], axis=1)
    for name, case in cases.items():
        values = np.nansum(case['intensity'], axis=1)
        values[~valid_rows] = np.nan
        profile.add_trace(go.Scatter(x=result['v_mm'], y=values, name=name))
    profile.update_layout(title='Row sums over resolved detector columns', xaxis_title='v (mm)',
                          yaxis_title='Sum of morphology intensity', yaxis_type='log', height=360,
                          legend=dict(orientation='h'))
    st.plotly_chart(profile, width='stretch')
    st.caption('Blank Fourier-range pixels are excluded from these row sums; all cases use exactly the same mask. '
               'Cyan markers show ideal reference rods, not predictions of disordered-surface intensity maxima.')
    metadata = dict(model='coherent_same_material_height_columns_v1', **saved,
                    statistics={name: case['statistics'] for name, case in cases.items()})
    export_button('Download surface settings JSON', json.dumps(metadata, indent=2),
                  'surface_settings.json', 'application/json')
    export_button('Download surfaces + detector simulations (NPZ)', export_npz(result, metadata),
                  'surface_forward.npz', 'application/octet-stream')
    with st.expander('Forward model and interpretation'):
        st.latex(r'M(\mathbf q)=\left|\frac{\sum_j W_j e^{i[(q_x-G_x)x_j+(q_y-G_y)y_j+q_z a n_j]}}{\sum_j W_j}\right|^2')
        st.write('The implementation integrates constant-height lateral tiles, with the corresponding sinc factors. '
                 'It sums complex amplitudes before squaring, preserving interference between terraces and islands. '
                 'It evaluates each pixel’s qz, not one constant-qz slice. Independent coherent patches are averaged as intensities.')
        st.write('The paper separates Ip, Iw and Is under its Gaussian step-position correlation assumptions. '
                 'Here independently sampled positive widths accumulate into a different step-position process; '
                 'the jagged edges have a finite correlation length, and islands are explicit. '
                 'The comparison maps are controlled numerical experiments, not the paper’s additive Ip/Iw/Is components.')
        st.write('Same lattice and material for substrate and islands; no film structure factors, absorption, '
                 'refraction, detector response or growth dynamics. Coherence is a Gaussian patch approximation. '
                 'Detector pixels sample their centres; narrow peaks require checking detector pitch as well as the surface grid. '
                 'For quantitative LNO/STO fitting, film/substrate amplitudes and instrument calibration must be added.')
