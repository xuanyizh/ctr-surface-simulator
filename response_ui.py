"""Shared detector response controls and numerical diagnostics."""
import streamlit as st
from detector_response import DetectorResponse, log_upper_limit


def response_controls():
    with st.expander('Intensity & detector response', expanded=True):
        mode = st.selectbox('Displayed intensity', ['Detector signal (relative)', 'Intrinsic intensity'],
                            key='response_view')
        samples = st.selectbox('Samples per pixel axis', [1, 2, 4, 8], index=1, key='response_samples',
                               help='2 means 2 × 2 subpixels. Intensities are averaged, not amplitudes. '
                                    '1 samples only the center. Refine until the convergence check stabilizes.')
        polarization = st.checkbox('Apply polarization', value=False, key='response_polarization')
        fraction = st.slider('Horizontal polarization fraction', 0., 1., 1., .01,
                             disabled=not polarization, key='response_horizontal',
                             help='Lab horizontal E is y; incident beam is +x. 1 = horizontal, '
                                  '0 = vertical, 0.5 = unpolarized. No outgoing analyzer is modeled.')
        column = st.checkbox('Include ideal column CTR factor', value=False, key='response_column')
        st.caption('Detector signal integrates intensity × polarization × pixel solid angle. '
                   'Intrinsic intensity is the unweighted pixel average. Both modes use the exact Ewald mapping.')
        if column:
            length = st.number_input('Amplitude attenuation length (unit cells)', 1., 100000., 1000.,
                                     key='response_attenuation',
                                     help='Phenomenological damping prevents the semi-infinite Bragg divergence. '
                                          'This is not a calculated absorption length or film thickness.')
            a, b = st.columns(2)
            br = a.number_input('Bulk F real', value=1., key='response_bulk_real')
            bi = b.number_input('Bulk F imaginary', value=0., key='response_bulk_imag')
            sr = a.number_input('Surface F real', value=1., key='response_surface_real')
            si = b.number_input('Surface F imaginary', value=0., key='response_surface_imag')
            st.caption('Constant unit-cell amplitudes over this image; defaults Fbulk = Fsurface = 1. '
                       'This ideal column does not supply LNO/STO atomic structure factors or film interference.')
        else:
            length, br, bi, sr, si = 1000., 1., 0., 1., 0.
    response = DetectorResponse(samples, fraction if polarization else None, column, length, br, bi, sr, si)
    return response, 'detector_signal' if mode.startswith('Detector') else 'intrinsic_intensity'


def response_caption(response, key):
    label = 'Relative detector signal' if key == 'detector_signal' else 'Pixel-averaged intrinsic intensity'
    model = 'including the specified ideal column factor' if response.column_factor else 'morphology only'
    text = f'{label}; {response.subpixels} × {response.subpixels} subpixels; {model}.'
    if key == 'detector_signal':
        text += (' Fixed scale: ∫ I P dΩ / 10⁻⁹ sr. Pixel area and distance affect collected signal. '
                 'No flux or detector efficiency calibration: these are not absolute photon counts.')
    return text


def show_convergence(metrics, coarse_order, fine_order):
    for name, metric in metrics.items():
        name = name.replace('intrinsic_intensity', 'Intrinsic intensity').replace('detector_signal', 'Detector signal')
        error, flux = metric['l1_relative'], metric['flux_relative']
        if error is None:
            st.warning(f'{name}: convergence cannot be assessed (no resolved signal).')
            continue
        text = (f'{name}: {coarse_order} × {coarse_order} → {fine_order} × {fine_order}: '
                f'image L1 change {error:.2%}; total-signal change {flux:.2%}; '
                f'common resolved pixels {metric["valid_fraction"]:.1%}.')
        (st.success if error < .01 and metric['valid_fraction'] == 1 else st.warning)(text)
    st.caption('A small change is a convergence diagnostic, not a guarantee: extremely narrow peaks can be '
               'missed at both orders. Repeat at higher sampling, particularly near Bragg peaks. '
               'This check does not test surface-grid, FFT-padding or model accuracy.')
