"""Linked real-space / detector comparison with shared, preview-matched scales."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from physics import Geometry, rod_intersections
from plot_colors import MAGMA, VIRIDIS


def surface_comparison(result, geometry, central_surface=True,
                       detector_closeup=True, log_floor=-6):
    cases = result['cases']
    names = list(cases)
    fig = make_subplots(rows=2, cols=len(names), subplot_titles=names + [''] * len(names),
                        horizontal_spacing=.035, vertical_spacing=.16)
    crops = []
    for case in cases.values():
        s = case['surface']
        x, y = s.x / 10, s.y / 10
        use_x = np.abs(x) <= 700 if central_surface else np.ones(x.shape, dtype=bool)
        use_y = np.abs(y) <= 500 if central_surface else np.ones(y.shape, dtype=bool)
        crops.append((x[use_x], y[use_y], s.height_uc[np.ix_(use_y, use_x)]))
    height_limit = max(1, max(int(np.max(np.abs(z))) for _, _, z in crops))
    hkl = result['q_crystal'] * geometry['a'] / (2 * np.pi)
    peaks = [r for r in rod_intersections(Geometry(**geometry)) if r['status'] == 'Captured']
    u, v = result['u_mm'], result['v_mm']
    u_edge = float(np.max(np.abs(u)) + geometry['pitch'] / 2)
    v_edge = float(np.max(np.abs(v)) + geometry['pitch'] / 2)
    u_limit = min(1.4, u_edge) if detector_closeup else u_edge
    v_limit = min(8.5, v_edge) if detector_closeup else v_edge
    for column, ((name, case), (x, y, heights)) in enumerate(zip(cases.items(), crops), 1):
        fig.add_trace(go.Heatmap(
            x=x, y=y, z=heights, coloraxis='coloraxis', zsmooth=False,
            name=name, hovertemplate='x=%{x:.1f} nm<br>y=%{y:.1f} nm<br>Height=%{z:d} uc<extra>%{fullData.name}</extra>'),
            row=1, col=column)
        intensity = case['intensity']
        log_intensity = np.log10(np.maximum(intensity, 1e-300))
        log_intensity = np.where(result['sampling_valid'], log_intensity, np.nan)
        fig.add_trace(go.Heatmap(
            x=u, y=v, z=log_intensity, coloraxis='coloraxis2', name=name,
            zsmooth=False, connectgaps=False, hoverongaps=False,
            customdata=np.concatenate((intensity[..., None], hkl), axis=-1),
            hovertemplate='u=%{x:.3f} mm<br>v=%{y:.3f} mm<br>M=%{customdata[0]:.3e}'
                          '<br>h=%{customdata[1]:.6f}<br>k=%{customdata[2]:.6f}'
                          '<br>l=%{customdata[3]:.6f}<extra>%{fullData.name}</extra>'),
            row=2, col=column)
        fig.add_trace(go.Scatter(
            x=[r['u'] for r in peaks], y=[r['v'] for r in peaks], mode='markers',
            marker=dict(symbol='circle-open', size=7, color='rgb(73, 222, 255)', line=dict(width=1)),
            name='Ideal rod centres', showlegend=False, hoverinfo='skip'), row=2, col=column)
        fig.update_xaxes(title_text='x (nm)', row=1, col=column, nticks=3)
        fig.update_yaxes(title_text='y (nm)' if column == 1 else None,
                         row=1, col=column, nticks=5)
        fig.update_xaxes(title_text='u (mm)', range=[-u_limit, u_limit],
                         row=2, col=column, nticks=3)
        fig.update_yaxes(title_text='v (mm)' if column == 1 else None,
                         range=[-v_limit, v_limit], row=2, col=column, nticks=5)
    exponents = list(range(log_floor, 1))
    fig.update_layout(
        template='plotly_white', height=740, margin=dict(l=55, r=100, t=55, b=50),
        paper_bgcolor='white', plot_bgcolor='#dce1e8', font=dict(color='#24364b', size=11),
        coloraxis=dict(colorscale=VIRIDIS, cmin=-height_limit, cmax=height_limit,
                       colorbar=dict(title=dict(text='Height (uc)', side='right'),
                                     x=1.02, y=.79, len=.42, thickness=14)),
        coloraxis2=dict(colorscale=MAGMA, cmin=log_floor, cmax=0,
                        colorbar=dict(title=dict(text='Morphology intensity M', side='right'),
                                      tickvals=exponents,
                                      ticktext=[f'10<sup>{n}</sup>' for n in exponents],
                                      x=1.02, y=.21, len=.42, thickness=14)),
        uirevision=f'comparison-{central_surface}-{detector_closeup}-{log_floor}')
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, mirror=True,
                     linecolor='#8793a0', ticks='outside', title_font_size=11)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, mirror=True,
                     linecolor='#8793a0', ticks='outside', title_font_size=11)
    fig.update_annotations(font_size=12)
    return fig
