"""Run: python -m streamlit run app.py"""
import io
import json
import sys
import base64
from pathlib import Path
from dataclasses import asdict
import numpy as np
import streamlit as st
from physics import Geometry, HC, preset, pixel_q, detector_image, rod_intersections, pixel_jacobian, project_direction
from plots import real_space, reciprocal, detector, terraces, project_2d
from surface_ui import render_surface_simulation

def export_button(label, data, filename, mime):
    if sys.platform == 'emscripten':
        payload=data.encode('utf-8') if isinstance(data,str) else data
        uri='data:'+mime+';base64,'+base64.b64encode(payload).decode('ascii')
        st.markdown(f'<a download="{filename}" href="{uri}" style="display:inline-block;padding:8px 14px;border:1px solid #16847d;border-radius:6px;margin:5px 0;color:#16847d;text-decoration:none">{label}</a>',unsafe_allow_html=True)
    else:
        st.download_button(label,data,filename,mime)

st.set_page_config(page_title='CTR Surface Simulator',page_icon='🔬',layout='wide')
st.markdown('''<style>.block-container{padding-top:1.4rem}h1{letter-spacing:-1px}
[data-testid="stMetricValue"]{font-size:1.45rem}</style>''',unsafe_allow_html=True)
if 'energy' not in st.session_state:
    st.session_state.update(asdict(preset('Oblique steps')))

def apply_preset():
    st.session_state.update(asdict(preset(st.session_state['preset_name'])))

def align_target():
    h,k,l=[st.session_state.get('target_'+x,d) for x,d in zip('hkl',[0.,0.,.5])]
    a=st.session_state.a; E=st.session_state.energy
    mag=np.linalg.norm([h,k,l]); arg=HC/E*mag/(2*a)
    if arg>1 or mag==0:
        st.session_state.target_message='Target is unreachable at this energy, or q=0.'
        return
    th=float(np.rad2deg(np.arcsin(arg)))
    phi=float(-np.rad2deg(np.arctan2(k,h))) if h or k else 0.
    alpha=float(np.rad2deg(np.arctan2(np.hypot(h,k),l))+th)
    if not -89.<=alpha<=89.:
        st.session_state.target_message='This simple reflection solution is outside the allowed sample tilt. Choose another reflection/orientation.'
        return
    st.session_state.update(alpha=alpha,phi=phi,elevation=2*th,azimuth=0.,target_message='Detector centre aligned to the requested q. Check actual surface incidence and the rod table; a fractional target is not automatically a peak.')

with st.sidebar:
    st.title('Geometry controls')
    webgl=True
    if Path('browser_capabilities.json').exists():
        webgl=json.loads(Path('browser_capabilities.json').read_text()).get('webgl',True)
    display_mode=st.selectbox('Diagram mode',['3D (rotate)','2D projections (no WebGL)'],index=0 if webgl else 1)
    if not webgl:
        st.caption('This browser cannot render WebGL. The 2D projections show the same geometry. Choose 3D in a WebGL-enabled browser.')
    st.selectbox('Start with an example',['Oblique steps','Across steps','Along steps','Flat surface','Paper-like L=1.6'],key='preset_name')
    st.button('Load example',on_click=apply_preset,width='stretch')
    st.caption('The paper-like example uses its energy and l; other settings are illustrative, not fitted experimental values.')
    with st.expander('Beam and crystal',expanded=True):
        st.number_input('Energy (keV)',min_value=1.,max_value=50.,step=.1,key='energy')
        st.number_input('Cubic lattice a (Å)',min_value=2.,max_value=20.,step=.001,format='%.4f',key='a')
        st.number_input('Crystal tilt α (deg)',min_value=-89.,max_value=89.,step=.01,format='%.4f',key='alpha',help='R = Ry(-α) Rz(φ). At φ=0 the incident beam crosses the step edges.')
        st.slider('Sample azimuth φ (deg)',-180.,180.,step=1.,key='phi',help='0°: scattering plane across steps. 90°: parallel to step edges, for detector azimuth 0°.')
        st.number_input('Miscut μ (deg)',min_value=0.,max_value=5.,step=.01,format='%.3f',key='miscut')
    with st.expander('Detector',expanded=True):
        st.number_input('Elevation δ (deg)',min_value=-179.,max_value=179.,step=.01,format='%.4f',key='elevation')
        st.number_input('Azimuth γ (deg)',min_value=-180.,max_value=180.,step=.1,key='azimuth')
        st.number_input('Roll (deg)',min_value=-180.,max_value=180.,step=1.,key='roll')
        st.number_input('Sample-to-centre distance (mm)',min_value=20.,max_value=5000.,step=50.,key='distance')
        st.number_input('Pixel pitch (mm)',min_value=.01,max_value=1.,step=.005,format='%.4f',key='pitch')
        st.select_slider('Columns',options=[128,256,384,512],key='nx')
        st.select_slider('Rows',options=[128,256,384,512],key='ny')
    with st.expander('Rod family and teaching intensity'):
        st.number_input('Rod family h',min_value=-5,max_value=5,step=1,key='h')
        st.number_input('Rod family k',min_value=-5,max_value=5,step=1,key='k')
        st.number_input('First parent Bragg L',min_value=-8,max_value=0,step=1,key='lmin')
        st.number_input('Last parent Bragg L',min_value=1,max_value=12,step=1,key='lmax')
        st.number_input('Across-step coherence ξx (µm)',min_value=.01,max_value=10.,step=.1,key='coherence_x')
        st.number_input('Along-step coherence ξy (µm)',min_value=.01,max_value=10.,step=.1,key='coherence_y')
        st.slider('Total terrace roughness σ / width',0.,.4,step=.01,key='roughness')
        st.caption('Only the selected (h,k) rod family is evaluated. Intensities exclude the common ideal-CTR factor and structure-factor extinctions.')

g=Geometry(**{k:st.session_state[k] for k in asdict(Geometry())})
u,v,I,qmap,qcmap=detector_image(g)
records=rod_intersections(g)
st.title('CTR Surface Simulator')
st.markdown('**Follow one detector pixel from a real-space ray to a reciprocal-space point.** Rotate the 3D views with the mouse; use the pixel controls to move the green cross.')
cols=st.columns(4)
cols[0].metric('Wavelength',f'{HC/g.energy:.4f} Å')
cols[1].metric('Actual incidence to mean surface',f'{g.incidence:.4f}°')
cols[2].metric('Mean terrace width','∞ (flat)' if not np.isfinite(g.terrace_width) else f'{g.terrace_width/10:.1f} nm')
cols[3].metric('Captured rod intersections',sum(r['status']=='Captured' for r in records))
if g.incidence<=0: st.warning('The beam is not incident onto the top surface. Adjust sample tilt; the physical image is blank.')
if np.isfinite(g.terrace_width) and g.coherence_x*1e4/g.terrace_width < 3:
    st.info('The across-step coherence spans fewer than three terraces. Distinct sub-rods may not resolve; the Gaussian teaching image is not a quantitative short-coherence model.')

cc=st.columns([1,1,1])
col=cc[0].slider('Selected column (0-based)',0,g.nx-1,(g.nx-1)//2,key=f'col_{g.nx}')
row=cc[1].slider('Selected row (0-based, upward)',0,g.ny-1,(g.ny-1)//2,key=f'row_{g.ny}')
layer=cc[2].selectbox('Detector colour map',['Intensity','h','k','l'])
us,vs=u[col],v[row]
q,qc,out=pixel_q(g,us,vs); hkl=qc*g.a/(2*np.pi)
st.caption(f'Selected pixel: u={us:.4f}, v={vs:.4f} mm → outgoing direction ({out[0]:.6f}, {out[1]:.6f}, {out[2]:.6f}) → (h,k,l)=({hkl[0]:.6f}, {hkl[1]:.6f}, {hkl[2]:.6f}).')

tab1,tab_surface,tab2,tab3,tab4=st.tabs(['Linked views','Surface forward simulation','Peak access & axes','Understand the construction','Save / model notes'])
with tab_surface:
    render_surface_simulation(g, export_button)
with tab1:
    a,b=st.columns(2)
    with a:
        rf=real_space(g,us,vs)
        if display_mode.startswith('2D'): rf=project_2d(rf,title='1 | Real space: lab X–Z projection',xlabel='Lab X (mm)',ylabel='Lab Z (mm)')
        st.plotly_chart(rf,width='stretch')
    with b:
        ef=reciprocal(g,us,vs)
        if display_mode.startswith('2D'): ef=project_2d(ef,title='2 | Reciprocal space: lab qX–qZ projection',xlabel='qX (Å⁻¹)',ylabel='qZ (Å⁻¹)')
        st.plotly_chart(ef,width='stretch')
    a,b=st.columns(2)
    with a:
        zf=reciprocal(g,us,vs,True)
        if display_mode.startswith('2D'): zf=project_2d(zf,title='3 | Reciprocal close-up: h–l projection',xlabel='h (r.l.u.)',ylabel='l (r.l.u.)',equal=False)
        st.plotly_chart(zf,width='stretch')
        if display_mode.startswith('2D'): st.caption('Projection only: overlap in h–l does not imply intersection in 3D; k must also match. Black markers are the actual 3D intersections.')
        st.caption('The close-up uses unequal axis scales so small miscut splitting is visible. Full Ewald view uses equal physical scales. Only the selected (h,k) rod family is shown.')
    with b:
        st.plotly_chart(detector(g,u,v,I,qcmap,us,vs,records,layer),width='stretch')
        st.caption('Cyan circles: exact rod/sphere intersections. A geometric intersection can have negligible intensity. The image is a relative teaching model, not predicted counts. Detector row +v points upward here; camera files may use downward row indexing.')
with tab2:
    st.subheader('Which sub-rods reach this detector?')
    st.write('Each row is one mathematical rod/sphere intersection. Only rays leaving the top surface and landing inside the detector are marked Captured. Bulk Bragg peaks additionally require integer h,k,l and a nonzero structure factor.')
    rows=[]
    for r in records:
        hk=r.get('hkl',[np.nan]*3)
        rows.append({'Parent L':'all (flat)' if g.miscut==0 else r['L'],'Status':r['status'],'u (mm)':r['u'],'v (mm)':r['v'],'h':hk[0],'k':hk[1],'l':hk[2],'Exit angle (deg)':r.get('exit_angle',np.nan)})
    st.dataframe(rows,width='stretch',hide_index=True)
    st.subheader('What do detector horizontal and vertical mean?')
    J=pixel_jacobian(g,us,vs)
    st.dataframe([{'Reciprocal coordinate':x,'Change per +1 column':J[i,0],'Change per +1 upward row':J[i,1]} for i,x in enumerate(['h','k','l'])],hide_index=True,width='stretch')
    st.write('These are local derivatives at the selected pixel. A detector axis generally mixes h, k, and l. Switch the detector colour map to h, k, or l to see the nonlinear mapping across the full image.')
    st.subheader('Try a target reciprocal point')
    ts=st.columns(3)
    for colw,key,default in zip(ts,'hkl',[0.,0.,.5]):
        colw.number_input('Target '+key,min_value=-12.,max_value=12.,value=default,step=.1,key='target_'+key)
    target=np.array([st.session_state['target_'+key] for key in 'hkl'])
    G=g.R@(2*np.pi/g.a*target)
    trial=g.ki+G; norm=np.linalg.norm(trial)
    mismatch=norm-g.wavevector
    ut,vt,front=project_direction(g,trial/norm) if norm>0 else (np.nan,np.nan,False)
    st.write(f'Current elastic-shell mismatch |ki + G| − |ki| = **{mismatch:.6g} Å⁻¹**. Exact elastic access requires zero. The directional projection would be u={ut:.3f}, v={vt:.3f} mm; this is a real diffraction position only when the target lies on the Ewald sphere.')
    if abs(mismatch)<1e-8:
        physical=(trial@g.normal>0 and g.incidence>0)
        inside=front and abs(ut)<=g.nx*g.pitch/2 and abs(vt)<=g.ny*g.pitch/2
        st.success('Target is on the Ewald sphere and captured geometrically.' if inside and physical else 'Target is on the Ewald sphere but not captured in this reflection geometry.')
    st.button('Align sample and detector centre to target',on_click=align_target)
    if 'target_message' in st.session_state: st.info(st.session_state.target_message)
    st.caption('The alignment button changes tilt, sample azimuth, detector elevation and detector azimuth. It does not change the selected rod family. It finds one simple orientation, not a full diffractometer motor solution.')
with tab3:
    st.markdown('''### 1. There are two different spaces
The sample, incident beam and detector plane live in **real space**. Crystal truncation rods (CTRs), Bragg points and the Ewald sphere live in **reciprocal space**. A CTR is not a physical rod travelling toward the detector.

### 2. Every pixel selects one direction
The sample is at the real-space origin. The detector centre is D nD; u and v are offsets along its two perpendicular axes. The ray to that pixel determines the outgoing direction.''')
    st.latex(r'\mathbf p=D\hat{\mathbf n}_D+u\hat{\mathbf e}_u+v\hat{\mathbf e}_v,\quad \mathbf k_f=\frac{2\pi}{\lambda}\frac{\mathbf p}{|\mathbf p|},\quad \mathbf q=\mathbf k_f-\mathbf k_i')
    st.markdown('''### 3. That direction selects one point on a sphere
Elastic scattering fixes the lengths of ki and kf. In the **q-space convention used here**, the Ewald sphere has centre **−ki**, radius 2π/λ, and passes through q=0. Translate the ki and kf arrows to the same tail at −ki: the arrow between their tips is q. Other books draw the sphere with a different origin; the scattering condition is unchanged.''')
    st.latex(r'|\mathbf q+\mathbf k_i|=\frac{2\pi}{\lambda}')
    st.markdown('''### 4. Why steps split a CTR
Here each terrace descends by one cubic unit cell along crystal +x, and step edges run along crystal y. The mean surface normal tilts toward +x by μ. Each sub-rod passes through its own parent Bragg point G(h,k,L) and runs parallel to that **mean surface normal**, rather than the crystal [001] direction.''')
    st.latex(r'\mathbf q=\mathbf G_{hkL}+t\hat{\mathbf n}_s,\qquad h_{\rm pixel}=h_0+\tan\mu\,(l-L),\qquad k_{\rm pixel}=k_0')
    st.write('At fixed l, adjacent sub-rods have |Δqx| = (2π/a) tan μ = 2π/terrace width. However, a detector samples a curved surface, not a fixed-l plane; its peaks generally occur at slightly different l values.')
    st.plotly_chart(terraces(g),width='stretch')
    st.markdown('''### 5. Which peaks appear?
First intersect the rods with the Ewald sphere. Then extend the corresponding outgoing rays to the detector plane. A spot can be outside the detector, below the surface, weak because of its structure factor, or unresolved because of finite coherence/pixel size. A nonzero rod width can also produce image intensity when the exact centreline falls just outside the detector.

### Try these experiments
1. Load **Oblique steps**. Match cyan detector markers to the black intersection points in the reciprocal close-up.
2. Switch the colour map from intensity to **h**, **k**, then **l**. Watch why detector horizontal cannot automatically be called L.
3. Compare **Across steps** (φ=0°) and **Along steps** (φ=90°). The sub-rods remain present, but their separation on the detector changes. Parallel alignment does not universally remove the signal.
4. Increase miscut: terraces narrow and sub-rods separate further in reciprocal space. Increase distance: the same angular separation spans more millimetres and pixels.
5. Load **Flat surface**: the sub-rods collapse onto one CTR. All parent L labels refer to the same line.
6. Increase teaching roughness: distant-parent sub-rods weaken in the linked-view model. Open **Surface forward simulation** to explicitly generate width disorder, jagged edges and islands, and compare their scattering.
7. Set target (0,0,1), align, and compare with (0,0,0.5). This demonstrates geometric Bragg and anti-Bragg access; the image is normalized and does not represent their enormous absolute intensity difference.

A selected q probes a Fourier component of the structure. It does **not** select a unique spatial patch or a unique island size. Features around different sub-rods may mix different terrace/height correlations and coherent interference. Geometry alone cannot predict XPCS two-time dynamics.''')
with tab4:
    st.subheader('Save the geometry and coordinate maps')
    settings=json.dumps(asdict(g),indent=2)
    export_button('Download geometry JSON',settings,'geometry.json','application/json')
    buf=io.BytesIO()
    np.savez_compressed(buf,u_mm=u,v_mm=v,relative_intensity=I,q_lab_Ainv=qmap,hkl=qcmap*g.a/(2*np.pi),geometry_json=np.array(settings))
    export_button('Download image + hkl maps (NPZ)',buf.getvalue(),'detector_maps.npz','application/octet-stream')
    uploaded=st.file_uploader('Load a saved geometry JSON',type=['json'])
    if uploaded:
        try:
            data=json.load(uploaded)
            # Validate by rerunning the dataclass and checking known fields; UI limits still govern use.
            Geometry(**data)
            def load_settings(data=data):
                defaults=asdict(Geometry()); defaults.update(data); st.session_state.update(defaults)
            st.button('Apply loaded geometry',on_click=load_settings)
        except Exception as exc: st.error(f'Cannot load settings: {exc}')
    st.markdown('''### What is exact and what is approximate?
**Exact within the stated geometry:** elastic scattering vectors; plane-detector projection; cubic crystal/lab transformations; straight mean-normal CTR/sphere intersections; local detector-to-hkl derivatives. Refraction is neglected.

**Teaching intensity:** Gaussian transverse rod profiles, with widths from coherence and an approximate pixel-resolution term. For nonzero miscut the longitudinal relative weight is sinc²(l−L) exp[−(2π(l−L)σ̃tot)²], taken from the per-sub-rod factor in Eq. (17). The common ideal-CTR intensity I0 is omitted. This does not reproduce absolute counts or Bragg enhancement. At zero miscut one Gaussian CTR is drawn with constant longitudinal weight. For non-specular families the same envelope is illustrative.

The full paper separates sharp peaks Ip, a broad component Iw from terrace-width variation, and diffuse Is from step jaggedness. **The Linked views teaching image includes only a simplified Ip-like component.** The **Surface forward simulation** tab instead calculates coherent scattering from explicit nonideal height maps; its width CV and edge RMS are not automatically the paper’s σw and σs. It includes islands of the same material and independent-patch intensity averaging. See SURFACE_MODEL.md for equations and statistical assumptions. No film thickness fringes, material structure factors, systematic absences, absorption, polarization correction, instrument calibration, multiple scattering, or growth/XPCS dynamics are included.

### Paper read
Both supplied files are byte-identical: Trevor A. Petach, David Goldhaber-Gordon, Apurva Mehta and Michael F. Toney, *Crystal truncation rods from miscut surfaces*, arXiv:1706.00484v1 (2017), 10 pages. The supplemental material is referenced in the paper but was not attached.

The implementation follows Fig. 1 for mean-normal sub-rods, Eq. (6) for the terrace phase, Eqs. (12)–(17) for peak spacing/coherence and relative weights, and Section III for the detector's Ewald-sphere patch. Figure 3 and Eqs. (22), (26) motivate the explicit exclusions above.

Software: NumPy, Plotly and Streamlit. Streamlit API reference: https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
''')
