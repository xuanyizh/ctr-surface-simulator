import numpy as np
import plotly.graph_objects as go
from physics import pixel_q
from plot_colors import MAGMA

COLORS=['#1e88e5','#f4511e','#43a047','#8e24aa','#ffb300','#00acc1','#e53935','#3949ab','#6d4c41']

def line(fig,points,name,color,width=5,dash=None):
    p=np.array(points)
    fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',name=name,line=dict(color=color,width=width,dash=dash)))

def arrow(fig,start,end,name,color):
    start,end=np.asarray(start),np.asarray(end); line(fig,[start,end],name,color)
    d=end-start
    fig.add_trace(go.Cone(x=[end[0]],y=[end[1]],z=[end[2]],u=[d[0]],v=[d[1]],w=[d[2]],sizemode='absolute',sizeref=np.linalg.norm(d)*.09,anchor='tip',showscale=False,colorscale=[[0,color],[1,color]],showlegend=False,hoverinfo='skip'))

def sheet(fig,p,name,color,opacity=.3):
    fig.add_trace(go.Surface(x=p[...,0],y=p[...,1],z=p[...,2],name=name,colorscale=[[0,color],[1,color]],showscale=False,opacity=opacity,showlegend=True,hoverinfo='skip'))

def style3d(fig,title,units='A^-1',equal=True):
    fig.update_layout(title=title,height=490,margin=dict(l=0,r=0,t=50,b=0),legend=dict(orientation='h',font=dict(size=10)),scene=dict(xaxis_title=f'X ({units})',yaxis_title=f'Y ({units})',zaxis_title=f'Z ({units})',aspectmode='data' if equal else 'auto',camera=dict(eye=dict(x=1.5,y=-1.7,z=1.1))),uirevision=title)
    return fig

def real_space(g,u,v):
    f=go.Figure(); n,eu,ev=g.basis; D=g.distance
    s=np.linspace(-.17*D,.17*D,2); X,Y=np.meshgrid(s,s)
    sx=g.R@np.array([np.cos(np.deg2rad(g.miscut)),0,-np.sin(np.deg2rad(g.miscut))]); sy=g.R@np.array([0,1,0])
    p=X[...,None]*sx+Y[...,None]*sy; sheet(f,p,'Mean sample surface (size schematic)','#37a89b',.55)
    for frac in [-.10,-.05,0,.05,.10]:
        line(f,[frac*D*sx-.17*D*sy,frac*D*sx+.17*D*sy],'Step edge' if frac==0 else '', '#176b61',2)
        f.data[-1].showlegend=(frac==0)
    arrow(f,[-.4*D,0,0],[0,0,0],'Incident beam','#1976d2')
    psel=D*n+u*eu+v*ev
    arrow(f,[0,0,0],psel,'Selected outgoing beam','#e45e38')
    line(f,[[0,0,0],D*n],'Detector centre ray','#999999',2,'dash')
    U,V=np.meshgrid(np.array([-1,1])*g.nx*g.pitch/2,np.array([-1,1])*g.ny*g.pitch/2)
    det=D*n+U[...,None]*eu+V[...,None]*ev; sheet(f,det,'Detector plane (true size)','#ffc857',.8)
    # An enlarged frame is explicitly marked to make a small area detector legible.
    mag=max(1.,.17*D/max(g.nx*g.pitch,g.ny*g.pitch))
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])
    cp=D*n+corners[:,0,None]*g.nx*g.pitch/2*mag*eu+corners[:,1,None]*g.ny*g.pitch/2*mag*ev
    line(f,cp,f'Detector outline x{mag:.1f} (visual guide)','#c8911b',2,'dash')
    for vec,name,c in [(g.normal,'Mean surface normal','#00897b'),(g.R[:,2],'Crystal [001]','#7e57c2'),(g.R[:,0],'Crystal [100]','#546e7a')]: arrow(f,[0,0,0],vec*.25*D,name,c)
    arrow(f,D*n,D*n+.12*D*eu,'Detector +u','#c48c13'); arrow(f,D*n,D*n+.12*D*ev,'Detector +v','#ab6c10')
    return style3d(f,'1 | Real space: rays meet a flat detector','mm')

def reciprocal(g,u,v,zoom=False):
    f=go.Figure(); q,qc,out=pixel_q(g,u,v); k=g.wavevector
    U,V=np.meshgrid(np.linspace(-g.nx*g.pitch/2,g.nx*g.pitch/2,20),np.linspace(-g.ny*g.pitch/2,g.ny*g.pitch/2,20))
    patch,_,_=pixel_q(g,U,V)
    if zoom:
        p=patch@g.R*g.a/(2*np.pi)
        sheet(f,p,'Detector acceptance on Ewald sphere','#eab844',.4)
        bounds=[(p[...,i].min(),p[...,i].max()) for i in range(3)]
        dz=max(bounds[2][1]-bounds[2][0],.04)
        lo=bounds[2][0]-.25*dz; hi=bounds[2][1]+.25*dz
        ls=np.linspace(lo,hi,100)
        labels=[0] if g.miscut==0 else range(g.lmin,g.lmax+1)
        for j,L in enumerate(labels):
            h=g.h+np.tan(np.deg2rad(g.miscut))*(ls-L)
            pts=np.stack([h,np.full_like(ls,g.k),ls],axis=-1)
            line(f,pts,'Flat CTR' if g.miscut==0 else f'Sub-rod L={L}',COLORS[j%len(COLORS)],4)
        qq=qc*g.a/(2*np.pi)
        f.add_trace(go.Scatter3d(x=[qq[0]],y=[qq[1]],z=[qq[2]],mode='markers',marker=dict(size=6,color='#ff3355'),name='Selected pixel q'))
        from physics import rod_intersections
        rec=[r for r in rod_intersections(g) if r['status']=='Captured']
        if rec:
            pts=np.array([r['hkl'] for r in rec]); f.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',marker=dict(size=4,color='black'),name='Captured intersections'))
        style3d(f,'3 | Reciprocal close-up: same detector patch','r.l.u.',False)
        f.update_layout(scene=dict(xaxis_title='h (crystal [100]*)',yaxis_title='k (crystal [010]*)',zaxis_title='l (crystal [001]*)'))
        # Clip rods to a useful local range; full long rods would hide the acceptance patch.
        for key,i in [('xaxis',0),('yaxis',1),('zaxis',2)]:
            low,high=bounds[i]; pad=max((high-low)*.15,.001)
            f.layout.scene[key].range=[low-pad,high+pad]
        return f
    theta=np.linspace(0,np.pi,32); phi=np.linspace(0,2*np.pi,60)
    T,P=np.meshgrid(theta,phi)
    sphere=np.stack([k*np.sin(T)*np.cos(P)-k,k*np.sin(T)*np.sin(P),k*np.cos(T)],axis=-1)
    sheet(f,sphere,'Ewald sphere: centre = -ki','#8caaca',.12)
    sheet(f,patch,'Detector acceptance patch','#eab844',.8)
    C=-g.ki
    arrow(f,C,[0,0,0],'ki (translated)','#1976d2')
    arrow(f,C,q,'kf (translated)','#e45e38')
    arrow(f,[0,0,0],q,'q = kf - ki','#b02c7c')
    f.add_trace(go.Scatter3d(x=[0,C[0]],y=[0,0],z=[0,0],mode='markers+text',text=['Reciprocal origin','Sphere centre -ki'],textposition='top center',marker=dict(size=4,color='#222'),showlegend=False))
    labels=[0] if g.miscut==0 else range(g.lmin,g.lmax+1)
    for j,L in enumerate(labels):
        G=g.R@(2*np.pi/g.a*np.array([g.h,g.k,L]))
        # display a segment of each infinite line, including its primary Bragg point
        t0=-2*k-(G@g.normal); t1=2*k-(G@g.normal)
        line(f,[G+t0*g.normal,G+t1*g.normal],f'Rod L={L}',COLORS[j%len(COLORS)],2)
        f.add_trace(go.Scatter3d(x=[G[0]],y=[G[1]],z=[G[2]],mode='markers',marker=dict(size=3,color=COLORS[j%len(COLORS)]),showlegend=False,hovertext=f'Bragg ({g.h},{g.k},{L})'))
    style3d(f,'2 | Reciprocal space: the scattering triangle')
    f.update_layout(scene=dict(xaxis=dict(range=[-2.1*k,.5*k]),yaxis=dict(range=[-1.2*k,1.2*k]),zaxis=dict(range=[-1.2*k,1.2*k])))
    return f

def detector(g,u,v,I,qc,selected_u,selected_v,records,layer='Intensity'):
    hkl=qc*g.a/(2*np.pi)
    if layer=='Intensity': z=np.log10(np.maximum(I,1e-8)); scale=MAGMA; label='log10 relative intensity'; limits=dict(zmin=-6,zmax=0)
    else: z=hkl[...,{'h':0,'k':1,'l':2}[layer]]; scale='RdBu';label=layer+' (r.l.u.)';limits={}
    f=go.Figure(go.Heatmap(x=u,y=v,z=z,customdata=hkl,colorscale=scale,colorbar=dict(title=label),hovertemplate='u=%{x:.3f} mm<br>v=%{y:.3f} mm<br>h=%{customdata[0]:.6f}<br>k=%{customdata[1]:.6f}<br>l=%{customdata[2]:.6f}<extra></extra>',**limits))
    rs=[r for r in records if r['status']=='Captured']
    if rs:
        f.add_trace(go.Scatter(x=[r['u'] for r in rs],y=[r['v'] for r in rs],mode='markers+text',text=[f"L={r['L']}" if g.miscut else 'CTR' for r in rs],textposition='top right',marker=dict(symbol='circle-open',color='#49deff',size=10),name='Geometric intersections'))
    f.add_trace(go.Scatter(x=[selected_u],y=[selected_v],mode='markers',marker=dict(symbol='cross',size=13,color='#53ffb6'),name='Selected pixel'))
    f.update_layout(title='4 | Detector image: hover for h, k, l',height=490,xaxis_title='u (mm): detector columns increase →',yaxis_title='v (mm): detector rows increase ↑',margin=dict(l=30,r=20,t=50,b=40),legend=dict(orientation='h'),uirevision='detector',yaxis=dict(scaleanchor='x',scaleratio=1))
    return f

def terraces(g):
    width=g.terrace_width if np.isfinite(g.terrace_width) else 1000
    x=np.linspace(-3*width,3*width,1000)
    z=-g.a*np.floor(x/width) if g.miscut else np.zeros_like(x)
    f=go.Figure(go.Scatter(x=x/10,y=z,mode='lines',name='Stepped surface',line=dict(color='#159787',width=3)))
    f.add_trace(go.Scatter(x=x/10,y=-np.tan(np.deg2rad(g.miscut))*x+g.a/2 if g.miscut else z,mode='lines',name='Mean surface',line=dict(color='#9b75c5',dash='dash')))
    f.update_layout(title='Surface cross-section: +x goes downhill, step edges run along y',height=260,xaxis_title='Crystal x (nm)',yaxis_title='Height z (Å); vertical scale exaggerated',margin=dict(l=20,r=20,t=50,b=20),legend=dict(orientation='h'))
    return f


def project_2d(fig, xindex=0, yindex=2, title='', xlabel='', ylabel='', equal=True):
    """Orthographic projection for browsers without WebGL.
    This is a projection of the same coordinates, not a reciprocal-space slice.
    """
    out=go.Figure()
    for tr in fig.data:
        if tr.type=='cone':
            continue
        if tr.type=='surface':
            grids=[np.asarray(tr.x),np.asarray(tr.y),np.asarray(tr.z)]
            X,Y=grids[xindex],grids[yindex]
            xx=[];yy=[]
            # Surface grid, shown as thin SVG lines; avoids any WebGL requirement.
            stride=max(1,X.shape[0]//8)
            for i in range(0,X.shape[0],stride):
                xx.extend(X[i].tolist()+[None]);yy.extend(Y[i].tolist()+[None])
            stride=max(1,X.shape[1]//8)
            for i in range(0,X.shape[1],stride):
                xx.extend(X[:,i].tolist()+[None]);yy.extend(Y[:,i].tolist()+[None])
            col=tr.colorscale[0][1] if tr.colorscale else '#999999'
            out.add_trace(go.Scatter(x=xx,y=yy,mode='lines',name=tr.name,line=dict(color=col,width=1),opacity=.45,showlegend=bool(tr.showlegend),hoverinfo='skip'))
        elif tr.type=='scatter3d':
            coords=[tr.x,tr.y,tr.z]
            mode=tr.mode or 'lines'
            params=dict(x=coords[xindex],y=coords[yindex],mode=mode,name=tr.name,showlegend=tr.showlegend)
            if 'lines' in mode:
                params['line']=dict(color=tr.line.color,width=min(tr.line.width or 2,3),dash=tr.line.dash)
            if 'markers' in mode:
                params['marker']=dict(color=tr.marker.color,size=max(5,tr.marker.size or 5))
            if 'text' in mode:
                params.update(text=tr.text,textposition=tr.textposition)
            out.add_trace(go.Scatter(**params))
    out.update_layout(title=title,height=480,xaxis_title=xlabel,yaxis_title=ylabel,margin=dict(l=45,r=15,t=60,b=45),legend=dict(orientation='h',font=dict(size=10)),uirevision=title)
    if equal:out.update_yaxes(scaleanchor='x',scaleratio=1)
    # Preserve the close-up's finite bounds, when supplied.
    for name,index in [('xaxis',xindex),('yaxis',yindex)]:
        source=fig.layout.scene[['xaxis','yaxis','zaxis'][index]]
        if source.range:out.layout[name].range=source.range
    return out
