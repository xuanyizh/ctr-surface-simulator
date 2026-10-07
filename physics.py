"""Exact elastic geometry, plus explicitly approximate teaching intensities.
Distances: detector mm, lattice/coherence Angstrom, reciprocal vectors rad/Angstrom.
Column-vector convention: v_lab = R @ v_crystal. No 2pi is hidden in hkl.
"""
from dataclasses import dataclass, asdict
import numpy as np

HC = 12.398419843320026  # keV Angstrom

def ry(t):
    t=np.deg2rad(t); c,s=np.cos(t),np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rz(t):
    t=np.deg2rad(t); c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

@dataclass(frozen=True)
class Geometry:
    energy: float=10.16
    a: float=3.905
    alpha: float=4.481
    phi: float=45.
    miscut: float=0.10
    elevation: float=8.962
    azimuth: float=0.
    roll: float=0.
    distance: float=1000.
    pitch: float=0.075
    nx: int=256
    ny: int=256
    h: int=0
    k: int=0
    lmin: int=-2
    lmax: int=6
    coherence_x: float=1.0 # micrometres, across steps
    coherence_y: float=0.4 # micrometres, along steps
    roughness: float=0.07 # sigma_total / mean terrace width

    @property
    def wavevector(self): return 2*np.pi*self.energy/HC
    @property
    def R(self): return ry(-self.alpha) @ rz(self.phi)
    @property
    def ns(self):
        mu=np.deg2rad(self.miscut)
        return np.array([np.sin(mu),0,np.cos(mu)])
    @property
    def normal(self): return self.R @ self.ns
    @property
    def ki(self): return np.array([self.wavevector,0.,0.])
    @property
    def basis(self):
        d,g,r=np.deg2rad([self.elevation,self.azimuth,self.roll])
        n=np.array([np.cos(d)*np.cos(g),np.cos(d)*np.sin(g),np.sin(d)])
        u=np.array([-np.sin(g),np.cos(g),0.])
        v=np.cross(n,u)
        return n, np.cos(r)*u+np.sin(r)*v, -np.sin(r)*u+np.cos(r)*v
    @property
    def terrace_width(self):
        return np.inf if self.miscut==0 else self.a/np.tan(np.deg2rad(self.miscut))
    @property
    def incidence(self): return np.rad2deg(np.arcsin(-self.normal[0]))

def pixel_q(g,u,v):
    """Detector offsets in mm -> Q_lab, Q_crystal, outgoing unit direction."""
    n,eu,ev=g.basis
    u,v=np.broadcast_arrays(u,v)
    p=g.distance*n+u[...,None]*eu+v[...,None]*ev
    direction=p/np.linalg.norm(p,axis=-1,keepdims=True)
    q=g.wavevector*direction-g.ki
    return q, q @ g.R, direction

def project_direction(g,direction):
    n,eu,ev=g.basis
    den=float(np.dot(direction,n))
    if den<=1e-12: return np.nan,np.nan,False
    p=g.distance*np.asarray(direction)/den
    return float(p@eu),float(p@ev),True

def rod_intersections(g):
    """Solve |G+t n + ki|^2=k^2, then ray/plane intersect.
    At zero miscut all L labels are the same geometric rod, counted once.
    """
    records=[]
    labels=[0] if g.miscut==0 else range(g.lmin,g.lmax+1)
    for L in labels:
        Gs=2*np.pi/g.a*np.array([g.h,g.k,L]); G=g.R@Gs
        b=float(g.normal@(G+g.ki)); c=float((G+g.ki)@(G+g.ki)-g.wavevector**2)
        disc=b*b-c
        if disc < -1e-10:
            records.append(dict(L=L,status='No Ewald intersection',u=np.nan,v=np.nan,q=None))
            continue
        for t in np.unique([-b-np.sqrt(max(0.,disc)),-b+np.sqrt(max(0.,disc))]):
            q=G+t*g.normal; out=(q+g.ki)/g.wavevector
            u,v,front=project_direction(g,out)
            physical=(out@g.normal>1e-10 and g.incidence>0)
            inside=front and abs(u)<=g.nx*g.pitch/2 and abs(v)<=g.ny*g.pitch/2
            status='Captured' if inside and physical else ('Below surface / invalid incidence' if not physical else ('Outside detector' if front else 'Behind detector'))
            records.append(dict(L=L,status=status,u=u,v=v,q=q,hkl=(q@g.R)*g.a/(2*np.pi),exit_angle=np.rad2deg(np.arcsin(np.clip(out@g.normal,-1,1)))))
    return records

def intensity_at(g,qc):
    """Normalized rod weights from Eq.17 times Gaussian transverse profiles.
    Not a full evaluation of paper Eqs.12/22/26. No absolute counts.
    Gaussian pixel blur approximates resolution; geometric centres are exact.
    """
    scale=2*np.pi/g.a
    h,k,l=np.moveaxis(qc/scale,-1,0)
    tan=np.tan(np.deg2rad(g.miscut))
    # Beam coherence width plus an isotropic approximation to pixel integration.
    pix=g.wavevector*g.pitch/g.distance/np.sqrt(12)
    sx=np.hypot(1/(g.coherence_x*1e4),pix)
    sy=np.hypot(1/(g.coherence_y*1e4),pix)
    dy=(k-g.k)*scale
    if g.miscut==0:
        return np.exp(-0.5*(((h-g.h)*scale/sx)**2+(dy/sy)**2))
    result=np.zeros_like(l)
    for L in range(g.lmin,g.lmax+1):
        dx=((h-g.h)-tan*(l-L))*scale
        # Relative integrated weight cp,L; excludes common ideal CTR factor I0.
        w=np.sinc(l-L)**2*np.exp(-(2*np.pi*(l-L)*g.roughness)**2)
        result+=w*np.exp(-0.5*((dx/sx)**2+(dy/sy)**2))
    return result

def detector_image(g):
    u=(np.arange(g.nx)-(g.nx-1)/2)*g.pitch
    v=(np.arange(g.ny)-(g.ny-1)/2)*g.pitch
    U,V=np.meshgrid(u,v)
    q,qc,out=pixel_q(g,U,V)
    intensity=intensity_at(g,qc)
    intensity=np.where((out@g.normal>0)&(g.incidence>0),intensity,0.)
    return u,v,intensity,q,qc

def pixel_jacobian(g,u,v):
    """Columns = change in (h,k,l) per +1 detector column / row."""
    cols=[]
    for du,dv in [(g.pitch/2,0),(0,g.pitch/2)]:
        _,qp,_=pixel_q(g,u+du,v+dv); _,qm,_=pixel_q(g,u-du,v-dv)
        cols.append((qp-qm)*g.a/(2*np.pi))
    return np.array(cols).T

def preset(name):
    a=3.905; energy=10.16; l=0.5
    if name=='Paper-like L=1.6': energy=15.5; l=1.6
    theta=float(np.rad2deg(np.arcsin(HC/energy*l/(2*a))))
    phi={'Across steps':0.,'Along steps':90.,'Oblique steps':45.,'Paper-like L=1.6':45.,'Flat surface':0.}[name]
    return Geometry(energy=energy,alpha=theta,elevation=2*theta,phi=phi,miscut=0. if name=='Flat surface' else 0.10)
