"""Hann spectra, conventional TFA and frozen sinusoidal probing."""
import numpy as np
import torch
from scipy.signal import get_window
from .statistics import aggregate
FREQ = np.round(np.arange(.02, .501, .01), 2)


def participant_curves(data, net=None, path='MS-aux', device='cpu', baseline='T'):
    """Return person curves and visit-first band features in relative gain units."""
    from .models import encode_arrays
    from .fir import step_to_fir
    from ._io import config
    keys = sorted(set(zip(data['labels'].tolist(), data['person_ids'].tolist())))
    if net is None and baseline == 'T':
        px, py, pc = spectra(data['abp'], data['cbv'])
        window_curves = None
    elif net is None and baseline == 'F':
        # The manuscript's window-wise FIR comparator uses the directly fitted
        # coefficients, before neural step reconstruction.
        window_curves = fir_curves(step_to_fir(data['step']))
    elif net is None:
        raise ValueError('Clinical baseline must be T (spectral TFA) or F (window-wise FIR)')
    else:
        branch = 'main' if path in ['M', 'MS-main'] else 'auxiliary'
        mu, _ = encode_arrays(net, data['y_z'], data['x_z'], data['step'], branch, device)
        if path == 'S':
            with torch.inference_mode():
                steps = net.ds(torch.as_tensor(mu, device=device)).cpu().numpy()
            window_curves = fir_curves(step_to_fir(steps))
        else:
            window_curves = neural_curves(net, mu, device)
    if window_curves is not None:
        signals = config()['scaler']['signals']
        window_curves[..., 0] *= signals['cbv']['std'] / signals['abp']['std']
    person_curves, person_bands = [], []
    for label, person in keys:
        mask = data['person_ids'] == person
        visits = []
        for record in sorted(set(data['record_ids'][mask])):
            rm = mask & (data['record_ids'] == record)
            side_curves = []
            for side in sorted(set(data['sides'][rm])):
                sm = rm & (data['sides'] == side)
                if window_curves is None:
                    curve = v2_spectra(px[sm].mean(0), py[sm].mean(0), pc[sm].mean(0))[2:]
                else:
                    curve = aggregate(window_curves[sm], axis=0)
                side_curves.append(curve)
            visits.append(aggregate(np.stack(side_curves), axis=0))
        visits = np.stack(visits)
        person_curves.append(aggregate(visits, axis=0))
        person_bands.append(aggregate(bands(visits), axis=0).reshape(9))
    return np.stack(person_curves), np.stack(person_bands), keys

def spectra(x, y, fs=1.):
    """Per-100-point-window Hann/constant-detrended one-sided density spectra."""
    x, y = np.asarray(x,float), np.asarray(y,float)
    if x.shape[-1] != y.shape[-1]: raise ValueError('Mismatched sample dimensions')
    length=x.shape[-1]; w=get_window('hann',length,fftbins=True)
    xf=np.fft.rfft((x-x.mean(-1,keepdims=True))*w,axis=-1)
    yf=np.fft.rfft((y-y.mean(-1,keepdims=True))*w,axis=-1)
    scale=1/(fs*np.sum(w*w))
    pxx=(xf.conj()*xf).real*scale; pyy=(yf.conj()*yf).real*scale; pxy=xf.conj()*yf*scale
    interior=slice(1,-1) if length%2==0 else slice(1,None)
    for p in (pxx,pyy,pxy): p[...,interior] *= 2
    return pxx,pyy,pxy


def probes(frequencies):
    t=np.arange(300,dtype=float)
    x=[]
    for f in frequencies:
        p=(np.cos(np.pi*t)/np.sqrt(2) if np.isclose(f,.5) else np.sin(2*np.pi*f*t)).astype(np.float32)
        x.append(np.stack([p[s:s+100] for s in range(0,201,50)]))
    return np.stack(x)


def v2_spectra(pxx,pyy,pxy):
    pxx,pyy,pxy=np.asarray(pxx),np.asarray(pyy),np.asarray(pxy)
    h=pxy/np.where(np.abs(pxx)<1e-12,1e-12,pxx)
    denominator=np.abs(pxx)*np.abs(pyy);numerator=np.abs(pxy)**2
    valid=np.isreal(pxx)&np.isreal(pyy)&(pxx.real>0)&(pyy.real>0)&np.isfinite(pxx)&np.isfinite(pyy)&np.isfinite(pxy)&np.isfinite(denominator)&np.isfinite(numerator)&(denominator>0)
    c=np.full(np.broadcast_shapes(denominator.shape,numerator.shape),np.nan)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        np.divide(numerator,denominator,out=c,where=valid)
    c=np.where(np.isfinite(c),np.clip(c,0,1),np.nan)
    return np.stack([np.abs(h),np.rad2deg(np.angle(h)),c],axis=-1)


@torch.inference_mode()
def neural_curves(net,mu,device='cpu',batch=16):
    probe=probes(FREQ);nf,nw,nt=probe.shape;all_curves=[];idx=np.rint(FREQ*100).astype(int)
    for first in range(0,len(mu),batch):
        latent=mu[first:first+batch];n=len(latent)
        zz=torch.tensor(np.repeat(latent,nf*nw,axis=0),device=device)
        xx=torch.tensor(np.tile(probe.reshape(-1,nt),(n,1)),device=device)
        yy=net.decode_y(zz,xx).cpu().numpy().reshape(n,nf,nw,nt)
        px,py,pc=spectra(probe[None],yy)
        all_curves.append(v2_spectra(px.mean(-2),py.mean(-2),pc.mean(-2))[:,np.arange(nf),idx,:])
    return np.concatenate(all_curves)


def fir_curves(coefficients,batch=32):
    h=np.asarray(coefficients,float);nf=len(FREQ);probe=probes(FREQ)
    t=np.arange(300);cosine=np.stack([np.stack([np.cos(2*np.pi*f*t)[j:j+100] for j in [0,50,100,150,200]]) for f in FREQ])
    response=h@np.exp(-2j*np.pi*np.arange(12)[:,None]*FREQ[None,:])
    response[:,-1]=response[:,-1].real
    result=[];idx=np.rint(FREQ*100).astype(int)
    for first in range(0,len(h),batch):
        a=response[first:first+batch]
        # Equivalent to steady-state sinusoidal convolution including known probe history.
        yy=a.real[:,:,None,None]*probe[None]+a.imag[:,:,None,None]*cosine[None]
        px,py,pc=spectra(probe[None],yy)
        curve=v2_spectra(px.mean(-2),py.mean(-2),pc.mean(-2))[:,np.arange(nf),idx,:]
        if not np.allclose(curve[...,0],abs(a),rtol=2e-5,atol=2e-7):raise AssertionError('FIR numerical probe and analytic magnitude differ')
        good=abs(a)>1e-8
        delta=np.angle(np.exp(1j*(np.deg2rad(curve[...,1])-np.angle(a))))
        if np.max(abs(delta[good]),initial=0)>2e-5:raise AssertionError('FIR phase sign differs from analytic response')
        result.append(curve)
    return np.concatenate(result)


def bands(curves, no_nyquist=False):
    masks=[FREQ<.07,(FREQ>=.07)&(FREQ<.20),(FREQ>=.20)&(FREQ<.5 if no_nyquist else True)]
    return np.stack([aggregate(curves[...,mask,:],axis=-2) for mask in masks],axis=-2)
