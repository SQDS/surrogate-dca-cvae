"""Valid-window 12-tap ridge FIR and step-response conversions."""
import numpy as np

def fir_step(abp, cbv, n_taps=12, ridge=1e-4, batch_size=8192):
    abp = np.asarray(abp, np.float32); cbv = np.asarray(cbv, np.float32)
    if abp.shape != cbv.shape or abp.ndim != 2: raise ValueError('Expected paired N x T arrays')
    n, length = abp.shape; h = np.empty((n, n_taps), np.float32)
    for start in range(0,n,batch_size):
        p, v = abp[start:start+batch_size], cbv[start:start+batch_size]
        a = np.stack([p[:, n_taps-1-k:length-k] for k in range(n_taps)], axis=-1)
        ata = np.einsum('bmk,bml->bkl', a, a, optimize=True)
        ata += np.float32(ridge)*np.eye(n_taps,dtype=np.float32)[None]
        aty = np.einsum('bmk,bm->bk', a, v[:, n_taps-1:], optimize=True)
        try:
            block = np.linalg.solve(ata.astype(float), aty.astype(float)[...,None])[...,0]
        except np.linalg.LinAlgError:
            block = np.stack([np.linalg.lstsq(aa.astype(float),bb.astype(float),rcond=None)[0] for aa,bb in zip(ata,aty)])
        h[start:start+len(p)] = block.astype(np.float32)
    return h, np.cumsum(h, axis=1, dtype=np.float32)


def step_to_fir(s):
    s=np.asarray(s);return np.diff(s,axis=-1,prepend=np.zeros(s.shape[:-1]+(1,),dtype=s.dtype))


def fir_design(x):
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x),12,axis=-1)[...,::-1]


def apply_fir(x,h):
    x=np.asarray(x);h=np.asarray(h)
    result=np.full(x.shape,np.nan,dtype=np.float64)
    result[...,11:]=np.einsum('...tk,...k->...t',fir_design(x).astype(float),h.astype(float))
    return result


def pooled_fir(x,y,ridge=1e-4):
    # Calibration dimensions: (target, five windows, time).
    a=fir_design(x).astype(np.float64);b=np.asarray(y,dtype=float)[...,11:]
    aa=np.einsum('bwti,bwtj->bij',a,a)+ridge*np.eye(12)[None]
    ab=np.einsum('bwti,bwt->bi',a,b)
    return np.linalg.solve(aa,ab[...,None])[...,0]
