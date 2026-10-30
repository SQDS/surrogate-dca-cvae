"""Participant/source bootstrap, paired DeLong and planned-family Holm."""
import numpy as np
from scipy.stats import norm

def aggregate(values, axis=0):
    axis %= values.ndim
    if axis == values.ndim-1: raise ValueError('Cannot average across metrics')
    out=np.mean(values,axis=axis)
    out[...,1]=circ(values[...,1],axis=axis)
    return out


def kernel(y,p):
    a,b=np.asarray(p)[y==1],np.asarray(p)[y==0]
    return (a[:,None]>b).astype(float)+.5*(a[:,None]==b)


def circ(a,axis=0):
    r=np.deg2rad(a);return np.rad2deg(np.arctan2(np.sin(r).mean(axis),np.cos(r).mean(axis)))


def circ_sd(a,axis=0):
    r=np.deg2rad(a);v=np.hypot(np.sin(r).mean(axis),np.cos(r).mean(axis))
    return np.rad2deg(np.sqrt(-2*np.log(np.clip(v,1e-15,1))))


def holm(pvalues):
    p=np.asarray(pvalues,float)
    if np.any(np.isfinite(p)&((p<0)|(p>1))):raise ValueError('Invalid p value')
    # Missing planned tests retain their place/family size without claiming a p.
    valid=np.isfinite(p);filled=np.where(valid,p,1.);order=np.argsort(filled,kind='stable')
    adjusted=np.minimum(1.,np.maximum.accumulate(filled[order]*(len(p)-np.arange(len(p)))))
    result=np.empty(len(p));result[order]=adjusted;result[~valid]=np.nan
    return result


def delong_raw(y,left,right):
    a,b=kernel(y,left),kernel(y,right)
    v=np.vstack([a.mean(1),b.mean(1)]);w=np.vstack([a.mean(0),b.mean(0)])
    covariance=np.cov(v,ddof=1)/len(a)+np.cov(w,ddof=1)/a.shape[1]
    contrast=np.array([1.,-1.]);se=float(np.sqrt(max(float(contrast@covariance@contrast),0)))
    delta=float(a.mean()-b.mean())
    if se<=1e-15:
        p=1. if abs(delta)<=1e-15 else np.nan
        status='identical AUC / zero variance' if np.isfinite(p) else 'degenerate variance; normal approximation unavailable'
    else:p=float(2*norm.sf(abs(delta/se)));status='ok'
    return dict(auc_left=float(a.mean()),auc_right=float(b.mean()),delta_auc=delta,delong_se=se,
                raw_p=p,delong_status=status)


def interval(a,level=.95):
    tail=(1-level)/2;v=np.quantile(np.asarray(a,float),[tail,1-tail]);return float(v[0]),float(v[1])


def source_bootstrap(values,draws=10000,seed=20261013):
    """Rows are equally weighted independent source units; columns are methods."""
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    if not np.isfinite(values).all():raise ValueError('Nonfinite source statistic')
    n=len(values);rng=np.random.default_rng(seed);result=[]
    for first in range(0,draws,500):
        w=rng.multinomial(n,np.full(n,1/n),size=min(500,draws-first))
        result.append((w@values)/n)
    return np.concatenate(result)


def auc_bootstrap(labels,probabilities,draws=10000,seed=20261013):
    labels=np.asarray(labels);n0=int(np.sum(labels==0));n1=int(np.sum(labels==1))
    rng=np.random.default_rng(seed);w1=rng.multinomial(n1,np.full(n1,1/n1),size=draws)
    w0=rng.multinomial(n0,np.full(n0,1/n0),size=draws)
    result={}
    for name,p in probabilities.items():
        k=kernel(labels,p);result[name]=np.sum((w1@k)*w0,axis=1)/(n1*n0)
    return result
