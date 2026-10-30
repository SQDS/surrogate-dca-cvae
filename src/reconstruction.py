"""Deterministic same-window and target-blind cross-window inference."""
import numpy as np
import torch
from .fir import fir_step, step_to_fir, fir_design, apply_fir, pooled_fir
from .tfa import spectra
PRIMARY = slice(11, 100)
CENTER = slice(25, 75)

def inverse_y(a,scaler):
    s=scaler['signals']['cbv'];return np.asarray(a,dtype=np.float64)*s['std']+s['mean']


def fit_baselines(x_cal_relative,y_cal_relative,x_cal_z,y_cal_z,step_cal):
    px,_,pc=spectra(x_cal_relative,y_cal_relative)
    return dict(pxx=px.mean(1),pxy=pc.mean(1),
                dc=np.asarray(y_cal_relative,float).mean((1,2)),
                fir_pooled=pooled_fir(x_cal_z,y_cal_z),
                fir_mean=step_to_fir(step_cal).mean(1))


def predict_baselines(context,x_target_relative,x_target_z,scaler,lam=0.):
    x=np.asarray(x_target_relative,dtype=float)
    den=context['pxx']+lam*context['pxx'][:,1:].mean(1,keepdims=True)
    h=context['pxy']/np.where(np.abs(den)<1e-12,1e-12,den)
    fft=np.fft.rfft(x-x.mean(1,keepdims=True),axis=-1)*h
    fft[:,0]=100*context['dc'];fft[:,-1]=fft[:,-1].real
    y=np.fft.irfft(fft,n=100,axis=-1)
    return dict(T=y,F_pooled=inverse_y(apply_fir(x_target_z,context['fir_pooled']),scaler),
                F_mean=inverse_y(apply_fir(x_target_z,context['fir_mean']),scaler),
                Constant=np.broadcast_to(context['dc'][:,None],x.shape).copy())


@torch.inference_mode()
def forecast(net,cal_x,cal_y,cal_s,target_x,path,device='cpu',batch=512):
    # Deliberately no target CBFV argument.
    output=[];ncal=cal_x.shape[1]
    for first in range(0,len(target_x),batch):
        sl=slice(first,first+batch);n=len(target_x[sl])
        xx=torch.as_tensor(np.array(cal_x[sl]).reshape(-1,100),device=device)
        ss=torch.as_tensor(np.array(cal_s[sl]).reshape(-1,12),device=device)
        xt=torch.as_tensor(np.array(target_x[sl]),device=device)
        if path in ['M','MS-main']:
            yy=torch.as_tensor(np.array(cal_y[sl]).reshape(-1,100),device=device)
            mu,_=net.encode_main(yy,xx)
        else:mu,_=net.encode_aux(ss)
        if path=='S':
            step=net.ds(mu).reshape(n,ncal,12).mean(1)
            h=torch.diff(step,dim=1,prepend=torch.zeros_like(step[:,:1]))
            pred=torch.full_like(xt,float('nan'))
            pred[:,11:]=(xt.unfold(1,12,1).flip(-1)*h[:,None,:]).sum(-1)
        else:
            pred=net.decode_y(mu,xt.repeat_interleave(ncal,dim=0)).reshape(n,ncal,100).mean(1)
        output.append(pred.cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def reconstruct(net,x,y,s,path,device='cpu',batch=4096):
    outputs=[];mus=[];logvars=[];steps=[]
    for first in range(0,len(x),batch):
        sl=slice(first,first+batch)
        xx=torch.as_tensor(np.array(x[sl]),device=device)
        ss=torch.as_tensor(np.array(s[sl]),device=device)
        if path in ['M','MS-main']:
            mu,lv=net.encode_main(torch.as_tensor(np.array(y[sl]),device=device),xx)
        else:mu,lv=net.encode_aux(ss)
        if path=='S':
            rs=net.ds(mu);h=torch.diff(rs,dim=1,prepend=torch.zeros_like(rs[:,:1]))
            pred=torch.full_like(xx,float('nan'));pred[:,11:]=(xx.unfold(1,12,1).flip(-1)*h[:,None,:]).sum(-1)
        else:pred=net.decode_y(mu,xx)
        outputs.append(pred.cpu().numpy());mus.append(mu.cpu().numpy());logvars.append(lv.cpu().numpy())
        if net.ds is not None:steps.append(net.ds(mu).cpu().numpy())
    return dict(pred=np.concatenate(outputs),mu=np.concatenate(mus),logvar=np.concatenate(logvars),
                reconstructed_step=np.concatenate(steps) if steps else None)


def errors(pred,true):
    pred=np.asarray(pred,float);true=np.asarray(true,float)
    if pred.shape!=true.shape:raise ValueError('Prediction/target shape mismatch')
    if not np.isfinite(pred[...,PRIMARY]).all() or not np.isfinite(true).all():raise ValueError('Invalid scored waveform')
    a=pred[...,PRIMARY];b=true[...,PRIMARY];d=a-b
    ac=a-a.mean(-1,keepdims=True);bc=b-b.mean(-1,keepdims=True)
    denom=np.sqrt(np.sum(ac**2,axis=-1)*np.sum(bc**2,axis=-1))
    valid_correlation=(denom>0)&(np.ptp(a,axis=-1)>0)&(np.ptp(b,axis=-1)>0)
    corr=np.full(denom.shape,np.nan);np.divide(np.sum(ac*bc,axis=-1),denom,out=corr,where=valid_correlation)
    full=np.mean((pred-true)**2,axis=-1)
    return dict(mse89=np.mean(d*d,axis=-1),mse50=np.mean((pred[...,CENTER]-true[...,CENTER])**2,axis=-1),
                mse100=full,demeaned_mse89=np.mean((ac-bc)**2,axis=-1),correlation89=corr)


def build_pairs(rid,seg,ari,mask):
    rows=np.flatnonzero(mask);lookup={(int(rid[i]),int(seg[i]),int(ari[i])):int(i) for i in rows}
    assert len(lookup)==len(rows)
    accepted=[];excluded=[]
    for r in np.unique(rid[rows]):
        segments=sorted(set(seg[rows][rid[rows]==r].tolist()));chosen=None
        for first in segments:
            required=[first+i for i in [0,1,2,3,4,7]]
            if all((int(r),int(w),a) in lookup for w in required for a in range(10)):
                chosen=required;break
        if chosen is None:
            excluded.append(dict(source_record_id=int(r),reason='insufficient required calibration/target time supports',segments=';'.join(map(str,segments))))
            continue
        for a in range(10):
            accepted.append(dict(source_record_id=int(r),ari=a,cal_rows=[lookup[int(r),int(w),a] for w in chosen[:5]],
                                 target_row=lookup[int(r),int(chosen[-1]),a],first_segment=int(chosen[0]),
                                 target_start_sec=int((chosen[-1]-1)*50)))
    cal=np.array([r['cal_rows'] for r in accepted],dtype=np.int64)
    target=np.array([r['target_row'] for r in accepted],dtype=np.int64)
    return accepted,excluded,cal,target
