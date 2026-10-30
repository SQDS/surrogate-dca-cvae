"""Exact four-configuration reconstruction and KL objectives."""
import torch
LOSS_FIELDS = ("total", "cbfv_direct", "step_direct", "cbfv_cross", "step_cross", "kl_main", "kl_aux")

def objective(net,y,x,s,beta,epsilon_main=None,epsilon_aux=None):
    zero=y.new_zeros(());v={k:zero for k in LOSS_FIELDS}
    if net.em is not None:
        mm,lm=net.encode_main(y,x)
        zm=mm if epsilon_main is None else mm+torch.exp(.5*lm)*epsilon_main
        v['kl_main']=.5*(mm.square()+lm.exp()-1-lm).sum(-1).mean()
        v['cbfv_direct']=(net.decode_y(zm,x)-y).square().mean()
    if net.es is not None:
        ms,ls=net.encode_aux(s)
        zs=ms if epsilon_aux is None else ms+torch.exp(.5*ls)*epsilon_aux
        v['kl_aux']=.5*(ms.square()+ls.exp()-1-ls).sum(-1).mean()
        v['step_direct']=(net.ds(zs)-s).square().mean()
        if net.dy is not None:v['cbfv_cross']=(net.decode_y(zs,x)-y).square().mean()
    if net.variant=='MS':v['step_cross']=(net.ds(zm)-s).square().mean()
    v['total']=v['cbfv_direct']+v['step_direct']+.5*(v['cbfv_cross']+v['step_cross'])+beta*(v['kl_main']+v['kl_aux'])
    return v
