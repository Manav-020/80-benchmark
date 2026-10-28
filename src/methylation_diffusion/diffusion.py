"""Small custom conditional DDPM; not an official TabDDPM reproduction."""
import time
import numpy as np
import torch
from torch import nn
from scipy.special import expit,logit
from sklearn.preprocessing import StandardScaler

class Denoiser(nn.Module):
    def __init__(self,features,cfg):
        super().__init__()
        e,h=cfg['embedding'],cfg['hidden']
        self.time=nn.Embedding(cfg['steps'],e)
        self.label=nn.Embedding(2,e)
        self.net=nn.Sequential(nn.Linear(features+2*e,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,features))
    def forward(self,x,t,y):
        return self.net(torch.cat([x,self.time(t),self.label(y)],dim=1))

def generate(x,y,counts,seed,cfg,checkpoint):
    start=time.perf_counter()
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device=torch.device(cfg['device'] if cfg['device']!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    sc=StandardScaler().fit(logit(np.clip(x,cfg['epsilon'],1-cfg['epsilon'])))
    z=torch.tensor(sc.transform(logit(np.clip(x,cfg['epsilon'],1-cfg['epsilon']))),dtype=torch.float32,device=device)
    yy=torch.tensor(y,dtype=torch.long,device=device)
    model=Denoiser(x.shape[1],cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    beta=torch.linspace(cfg['beta_start'],cfg['beta_end'],cfg['steps'],device=device)
    alpha=1-beta; abar=torch.cumprod(alpha,dim=0)
    losses=[]
    for step in range(cfg['updates']):
        idx=torch.randint(len(z),(min(cfg['batch_size'],len(z)),),device=device)
        t=torch.randint(cfg['steps'],(len(idx),),device=device)
        noise=torch.randn_like(z[idx])
        noisy=abar[t,None].sqrt()*z[idx]+(1-abar[t,None]).sqrt()*noise
        loss=(model(noisy,t,yy[idx])-noise).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss; no numerical retries')
        opt.zero_grad();loss.backward();opt.step()
        if step%20==0 or step==cfg['updates']-1: losses.append({'update':step+1,'loss':float(loss.detach().cpu())})
    model.eval()
    labels=np.repeat([0,1],counts)
    # Sampling stream is independent of training minibatch consumption.
    torch.manual_seed(seed+100000)
    with torch.no_grad():
        v=torch.randn((len(labels),x.shape[1]),device=device)
        cy=torch.tensor(labels,dtype=torch.long,device=device)
        for i in reversed(range(cfg['steps'])):
            t=torch.full((len(v),),i,dtype=torch.long,device=device)
            mean=(v-beta[i]/torch.sqrt(1-abar[i])*model(v,t,cy))/torch.sqrt(alpha[i])
            if i:
                posterior=beta[i]*(1-abar[i-1])/(1-abar[i])
                v=mean+posterior.sqrt()*torch.randn_like(v)
            else: v=mean
    samples=expit(sc.inverse_transform(v.cpu().numpy().astype(float)))
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.cpu().state_dict(),'architecture':cfg,'features':x.shape[1],
        'logit_mean':sc.mean_,'logit_scale':sc.scale_,'seed':seed,'sampling_seed':seed+100000,
        'optimizer':opt.state_dict(),'updates':cfg['updates'],'selected_beta_input':True},checkpoint)
    if not np.isfinite(samples).all(): raise FloatingPointError('Invalid generation; no retries/discarding')
    return samples,labels,{'losses':losses,'seconds':time.perf_counter()-start,
                          'parameters':sum(p.numel() for p in model.parameters()),'device':str(device)}
