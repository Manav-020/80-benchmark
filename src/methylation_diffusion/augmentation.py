import numpy as np
from scipy.spatial.distance import cdist,pdist
from .diffusion import generate

def augment(x,y,method,seed,cfg,checkpoint):
    rng=np.random.default_rng(seed)
    target=2*np.bincount(y,minlength=2).max()
    counts=[int(target-(y==c).sum()) for c in [0,1]]
    info={}
    if method=='real': return x,y,info
    if method=='diffusion': sx,sy,info=generate(x,y,counts,seed,cfg,checkpoint)
    else:
        chunks=[]
        for c in [0,1]:
            xc=x[y==c]; n=counts[c]
            if method=='random': chunk=xc[rng.integers(len(xc),size=n)]
            elif method=='smote':
                k=min(5,len(xc)-1)
                if k<1: raise ValueError('Class too small for SMOTE')
                dist=cdist(xc,xc);np.fill_diagonal(dist,np.inf)
                neighbors=np.argsort(dist,axis=1,kind='stable')[:,:k]
                base=rng.integers(len(xc),size=n)
                partner=neighbors[base,rng.integers(k,size=n)]
                chunk=xc[base]+rng.random((n,1))*(xc[partner]-xc[base])
            else: raise ValueError(method)
            chunks.append(chunk)
        sx=np.concatenate(chunks);sy=np.repeat([0,1],counts)
    ax=np.concatenate([x,sx]);ay=np.concatenate([y,sy])
    assert np.array_equal(np.bincount(ay),[target,target])
    assert np.array_equal(ax[:len(x)],x)
    if method=='diffusion': info['diagnostics']=diagnose(x,y,sx,sy)
    info['real_counts']=np.bincount(y).tolist(); info['final_counts']=np.bincount(ay).tolist()
    return ax,ay,info

def diagnose(x,y,sx,sy):
    out={}
    for c in [0,1]:
        a=x[y==c];b=sx[sy==c]
        distances=cdist(b,a)
        rr=cdist(a,a);np.fill_diagonal(rr,np.inf)
        near=cdist(b,a,metric='chebyshev')
        with np.errstate(invalid='ignore',divide='ignore'):
            ca=np.corrcoef(a,rowvar=False);cb=np.corrcoef(b,rowvar=False)
        upper=np.triu_indices(a.shape[1],1)
        ok=np.isfinite(ca[upper]) & np.isfinite(cb[upper])
        corr_diff=float(np.mean(abs(ca[upper][ok]-cb[upper][ok]))) if ok.any() else None
        out[str(c)]={'real_mean':a.mean(0).tolist(),'synthetic_mean':b.mean(0).tolist(),
          'real_std':a.std(0).tolist(),'synthetic_std':b.std(0).tolist(),
          'correlation_mean_abs_difference':corr_diff,'finite_correlation_pairs':int(ok.sum()),
          'real_correlation':np.nan_to_num(ca).tolist(),'synthetic_correlation':np.nan_to_num(cb).tolist(),
          'synthetic_to_real_nearest_euclidean':distances.min(1).tolist(),
          'real_to_real_nearest_euclidean_excluding_self':rr.min(1).tolist(),
          'exact_real_duplicate_rows':int((near.min(1)==0).sum()),
          'near_real_duplicate_rows_maxabs_1e-6':int((near.min(1)<=1e-6).sum()),
          'synthetic_internal_exact_duplicate_pairs':int((pdist(b,metric='chebyshev')==0).sum()),
          'synthetic_internal_near_duplicate_pairs_1e-6':int((pdist(b,metric='chebyshev')<=1e-6).sum()),
          'boundary_fraction_epsilon_1e-4':float(((b<=1e-4)|(b>=1-1e-4)).mean()),
          'excessive_boundary_flag_gt_10pct':bool(((b<=1e-4)|(b>=1-1e-4)).mean()>.1),
          'invalid_count':int((~np.isfinite(b)).sum())}
    return out
