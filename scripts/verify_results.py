"""Independently inspect real-fit statistics, checkpoint budgets and OOF coverage."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from scipy.special import expit
from scipy.spatial.distance import cdist
from methylation_diffusion.diffusion import Denoiser
from methylation_diffusion.common import ROOT,config,fingerprint,save_json
from methylation_diffusion.data import load_data
from methylation_diffusion.evaluation import metrics

datasets=load_data()[0]
fp=fingerprint()
cfg=config('diffusion')
parts=json.loads((ROOT/'outputs/splits/splits.json').read_text())
records={}
for p in (ROOT/'outputs/metrics/primary').glob('*.json'):
    r=json.loads(p.read_text())
    if r.get('fingerprint')==fp and r.get('status')=='complete': records[r['run_id']]=r
assert len(records)==len(parts)*10, f'Incomplete suite: {len(records)}/{len(parts)*10}'
budgets=[]
neighbor_rows=[]
torch.set_num_threads(1)
for s in parts:
    sid=f"{s['experiment']}_r{s['repeat']}_f{s['fold']}"
    runs=[r for k,r in records.items() if k.startswith(sid+'_')]
    assert len(runs)==10
    base=next(r for r in runs if r['method']=='real')
    x,m=datasets[s['source']]
    y=m.set_index('sample').loc[s['train'],'label'].values
    assert not set(s['train'])&set(s['test'])
    a=x.loc[s['train'],base['selected_cpgs']].values.astype(float)
    median=np.nanmedian(a,axis=0)
    a=np.where(np.isnan(a),median,a)
    assert np.allclose(median,base['train_medians_selected'])
    assert np.allclose(a.mean(0),base['scaler_mean'])
    scale=a.std(0);scale[scale==0]=1
    assert np.allclose(scale,base['scaler_scale'])
    z=logit(np.clip(a,cfg['epsilon'],1-cfg['epsilon']))
    target=2*np.bincount(y).max()
    for r in runs:
        assert r['preprocessing_fit_samples']==s['train']
        assert r['selected_cpgs']==base['selected_cpgs']
        assert r['scaler_mean']==base['scaler_mean'] and r['scaler_scale']==base['scaler_scale']
        if r['method']!='real': assert r['final_train_n']==2*target
        p=pd.read_csv(ROOT/f"outputs/predictions/primary/{r['run_id']}.csv")
        assert p['sample'].tolist()==s['test']
        computed=metrics(p.label,p.probability)
        for k,v in computed.items():
            assert (v is None and r['metrics'][k] is None) or np.isclose(v,r['metrics'][k])
        if r['method']=='diffusion':
            ck=torch.load(ROOT/f"outputs/checkpoints/primary/{r['run_id']}.pt",map_location='cpu',weights_only=False)
            assert ck['updates']==2000 and ck['architecture']==cfg
            assert np.allclose(ck['logit_mean'],z.mean(0))
            zs=z.std(0);zs[zs==0]=1
            assert np.allclose(ck['logit_scale'],zs)
            d=json.loads((ROOT/f"outputs/diagnostics/primary/{r['run_id']}.json").read_text())
            assert d['losses'][-1]['update']==2000
            assert all(v['invalid_count']==0 for v in d['diagnostics'].values())
            # Replay the saved sampling stream (no refitting, filtering or fresh seed).
            net=Denoiser(a.shape[1],cfg)
            net.load_state_dict(ck['state_dict']);net.eval()
            torch.manual_seed(ck['sampling_seed'])
            counts=[int(target-(y==c).sum()) for c in [0,1]]
            sy=np.repeat([0,1],counts)
            cy=torch.tensor(sy,dtype=torch.long)
            beta=torch.linspace(cfg['beta_start'],cfg['beta_end'],cfg['steps'])
            alpha=1-beta;abar=torch.cumprod(alpha,0)
            with torch.no_grad():
                v=torch.randn((len(sy),a.shape[1]))
                for i in reversed(range(cfg['steps'])):
                    t=torch.full((len(v),),i,dtype=torch.long)
                    mean=(v-beta[i]/torch.sqrt(1-abar[i])*net(v,t,cy))/torch.sqrt(alpha[i])
                    v=mean+(beta[i]*(1-abar[i-1])/(1-abar[i])).sqrt()*torch.randn_like(v) if i else mean
            generated=expit(v.numpy().astype(float)*ck['logit_scale']+ck['logit_mean'])
            full_rr=cdist(a,a);np.fill_diagonal(full_rr,np.inf)
            all_neighbors={}
            for c in [0,1]:
                b=generated[sy==c]
                assert np.allclose(b.mean(0),d['diagnostics'][str(c)]['synthetic_mean'],atol=1e-12)
                dist=cdist(b,a);near=cdist(b,a,metric='chebyshev')
                extra={'synthetic_to_all_real_training_nearest_euclidean':dist.min(1).tolist(),
                    'real_to_all_real_training_nearest_excluding_self':full_rr[y==c].min(1).tolist(),
                    'exact_real_duplicate_rows_all_classes':int((near.min(1)==0).sum()),
                    'near_real_duplicate_rows_all_classes_maxabs_1e-6':int((near.min(1)<=1e-6).sum())}
                all_neighbors[str(c)]=extra
                neighbor_rows.append({'run_id':r['run_id'],'experiment':s['experiment'],'seed':r['seed'],'class':c,
                    'generated_to_all_training_nearest_mean':float(dist.min(1).mean()),
                    'real_to_all_training_nearest_mean':float(full_rr[y==c].min(1).mean()),
                    'exact_duplicates_all_classes':extra['exact_real_duplicate_rows_all_classes'],
                    'near_duplicates_all_classes':extra['near_real_duplicate_rows_all_classes_maxabs_1e-6']})
            d['all_training_neighbor_diagnostics_replayed_from_checkpoint']=all_neighbors
            save_json(ROOT/f"outputs/diagnostics/primary/{r['run_id']}.json",d)
            budgets.append(ck['updates'])
pd.DataFrame(neighbor_rows).to_csv(ROOT/'reports/all_training_neighbor_diagnostics.csv',index=False)
save_json(ROOT/'reports/artifact_verification.json',{'completed_runs':len(records),'partitions':len(parts),
    'diffusion_fits':len(budgets),'all_generator_updates':2000,'matched_features_and_scalers':True,
    'classifier_statistics_match_real_training_rows':True,'generator_statistics_match_real_training_rows':True,
    'train_test_disjoint':True,'metrics_match_predictions':True,'sampler_replay_matches_saved_synthetic_means':True,
    'all_class_training_neighbor_checks_saved':True,'fingerprint':fp})
print('Verified all fitted statistics, checkpoint budgets, matched counts and predictions.')
