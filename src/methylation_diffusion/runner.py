import argparse
import json
import time
import traceback
import importlib.metadata
import platform
import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .common import ROOT,config,save_json,fingerprint,digest
from .data import load_data
from .preprocessing import TrainPreprocessor
from .augmentation import augment
from .evaluation import metrics

def run(smoke=False,limit=None,experiment=None):
    cfg=config('protocol');dcfg=config('diffusion')
    if smoke: dcfg={**dcfg,'updates':5}
    for source in json.loads((ROOT/'outputs/audits/sources.json').read_text()):
        if digest(source['filename'])!=source['sha256']:
            raise RuntimeError('Source hash changed; rerun audit and splits before resuming: '+source['filename'])
    datasets=load_data()[0]
    splits=json.loads((ROOT/'outputs/splits/splits.json').read_text())
    if experiment: splits=[s for s in splits if s['experiment']==experiment]
    if smoke:
        splits=[next(s for s in splits if s['experiment']==e) for e in ['GSE99511_loco','GSE46306_strict_cv','transfer_46306_99511']]
    if limit: splits=splits[:limit]
    tag='smoke' if smoke else 'primary'
    fp=fingerprint()
    save_json(ROOT/f'outputs/logs/{tag}_environment.json',{'python':platform.python_version(),
        'platform':platform.platform(),'torch_cuda':torch.cuda.is_available(),
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','pyarrow']},
        'fingerprint':fp,'smoke_non_scientific':smoke,'diffusion':dcfg})
    started=time.perf_counter();completed=0
    for split in splits:
        sid=f"{split['experiment']}_r{split['repeat']}_f{split['fold']}"
        sx,sm=datasets[split['source']];tx,tm=datasets[split['target']]
        names=sx.columns if split['source']==split['target'] else sx.columns.intersection(tx.columns).sort_values()
        tr=split['train'];te=split['test'];assert not set(tr)&set(te)
        if split['experiment']=='GSE99511_loco':
            mm=sm.set_index('sample');assert not set(mm.loc[tr,'chip'])&set(mm.loc[te,'chip'])
        y=sm.set_index('sample').loc[tr,'label'].values.astype(int)
        yt=tm.set_index('sample').loc[te,'label'].values.astype(int)
        prep=TrainPreprocessor(cfg['features'],cfg['missingness_max'],cfg['variance_min']).fit(sx.loc[tr,names].values,y,names)
        a=prep.transform(sx.loc[tr,names].values);b=prep.transform(tx.loc[te,names].values)
        scaler=StandardScaler().fit(a)
        for method in config('experiments')['methods']:
            seeds=[0] if method=='real' else cfg['augmentation_seeds']
            if smoke: seeds=[0] if method=='real' else [17]
            for seed in seeds:
                run_id=f'{sid}_{method}_s{seed}'
                result=ROOT/f'outputs/metrics/{tag}/{run_id}.json'
                predpath=ROOT/f'outputs/predictions/{tag}/{run_id}.csv'
                if result.exists() and predpath.exists():
                    old=json.loads(result.read_text())
                    if old.get('fingerprint')==fp and old.get('status')=='complete': continue
                t=time.perf_counter()
                record={**split,'method':method,'seed':seed,'fingerprint':fp,'run_id':run_id,
                        'selected_cpgs':prep.names.tolist(),'preprocessing_fit_samples':tr,
                        'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
                        'train_medians_selected':prep.median[prep.selected].tolist(),
                        'source_schema_probes':len(names),'smoke_non_scientific':smoke}
                try:
                    ck=ROOT/f'outputs/checkpoints/{tag}/{run_id}.pt'
                    with threadpool_limits(limits=1):
                        ax,ay,info=augment(a,y,method,seed,dcfg,ck)
                        model=LinearDiscriminantAnalysis(**cfg['classifier']).fit(scaler.transform(ax),ay)
                        prob=model.predict_proba(scaler.transform(b))[:,1]
                    if not np.isfinite(prob).all(): raise FloatingPointError('Invalid probabilities')
                    prediction=pd.DataFrame({'sample':te,'label':yt,'probability':prob,
                         'predicted':(prob>=cfg['threshold']).astype(int),'chip':tm.set_index('sample').loc[te,'chip'].values})
                    predpath.parent.mkdir(parents=True,exist_ok=True)
                    prediction.to_csv(predpath,index=False)
                    record.update(status='complete',metrics=metrics(yt,prob),seconds=time.perf_counter()-t,
                                  real_train_n=len(a),final_train_n=len(ax))
                    if method=='diffusion':
                        # Checkpoint sidecar carries feature IDs, real-fit preprocessing and run identity.
                        save_json(ck.with_suffix('.json'),{**record,'diffusion_config':dcfg})
                    save_json(ROOT/f'outputs/diagnostics/{tag}/{run_id}.json',info)
                    completed+=1
                    print(f"{run_id}: BA={record['metrics']['balanced_accuracy']} seconds={record['seconds']:.2f}",flush=True)
                except Exception as e:
                    record.update(status='failed',error=str(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-t)
                    print(f'FAILED {run_id}: {e}',flush=True)
                save_json(result,record)
    print(f'{tag}: {completed} newly completed runs in {time.perf_counter()-started:.1f}s',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--experiment')
    args=p.parse_args();run(args.smoke,args.limit,args.experiment)

if __name__=='__main__': main()
