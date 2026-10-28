import numpy as np
from sklearn.model_selection import StratifiedKFold
from .common import ROOT,config,save_json
from .data import load_data

def make_splits(datasets=None):
    if datasets is None: datasets=load_data()[0]
    cfg=config('protocol');splits=[]
    def add(experiment,source,target,train,test,repeat,fold):
        sm=datasets[source][1];tm=datasets[target][1]
        tr=sm.iloc[train]['sample'].tolist();te=tm.iloc[test]['sample'].tolist()
        assert not set(tr)&set(te)
        splits.append(dict(experiment=experiment,source=source,target=target,train=tr,test=te,repeat=repeat,fold=str(fold)))
    m=datasets['GSE99511'][1]
    for chip in sorted(m.chip.unique()):
        add('GSE99511_loco','GSE99511','GSE99511',np.flatnonzero(m.chip!=chip),np.flatnonzero(m.chip==chip),0,chip)
    for source in ['GSE99511','GSE46306_strict']:
        m=datasets[source][1]
        for repeat,seed in enumerate(cfg['cv_seeds']):
            for fold,(tr,te) in enumerate(StratifiedKFold(cfg['folds'],shuffle=True,random_state=seed).split(m,m.label)):
                add(source+'_cv',source,source,tr,te,repeat,fold)
                if source=='GSE46306_strict':
                    rm=datasets['GSE46306_reconciled'][1]
                    # Preserve every original assignment; added CIN3 cycles over fold 0,1,2 by repeat.
                    assigned=repeat%cfg['folds']
                    train=list(m.iloc[tr]['sample']);test=list(m.iloc[te]['sample'])
                    (test if fold==assigned else train).append('GSM1128506')
                    idx={s:i for i,s in enumerate(rm['sample'])}
                    add('GSE46306_reconciled_cv','GSE46306_reconciled','GSE46306_reconciled',
                        [idx[s] for s in train],[idx[s] for s in test],repeat,fold)
    for exp,source,target in [('transfer_99511_strict','GSE99511','GSE46306_strict'),
        ('transfer_99511_reconciled','GSE99511','GSE46306_reconciled'),('transfer_46306_99511','GSE46306_strict','GSE99511')]:
        add(exp,source,target,np.arange(len(datasets[source][1])),np.arange(len(datasets[target][1])),0,'all')
    save_json(ROOT/'outputs/splits/splits.json',splits)
    print(f'Saved {len(splits)} train/test partitions',flush=True)
    return splits
