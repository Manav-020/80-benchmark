import numpy as np
from methylation_diffusion.preprocessing import TrainPreprocessor
from methylation_diffusion.augmentation import augment
from methylation_diffusion.common import config
from methylation_diffusion.evaluation import metrics

def test_train_only_selection_and_imputation():
    x=np.array([[.1,np.nan,.5,.2],[.2,.4,.5,.3],[.8,.6,.5,.7],[.9,.8,.5,.8]])
    prep=TrainPreprocessor(k=2).fit(x,np.array([0,0,1,1]),['cg1','cg2','cg3','cg4'])
    med=prep.median.copy();names=prep.names.copy()
    heldout=np.full((2,4),np.nan);z=prep.transform(heldout)
    assert np.isfinite(z).all() and np.array_equal(prep.median,med) and np.array_equal(names,prep.names)
    assert 'cg3' not in names

def test_methods_counts_bounds_determinism(tmp_path):
    rng=np.random.default_rng(9);x=rng.uniform(.01,.99,(9,5));y=np.array([0]*4+[1]*5)
    cfg={**config('diffusion'),'updates':3}
    for method in ['real','random','smote','diffusion']:
        a,b,_=augment(x,y,method,17,cfg,tmp_path/'a.pt')
        c,d,_=augment(x,y,method,17,cfg,tmp_path/'b.pt')
        assert np.array_equal(a,c) and np.array_equal(b,d)
        assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
        if method!='real': assert np.array_equal(np.bincount(b),[10,10])
        assert np.array_equal(a[:9],x)

def test_metrics():
    m=metrics([0,0,1,1],[.1,.6,.4,.9])
    assert m['balanced_accuracy']==.5 and m['tn']==1 and m['tp']==1
    assert metrics([0,0],[.2,.3])['balanced_accuracy'] is None

def test_saved_splits_and_fixed_sensitivity():
    import json
    import pytest
    from methylation_diffusion.common import ROOT
    if not (ROOT/'outputs/splits/splits.json').exists():
        pytest.skip('Integration check requires audit_data.py and make_splits.py with external research data')
    splits=json.loads((ROOT/'outputs/splits/splits.json').read_text())
    for s in splits: assert not set(s['train'])&set(s['test'])
    for s in splits:
        if s['experiment']=='GSE46306_strict_cv':
            r=next(t for t in splits if t['experiment']=='GSE46306_reconciled_cv' and t['repeat']==s['repeat'] and t['fold']==s['fold'])
            assert set(r['test'])-{'GSM1128506'}==set(s['test'])
            assert set(r['train'])-{'GSM1128506'}==set(s['train'])

def test_tie_breaking_and_fold_missingness():
    y=np.array([0,0,1,1])
    x=np.array([[.1,.1,np.nan],[.2,.2,np.nan],[.8,.8,.3],[.9,.9,.6]])
    p=TrainPreprocessor(k=1).fit(x,y,['cgZ','cgA','cgMissing'])
    assert p.names.tolist()==['cgA']
    assert 2 not in p.keep
    before=p.transform(x)
    p.transform(np.full((3,3),99.))
    assert np.array_equal(before,p.transform(x))
