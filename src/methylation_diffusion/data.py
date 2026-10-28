"""Read deposited files without modifying the external data tree."""
import csv
import gzip
import re
import numpy as np
import pandas as pd
from .common import ROOT, data_root, digest, save_json

def series_header(path):
    rows = {}
    with gzip.open(path, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            if not row: continue
            if row[0] == '!series_matrix_table_begin':
                break
            rows.setdefault(row[0], []).append(row[1:])
    return rows

def load_data():
    root = data_root()
    meta = pd.read_csv(root/'GSE99511/metadata/sample_metadata.csv', dtype=str)
    evidence = pd.read_csv(root/'GSE99511/metadata/metadata.csv', dtype=str).set_index('gsm_id')
    assert meta.GSM_ID.is_unique and meta.Sentrix_Key.is_unique
    assert (evidence.loc[meta.GSM_ID, 'disease'].values == meta.Diagnosis.values).all()
    beta = pd.read_parquet(root/'GSE99511/intermediate/beta_values.parquet')
    assert beta.index.is_unique and beta.columns.is_unique
    assert set(beta.columns) == set(meta.Sentrix_Key)
    meta = meta[meta.Diagnosis.isin(['Normal', 'CIN3'])].copy()
    assert meta.Diagnosis.value_counts().to_dict() == {'CIN3':36, 'Normal':28}
    a = beta.loc[beta.index.str.startswith('cg'), meta.Sentrix_Key].T
    a.index = meta.GSM_ID.values
    ma = pd.DataFrame({'sample':meta.GSM_ID.values, 'label':(meta.Diagnosis=='CIN3').astype(int).values,
                       'chip':meta.Sentrix_ID.values, 'author':evidence.loc[meta.GSM_ID,'sample_title'].values})
    assert ma.chip.nunique() == 6
    del beta
    series = root/'GSE46306/GSE46306_series_matrix.txt.gz'
    hdr = series_header(series)
    ids = hdr['!Sample_geo_accession'][0]
    titles = hdr['!Sample_title'][0]
    authors = [re.search(r'NT2_(\d+)\.', t).group(1) for t in titles]
    disease = next(r for r in hdr['!Sample_characteristics_ch1'] if any('disease status:' in v for v in r))
    labels = [v.split(': ',1)[1] if ': ' in v else '' for v in disease]
    mb = pd.DataFrame({'sample':ids, 'author':authors, 'diagnosis':labels, 'chip':'unverified'})
    assert len(mb)==44 and mb['sample'].is_unique and mb.author.is_unique
    proc = root/'GSE46306/GSE46306_Matrix_processed2.txt.gz'
    with gzip.open(proc,'rt') as f:
        h1 = f.readline().rstrip('\n').split('\t')
        h2 = f.readline().rstrip('\n').split('\t')
    start = h1.index('samplename') + 1
    names = h1[start:start+44]
    groups = h2[start:start+len(names)]
    assert len(names)==44 and len(set(names))==44 and set(names)==set(authors)
    group_map = dict(zip(names,groups))
    assert group_map['2010041']=='g1'
    for r in mb.itertuples():
        if r.diagnosis:
            assert group_map[r.author] == {'normal':'g0','cin3':'g1','cancer':'g2'}[r.diagnosis]
    # Ignore all deposited supervised summary/ranking columns; read individual values only.
    b = pd.read_csv(proc, sep='\t', skiprows=2, header=None,
                    usecols=[0]+list(range(start,start+44)), na_values=['NA'], low_memory=False)
    b.columns = ['probe'] + names
    b = b.set_index('probe')
    assert b.index.is_unique
    b = b.loc[b.index.str.startswith('cg'), authors].astype(np.float32).T
    b.index = ids
    mb['strict'] = mb.diagnosis.isin(['normal','cin3'])
    mb['label'] = (mb.diagnosis=='cin3').astype(int)
    mb.loc[mb.author=='2010041','label'] = 1
    assert mb[mb.strict].label.value_counts().to_dict() == {0:20,1:17}
    rec = mb[mb.strict | (mb.author=='2010041')].copy().reset_index(drop=True)
    strict = mb[mb.strict].copy().reset_index(drop=True)
    return {'GSE99511':(a.astype(np.float32),ma.reset_index(drop=True)),
            'GSE46306_strict':(b.loc[strict['sample']],strict),
            'GSE46306_reconciled':(b.loc[rec['sample']],rec)}, (hdr, mb, start, names, groups)

def audit():
    datasets, extra = load_data()
    hdr, mb, start, names, groups = extra
    root = data_root()
    paths = [root/'GSE99511/intermediate/beta_values.parquet',root/'GSE99511/metadata/sample_metadata.csv',
             root/'GSE99511/metadata/metadata.csv'] + sorted((root/'GSE46306').glob('*.gz'))
    save_json(ROOT/'outputs/audits/sources.json', [dict(filename=str(p), bytes=p.stat().st_size,sha256=digest(p),
                accession_url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+('GSE99511' if 'GSE99511' in str(p) else 'GSE46306')) for p in paths])
    summary={}
    for name,(x,m) in datasets.items():
        assert x.index.is_unique and x.columns.is_unique and list(x.index)==list(m['sample'])
        v=x.values
        assert np.nanmin(v)>=0 and np.nanmax(v)<=1 and not np.isinf(v).any()
        summary[name]={'samples':len(m),'probes':x.shape[1],'normal':int((m.label==0).sum()),'cin3':int(m.label.sum()),
                       'min':float(np.nanmin(v)),'max':float(np.nanmax(v)),'missing':int(np.isnan(v).sum()),
                       'missing_fraction':float(np.isnan(v).mean()),'duplicate_sample_ids':0,'duplicate_probe_ids':0}
        m.to_csv(ROOT/f'outputs/audits/{name}_manifest.csv',index=False)
    common = datasets['GSE99511'][0].columns.intersection(datasets['GSE46306_strict'][0].columns)
    summary['shared_probes']=len(common)
    # Representative checks: first 12 deposited processed rows, all 44 samples.
    proc=root/'GSE46306/GSE46306_Matrix_processed2.txt.gz'
    rows=[]
    with gzip.open(proc,'rt') as f:
        next(f); next(f)
        for i in range(12): rows.append(next(f).rstrip('\n').split('\t'))
    checks=[]; wanted={r[0]:r for r in rows}; signals={}; special=None
    sig=root/'GSE46306/GSE46306_Matrix_signal_intensities2.txt.gz'
    with gzip.open(sig,'rt') as f:
        for line in f:
            if line.startswith('Index\t'): sh=line.rstrip('\n').split('\t'); break
        cols={re.search(r'NT2_(\d+)\.',v).group(1):i for i,v in enumerate(sh) if v.endswith('.AVG_Beta')}
        assert set(cols)==set(names)
        for line in f:
            r=line.rstrip('\n').split('\t')
            if r[1] in wanted: signals[r[1]]={n:float(r[j]) for n,j in cols.items()}
            if r[1]=='cg00979000': special={n:r[j] for n,j in cols.items()}
    for row in rows:
        vals=np.array([float(v) for v in row[start:start+44]])
        sg=np.array([signals[row[0]][n] for n in names])
        cin=np.array(groups)=='g1'; normal=np.array(groups)=='g0'
        checks.append({'probe':row[0],'signal_processed_max_abs_error':float(np.max(abs(vals-sg))),
                       'deposited_mean_cin3':float(row[6].replace(',','.')),
                       'arcsin_mean_cin3':float(np.arcsin(np.sqrt(vals[cin])).mean()),
                       'deposited_mean_normal':float(row[7].replace(',','.')),
                       'arcsin_mean_normal':float(np.arcsin(np.sqrt(vals[normal])).mean())})
    assert max(c['signal_processed_max_abs_error'] for c in checks)<1e-7
    assert max(abs(c['arcsin_mean_cin3']-c['deposited_mean_cin3']) for c in checks)<1e-6
    assert max(abs(c['arcsin_mean_normal']-c['deposited_mean_normal']) for c in checks)<1e-6
    pd.DataFrame(checks).to_csv(ROOT/'outputs/audits/scale_checks.csv',index=False)
    # Series values separately checked against processed values for representative probes.
    with gzip.open(root/'GSE46306/GSE46306_series_matrix.txt.gz','rt') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'): break
        series_values=pd.read_csv(f,sep='\t',index_col=0,comment='!',nrows=1000)
    xb=datasets['GSE46306_reconciled'][0]
    shared=series_values.index.intersection(xb.columns)
    sample_ids=list(xb.index)
    err=float(np.nanmax(abs(series_values.loc[shared,sample_ids].values.astype(float)-xb.loc[sample_ids,shared].values.T)))
    # float32 conversion accounts for at most ~3e-8 discrepancy.
    assert err<1e-6
    try:
        with gzip.open(root/'GSE46306/GSE46306_series_matrix.txt.gz','rb') as f:
            while f.read(8*1024*1024): pass
        integrity='valid gzip stream'
    except EOFError:
        integrity='TRUNCATED gzip: intact metadata header and first 1000 table rows checked; series matrix not used as modeling input'
    summary['series_integrity']=integrity
    summary['scale_evidence']={'representative_probes':len(checks),'representative_series_shared_probes':len(shared),
       'series_processed_max_abs_error':err,'decision':'Use deposited individual beta values directly',
       'cg00979000_in_processed': 'cg00979000' in xb.columns,
       'cg00979000_signal_missing_authors':[n for n,v in special.items() if v.strip() in ['NA','','NaN']],
       'reconciled_label':{'sample':'GSM1128506','author':'2010041','supplement_group':'g1','g1_verified_diagnosis':'cin3'}}
    summary['grouping']='GSE99511: verified Sentrix IDs; GSE46306: no independently verified chip IDs. Title position prefixes not used as chip identities. No explicit person IDs; unique author IDs, repeated persons cannot be ruled out.'
    summary['metadata_scale_contradiction']=hdr.get('!Sample_data_processing',[])[0][0]
    save_json(ROOT/'outputs/audits/audit.json',summary)
    save_json(ROOT/'outputs/audits/series_metadata.json',hdr)
    print(summary,flush=True)
    return datasets
