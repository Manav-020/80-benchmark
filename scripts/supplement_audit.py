"""Additional full-cohort checks; no writes to the external data root."""
import gzip
import json
import numpy as np
import pandas as pd
from methylation_diffusion.common import ROOT,data_root,save_json

hdr=json.loads((ROOT/'outputs/audits/series_metadata.json').read_text())
labels=next(r for r in hdr['!Sample_characteristics_ch1'] if any('disease status:' in v for v in r))
counts=pd.Series([v.split(': ',1)[-1] if v else 'missing diagnosis' for v in labels]).value_counts().to_dict()
path=data_root()/'GSE46306/GSE46306_Matrix_processed2.txt.gz'
with gzip.open(path,'rt') as f:
    h=f.readline().rstrip('\n').split('\t')
start=h.index('samplename')+1
vals=pd.read_csv(path,sep='\t',skiprows=2,header=None,usecols=list(range(start,start+44)),dtype=float)
assert vals.shape[1]==44 and np.isfinite(vals.values).all() and vals.values.min()>=0 and vals.values.max()<=1
assert counts=={'normal':20,'cin3':17,'cancer':6,'missing diagnosis':1}
save_json(ROOT/'reports/full_cohort_audit.json',{'GSE46306_total_samples':44,'series_diagnosis_counts':counts,
    'processed_total_rows':len(vals),'all44_beta_min':float(vals.values.min()),'all44_beta_max':float(vals.values.max()),
    'all44_missing_values':int(vals.isna().sum().sum()),'processed_gzip_complete':True})
print('Verified all 44 processed samples and strict/cancer/missing-diagnosis counts.')
