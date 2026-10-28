"""Build all scientific summaries from saved real held-out predictions."""
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,save_json,fingerprint
from .evaluation import metrics

def build():
    out=ROOT/'reports';out.mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    records=[];preds=[];failures=[];diagnostics=[]
    fp=fingerprint()
    for path in sorted((ROOT/'outputs/metrics/primary').glob('*.json')):
        r=json.loads(path.read_text())
        if r['fingerprint']!=fp: continue
        if r['status']!='complete': failures.append({k:r.get(k) for k in ['run_id','error','seconds']});continue
        p=pd.read_csv(ROOT/f"outputs/predictions/primary/{r['run_id']}.csv")
        check=metrics(p.label,p.probability)
        for key,value in check.items():
            saved=r['metrics'][key]
            assert (value is None and saved is None) or np.isclose(value,saved), (path,key)
        keys=['experiment','repeat','fold','method','seed','seconds']
        records.append({**{k:r[k] for k in keys},**check})
        for k in keys[:-1]: p[k]=r[k]
        preds.append(p)
        if r['method']=='diffusion':
            d=json.loads((ROOT/f"outputs/diagnostics/primary/{r['run_id']}.json").read_text())
            for c,v in d['diagnostics'].items():
                diagnostics.append({'run_id':r['run_id'],'experiment':r['experiment'],'seed':r['seed'],'class':c,
                    'correlation_mean_abs_difference':v['correlation_mean_abs_difference'],
                    'synthetic_nearest_mean':np.mean(v['synthetic_to_real_nearest_euclidean']),
                    'real_nearest_mean':np.mean(v['real_to_real_nearest_euclidean_excluding_self']),
                    **{k:v[k] for k in ['exact_real_duplicate_rows','near_real_duplicate_rows_maxabs_1e-6','boundary_fraction_epsilon_1e-4','invalid_count']},
                    'parameters':d['parameters'],'generator_seconds':d['seconds']})
    if not records: raise RuntimeError('No compatible primary predictions')
    fold=pd.DataFrame(records);allp=pd.concat(preds,ignore_index=True)
    pooled=[]
    for (ex,repeat,method,seed),g in allp.groupby(['experiment','repeat','method','seed']):
        assert not g['sample'].duplicated().any()
        # Completeness per repeat: prevent partial OOF tables masquerading as full results.
        expected=64 if ex.startswith('GSE99511') or ex=='transfer_46306_99511' else (38 if 'reconciled' in ex else 37)
        complete=len(g)==expected
        pooled.append(dict(experiment=ex,repeat=repeat,method=method,seed=seed,complete=complete,**metrics(g.label,g.probability)))
    pool=pd.DataFrame(pooled)
    summaries=[]
    for (ex,method),g in pool[pool.complete].groupby(['experiment','method']):
        row={'experiment':ex,'method':method,'pooled_evaluations':len(g),'participants_per_evaluation':int(g.n.iloc[0])}
        for k in ['balanced_accuracy','accuracy','sensitivity','specificity','roc_auc']:
            row[k]=g[k].mean();row[k+'_descriptive_sd']=g[k].std(ddof=0)
        row['benchmark_80_mean']=bool(row['balanced_accuracy']>=.8)
        row['benchmark_80_any']=bool(g.benchmark_80.any())
        base=pool[(pool.experiment==ex)&(pool.method=='real')&pool.complete].set_index('repeat')
        row['matched_mean_ba_difference']=np.mean([r.balanced_accuracy-base.loc[r.repeat,'balanced_accuracy'] for r in g.itertuples()]) if len(base) else np.nan
        row['total_seconds']=fold[(fold.experiment==ex)&(fold.method==method)].seconds.sum()
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    chip=fold[fold.experiment=='GSE99511_loco'].copy()
    # Explicit single-class chips remain undefined and visible.
    chipmeans=chip.groupby(['method','seed']).balanced_accuracy.mean().reset_index(name='unweighted_mean_chip_ba_excluding_undefined')
    chipmeans['defined_chip_count']=chip.groupby(['method','seed']).balanced_accuracy.count().values
    chipmeans['total_chip_count']=chip.groupby(['method','seed']).size().values
    for name,table in [('fold_metrics',fold),('pooled_metrics',pool),('summary',summary),('chip_metrics',chip),('chip_mean_metrics',chipmeans),('diagnostic_summary',pd.DataFrame(diagnostics)),('failed_runs',pd.DataFrame(failures,columns=['run_id','error','seconds']))]:
        table.to_csv(out/f'{name}.csv',index=False)
    if len(chip):
        pivot=chip.groupby(['fold','method']).balanced_accuracy.mean().unstack()
        pivot.plot.bar(figsize=(11,5),ylim=(0,1),ylabel='Balanced accuracy (undefined for single-class chips)')
        plt.tight_layout();plt.savefig(out/'figures/per_chip.png');plt.close()
    # Confusions use one prespecified prediction stream, avoiding repeated-person pseudo-counts.
    for ex in allp.experiment.unique():
        fig,axes=plt.subplots(1,4,figsize=(12,3))
        for ax,method in zip(axes,['real','random','smote','diffusion']):
            g=allp[(allp.experiment==ex)&(allp['method']==method)&(allp['repeat']==0)&(allp.seed==(0 if method=='real' else 17))]
            if len(g):
                mm=metrics(g.label,g.probability);cm=np.array([[mm['tn'],mm['fp']],[mm['fn'],mm['tp']]])
                ax.imshow(cm,cmap='Blues');ax.set_title(method)
                for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
                ax.set_xticks([0,1],['Normal','CIN3']);ax.set_yticks([0,1],['Normal','CIN3'])
        fig.suptitle(ex+'; repeat 0, augmentation seed 17');fig.tight_layout();fig.savefig(out/f'figures/confusion_{ex}.png');plt.close(fig)
    dg=pd.DataFrame(diagnostics)
    if len(dg):
        dg[['synthetic_nearest_mean','real_nearest_mean']].plot.box(figsize=(6,4),ylabel='Euclidean distance in selected beta space')
        plt.tight_layout();plt.savefig(out/'figures/synthesis_distances.png');plt.close()
    expected_runs=len(json.loads((ROOT/'outputs/splits/splits.json').read_text()))*10
    complete=len(records)==expected_runs and not failures
    intro=f'''# Exploratory diffusion augmentation results

Primary suite complete: **{complete}**. Compatible completed runs: {len(records)}/{expected_runs}; failed runs: {len(failures)}.
All results are exploratory: both cohorts were examined during earlier development. Smoke outputs are excluded.
ANOVA top 50, real-training-only preprocessing, shrinkage LDA, threshold 0.5. Each augmentation reaches twice the largest real class count per class. No tuning or seed selection.
Numbers below average complete pooled evaluations across prespecified repeats and augmentation seeds; baseline is evaluated once per split. Standard deviations are descriptive, not independent-person uncertainty. No inferential significance claims.

GSE99511 LOCO pools predictions over 64 real people separately for each seed. Chip metrics and the unweighted mean of defined chip balanced accuracies are separate CSVs. A chip with one true class has undefined balanced accuracy; its class recall and confusion counts remain reported. Repeated CV pools each repeat separately; it does not establish unseen-chip transfer for GSE46306.

The strict 37-sample GSE46306 cohort is primary. The 38-sample reconciled cohort is predefined sensitivity only. Original assignments are fixed; GSM1128506 is held out in fold equal to repeat index (0,1,2). Transfer uses shared probe IDs only and source-only learned statistics. Targets never train any component. These are exploratory transfer experiments, not untouched external confirmation.
'''
    sections=[]
    for title,mask in [('GSE99511 within-dataset',summary.experiment.str.startswith('GSE99511')),('GSE46306 within-dataset',summary.experiment.str.startswith('GSE46306')),('Cross-dataset transfer',summary.experiment.str.startswith('transfer'))]:
        cols=['experiment','method','balanced_accuracy','matched_mean_ba_difference','accuracy','sensitivity','specificity','roc_auc','benchmark_80_mean']
        table=summary.loc[mask,cols]
        sections.append((title,table))
    narrative=[]
    for r in summary[summary.method=='diffusion'].itertuples():
        direction='improved' if r.matched_mean_ba_difference>0 else ('degraded' if r.matched_mean_ba_difference<0 else 'equaled')
        narrative.append(f'{r.experiment}: diffusion {direction} real-only mean held-out balanced accuracy by {r.matched_mean_ba_difference:+.4f}; diffusion mean BA {r.balanced_accuracy:.4f}; >=80% mean flag {r.benchmark_80_mean}.')
    limitations='''Small cohorts, uncertain repeated-person identity, incompatible cohort sampling and preprocessing, and the contradictory GSE46306 scale description limit interpretation. GSE46306 Normal/ CIN3 also differs in HPV status. No causal explanation of chip effects is established. Generator losses and distances do not establish biological validity or privacy. Diffusion uses a custom small DDPM, not an official TabDDPM reproduction. Historical scores are not comparable protocols. No low-performing chip was removed. The linear 100-step schedule leaves nonzero terminal signal and may impair synthesis; this was fixed before training and not selected on outcomes.'''
    md=intro+'\n'.join('\n## '+title+'\n\n'+table.to_string(index=False)+'\n' for title,table in sections)+'\n\n'+'\n\n'.join(narrative)+'\n\n'+limitations
    (out/'final_report.md').write_text(md,encoding='utf-8')
    body='<h1>Exploratory methylation diffusion augmentation</h1><pre>'+html.escape(intro)+'</pre>'
    for title,table in sections: body+='<h2>'+title+'</h2>'+table.to_html(index=False,float_format=lambda v:f'{v:.6f}',border=0)
    body+=''.join('<p>'+html.escape(s)+'</p>' for s in narrative)+ '<p>'+limitations+'</p>'
    body+='<h2>Seed and repeat results</h2>'+pool.to_html(index=False,border=0)
    body+='<h2>Unweighted chip means</h2>'+chipmeans.to_html(index=False,border=0)
    for p in sorted((out/'figures').glob('*.png')): body+=f'<img style="max-width:100%" src="figures/{p.name}" alt="{p.stem}">'
    document='<!doctype html><html lang="en"><meta charset="utf-8"><title>Methylation diffusion results</title><style>body{font:16px system-ui;margin:32px;max-width:1500px}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}table{font-size:13px}</style><body>'+body+'</body></html>'
    (out/'results.html').write_text(document,encoding='utf-8')
    # Round-trip the HTML tables and compare values with CSV-backed source tables.
    from io import StringIO
    tables=pd.read_html(StringIO(document))
    for actual,(_,expected) in zip(tables,sections):
        assert len(actual)==len(expected)
        assert np.allclose(actual.balanced_accuracy,expected.balanced_accuracy,atol=1e-6)
    save_json(out/'verification.json',{'complete':complete,'completed_runs':len(records),'expected_runs':expected_runs,
        'failed_runs':len(failures),'metrics_recomputed_from_predictions':True,'html_tables_roundtrip_match':True,'fingerprint':fp})
    print(summary[['experiment','method','balanced_accuracy','matched_mean_ba_difference']].to_string(index=False),flush=True)
    print(f'Complete: {complete}; runs {len(records)}/{expected_runs}',flush=True)
