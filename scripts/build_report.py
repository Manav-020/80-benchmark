"""Rebuild tables, figures, HTML and README results without fitting any model."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from methylation_diffusion.reporting import build
from methylation_diffusion.common import ROOT,fingerprint

figdir=ROOT/'reports/figures'
figdir.mkdir(parents=True,exist_ok=True)
losses=[];corr=[];boundary=[]
for p in sorted((ROOT/'outputs/diagnostics/primary').glob('*diffusion*.json')):
    metric=ROOT/'outputs/metrics/primary'/p.name
    if not metric.exists(): continue
    if json.loads(metric.read_text()).get('fingerprint')!=fingerprint(): continue
    d=json.loads(p.read_text())
    losses.append(pd.Series({v['update']:v['loss'] for v in d['losses']}))
    for v in d['diagnostics'].values():
        corr.append(v['correlation_mean_abs_difference'])
        boundary.append(v['boundary_fraction_epsilon_1e-4'])
if losses:
    frame=pd.DataFrame(losses)
    fig,ax=plt.subplots(figsize=(8,4))
    xx=frame.columns.to_numpy(dtype=float)
    ax.plot(xx,frame.mean(0),label='Mean across fold/seed fits')
    ax.fill_between(xx,frame.quantile(.1),frame.quantile(.9),alpha=.2,label='10th–90th percentile, descriptive')
    ax.set(xlabel='Optimizer update',ylabel='Training noise-prediction MSE',title='Training loss is not evidence of useful synthesis')
    ax.legend();fig.tight_layout();fig.savefig(figdir/'training_losses.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    axes[0].hist([v for v in corr if v is not None],bins=20);axes[0].set(xlabel='Mean absolute correlation difference',ylabel='Class-conditioned fits')
    axes[1].hist(boundary,bins=20);axes[1].set(xlabel='Beta boundary concentration',ylabel='Class-conditioned fits')
    fig.tight_layout();fig.savefig(figdir/'synthesis_quality.png');plt.close(fig)
neighbors=ROOT/'reports/all_training_neighbor_diagnostics.csv'
if neighbors.exists():
    nd=pd.read_csv(neighbors)
    nd[['generated_to_all_training_nearest_mean','real_to_all_training_nearest_mean']].plot.box(figsize=(8,4),ylabel='Euclidean distance in beta space; all real training rows')
    plt.tight_layout();plt.savefig(figdir/'all_training_neighbor_distances.png');plt.close()
build()

summary=pd.read_csv(ROOT/'reports/summary.csv')
pool=pd.read_csv(ROOT/'reports/pooled_metrics.csv')
pool[pool.complete].groupby(['experiment','method','seed'])[['balanced_accuracy','accuracy','sensitivity','specificity','roc_auc']].mean().reset_index().to_csv(ROOT/'reports/seed_average_metrics.csv',index=False)

def markdown(table):
    rows=[list(table.columns)]+[[f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row] for row in table.itertuples(index=False,name=None)]
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]])

report=ROOT/'reports/final_report.md'
text=report.read_text(encoding='utf-8')
cols=['experiment','method','balanced_accuracy','matched_mean_ba_difference','accuracy','sensitivity','specificity','roc_auc','benchmark_80_mean']
for mask in [summary.experiment.str.startswith('GSE99511'),summary.experiment.str.startswith('GSE46306'),summary.experiment.str.startswith('transfer')]:
    table=summary.loc[mask,cols]
    # Replace the exact plain-text serialization with a proper Markdown table.
    text=text.replace(table.to_string(index=False),markdown(table))
# Float parsing can change default display precision; reconstruct sections independently if needed.
start=text.index('\n## GSE99511 within-dataset')
end=text.index('\n\nGSE',start) if '\n\nGSE' in text[start:] else -1
# Preserve the narrative by finding its first generated diffusion sentence.
needle=next((f'\n\n{r.experiment}: diffusion' for r in summary[summary.method=='diffusion'].itertuples() if f'\n\n{r.experiment}: diffusion' in text),None)
if needle:
    end=text.index(needle,start)
    sections=[]
    for title,mask in [('GSE99511 within-dataset',summary.experiment.str.startswith('GSE99511')),('GSE46306 within-dataset',summary.experiment.str.startswith('GSE46306')),('Cross-dataset transfer',summary.experiment.str.startswith('transfer'))]:
        sections.append('\n## '+title+'\n\n'+markdown(summary.loc[mask,cols])+'\n')
    text=text[:start]+''.join(sections)+text[end:]
text+='\n\nThe local GSE46306 series gzip is truncated. Readable metadata supplies labels; the separate processed supplementary matrix supplies modeling values. See data_audit.json and source_hashes.json.\n'
report.write_text(text,encoding='utf-8')
hp=ROOT/'reports/results.html'
htext=hp.read_text(encoding='utf-8')
htext=htext.replace('</body>','<h2>Additional provenance and verification</h2><p>The local GSE46306 series gzip is truncated. Its readable metadata header supplies diagnoses and sample mappings; modeling uses the separate processed supplementary matrix, whose stream was read completely. Numerical scale checks support deposited beta values despite contradictory GEO VALUE text. The absent cg00979000 probe is not restored.</p><p>Nearest-neighbor diagnostics include both same-class and all-class real training references; saved checkpoint sampling streams were replayed and checked against original synthetic feature means. No held-out sample enters synthesis diagnostics. See artifact_verification.json, data_audit.json, full_cohort_audit.json, scale_checks.csv and source_hashes.json.</p></body>')
hp.write_text(htext,encoding='utf-8')

diff=summary[summary.method=='diffusion']
direction='lower' if (diff.matched_mean_ba_difference<0).all() else 'mixed'
qualified=summary[summary.benchmark_80_mean]
names=', '.join(f'{r.experiment}/{r.method} ({r.balanced_accuracy:.2%})' for r in qualified.itertuples())
dp=pool[(pool.method=='diffusion') & pool.complete & (pool.balanced_accuracy>=.8)]
conclusion=f'Diffusion mean held-out balanced accuracy was {direction} than real-only in all {len(diff)} evaluated protocols. Mean >=80% balanced accuracy was observed for: {names}. Diffusion reached >=80% in {len(dp)} individual pooled repeat/seed evaluations, all visible in pooled_metrics.csv; none of its protocol means reached 80%. These descriptive results do not establish a useful augmentation benefit.'
diag=pd.read_csv(ROOT/'reports/diagnostic_summary.csv')
extra=f"Across {len(diag)//2} diffusion fits, invalid synthetic values totaled {int(diag.invalid_count.sum())}; same-class exact and near-real duplicate row counts totaled {int(diag.exact_real_duplicate_rows.sum())} and {int(diag['near_real_duplicate_rows_maxabs_1e-6'].sum())}. Maximum boundary fraction was {diag['boundary_fraction_epsilon_1e-4'].max():.4f}. Absence of these flags does not establish biological validity or privacy."
report.write_text(report.read_text(encoding='utf-8')+'\n\n'+conclusion+'\n\n'+extra+'\n',encoding='utf-8')
import html
hp.write_text(hp.read_text(encoding='utf-8').replace('</body>','<h2>Observed conclusions</h2><p>'+html.escape(conclusion)+'</p><p>'+html.escape(extra)+'</p></body>'),encoding='utf-8')
import re
page=hp.read_text(encoding='utf-8')
intro=re.search(r'<pre>(.*?)</pre>',page,flags=re.S)
if intro:
    plain=html.unescape(intro.group(1)).replace('**','')
    paragraphs=plain.split('\n\n')
    intro_html=''.join('<p>'+html.escape(p.replace('\n',' '))+'</p>' for p in paragraphs if p.strip() and not p.startswith('# '))
    page=page[:intro.start()]+intro_html+page[intro.end():]
    hp.write_text(page,encoding='utf-8')

compact=summary.pivot(index='experiment',columns='method',values='balanced_accuracy').reset_index()
readme=ROOT/'README.md'
content=readme.read_text(encoding='utf-8').split('\n<!-- EXECUTED_RESULTS -->')[0]
content+='\n<!-- EXECUTED_RESULTS -->\n\n## Executed results\n\nMean pooled held-out balanced accuracy, averaging all predefined repeats/seeds. Strict GSE46306 is primary; reconciled is sensitivity.\n\n'+markdown(compact)+'\n\nSee reports/summary.csv for paired changes and descriptive variability, reports/seed_average_metrics.csv for every seed, and reports/verification.json for completeness.\n'
content+='\n'+conclusion+'\n'
readme.write_text(content,encoding='utf-8')
