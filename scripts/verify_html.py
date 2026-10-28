"""Check report CSV agreement, local images, and rendering in Chrome/Edge."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from io import StringIO
import numpy as np
import pandas as pd
from methylation_diffusion.common import ROOT,save_json

report=ROOT/'reports/results.html'
document=report.read_text(encoding='utf-8')
summary=pd.read_csv(ROOT/'reports/summary.csv')
tables=pd.read_html(StringIO(document))
for actual,mask in zip(tables,[summary.experiment.str.startswith('GSE99511'),summary.experiment.str.startswith('GSE46306'),summary.experiment.str.startswith('transfer')]):
    expected=summary[mask]
    assert actual.experiment.tolist()==expected.experiment.tolist()
    assert actual.method.tolist()==expected.method.tolist()
    for metric in ['balanced_accuracy','matched_mean_ba_difference','accuracy','sensitivity','specificity','roc_auc']:
        assert np.allclose(actual[metric],expected[metric],atol=1e-6)
for source in re.findall(r'<img[^>]+src="([^"]+)"',document):
    assert (report.parent/source).is_file(),source
candidates=[shutil.which('google-chrome'),shutil.which('chromium'),
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']
browser=next((p for p in candidates if p and Path(p).exists()),None)
if browser is None: raise RuntimeError('Chrome/Edge needed for headless render verification')
logs=ROOT/'outputs/logs';logs.mkdir(parents=True,exist_ok=True)
command=[browser,'--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--allow-file-access-from-files',
    f'--user-data-dir={ROOT/"outputs/browser-profile"}','--window-size=1600,1100',
    f'--screenshot={logs/"report_screenshot.png"}','--dump-dom',report.as_uri()]
p=subprocess.run(command,capture_output=True,text=True,timeout=45,
    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
(logs/'browser_dom.html').write_text(p.stdout,encoding='utf-8')
assert p.returncode==0 and '540/540' in p.stdout and (logs/'report_screenshot.png').exists(),p.stderr[-1000:]
save_json(ROOT/'reports/browser_verification.json',{'browser':Path(browser).name,'returncode':p.returncode,
    'report_loaded':True,'screenshot_saved_locally':True,'all_images_exist':True,'six_report_metrics_match_csv':True})
print('HTML rendered; tables match CSVs and all image links resolve.')
