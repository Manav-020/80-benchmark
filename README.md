# Diffusion augmentation for cervical DNA methylation

Does adding class-conditional diffusion samples improve real held-out Normal-versus-CIN3 classification compared with real-only training, random oversampling and SMOTE? This project implements and executes that comparison using ANOVA top 50 CpGs and shrinkage LDA. It is a custom PyTorch DDPM, **not an official TabDDPM reproduction**.

All results are exploratory: both datasets were examined during earlier development. An 80% balanced-accuracy benchmark is descriptive, never a required outcome. See [the generated report](reports/final_report.md), [HTML report](reports/results.html), and [aggregate CSV](reports/summary.csv) for actual completed results; smoke-test outputs are excluded.

## Data and cohorts

- [GSE99511](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE99511): 28 Normal, 36 CIN3; four SCC excluded using diagnosis metadata. Primary evaluation leaves each of six verified Sentrix chips out in turn. Secondary evaluation uses five-fold stratified CV repeated three times.
- [GSE46306](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE46306): primary strict cohort is 20 Normal and 17 explicit CIN3. Six invasive cancers are excluded. Sensitivity adds GSM1128506/author 2010041 only after verifying supplementary g1=CIN3, giving 38 participants. No reliable chip IDs were independently verified; repeated stratified CV does not demonstrate unseen-chip transfer.
- Transfer trains on GSE99511 and evaluates each GSE46306 cohort, then trains on strict GSE46306 and evaluates GSE99511. Shared probe IDs are a schema restriction; all fitted statistics and models use source samples only. These are exploratory transfer experiments.

The existing external data root is read-only. Large datasets are not copied into this repository. The GSE99511 reconstructed matrix precedes the reference pipeline's global missingness filter; this project learns that filter within each fold. GSE46306 processed supplementary individual values are used directly as beta, supported by signal-file and group-summary checks. Contradictory GEO scale text is retained as a limitation. The local series-matrix gzip is truncated; readable metadata and representative table rows are audited, while model inputs come from the separate processed supplement. cg00979000 is not restored.

## Windows installation

Use Python 3.10 or newer. The executed environment uses Python 3.10.16 and CPU PyTorch; the host also has a 6 GB RTX 3050, but CPU is appropriate for this small network. The exact executed dependency versions are in requirements.lock.txt.

```powershell
Set-Location 'D:\coding\python\ML RESEARCH PROJECT\diffusion+methylation'
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item configs\paths.example.yaml configs\paths.local.yaml
```

Set `data_root` in configs/paths.local.yaml to your external data directory; use forward slashes in YAML. The default sibling path matches this workspace. The loader discovers the verified files under GSE99511 and GSE46306 and fails on inconsistent mappings, labels, or scales. No synthetic substitute data are used. Local path configuration and participant-level outputs are ignored by Git.

## Reproduce and resume

```powershell
.\.venv\Scripts\python.exe scripts\audit_data.py
.\.venv\Scripts\python.exe scripts\supplement_audit.py
.\.venv\Scripts\python.exe scripts\make_splits.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_smoke_test.py
.\.venv\Scripts\python.exe scripts\run_experiments.py
.\.venv\Scripts\python.exe scripts\verify_results.py
.\.venv\Scripts\python.exe scripts\build_report.py
.\.venv\Scripts\python.exe scripts\verify_html.py
Start-Process reports\results.html
```

The same `run_experiments.py` command resumes compatible runs. It checks actual input hashes, saved splits, code and config fingerprints. Each completed fold/method/seed has a prediction CSV, metric JSON and diagnostics; diffusion adds checkpoint and preprocessing sidecar. Changing code or protocol invalidates reuse. Failed runs are recorded and retried on rerun. Smoke uses five updates and one augmentation seed, is non-scientific, and lives in separate output directories.

Run an individual protocol using `--experiment GSE99511_loco`, `GSE99511_cv`, `GSE46306_strict_cv`, `GSE46306_reconciled_cv`, `transfer_99511_strict`, `transfer_99511_reconciled` or `transfer_46306_99511`. `--limit 1` is a representative full-budget partition, not a complete study. Build the report after resuming the full suite. Report completeness is explicit.

## Pipeline in plain language

First hide real test samples. From the remaining real samples, remove probes with too much missing data, fill remaining gaps using training medians, remove constant probes, and select 50 sites associated with diagnosis. Fit a scaler to those real training values. Every method uses those same sites and that same scaler.

The real-only model learns from the original rows. Random oversampling repeats rows; SMOTE interpolates same-class neighbors. Diffusion learns to remove noise while conditioned on diagnosis, then generates new rows. All three augmentation methods preserve every real row and reach twice the largest original class count **per class**, using seeds 17, 42, 2026 without selecting a winner. Classifier preprocessing is separate from the generator's bounded-beta logit transform. Every scored sample is real and held out.

DDPM settings are fixed engineering choices: two width-128 hidden layers, timestep/class embeddings, 100 linear diffusion steps, AdamW lr=.001, batch <=32 and exactly 2000 optimizer updates. No test-based tuning, model-duration selection, sample rejection or numerical retry occurs. CPU is the fixed default; `device: auto` or `cuda` is available in configs/diffusion.yaml. Changing settings creates a different experiment fingerprint.

## Reading the outputs

reports/summary.csv compares methods, including paired baseline differences and >=80% flags. pooled_metrics.csv reports every repeat/seed. fold_metrics.csv preserves every held-out fold. chip_metrics.csv and chip_mean_metrics.csv distinguish pooled participant performance from unweighted chip averages. Single-class chip balanced accuracy is undefined, with class recall still reported. Confusion figures use repeat 0 and prespecified augmentation seed 17, so repeated predictions are not counted as extra people. Timing, failures and synthesis diagnostics are included. HTML tables are round-tripped and checked against CSV-backed data; saved metrics are recomputed from prediction files.

See [fixed protocol](docs/experiment_protocol.md), [methodology](docs/methodology.md), [provenance](docs/data_provenance.md), [limitations](docs/limitations.md) and notebooks/explore_results.ipynb. Low denoising loss or apparent synthetic similarity does not establish usefulness, biology or privacy. Small cohorts, unknown person IDs, cohort differences and preprocessing uncertainty limit conclusions. No low-performing chip is removed, no seed is selected, and historical scores are not comparable baselines.

## Publish manually

No remote is created and nothing is pushed by this project. No code license or dataset redistribution rights are assumed. Small aggregate reports and figures are suitable for review; datasets, local paths, participant-level predictions/manifests and checkpoints remain ignored. Review permissions and content before publishing.

Create an empty GitHub repository yourself, then replace YOUR_ACCOUNT and REPOSITORY below:

```powershell
git status --short
git diff --cached --stat
git add .
git diff --cached --stat
git commit -m "Implement exploratory methylation diffusion augmentation study"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/REPOSITORY.git
git push -u origin main
```

<!-- EXECUTED_RESULTS -->

## Executed results

Mean pooled held-out balanced accuracy, averaging all predefined repeats/seeds. Strict GSE46306 is primary; reconciled is sensitivity.

| experiment | diffusion | random | real | smote |
| --- | --- | --- | --- | --- |
| GSE46306_reconciled_cv | 0.6873 | 0.7537 | 0.7833 | 0.7525 |
| GSE46306_strict_cv | 0.7760 | 0.8013 | 0.8240 | 0.8124 |
| GSE99511_cv | 0.5335 | 0.5481 | 0.5820 | 0.5280 |
| GSE99511_loco | 0.5437 | 0.5642 | 0.5655 | 0.5966 |
| transfer_46306_99511 | 0.4418 | 0.4722 | 0.4583 | 0.4623 |
| transfer_99511_reconciled | 0.3926 | 0.3722 | 0.5083 | 0.4231 |
| transfer_99511_strict | 0.3882 | 0.3559 | 0.4838 | 0.4123 |

See reports/summary.csv for paired changes and descriptive variability, reports/seed_average_metrics.csv for every seed, and reports/verification.json for completeness.

Diffusion mean held-out balanced accuracy was lower than real-only in all 7 evaluated protocols. Mean >=80% balanced accuracy was observed for: GSE46306_strict_cv/random (80.13%), GSE46306_strict_cv/real (82.40%), GSE46306_strict_cv/smote (81.24%). Diffusion reached >=80% in 3 individual pooled repeat/seed evaluations, all visible in pooled_metrics.csv; none of its protocol means reached 80%. These descriptive results do not establish a useful augmentation benefit.
