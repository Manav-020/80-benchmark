# Exploratory diffusion augmentation results

Primary suite complete: **True**. Compatible completed runs: 540/540; failed runs: 0.
All results are exploratory: both cohorts were examined during earlier development. Smoke outputs are excluded.
ANOVA top 50, real-training-only preprocessing, shrinkage LDA, threshold 0.5. Each augmentation reaches twice the largest real class count per class. No tuning or seed selection.
Numbers below average complete pooled evaluations across prespecified repeats and augmentation seeds; baseline is evaluated once per split. Standard deviations are descriptive, not independent-person uncertainty. No inferential significance claims.

GSE99511 LOCO pools predictions over 64 real people separately for each seed. Chip metrics and the unweighted mean of defined chip balanced accuracies are separate CSVs. A chip with one true class has undefined balanced accuracy; its class recall and confusion counts remain reported. Repeated CV pools each repeat separately; it does not establish unseen-chip transfer for GSE46306.

The strict 37-sample GSE46306 cohort is primary. The 38-sample reconciled cohort is predefined sensitivity only. Original assignments are fixed; GSM1128506 is held out in fold equal to repeat index (0,1,2). Transfer uses shared probe IDs only and source-only learned statistics. Targets never train any component. These are exploratory transfer experiments, not untouched external confirmation.

## GSE99511 within-dataset

| experiment | method | balanced_accuracy | matched_mean_ba_difference | accuracy | sensitivity | specificity | roc_auc | benchmark_80_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSE99511_cv | diffusion | 0.5335 | -0.0485 | 0.5347 | 0.5432 | 0.5238 | 0.5412 | False |
| GSE99511_cv | random | 0.5481 | -0.0340 | 0.5486 | 0.5525 | 0.5437 | 0.5513 | False |
| GSE99511_cv | real | 0.5820 | 0.0000 | 0.5833 | 0.5926 | 0.5714 | 0.6055 | False |
| GSE99511_cv | smote | 0.5280 | -0.0540 | 0.5295 | 0.5401 | 0.5159 | 0.5587 | False |
| GSE99511_loco | diffusion | 0.5437 | -0.0218 | 0.5521 | 0.6111 | 0.4762 | 0.5858 | False |
| GSE99511_loco | random | 0.5642 | -0.0013 | 0.5677 | 0.5926 | 0.5357 | 0.5956 | False |
| GSE99511_loco | real | 0.5655 | 0.0000 | 0.5781 | 0.6667 | 0.4643 | 0.5804 | False |
| GSE99511_loco | smote | 0.5966 | 0.0311 | 0.6042 | 0.6574 | 0.5357 | 0.5962 | False |

## GSE46306 within-dataset

| experiment | method | balanced_accuracy | matched_mean_ba_difference | accuracy | sensitivity | specificity | roc_auc | benchmark_80_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GSE46306_reconciled_cv | diffusion | 0.6873 | -0.0960 | 0.6901 | 0.6358 | 0.7389 | 0.7657 | False |
| GSE46306_reconciled_cv | random | 0.7537 | -0.0296 | 0.7573 | 0.6852 | 0.8222 | 0.8019 | False |
| GSE46306_reconciled_cv | real | 0.7833 | 0.0000 | 0.7895 | 0.6667 | 0.9000 | 0.8231 | False |
| GSE46306_reconciled_cv | smote | 0.7525 | -0.0309 | 0.7573 | 0.6605 | 0.8444 | 0.8119 | False |
| GSE46306_strict_cv | diffusion | 0.7760 | -0.0480 | 0.7748 | 0.7908 | 0.7611 | 0.8551 | False |
| GSE46306_strict_cv | random | 0.8013 | -0.0227 | 0.8048 | 0.7582 | 0.8444 | 0.8730 | True |
| GSE46306_strict_cv | real | 0.8240 | 0.0000 | 0.8288 | 0.7647 | 0.8833 | 0.8912 | True |
| GSE46306_strict_cv | smote | 0.8124 | -0.0116 | 0.8168 | 0.7582 | 0.8667 | 0.8786 | True |

## Cross-dataset transfer

| experiment | method | balanced_accuracy | matched_mean_ba_difference | accuracy | sensitivity | specificity | roc_auc | benchmark_80_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transfer_46306_99511 | diffusion | 0.4418 | -0.0165 | 0.4688 | 0.6574 | 0.2262 | 0.3702 | False |
| transfer_46306_99511 | random | 0.4722 | 0.0139 | 0.5208 | 0.8611 | 0.0833 | 0.3459 | False |
| transfer_46306_99511 | real | 0.4583 | 0.0000 | 0.5156 | 0.9167 | 0.0000 | 0.3383 | False |
| transfer_46306_99511 | smote | 0.4623 | 0.0040 | 0.5156 | 0.8889 | 0.0357 | 0.3629 | False |
| transfer_99511_reconciled | diffusion | 0.3926 | -0.1157 | 0.3860 | 0.5185 | 0.2667 | 0.3569 | False |
| transfer_99511_reconciled | random | 0.3722 | -0.1361 | 0.3684 | 0.4444 | 0.3000 | 0.3690 | False |
| transfer_99511_reconciled | real | 0.5083 | 0.0000 | 0.5263 | 0.1667 | 0.8500 | 0.4306 | False |
| transfer_99511_reconciled | smote | 0.4231 | -0.0852 | 0.4298 | 0.2963 | 0.5500 | 0.3838 | False |
| transfer_99511_strict | diffusion | 0.3882 | -0.0956 | 0.3784 | 0.5098 | 0.2667 | 0.3446 | False |
| transfer_99511_strict | random | 0.3559 | -0.1279 | 0.3514 | 0.4118 | 0.3000 | 0.3417 | False |
| transfer_99511_strict | real | 0.4838 | 0.0000 | 0.5135 | 0.1176 | 0.8500 | 0.4000 | False |
| transfer_99511_strict | smote | 0.4123 | -0.0716 | 0.4234 | 0.2745 | 0.5500 | 0.3603 | False |


GSE46306_reconciled_cv: diffusion degraded real-only mean held-out balanced accuracy by -0.0960; diffusion mean BA 0.6873; >=80% mean flag False.

GSE46306_strict_cv: diffusion degraded real-only mean held-out balanced accuracy by -0.0480; diffusion mean BA 0.7760; >=80% mean flag False.

GSE99511_cv: diffusion degraded real-only mean held-out balanced accuracy by -0.0485; diffusion mean BA 0.5335; >=80% mean flag False.

GSE99511_loco: diffusion degraded real-only mean held-out balanced accuracy by -0.0218; diffusion mean BA 0.5437; >=80% mean flag False.

transfer_46306_99511: diffusion degraded real-only mean held-out balanced accuracy by -0.0165; diffusion mean BA 0.4418; >=80% mean flag False.

transfer_99511_reconciled: diffusion degraded real-only mean held-out balanced accuracy by -0.1157; diffusion mean BA 0.3926; >=80% mean flag False.

transfer_99511_strict: diffusion degraded real-only mean held-out balanced accuracy by -0.0956; diffusion mean BA 0.3882; >=80% mean flag False.

Small cohorts, uncertain repeated-person identity, incompatible cohort sampling and preprocessing, and the contradictory GSE46306 scale description limit interpretation. GSE46306 Normal/ CIN3 also differs in HPV status. No causal explanation of chip effects is established. Generator losses and distances do not establish biological validity or privacy. Diffusion uses a custom small DDPM, not an official TabDDPM reproduction. Historical scores are not comparable protocols. No low-performing chip was removed. The linear 100-step schedule leaves nonzero terminal signal and may impair synthesis; this was fixed before training and not selected on outcomes.

The local GSE46306 series gzip is truncated. Readable metadata supplies labels; the separate processed supplementary matrix supplies modeling values. See data_audit.json and source_hashes.json.


Diffusion mean held-out balanced accuracy was lower than real-only in all 7 evaluated protocols. Mean >=80% balanced accuracy was observed for: GSE46306_strict_cv/random (80.13%), GSE46306_strict_cv/real (82.40%), GSE46306_strict_cv/smote (81.24%). Diffusion reached >=80% in 3 individual pooled repeat/seed evaluations, all visible in pooled_metrics.csv; none of its protocol means reached 80%. These descriptive results do not establish a useful augmentation benefit.

Across 162 diffusion fits, invalid synthetic values totaled 0; same-class exact and near-real duplicate row counts totaled 0 and 0. Maximum boundary fraction was 0.0000. Absence of these flags does not establish biological validity or privacy.
