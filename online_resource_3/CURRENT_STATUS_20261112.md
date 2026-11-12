# Current evidence status 12 November 2026

The A3 CSV is 12 November reproduction output, following the author's accepted correction. New post hoc evaluation artifacts now support all 30 saved-model replays; the separate SECOM conditional reanalysis remains unchanged.

## A3 complete reproduction verification

`reproduction/a3_model_seed_verification_20261112/` contains the complete new delivery, strict verifier and independent results. Its `received/01_历史原件/` contains supplied historical records; `02_11-12复现产物/` contains reproduction outputs; `03_本次事后重建与核验/` contains post hoc evaluation masks and imputers, dependencies and verification records. The large raw APS CSV is acquired separately using `python scripts/prepare_a3_data.py --download` and its original fingerprint; this restores the complete received layout for replay. These identities must remain separate.

The 119 payload hashes match. All 30 saved models were loaded locally; maximum probability deviation from saved NPZ/CSV values is 1.1102230246251565e-16, with all 30 AP values exactly equal to the separate historical scalar files. All 270,000 sample/label mappings, ten training masks, five evaluation-mask sets, training-only medians and five M3 matrix hashes passed independent reconstruction checks. Local environment and cross-version warnings are retained. The original producer's pandas/joblib versions remain unspecified.

The previous evaluation-mask, preprocessing-object and model-replay gaps are closed for the reproduction, by explicitly post hoc artifacts. Historical fitted models, masks, original prediction vectors, effective environment and complete execution linkage remain unrecovered. Equal predictions do not prove unique historical input identity. Full command content is supplied as a transcription matching the script and terminal prefix, but the raw conversation projection is absent. Terminal exit code 0 belongs to the background launcher, not a separately captured training process.

Earlier delivery remains in `reproduction/a3_model_seed_20261112/`; its dated report is superseded on current availability. The legacy `historical_recovery/a3_predictions_20261112/` location retains the same CSV and accepted provenance correction. Directory names are not provenance evidence.

## SECOM conditional reanalysis

`reanalysis/secom_bootstrap_20261112/` contains the delivered scripts, metadata, log and execution receipts, four (20000, 5) difference arrays and five (20000, n_test) resampling-count matrices. Input fold NPZ files are byte-identical to `historical_recovery/protocol_content_recovery_20261111/sources/secom_fold{0..4}.npz`. The deterministic verifier uses these packaged inputs and the saved counts; it generates no new draws and performs no model training.

The reanalysis preserves class counts within each outer fold, uses paired weights across arms and pressure inputs, averages AP differences across S1 inputs and then equally across folds, and uses default_rng(20261101 + fold index). This random stream differs from the archived 20,000-draw summary with seed 20261108. All 400,000 saved fold/draw differences reproduce exactly. The first 1,000 draws reproduce the archived 1,000-draw summary intervals, not historical draw arrays (which are unavailable).

| Comparison | Mean AP difference | New pointwise 95% percentile interval |
| --- | ---: | --- |
| Fixed alpha 0.25, S0 | +0.011272 | [-0.012829, +0.030606] |
| Fixed alpha 0.25, S1 30% | +0.003076 | [-0.010893, +0.019856] |
| Validation-selected, S0 | -0.004358 | [-0.028410, +0.020135] |
| Validation-selected, S1 30% | -0.010199 | [-0.023841, +0.002893] |

All intervals include zero; equivalence is not established. They condition on saved fitted models, masks and selections, omit retraining and reselection, and do not model dependence from overlapping training folds. They are pointwise, without multiplicity adjustment. The historical 20261108 script and summary remain preserved under historical_recovery/v22_or3_zip_recovery_20261111; their missing draw/execution artifacts are not replaced or relabeled as recovered originals.

## Verification from the unpacked resource directory

With NumPy, pandas, scikit-learn and joblib installed, acquire the fingerprint-checked APS CSV and run the strict saved-model check:

```sh
python scripts/prepare_a3_data.py --download
python reproduction/a3_model_seed_verification_20261112/verify_complete.py --out a3_replay_verification.json
```

Its README records the tested local versions. This reconstructs deterministic masks and preprocessing for comparison, reads saved models and returns nonzero on failed checks. It does not train classifiers or alter received artifacts.

The earlier NumPy-only saved-artifact checks remain available:

```sh
python historical_recovery/a3_predictions_20261112/verify_a3_predictions.py
python reproduction/a3_model_seed_20261112/verify_artifacts.py
python reanalysis/secom_bootstrap_20261112/verify_saved_counts.py
```

The SECOM supplied README describes an earlier 14:43–14:44 run. The retained final shell/platform receipts instead record 07:09:54–07:10:43 UTC (15:09:54–15:10:43 Asia/Shanghai), with exit code 0 and matching script/log hashes. We retain the received bytes and use the final receipts as the timestamp evidence for this delivery. Source claims that the old originals never existed or are globally absent are not independently established.

This public edition supersedes the 10 November base for current artifact availability. It includes the additions above as unpacked, browsable files. Publication does not extend the prior 211-fit acceptance scope. See PUBLIC_RELEASE_20261112.md for launch commands, distribution notices and the checks performed on this edition.
