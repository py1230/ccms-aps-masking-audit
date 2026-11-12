# Reproduce the nine training families

The portable entry prepares a new work directory, checks exact input fingerprints, compiles the finite-swap C code when required, records a fresh input manifest and calls the archived experiment's training functions. Raw observations, per-model iteration counts, model settings, masks, candidate definitions and selection rules follow the archived protocol. `--unit` selects one original split, repeat, window or data seed with the complete observations and full candidate set for that unit. It changes the execution schedule only. `PORTABILITY_CHANGES.json` records the small scheduling/count-assertion changes in the portable copies. Original frozen bytes remain separately preserved.

Use a new output directory for every invocation. Existing output directories raise an error. Python environments are deliberately separate because the hydraulic comparator and the earlier APS/SECOM evaluations used different numerical environments. A C compiler with shared-library support is required for A5. The tested host is macOS arm64. The shared library is built on the reader's host and is never redistributed as a universal binary.

## Obtain public inputs

Run these commands from the package directory. Each download verifies all expected raw member SHA256 values before writing usable inputs. Changed sources fail explicitly.

```sh
python scripts/acquire_data.py aps --download
python scripts/acquire_data.py secom --download
python scripts/acquire_data.py hydraulic --download
```

`--archive /path/to/archive.zip` accepts an existing official download. See `DATA_ACCESS.md` for dataset identities and notices. Raw measurements are obtained separately. The wrapper reconstructs a ZIP with verified member contents for hydraulic feature extraction. The reconstructed ZIP container bytes may differ from the historical container, while raw member bytes and features remain checked.

## Main matched environment

Use Python 3.12.15. Create and install a new environment.

```sh
python3.12 -m venv .venv-main
.venv-main/bin/python -m pip install -r requirements-training-main.txt
```

The manuscript-used APS rerun and six further matched-environment families use the following normal entries. These commands run one complete original unit and retain all its predictions, candidate selection, models and input fingerprints.

```sh
.venv-main/bin/python scripts/run_workflow.py aps_rerun --aps-environment matched --unit --output runs/aps_rerun_matched_unit
.venv-main/bin/python scripts/run_workflow.py aps_channels --unit --output runs/aps_channels_unit
.venv-main/bin/python scripts/run_workflow.py aps_resample --unit --output runs/aps_resample_unit
.venv-main/bin/python scripts/run_workflow.py aps_a5 --unit --output runs/aps_a5_unit
.venv-main/bin/python scripts/run_workflow.py aps_a5_sensitivity --unit --output runs/aps_a5_sensitivity_unit
.venv-main/bin/python scripts/run_workflow.py simulation_continuous --unit --output runs/simulation_continuous_unit
.venv-main/bin/python scripts/run_workflow.py simulation_feature_dependent --unit --output runs/simulation_feature_dependent_unit
```

The selected units are M3 repeat 1 split 11 for information channels, the independent anchor for training resampling, split 11 randomization repeat 1 for A5, split 11 C initial state repeat 1 at 5E for finite-step sensitivity, and data seeds 601 and 701 for the two separate simulation mechanisms. The A5 and sensitivity units each retain the full three-reference ensemble.

Remove `--unit` and choose another new output directory to execute the full prespecified family. No model selection or hypothesis family is changed. The full simulation commands generate each of the 30 prespecified seeds within its own mechanism. The full resampling command fits the anchor and all 20 empirical bootstrap repetitions. All settings remain in `training/<family>/config.json` and the accompanying `PROTOCOL.md`. The original postprocessing and verification scripts are retained in each materialized work directory, with evidence limits in the historical notes.

## Earlier APS and chronological SECOM environment

Use Python 3.13.9.

```sh
python3.13 -m venv .venv-legacy
.venv-legacy/bin/python -m pip install -r requirements-training-legacy.txt
.venv-legacy/bin/python scripts/run_workflow.py aps_rerun --aps-environment first --unit --output runs/aps_rerun_unit
.venv-legacy/bin/python scripts/run_workflow.py secom_temporal --unit --output runs/secom_temporal_unit
```

These additional diagnostic and chronological units contain all 25 models for original APS split 11 and all 38 selection/refit models for original chronological SECOM window 1. The `first` APS profile corresponds to the first rerun environment retained in the source archive. The manuscript-used `matched` profile above uses Python 3.12.15 and scikit-learn 1.9.1. Both profiles preserve their own frozen source and numerical references. Removing `--unit` runs all five original APS splits or all three SECOM windows, respectively.

## Hydraulic and missingness-avoiding tree environment

Use Python 3.12.15.

```sh
python3.12 -m venv .venv-hydraulic
.venv-hydraulic/bin/python -m pip install -r requirements-training-hydraulic.txt
.venv-hydraulic/bin/python scripts/run_workflow.py hydraulic --unit --output runs/hydraulic_unit
```

The entry reconstructs all features, original group splits, artificial masks and transformations from the fingerprint-verified raw members before fitting the complete candidate collection on fold 1. The original MA-DT implementation and MIT notice are included in `training/external_baseline_20261109`. Removing `--unit` executes all five folds. The cross-family comparisons remain descriptive.

## Acceptance scope

`acceptance/acceptance_summary.json` records actual fresh-environment executions and numerical comparisons with the corresponding saved original units. The acceptance uses full observations and the original per-model settings. It does not claim that every fit from all nine archives was newly rerun or that deterministic agreement proves the historical training events. The saved-result verifier and statistical replay commands in the README remain independently executable.

## Compare with the published original scores

The package retains compact original ensemble and selection/refit prediction arrays for every accepted unit. After a new unit run, compare its scores, labels and indices with the public reference. For example run the following command after the primary APS entry.

```sh
.venv-main/bin/python scripts/check_unit_predictions.py aps_rerun_matched --work-directory runs/aps_rerun_matched_unit/aps_rerun
```

The other unit keys are listed by `--help` and `acceptance/reference_prediction_manifest.json`. The retained first-environment APS diagnostic uses key `aps_rerun`. Exact comparisons use the corresponding matching environment. The checker loads only non-pickled NPZ arrays, verifies the reference fingerprint and fails on any differing dtype, shape or value.

## Historical A3 source recovery

The 11 November addendum under `historical_recovery/model_seed_recovery_20261111/` preserves a separate original runner with cloud paths. It is not a tenth portable or fresh-environment-accepted family. Use its standard-library verifier for artifact/log checks. A3 historical runtime and prediction-level recovery remain incomplete; see the addendum note before attempting any new training.
