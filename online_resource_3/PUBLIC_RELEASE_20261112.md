# Public reproducibility edition 12 November 2026

This edition accompanies **Auditing train-time class-conditional masking under missingness shift: budget-matched evidence from industrial fault classification**, by Yuchen Yang and Jian Guo, Beijing Institute of Petrochemical Technology, Beijing, China. Corresponding author: Jian Guo, gj2021@bipt.edu.cn. Prepared for Pattern Analysis and Applications; this repository does not indicate acceptance or submission status.

## What changed

The 11 November recovered sources and original ZIPs, corrected A3 reproduction identity, 30 saved A3 models with post hoc replay artifacts, and the 12 November SECOM 20,000-draw conditional reanalysis are now publicly browsable. Historical originals remain in their own dated locations. Original ZIPs are retained as provenance containers, with their content also available unpacked. The supplied manifests and received payload bytes are preserved, except that the large upstream APS CSV is acquired separately from UCI with its exact original fingerprint. Read CURRENT_STATUS_20261112.md before interpreting old dated reports.

## Reproduce the saved checks

From `online_resource_3`, create a Python 3.12 environment and install `requirements-training-main.txt`. This is the tested A3 replay profile; the producer used scikit-learn 1.9.0 and replay uses 1.9.1, so the retained version warnings are expected. Other training families retain their separately documented profiles in TRAINING.md.

```sh
python3.12 -m venv .venv-main
.venv-main/bin/python -m pip install -r requirements-training-main.txt
.venv-main/bin/python scripts/verify_saved_results.py
.venv-main/bin/python historical_recovery/verify_historical_tables.py --table-only
.venv-main/bin/python scripts/prepare_a3_data.py --download
.venv-main/bin/python reproduction/a3_model_seed_verification_20261112/verify_complete.py --out a3_replay_verification.json
.venv-main/bin/python reanalysis/secom_bootstrap_20261112/verify_saved_counts.py --out secom_replay_verification.json
```

These checks load saved data/models and reconstruct deterministic preprocessing or differences from saved weights. They fit no new classifiers and generate no new bootstrap draws. A3 compares 30 models and 270,000 row/label mappings; SECOM recomputes 400,000 fold/draw differences and four percentile intervals. See `release_validation/` for the actual checks on this staged public edition. These checks do not establish historical training provenance or arbitrary-environment compatibility.

## Start a fresh run

The received generator scripts retain their original language, paths and metadata as provenance. The English launcher adapts paths and the A3 run identifier in memory and verifies all generator/input fingerprints. It does not change model or statistical code. Run without `--execute` first to inspect the plan:

```sh
.venv-main/bin/python scripts/run_nov12.py a3 --output outputs/a3_fresh
.venv-main/bin/python scripts/run_nov12.py secom --output outputs/secom_fresh
```

Append `--execute` to intentionally perform 30 new A3 fits or 20,000 new SECOM bootstrap draws. The output directory must be new. `portable_launch.json` identifies the fresh run and every input. Native generator metadata retains dated source wording; it must not override the new launch receipt. In particular, claims that historical files never existed are not independently established. Only launcher input validation and saved-result checks were performed for this release; neither fresh generator was executed end to end. The previous 211-fit acceptance scope is unchanged.

## Distribution and provenance

The raw APS training CSV is acquired separately from UCI and checked against its preserved SHA-256; its original Scania notice remains embedded, and GPL text is supplied here; DATA_ACCESS.md records attribution and both notices on the current UCI page. Author-owned code and MA third-party code retain their separate MIT licenses. The 119-file A3 received-payload manifest is unchanged. The root payload manifest excludes the externally acquired raw CSV, which the original received manifest checks after acquisition. GitHub file APIs rejected the large CSV and native Git transfer timed out; the acquisition step provides the exact upstream data without modifying its fingerprint. Earlier reports stating that the public repository predates the additions describe their original date and are superseded by this edition.

A3 historical fitted models, masks, original prediction vectors and effective runtime remain unrecovered. The new SECOM random stream is 20261101 plus fold index; it is a new conditional analysis, not the missing historical 20261108 draw arrays. All four new intervals cross zero; this does not establish equivalence.

The acquisition helper was tested using the unchanged locally retained upstream CSV; a deliberately incorrect source was rejected. The live UCI endpoint connected and began transferring, but the full download was stopped because transfer was slow in this environment. Full live-network acquisition is not claimed as completed; `--file` supports an existing official CSV.
