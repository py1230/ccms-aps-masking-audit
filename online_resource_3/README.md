# Online Resource 3

This resource accompanies the CCMS industrial fault-classification manuscript. It provides the nine portable training families, their exact configurations and fixed indices, public-data acquisition with member fingerprints, third-party MA-DT source and notices, saved predictions and statistical replay programs. The authors selected MIT for their own software and authorized public release on 10 November 2026. The repository version is identified by its Git commit and this directory's SHA256 manifest.

Read `TRAINING.md` for the complete training commands and the three distinct Python environments. `acceptance/acceptance_summary.json` states which original units were actually executed in freshly installed environments and compared with their corresponding saved archives. These units retain full observations, full per-model settings and complete within-unit candidate collections. The entire set of historical fits was not newly retrained during acceptance. Agreement with saved outputs does not establish historical training provenance.

## Quick saved-result verification

Use Python 3.12 and NumPy. From this resource directory run the following commands in a new environment.

```sh
python3.12 -m venv .venv-verification
.venv-verification/bin/python -m pip install -r requirements-verification.txt
.venv-verification/bin/python scripts/verify_saved_results.py
```

The verifier checks the 153 preserved source fingerprints, recomputes 24 nominal AP intervals, eight Bonferroni intervals, the eight-comparison Holm family and historical sup-t values from the saved 20,000-draw matrices. It also checks 48 overall and 240 per-split legacy/strict alert interval pairs and six boundary/tie fixtures. These operations fit no models. Historical sup-t does not establish joint coverage across evaluation sets.

## Complete strict-capacity replay

```sh
.venv-verification/bin/python scripts/replay_strict_alert.py --output outputs/strict_replay_new
```

This replays all 1,000 paired draws from the five original APS score archives. It checks 600,000 components against a separate implementation that expands sampled row occurrences. Every sampled population uses exactly K selected occurrences, retaining original row order for tied scores and truncating boundary copies. Paired arms share the original-row bootstrap weights. The output directory must be new. Training and scores remain fixed during this conditional replay.

## Optional AP bootstrap regeneration

```sh
.venv-verification/bin/python -m pip install -r requirements-ap-bootstrap.txt
CCMS_NBOOT=1000 CCMS_OUT=outputs/ap_bootstrap1000_new .venv-verification/bin/python scripts/regenerate_ap_bootstrap.py
```

The explicit 1,000-draw run checks its contrast matrix against the original first 1,000 draws. Use `CCMS_NBOOT=20000` and another new output directory for a complete regeneration of the 20,000-draw archive. Regeneration uses saved predictions and preserves the conditional statistical scope.

## Training and numerical evidence

| Directory or file | Purpose |
|---|---|
| `scripts/run_workflow.py` | Normal portable entry for all nine families and either APS rerun environment profile |
| `training/` | Original configurations, protocols, fixed resampling/index/mask inputs, source copies and original freeze records |
| `training/shared/a5_reference/` | Exact paired C/B/BASE inputs and original reference models needed by both finite-swap workflows |
| `training/external_baseline_20261109/` | Adapted MA-DT source, source-commit extraction record and retained upstream MIT notice |
| `acceptance/` | Fresh environment/package records, executed-unit scopes and independent numerical comparisons |
| `data/aps_predictions/` | Five original label/score/index archives and first-1,000-draw AP references |
| `frozen/original_or3/` | Recovered original eight-member historical OR3 preserved unchanged |
| `reference/strict_alert/` | Corrected intervals, shared bootstrap weights, paired contrasts and boundary fixtures |
| `historical_recovery/` | Recovered historical scalar AP records, dependency modules and relocated standard-library table verifier |
| `RESULT_MAP.md` | Scientific object and executable evidence mapping with limits |
| `TRAINING_SOURCE_MANIFEST.json` | Current training-file fingerprints and original-source correspondence |
| `PORTABILITY_CHANGES.json` | Small disclosed scheduling and execution-receipt changes in portable source copies |
| `SHA256SUMS.txt` | Release payload fingerprints, excluding this manifest itself and subsequently generated outputs |

The primary APS training entry uses the historical-matched Python 3.12.15, NumPy 2.5.4 and scikit-learn 1.9.1 profile. The first Python 3.13.9/scikit-learn 1.7.2 rerun remains an explicitly labelled diagnostic profile. Hydraulic training uses its separate Python 3.12.15/NumPy 1.26.4/scikit-learn 1.5.2 environment. New runs record their observed environments and inputs before fitting. A C compiler is required for the finite-swap workflows.

## Historical records and interpretation

The recovered historical-table verifier recomputes all 284 reported aggregate cells from saved scalar AP records. Run it independently as follows.

```sh
python historical_recovery/verify_historical_tables.py --table-only
```

`historical_recovery/RECOVERY_SCOPE.md` explains the recovered runner's 9,000-row validation role and the incomplete metadata for the model-seed branch. Numerical recovery establishes the recorded aggregates and does not reconstruct missing prediction files, missing generator details or historical training events.

A5 finite-step switching has no proven uniform mixing guarantee. Zero candidates reuse a single BASE model while nonzero candidates use ens3. The 20 empirical training resamples selected BASE for C in every repeat. The two 30-seed simulation mechanisms retain separate 18-comparison Holm families. Intervals crossing zero do not establish equivalence. Retrospective overlapping APS splits, sparse dependent SECOM windows and hydraulic artificial missingness on one rig limit generalisation. Frozen old names such as `pure_*` remain historical fields and do not identify causal mechanisms.

## Data and software permissions

`DATA_ACCESS.md` links the official datasets, citations and notices. Raw measured features are downloaded separately and checked against exact historical member hashes. `LICENSE` grants MIT for author-owned software. `LICENSE_NOTICE.md` separates that grant from third-party software and dataset notices. The complete historical fitted-model archives remain local and can be regenerated with the full training schedules.
