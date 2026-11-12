# Online Resource 3

**Current update (12 November 2026):** A3 reproduction artifacts, separate historical receipts and a new SECOM conditional bootstrap reanalysis are included. The earlier identification of the A3 CSV as a historical original has been corrected. Post hoc A3 evaluation masks and imputers are now included, and all 30 saved models pass independent local replay. Read `CURRENT_STATUS_20261112.md` for current availability, new intervals and verification commands. Earlier recovery notes remain dated snapshots.

This resource accompanies the CCMS industrial fault-classification manuscript. It provides the nine portable training families, their exact configurations and fixed indices, public-data acquisition with member fingerprints, third-party MA-DT source and notices, saved predictions and statistical replay programs. The authors selected MIT for their own software and authorized public release on 10 November 2026. This public edition incorporates the 11–12 November recoveries, corrected A3 provenance, complete saved-model replay and new SECOM reanalysis. The payload manifest covers the unpacked files; earlier public commit c757fec4e9e029146441d1e7b5bed20433599d01 remains a historical base.

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
| `historical_recovery/` | Recovered scalar AP records, dependency modules, the A3 runner/log/source addendum and standard-library verifiers |
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

`historical_recovery/RECOVERY_SCOPE.md` explains the runners' 9,000-row validation role. The A3 recovery adds the model-seed runner, log and exported commands, establishing coded training-mask seed 9107 and evaluation input S1_30_101. Run `python historical_recovery/model_seed_recovery_20261111/verify_a3_recovery.py` for source/log checks. A3 per-sample predictions are now correctly identified as 12 November reproduction output, with new models, training masks and environment metadata. The complete post hoc check in `reproduction/a3_model_seed_verification_20261112/` replays 30 models, matches probabilities within 1.12e-16 and exactly reproduces the 30 historical AP scalars. Historical models, masks, effective environment and complete runtime/input associations remain unavailable. No training was performed in this local integration.

A5 finite-step switching has no proven uniform mixing guarantee. Zero candidates reuse a single BASE model while nonzero candidates use ens3. The 20 empirical training resamples selected BASE for C in every repeat. The two 30-seed simulation mechanisms retain separate 18-comparison Holm families. Intervals crossing zero do not establish equivalence. Retrospective overlapping APS splits, sparse dependent SECOM windows and hydraulic artificial missingness on one rig limit generalisation. Frozen old names such as `pure_*` remain historical fields and do not identify causal mechanisms.

## Data and software permissions

`DATA_ACCESS.md` links the official datasets, citations and notices. Raw features are acquired separately and checked against historical member hashes. Run `python scripts/prepare_a3_data.py --download` before A3 replay; it restores the exact APS training CSV, including its Scania notice, from the official UCI source. `LICENSE` grants MIT for author-owned software. `LICENSE_NOTICE.md` separates that grant from third-party software and dataset notices. The complete historical fitted-model archives remain local and can be regenerated with the full training schedules.

## Additional historical source review (11 November)

`historical_recovery/additional_source_review_20261111/` preserves 12 further historical files: freeze/deviation records, earlier SECOM and simulation scripts/results, and a packaging-time environment snapshot. Read its README for the earlier retained protocol mismatch and the distinction between SECOM 1,000-replicate statistics and simulation 20,000-replicate statistics. These files do not close the missing A3 runtime/prediction or SECOM 20,000-replicate provenance gaps. `verify_sources.py` checks identity and labels without training. This addition is included in the current public edition.

## Protocol content recovery and saved-prediction checks (11 November)

`historical_recovery/protocol_content_recovery_20261111/` adds ten byte-preserved files: the restored protocol matching the original `95f90d40...` target, corrected saved SECOM fold/OOF predictions and summary, the R6 input, and saved R9 contrasts. Its README distinguishes exact content recovery from unverified chat chronology and historical execution. `verify_addendum.py` checks 10 source hashes, two protocol checklist differences, 350 score-array correspondences, 84 AP summary values and saved R9 algebra without training or bootstrap resampling. The protocol content gap is closed. A later same-day recovery below also closes the two target-ZIP and SECOM script/summary gaps; At that recovery stage, A3 runtime/prediction originals and SECOM draw/execution evidence remained unavailable; current availability is updated above.

## Recovered target ZIPs and SECOM 20,000-replicate source

`historical_recovery/v22_or3_zip_recovery_20261111/` now preserves both original target ZIPs with full historical hash matches, the nine v2.2 members, provenance and a read-only verifier. The recovered SECOM script and JSON record 20,000 replicates and seed 20261108; five prediction hashes and 24 AP-difference scalars agree independently. Original SECOM draw arrays, independent execution logs and historical runtime evidence remain unavailable. Read the addendum README before interpreting old audit claims. The 11 November recovery did not regenerate historical intervals. The separate 12 November conditional reanalysis now supplies new intervals and saved arrays; see CURRENT_STATUS_20261112.md. The recovery itself performed no training. These records are now part of the public edition.
