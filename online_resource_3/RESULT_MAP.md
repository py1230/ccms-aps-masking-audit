# Results and executable evidence map

Table labels below are retained historical supplementary labels. The final manuscript's object destination map resolves any relocation. This table distinguishes a runnable numerical check from preservation of a source record.

| Scientific object | Package entry | What can be established here |
|---|---|---|
| Historical APS C/B/BASE three-way AP contrasts | `frozen/original_or3/CCMS_20k_bootstrap_summary.json`, field `results`; `aps_threeway_draws_20k.npz` | `verify_saved_results.py` reproduces 24 nominal intervals from 20,000 saved contrast draws. `regenerate_ap_bootstrap.py` can regenerate row-weight contrasts from packaged predictions. |
| Eight historical Bonferroni intervals | Same summary, field `bonferroni_ci` | Direct exact local numerical entry; eight recomputed intervals agree. Included in the public base and the submitted package. |
| Holm family and historical sup-t | Same summary, fields `holm_p`, `holm_adjusted_p`, `sup_t_c_975`, `sup_t_ci` | Numeric recomputation only; no cross-evaluation-set joint-coverage claim for sup-t. |
| Supplementary Table E5 | `reference/strict_alert/strict_alert_bootstrap_results.json`, field `E5_rows` | Eight corrected 2% alert intervals, including point estimates and old/new endpoints. Full replay verifies 600,000 components. |
| All strict-capacity audit conditions | Same JSON, `datasets`; `val_paired_recall_draws.npz`, `test_paired_recall_draws.npz` | 48 overall and 240 per-split old/new interval pairs, including the additional related contrasts and fixed-original-prevalence budgets. These are audit outputs, not added confirmatory hypotheses. |
| C/B core | `source/ccms.py`; frozen `aps_rerun/source/ccms.py` | English code annotations, unchanged numeric logic, 160 paired masks and AST comparison. No performance claim from this module alone. |
| A3/A4 implementation | Frozen `aps_rerun/run_aps.py` and `historical/run_arm3_countmatched.py` | Source-preservation evidence; `CHANGELOG.md` corrects old pure-mechanism comments. The portable APS entry fits the full candidate collection. Primary matched and retained first-environment split 11 acceptance are recorded separately. |
| A5 and chain sensitivity | Frozen `aps_a5/`, `aps_a5_sensitivity/`; `reference/audit/chain_replay.json` | Portable chain generation, shared original reference models, masks and prediction inputs are supplied. Fresh acceptance regenerates original chains and retrains the complete selected three-reference ensemble for each A5 family. |
| APS information channels | Frozen `aps_channels/` | Full paired scheduling and all fixed indices are supplied. Fresh acceptance retrains all 19 candidates of M3 repeat 1 split 11 and compares all generated arrays with the original unit. |
| APS 20 training resamples | Frozen `aps_resample/` | All 21 original ordered resample index plans, fixed evaluation masks and the fixed-iteration protocol are supplied. Fresh acceptance retrains the full 19-model independent anchor. The original 20-repeat C selection of BASE remains a saved-study finding. |
| Two 30-seed simulations | Frozen `simulation_continuous/`, `simulation_feature_dependent/` | The full generators and both 30-seed schedules are supplied. Fresh acceptance independently regenerates X, masks, all 20 models, selection and predictions for original seeds 601 and 701, with exact saved-array comparison. Mechanisms and 18-comparison Holm families stay separate. |
| SECOM chronological windows | Frozen `secom_temporal/`; `reference/audit/prediction_recompute.json` | Full chronological split indices and all three windows are supplied. Fresh acceptance retrains all 38 selection/refit models for original window 1 and compares its saved arrays. |
| Hydraulic external data and MA comparators | Frozen `hydraulic/`; same prior audit summary | Full raw-feature extraction, group splits, masks, adapted MA-DT vendor source and original MIT notice are supplied. Fresh acceptance rebuilds every prepared input and retrains the entire fold 1 candidate collection, including its selections and test predictions. |
| Historical A2/A3/B3/B4 | `historical_recovery/verify_historical_tables.py --table-only` and `RECOVERY_SCOPE.md` | All 284 reported scalar AP aggregate cells recomputed from recovered saved records. The recovered A3 runner/log specify validation and mask settings; A3 reproduction predictions support numerical checks; historical runtime/model/mask gaps remain. See `historical_recovery/model_seed_recovery_20261111/RECOVERY_NOTE.md`. |
| Formula, parameter and object mapping | `reference/audit/formula_parameter_mapping.md`, `protocol_parameter_snapshot.json`, `table_figure_source_map.json` | Original Stage 4 mappings retain historical local path strings as source labels; they are not runnable path dependencies or public URLs. |

The `frozen/source_snapshots` tree preserves original source bytes, including old paths and language. Current portable execution uses `training/` and `scripts/run_workflow.py`. The complete family schedules are available, and fresh acceptance is explicitly limited to the original units listed in the acceptance report. The tested host is macOS arm64. Historical full-archive model snapshots are regenerated by training commands and are not all redistributed in this package.

## Portable training and fresh acceptance

All nine source and input families now have the normal entry `scripts/run_workflow.py`. See `TRAINING.md` for the exact separate environments, raw-data acquisition, the single-unit acceptance schedule and full schedules. `acceptance/acceptance_summary.json` records actual checks. Full-size original units and their complete candidate models were executed, with the comparison scope specified per family. No new run authenticates the unrecorded historical training events.

## Historical context added on 11 November

`historical_recovery/additional_source_review_20261111/README.md` maps 12 additional original files to their historical roles. Its verifier establishes source identity and labels, not a new numerical result or historical execution certification. In particular, SECOM's retained formal record reports 1,000 replicates; R6's 20,000 replicates operate on five simulation seeds. The older retained freeze protocol still differs from the recorded target hash; the exact target content was subsequently restored in the addendum below. This earlier addition did not recover the target archives; the later same-day recovery below now preserves them without changing the main paper's evidence scope.

## Additional restored content and saved results

| Object | Package entry | Verification and limit |
|---|---|---|
| Original frozen protocol content | `historical_recovery/protocol_content_recovery_20261111/sources/protocol_v2_冻结_2026-10-31.95f90d40.md` | Exact full hash matches the pre-existing freeze record. Only two checklist lines differ from the older retained copy. Raw restoration chat export was not supplied; no certification of complete execution history. |
| Corrected saved SECOM results | Same addendum, `sources/secom_oof.npz`, `secom_fold*.npz`, `secom_summary.json` | 350 fold/OOF score arrays and 84 AP scalars agree. Predictions and a 1,000-replicate summary do not supply the missing historical SECOM 20,000-draw matrices. |
| R6 input and R9 APS contrasts | Same addendum, `simulation_continuous.json`, `r9_arm3_bootstrap_draws.npz` | R6 input fingerprint agrees; 50 R9 arrays (1,000 by 5) and ten algebraic decompositions checked. Separate from the A3 model-seed experiment and from SECOM. |

Run the addendum's `verify_addendum.py` for these saved-result checks. This addition changes provenance completeness, not reported scientific values or prior training-acceptance scope.

## Recovered historical target archives

| Object | Package entry | What is verified / still missing |
|---|---|---|
| v2.2 and OR3 path-revised original ZIPs | `historical_recovery/v22_or3_zip_recovery_20261111/original_zips/` | Full historical target hashes and 15 internal manifest entries match. The original ZIPs are distinct from the previously retained APS base. |
| SECOM historical 20k script, seed and summary | Same addendum, `v22_sources/secom_bootstrap_recompute_20261108.py`, `secom_bootstrap_20k_20261108.json` | 20,000 and seed 20261108 agree in code/JSON. Five source hashes and 24 point scalars verified. Per-draw matrices/indices/weights, independent execution logs and historical runtime remain unavailable; stored intervals are not regenerated or added to current tables. |
| Historical APS selector and r15 replay | Same addendum, original selector/r15 source and JSON | Preserved historical source artifacts. The old alert results do not replace current corrected strict-capacity results. No new replay claimed. |

`verify_recovery.py` validates this addendum without training or bootstrap resampling. Current gap status is narrower than earlier dated source-audit gap lists.

## 12 November 2026 update

| Object | Package entry | Evidence |
|---|---|---|
| A3 reproduction predictions; original identity corrected | `historical_recovery/a3_predictions_20261112/` | 270,000 reproduction records; 30 AP values and sample/label mappings verified. Historical model/mask/runtime gaps remain. |
| SECOM new conditional bootstrap reanalysis | `reanalysis/secom_bootstrap_20261112/` | Four 20,000-by-5 differences and five count matrices; all 400,000 differences replay exactly from packaged saved predictions. New random stream, not old run recovery. |

See `CURRENT_STATUS_20261112.md` for current intervals, execution timestamps, limitations and portable verification commands.

## A3 reproduction provenance correction

`reproduction/a3_model_seed_20261112/` contains the received new-run models, training masks, predictions and environment, together with separate historical terminal receipts. See CURRENT_STATUS_20261112.md for the corrected identity and verification limits.

## A3 post hoc evaluation and model replay update

`reproduction/a3_model_seed_verification_20261112/` adds the complete 120-file received package and strict saved-model verifier. Five evaluation-mask/imputer sets and all 30 models pass local replay; probability deviation is at most 1.12e-16, AP equality holds for all 30 historical scalars. Full command transcription is checked, but historical runtime linkage remains incomplete and the raw conversation projection is absent. Historical originals and post hoc artifacts remain distinct.
