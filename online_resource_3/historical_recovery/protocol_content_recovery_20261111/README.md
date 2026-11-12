> Later 11 November update: both target ZIPs and the SECOM 20,000-replicate script/summary are now recovered in `../v22_or3_zip_recovery_20261111/`. The missing-target and missing-script statements below describe this earlier stage. Its stored verification outputs remain unchanged; consult the later addendum for current scope. SECOM per-draw/execution/runtime evidence and A3 runtime/prediction originals are still missing.

# Restored protocol content and additional saved evidence (11 November 2026)

This addendum preserves ten files from the third received recovery archive. It adds one hash-matched restored protocol and nine saved-data files. `provenance.json` records the received archive hash and member identity; all selected sources retain their received bytes. Original commentary is retained separately in the author's internal audit archive, because several claims need the qualifications below.

## Protocol: target content recovered

`sources/protocol_v2_冻结_2026-10-31.95f90d40.md` has SHA-256 `95f90d409d8d9cb1899aed623f02eee5a40300b41e6320e5d62de51e51de7352`, matching the pre-existing 31 October freeze record. The retained 4,010-byte protocol (hash beginning `88481a04`) stays unchanged in the preceding addendum. Comparing both complete texts identifies only two checklist-line changes; substantive protocol terms are identical between these two files. `protocol_diff.patch` shows the comparison.

The received note reports restoration from a historical chat write command (conversation and message IDs are in provenance.json). The raw chat export/tool response was not supplied. Thus target content is recovered against an independently retained hash; the chat restoration chronology and all historical execution events have not been independently authenticated. This is a restored content copy, not a claim of recovering an untouched old filesystem artifact. The old hash-mismatch check remains valid for the older retained file, but is no longer an unresolved target-content gap.

## Saved-data roles

| Files | Verified role and limits |
|---|---|
| `secom_oof.npz`, `secom_fold0.npz` through `secom_fold4.npz`, `secom_summary.json` | Corrected saved SECOM results attributed to 1 November. The OOF archive has 1,567 rows and 104 positive labels. The five test folds partition the OOF rows; 350 fold-to-OOF score arrays match exactly. An independent tie-aware AP calculation agrees with 84 summary scalars to less than 1e-12. The summary declares 1,000 bootstrap replicates, but these files contain predictions and summaries, not the missing 20,000-draw SECOM matrix or its original generator. |
| `simulation_continuous.json` | Exact input hash recorded by the already archived R6 five-seed simulation uncertainty analysis. It is separate from the later 30-seed simulation families and from SECOM. |
| `r9_arm3_bootstrap_draws.npz` | Fifty saved arrays, each shaped (1,000, 5), for the APS count-matched A3 intervention analysis. Ten saved C-B = (C-A3) + (A3-B) decompositions agree to less than 1e-12. This is not SECOM and not the historical supplementary A3 model-seed experiment. Algebraic agreement does not identify causal effects. |

The two received APS bootstrap files already match `data/aps_predictions/aps_matched_bootstrap.json` and `aps_bootstrap_draws.npz` byte-for-byte, so they are not duplicated. Those bootstrap matrices hold contrasts, not raw per-arm AP values. Existing APS 20,000-draw records remain distinct from the unresolved SECOM records.

## Verification and remaining scope

From the package root, using the existing NumPy verification environment:

```sh
python historical_recovery/protocol_content_recovery_20261111/verify_addendum.py
```

The check validates source hashes, the exact frozen target, the two-line difference, fold membership/labels/scores, AP summaries, R9 array shapes and algebra, and the R6 input hash. It creates no new training fits or bootstrap resamples. `verification.json` records the check. No previously reported scientific value, frozen source byte or prior acceptance result was changed.

Still missing: the A3 model-seed run's effective runtime/prediction/model/mask/index originals; the separately described historical SECOM 20,000-draw scripts/matrices; the two named historical recheck/path-revision ZIPs. Absence from these exports does not establish that those artifacts never existed elsewhere. Current author approvals and disclosures are not overridden by old or unsupported commentary. The fixed 10 November public commit predates this local submission addendum.
