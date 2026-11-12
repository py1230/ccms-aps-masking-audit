> Later 11 November update: both target ZIPs and the SECOM 20,000-replicate script/summary are now recovered in `../v22_or3_zip_recovery_20261111/`. The missing-target and missing-script statements below describe this earlier stage. Its stored verification outputs remain unchanged; consult the later addendum for current scope. SECOM per-draw/execution/runtime evidence and A3 runtime/prediction originals are still missing.

# Additional historical source review — 11 November 2026

This directory adds 12 byte-preserved source files from a user-supplied recovery archive. They match existing local archival copies and were absent from the preceding submitted ZIP. They provide historical context, not newly verified primary results or a newly accepted training workflow. The complete received archive, its original commentary and the older title-page extract are retained in the author's submission-preparation records; only the checked research source files are copied here.

## What is preserved

- Four freeze-related files: the draft, retained protocol, original 39-file freeze record and deviations log. The recorded protocol hash starts `95f90d40`, whereas the retained file starts `88481a04`. This retained file does not match the target; a later same-day addendum now supplies hash-matched restored target content (see the update below).
- Seven earlier SECOM/simulation files: SECOM pilot and formal runners/results plus a formal-result note, and the R6 continuous-simulation runner/result. The SECOM formal record contains 1,000 bootstrap replicates for each of its 27 contrasts. The 20,000-replicate R6 analysis resamples five simulation seeds (501–505), not SECOM rows. These records do not replace the missing separately described SECOM 20,000-replicate package.
- One environment snapshot explicitly captured at packaging time on 1 November 2026. It does not establish the effective environment of the later A3 model-seed run.

The A3 runner and seed log in the received archive are byte-identical to the already preserved files; no second copy is added. Missing runtime records, per-sample A3 predictions, fitted models, masks and row-index outputs remain unavailable. Static code inspection establishes that the recovered runner does not save these artifacts itself; it cannot prove they were never saved by another process or elsewhere.

## Verification

From the package root run:

```sh
python historical_recovery/additional_source_review_20261111/verify_sources.py
```

The standard-library verifier checks 12 source fingerprints, the retained protocol mismatch, experiment identities and bootstrap counts, the packaging-snapshot label, and the existing A3 runner/log identity. It fits no model, draws no bootstrap samples and imports none of the recovered research scripts. `verification.json` records the successful check.

Original paths, code and contemporaneous statements remain unchanged. For example, an old 'one-time test' claim is scoped to that historical record, not a study-wide guarantee. Current manuscripts and source-audit limits take precedence over historical wording. The prior portable-training acceptance scope remains unchanged. This local update is not included in the fixed 10 November public commit.

## Later same-day update: target content recovered

`../protocol_content_recovery_20261111/README.md` records recovery of the exact target bytes and checks against the pre-existing freeze hash. Only two checklist lines differ between the restored and retained files; substantive terms are identical in these files. The raw chat export was not supplied. This supersedes the earlier unresolved-content status, while this directory's source files and verification.json remain dated evidence of the older retained copy. Other A3/SECOM/target-ZIP gaps remain.
