# Release candidate changes

## Stage 5 portability and English documentation

All files under `frozen/` and `data/aps_predictions/` were copied without changing bytes; `SOURCE_MANIFEST.json` records their exact fingerprints. The historical manuscripts and prior stages were not edited.

`scripts/replay_strict_alert.py` is a separate copy of the Stage 4 correction. Only package-relative input paths, an explicit output-directory argument, and the introductory documentation changed. The strict-capacity arithmetic, stable score ordering, shared resampling weights, independent expanded-occurrence checks and old/new comparison loops are retained. Its full 1,000-draw replay is compared with the preserved Stage 4 results.

`scripts/regenerate_ap_bootstrap.py` adapts the frozen original OR3 script to the current package layout. It resolves the five saved prediction files, the original first-1,000-draw references and the archived `summarize.py` from explicit relative locations. It refuses an existing output directory. The historical incorrect metadata string describing M3 as an “alpha0.25 ens3 reference model” is corrected to the stored unmodified single BASE model; this changes metadata only. The AP and bootstrap computations are unchanged. Its generated historical filenames contain `20k` even when the operator explicitly selects `CCMS_NBOOT=1000`; always read `protocol.n_bootstrap` in the generated JSON. The full historical 20,000-draw matrix is retained separately and verified arithmetically.

`source/ccms.py` translates the operator's docstrings and exception messages into English without changing control flow or numeric expressions. An AST comparison after removing documentation/message text and 160 paired mask checks passed. `scripts/verify_saved_results.py` and `scripts/acquire_data.py` are new English verification/acquisition entry points.

## Stage 4 strict alert capacity correction

The old alert bootstrap included all sampled copies of the boundary row, sometimes exceeding the claimed alert capacity. The corrected procedure uses exact integer ceiling `K`, stable original-row tie order, and only the required copies at the boundary. Three arms and all dependent splits/stress inputs share the sampled original-row weights. The correction changes conditional interval endpoints, not saved model scores, AP or alert point estimates. All eight E5 zero-inclusion decisions are unchanged. Original scripts, interval values and paired old/new contrasts are retained.

## Historical semantic corrections

C and B match expected new-deletion counts per column; individual realisations need not match. A3 redraws cells conditional on C's realised column counts; A4 does so conditional on B's realised counts. The historical A3 seed is `mask_seed*1000+7`; A4 uses `mask_seed*1000+11`. Both redraw only originally visible cells.

The identity `C-B = (C-A3) + (A3-B)` is algebraic. A3 changes cell locations, row burden and joint masks. Neither term identifies a pure causal mechanism. Historical fields containing `pure`, `share`, `allocation effect` or `column-profile effect` are preserved only as old names. The reported 95.6% and 88.4% ratios are path-specific descriptive ratios, not causal shares. A5 additionally fixes row counts and thus inherits class-associated row burden; its finite switch chains are not proof of uniform sampling or sufficient mixing.

The recovered original APS OR3 is distinct from the later path-revision ZIP and recheck ZIP, both subsequently recovered in the final 11 November addendum below; it does not supply the missing SECOM 20,000-draw matrix. The frozen protocol content was subsequently recovered in the separate 11 November addendum below. That original OR3 alone does not recover the four supplementary table generators; later recoveries are documented separately in `historical_recovery/RECOVERY_SCOPE.md`. No absent artifact has been silently replaced with a nearby experiment.

## 11 November 2026 A3 recovery update

Added the unchanged model-seed runner, seed log, original exported source package, provenance, static settings, dependency fingerprints and a standard-library consistency verifier. Updated current recovery documentation and the table verifier's limitation message; all 284 table values and 30 A3 JSON files are unchanged. Frozen source snapshots and prior acceptance artifacts are preserved. Historical runtime versions and per-sample predictions are still missing; no training was executed. This local submitted version extends the fixed 10 November public commit and has not been published remotely.

## 11 November: additional historical source review

Added 12 unchanged historical source files and a separate README, provenance record, standard-library identity verifier and its output. Original recovery commentary is not adopted as scientific evidence: absent save calls do not establish global nonexistence; a packaging snapshot is not an A3 runtime receipt; the R6 20,000-replicate simulation is distinct from the missing SECOM analysis; retained author approvals are not rolled back. All pre-existing research data, training files, accepted-run records and scientific results remain byte-identical. No training or bootstrap recomputation was performed. This local addendum has not been published remotely.

## 11 November: hash-matched protocol content recovery

Added the restored protocol whose full SHA-256 matches the original freeze record, and nine further saved-data files. The two retained protocol versions differ only in two checklist lines. Raw chat/tool export was not supplied, so the received restoration chronology is attributed rather than independently authenticated. Added source-hash, saved SECOM AP/fold and R9 contrast checks; no new fitting or resampling. Updated the two supplementary Word files and English supplementary PDF outside this ZIP to close the target-content gap while retaining the other limitations. The previously archived source and acceptance bytes are unchanged; only current explanatory notes and the root manifest were updated. No remote publication occurred.

## 11 November: original v2.2 and path-revised OR3 ZIPs recovered

Added two byte-preserved historical target ZIPs, nine extracted v2.2 members, and four documentation/provenance/verification files. Both full target hashes, 15 internal manifest entries, five SECOM input hashes and 24 point scalars pass. The APS revision preserves numerical artifacts, changes six JSON path strings and one script docstring path, and has no executable AST difference. Updated current recovery explanations and supplementary provenance text. The old r15 archive does not override the current strict-capacity correction. Existing research payload and acceptance bytes remain unchanged; no bootstrap or training was executed. SECOM per-draw/execution/runtime evidence and A3 runtime/prediction originals remain missing.
