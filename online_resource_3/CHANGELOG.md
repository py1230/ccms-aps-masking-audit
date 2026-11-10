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

The recovered original APS OR3 is not the missing later path-revision ZIP, recheck ZIP, frozen protocol byte sequence or historical SECOM 20,000-draw matrix. It also does not recover the exact five-split generators of A2/A3/B3/B4. No absent artifact has been silently replaced with a nearby experiment.
