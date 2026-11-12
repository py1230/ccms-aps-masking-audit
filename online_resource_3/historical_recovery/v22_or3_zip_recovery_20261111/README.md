# Recovered v2.2 and path-revised OR3 originals (11 November 2026)

Both historical target ZIPs have now been recovered from the original ChatGPT conversation and independently checked against their previously recorded full hashes. The source-download evidence is retained in the author's internal submission audit. This directory contains unchanged original ZIPs and an unchanged extraction of the nine v2.2 members for inspection.

| Original | Bytes | SHA-256 |
|---|---:|---|
| `original_zips/CCMS_v2.2_复核脚本与结果.zip` | 45,407 | `a99cdec7b7582b5d9464ab57b927e88253de4869d8d4f1a16634c5129a16a4e2` |
| `original_zips/CCMS_Online_Resource_3_v2.1_路径修订版.zip` | 18,081,470 | `a207d8bc98f11320ad9bff5fdefc2347e90e6af927bd27fc16d9bcc826637f0d` |

Both ZIP CRC checks and all 15 internal manifest entries pass. File identity is established against retained targets; this does not authenticate all historical computational events.

## Recovered SECOM material

`v22_sources/secom_bootstrap_recompute_20261108.py` and `secom_bootstrap_20k_20261108.json` provide the original analysis definition, defaults of 20,000 replicates and seed 20261108, and four saved result summaries. The five source hashes match `../protocol_content_recovery_20261111/sources/secom_fold0.npz` through `secom_fold4.npz`. An independent tie-aware AP calculation reproduces 20 fold differences and four overall means (24 scalars, maximum absolute error below 1e-12). It creates no bootstrap samples.

The original method resamples positive and negative rows within each disjoint outer test fold, shares row-count weights across paired arms and S1_30 evaluation repeats, and averages the five fold differences equally. It conditions on fixed fitted models, masks and selections, and does not model dependence from overlapping training sets. The stored intervals remain historical archived summaries; they have not been regenerated or added to current paper tables in this update.

Static inspection finds only the final summary-JSON data write at line 237. Four bootstrap arrays are held in memory; the source does not save them, resampling indices or weights. This observation is limited to the recovered script and does not exclude external logging or other storage. The recovered ZIPs contain no independent SECOM 20k stdout/stderr, actual invocation/exit record, or historical environment lockfile. These remain missing, along with the A3 model-seed run's runtime and prediction-level originals. Original chat replies report further searches, but their underlying cloud scan traces were not supplied for independent audit.

The original SECOM script expects specific member names in an ESM_2 ZIP. Some retained ESM_2 copies lack the UTF-8 filename flag and need separate path-name handling; matching prediction bytes does not determine which ZIP/name-resolution environment was used historically. The source is preserved, not silently patched or newly accepted as a portable training workflow.

## Path-revised APS OR3

The restored ZIP is a distinct historical artifact from `frozen/original_or3/`. Its four numerical/figure artifacts (summary JSON, draw NPZ, PNG, PDF) are byte-identical to that earlier package. The r13 JSON differs only in six source-path strings. The script differs only in its module docstring path; its executable AST is identical after removing that docstring. README and manifest changes remain preserved. The 24 saved APS arrays each have shape (20,000, 5); they are not SECOM draws. The original ZIP is retained intact without adding another expanded copy of the same APS matrix.

## Earlier claims and current scope

`v22_sources/README.md` and `AUDIT_NOTE_2026-11-08.md` preserve dated claims and then-pending work. They are source records, not current status: the protocol content and later experiments have since been recovered or completed, while execution limits remain. The old r15 alert calculations are not substituted for the current strict-capacity correction in `reference/strict_alert/`. Original completion statements do not supply missing logs or runtime evidence.

Earlier addenda's verification JSON and scripts retain their stage-specific gap lists; this addendum supersedes their missing-target-ZIP and missing-SECOM-script/summary status. The fixed 10 November public commit predates this local addition. Existing paper results, source bytes and prior training acceptance are unchanged.

## Read-only verification

From the Online Resource 3 root, using the existing NumPy verification environment:

```sh
python historical_recovery/v22_or3_zip_recovery_20261111/verify_recovery.py
```

The verifier checks original ZIP hashes/CRCs/manifests, extracted-member identity, five input hashes, fold partition, 24 point-estimate scalars, script defaults/write calls and the APS revision differences. It does not import or execute the recovered research scripts, fit models or generate bootstrap draws. `verification.json` records the checks and remaining limitations.
