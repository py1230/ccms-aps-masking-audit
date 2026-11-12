# Recovered model-seed experiment for historical Table A3

The 11 November 2026 submission update supplies the recovered `run_seed_dispersion.py`, `seed.log` and exported creation/launch records. It addresses historical Table A3 (formerly B5), not the count-matched A3 intervention. The runner and log are copied without modification into `../recovered/new_experiments_2026-11-05/`; the full supplied export is preserved in `original_recovery.zip` and `original_export/`.

The 30 JSON records are byte-identical to the pre-existing archive, the runner is byte-identical to the heredoc in the exported creation command, and all 30 printed AP values match the records at four decimals. All 42 supplied manifest entries pass. The ordinary historical-table verifier continues to check all 284 cells, including 45 A3 cells. These are consistency checks of supplied evidence; the remote session and runtime were not independently accessed.

## Coded settings

The runner reads the public APS training CSV, takes stratified splits 11, 22, 33, 44 and 55, fits on `train` and evaluates `val`. The retained `aps_io.py` defaults to 42,000/9,000/9,000 partitions. C and B use alpha 0.25 and training-mask seed 9107. Model seeds are 2026, 2027 and 2028. Evaluation uses `build_eval_inputs(Xva)["S1_30_101"][0]`: seed 101 adds missingness independently to originally observed validation cells with probability 0.30. Medians are fitted on Xtr, and missingness flags are rebuilt after masking. `_make_model(ms)` constructs HGB with that random state and library defaults.

`recovered_settings.json` is a new static transcription, not a historical runtime snapshot. The four corresponding dependencies are already in `../recovered/historical_dependencies/`; their archived bytes are identified in `dependency_fingerprints.json`. Their availability does not independently establish which remote modules were imported during the historic run.

## Verify without training

From the Online Resource 3 root:

```sh
python historical_recovery/model_seed_recovery_20261111/verify_a3_recovery.py
python historical_recovery/verify_historical_tables.py --table-only
```

The first command uses only the standard library and prints a report. An optional `--output NEW_PATH.json` writes to a new file. Neither command runs the training program. The original runner retains cloud absolute paths and is supplied as source evidence, not as an additional tested portable training family. The nine portable families and their prior acceptance scope are unchanged. No A3 retraining was performed for this update.

## Corrections to the preserved export notes

The original Chinese notes are source documents and are not the current verification conclusion. `pipeline.py` exists under `src`, not `scripts`, and the recovered runner adds both directories to the import path. The terminal metadata command contains 500 characters, not 260. The launch command uses `nohup ... &` and then `echo started`, so the terminal's exit code alone is not the training child's exit code; the matching 30-line log and `SEED_DONE` are separate evidence. A mentioned terminal `.txt` file is not included in the supplied ZIP. The manifest should be checked from the export root. The script does not save predictions/models/masks/indices, but this does not prove that no other historic process ever saved them.

## Remaining limits and versions

The effective historical Python, NumPy and scikit-learn versions are not directly recorded. Per-sample predictions, fitted models, mask matrices and row-index output for these 30 results are not supplied. Thus the AP values cannot be recomputed from historical prediction scores here, and complete historical training reproduction is not claimed. The 15 split/model-seed pairs share samples and remain auxiliary evidence excluded from core inference.

The public fixed commit c757fec4e9e029146441d1e7b5bed20433599d01 is the 10 November base. This local submission ZIP adds the recovery on 11 November; the fixed public commit does not contain this addendum. No remote publication was performed during this update. Earlier frozen audit reports describe the evidence available on their own dates and are retained unchanged; this note and `../RECOVERY_SCOPE.md` state the current recovery scope.
