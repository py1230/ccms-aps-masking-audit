# Historical supplementary AP source recovery

The original scalar AP records behind Tables A2, A3, B3 and B4 have been recovered from the retained v5 submission archive. All 284 numeric table cells reproduce at the published four-decimal precision. These remain historical auxiliary results excluded from core inference.

Run the portable saved-AP check with Python 3.12 or later.

```bash
python verify_historical_tables.py --table-only
```

The script uses Python's standard library. Keep `historical_table_targets.json` and `recovered/new_experiments_2026-11-05/` beside it. Its output `historical_table_verification.json` records all 284 checks. The target file records the exact SHA256 of the Stage 6 historical-archive Word file from which the table cells were extracted. If that baseline Word file is available in its original project layout, the verifier reads its cells directly. Otherwise it uses the packaged target transcription.

The recovered branch has 140 AP records from four configurations over five splits and 30 records from a model-seed check. The A–D training runner and summary script are frozen historical files and retain their original paths and language. Current portable execution guidance lives in the main reproducibility package.

The A–D runner evaluates within-split validation inputs. Its saved AP values reproduce A2, B3 and B4. The separate model-seed training script is absent. A3 records identify split, arm, model seed and AP but omit evaluation-partition and mask metadata. The historic S1 30% attribution is retained with that limitation.

The recovered scalar records do not include their per-sample predictions, fitted models or training-mask arrays. This verifier recalculates averages and differences from scalar AP values. It does not fit models or independently recalculate AP from prediction scores. The 140-record run log contains 133 fit-progress entries because the runner can resume by skipping existing files.

The nested source ZIP has SHA256 `d810bc6fefa016464509c64faf637eba3952db475f09ab9aa59940e54aa86632`. Its full 441-entry manifest has one mismatch in `PACKAGE_INDEX.md`. All 175 files under the recovered experiment branch match their archived bytes, and the 140 configuration records plus two scripts also match 142 separately retained fingerprints from the 6 November 2026 audit. These consistency checks do not authenticate historical training events.
