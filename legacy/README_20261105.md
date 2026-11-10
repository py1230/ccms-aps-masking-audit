# Code and reproduction material for the APS audit of train-time class-conditional masking

This repository accompanies the manuscript *Auditing train-time class-conditional masking under
missingness shift: budget-matched evidence from industrial fault classification* (Pattern Analysis and
Applications). It contains the analysis code, the operator implementation, and the scripts that turn the
archived per-sample predictions into the tables and figures of the paper.

Two kinds of work are kept apart in what follows. Some results are recomputed from stored predictions and
cost seconds; others require refitting the models. The tables below say which is which, so that a reader
can check a number without rebuilding the whole study.

## 1. Repository layout

```
code/
  07_formulas/ccms.py                      the two masking operators (C and B) and their bookkeeping
  08_repro_bundle/
    02_本轮代码与结果/scripts/              main pipeline: archive audit, verification, summaries, figures
    revision/analysis/                     analyses added during revision (three-way decomposition,
                                           multiplicity, deletion diagnostics, A3/A4 arms, decision layer,
                                           simulation uncertainty, row-weight sensitivity)
    revision/build/make_figures.py         figure generation
  05_figures/build/make_figures.py         figure generation (earlier path; byte-identical to
                                           08_repro_bundle/revision/build/make_figures.py)
  03_ExtraTrees_LightGBM/scripts/          robustness runs with other model families
  04_early_stopping/scripts/               early-stopping sensitivity (byte-identical copies of
                                           early_stopping.py and run_supplement.py above)
  new_experiments_2026-11-05/              Appendix B runs (native-NaN backbone, mask-only, seeds)
  10_repro_check_2026-11-05/               path-independent recomputation of Table 4
```

File and directory names inside `08_repro_bundle/` are kept exactly as they appear in the archived
package, including the Chinese directory names. They are part of the archive layout that the scripts
refer to; Section 4 maps them to English descriptions.

## 2. Environment

The archived predictions were produced with Python 3.12.15, NumPy 2.5.4 and scikit-learn 1.9.1 on macOS
(arm64). The robustness runs of Appendix B used Python 3.12.3, NumPy 2.5.1 and scikit-learn 1.9.0 on a
Linux machine with two cores and 4 GB of memory.

```
python>=3.10
numpy
scikit-learn
matplotlib
```

No model checkpoints are distributed. Nothing here retrains a model unless the command says so.

### 2.1 Environment variables

Two scripts of the Appendix B batch need files that are not part of this repository: the archive tree and
the APS training CSV. They read both from the environment and stop with a message naming the missing
variable instead of writing to a guessed location.

| Variable | Used by | Meaning |
|---|---|---|
| `CCMS_ARCHIVE` | `code/new_experiments_2026-11-05/run_new_experiments.py` | directory containing the archive's `src/` and `scripts/` subdirectories |
| `CCMS_APS_CSV` | same | path to `aps_failure_training_set.csv` |
| `CCMS_OUT` | same, and `summarize.py` | output directory; default `results/` next to the script |
| `CCMS_SUMMARY` | `summarize.py` | output markdown file; default `summary.md` next to the script |

```
export CCMS_ARCHIVE=/path/to/CCMS_archive
export CCMS_APS_CSV=/path/to/aps_failure_training_set.csv
```

## 3. Data

Both datasets are public and are not redistributed here.

- APS Failure at Scania Trucks: UCI Machine Learning Repository, https://doi.org/10.24432/C51S51
  (training file `aps_failure_training_set.csv`, 60,000 rows, 170 features, `neg`/`pos` labels).
- SECOM: UCI Machine Learning Repository, https://doi.org/10.24432/C54305

The Appendix B runner accepts the CSV location through an environment variable (Section 2.1). Other historical scripts may contain paths specific to the original working environment.

## 4. Archive layout and directory names

The scripts were written against the archived package layout. That layout uses Chinese directory names;
the table gives the English meaning of each so that a reader can rebuild it either way.

| Directory in the archive | Meaning | What must be inside |
|---|---|---|
| `02_本轮代码与结果/results` | pipeline results | `aps_seed{11,22,33,44,55}.npz`, `aps_reconstruction.json`, `archive_audit.json`, `secom_*.json/npz`, `simulation_*.json/npz` |
| `02_本轮代码与结果/scripts` | pipeline scripts | the scripts copied here |
| `03_原始项目归档/CCMS核查_2026-10-31` | frozen original project | `src/`, `records/`, `protocol/` |
| `04_实际使用的数据` | data actually used | `aps/aps_failure_training_set.csv`, `secom/` |
| `CCMS论文修订完整资料包_2026-11-01_192149` | the 2026-11-01 package root | the four directories above |
| `revision/analysis` | revision analyses | outputs are written here |

The revised `ESM_2.zip` (Online Resource 2) contains the saved arrays and archived scripts. Unpack it with UTF-8 filename support. The Table 4 command below was checked on the revised archive. Other historical scripts may depend on the original folder layout or additional data and have not all been verified in a clean environment.

## 5. Exhibit mapping and reproduction status

Only the Table 4 recomputation has been tested in a clean extracted copy of the revised Online Resource 2. The other commands below are historical references and may require path changes, additional inputs, or refitting; they are not verified instructions. The repository alone does not contain the prediction arrays or the original project archive.

The historical commands below were written for the package root. Their inputs, paths, runtime and whether they refit must be verified before use; only the Table 4 command has been tested in this revision.

| Exhibit | Command | Needs | Refit? |
|---|---|---|---|
| Table 3 (selection recovery) | `python3 08_repro_bundle/02_本轮代码与结果/scripts/audit_archive.py` | archive results | no |
| Table 4 (mean AP at fixed alpha) | `python3 10_repro_check_2026-11-05/reproduce_tables.py` | `aps_seed*.npz` | no |
| Table 5 (three-way decomposition) | `python3 08_repro_bundle/revision/analysis/run_r13_threeway.py` | `aps_seed*.npz`, `aps_bootstrap_draws.npz` | no |
| Table 5 (simultaneous correction) | `python3 08_repro_bundle/revision/analysis/run_r12_multiplicity.py` | draw matrices | no |
| Table 5 note (row weights) | `python3 08_repro_bundle/revision/analysis/lead_r10_rowweight_sensitivity.py` | draw matrices | no |
| Table 7 (deletion allocation) | `python3 08_repro_bundle/revision/analysis/run_r9_deletion.py` | `aps_reconstruction.json`, training CSV | no |
| Table 8 Panel A (local ablation) | `python3 08_repro_bundle/revision/analysis/run_r9_localization.py` | `localization_seed*.npz` | no |
| Table 8 Panel B (A3/A4 arms) | `python3 08_repro_bundle/revision/analysis/run_arm3_countmatched.py` | `arm3_seed*.npz`, `arm4_seed*.npz` | no |
| Table 9 (continuous simulation) | `python3 08_repro_bundle/revision/analysis/run_r6_sim.py` | `simulation_continuous.json` | no |
| Table 10 (recall at fixed budget) | `python3 08_repro_bundle/revision/analysis/run_r15_decision.py` | `aps_seed*.npz` | no |
| Figures 1 to 4 | `python3 08_repro_bundle/revision/build/make_figures.py` | the JSON files above | no |
| Tables B1 and B2 | `CCMS_ARCHIVE=… CCMS_APS_CSV=… python3 new_experiments_2026-11-05/run_new_experiments.py`, then `python3 new_experiments_2026-11-05/summarize.py` | archive `src/`, APS CSV | yes (140 fits) |
| ExtraTrees / LightGBM check | `python3 03_ExtraTrees_LightGBM/scripts/run_robustness.py` | APS CSV | yes |
| Early-stopping sensitivity | `python3 04_early_stopping/scripts/run_supplement.py` | APS CSV | yes |

Tables 1, 2 and A1 are written tables: Table 1 compares information conditions of the nearest work,
Table 2 defines the stress families, and Table A1 maps audit conclusions to artifacts. No script
generates them, and none is claimed to.

## 6. What this repository does not establish

The archive contains arrays and records supporting further checks of point estimates, intervals and counts; those checks have not all been independently rerun in this revision. They do
not show that the historical training runs happened exactly as described: there are no training logs, no
model files and no checkpoints, and the package does not re-run the original jobs. A matching SHA-256
shows that a file is the one distributed here, not that a run occurred.

The conditional intervals are conditional on the stored models. They cover neither retraining nor
reselection uncertainty, and the four Appendix B configurations are reported as per-split values and
means without recomputed bootstrap intervals.

## 7. Comments

All comments and docstrings in the code under `code/` are in English. Chinese remains in some string
literals that are directory names of the archive layout (Section 4); those are paths, not comments, and
were left unchanged so that the scripts keep pointing at the same files.

## 8. Citation and contact

If you use this material, please cite the manuscript. Correspondence: Yuchen Yang,
Beijing Institute of Petrochemical Technology, bgatechen@gmail.com.

