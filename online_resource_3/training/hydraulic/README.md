# Hydraulic independent-data and external-method experiment

Read `液压数据与外部方法实验报告.md` and `PROTOCOL.md` first. This is a new, fixed-design experiment dated 2026-11-09: one real hydraulic test rig, imposed missingness, grouped held-out condition combinations. It is not natural-missingness or cross-device confirmation.

Completed: 195 formal fits (105 HGB, 90 author MA-DT), five grouped outer folds, all prediction and selection checks passed. No failed or retried formal fits. Author MA-DT scientific definitions are unchanged; the small import-only extraction is documented in the sibling `external_baseline_20261109/extraction_manifest.json`.

## Replay saved results

From the project root:

```sh
experiments/external_baseline_20261109/.venv/bin/python experiments/hydraulic_external_20261109/audit.py
experiments/external_baseline_20261109/.venv/bin/python experiments/hydraulic_external_20261109/report.py
```

The audit reads the source ZIP, frozen inputs, all locally generated models and probabilities; it separately reconstructs AP, ensembles, selections, masks and split isolation. It checks 129 frozen files before computing results. The audit report is deterministic apart from elapsed-time metadata.

## Full rerun

Use a separate reproduction copy with the same sibling directory names. Retain `prepared`, source code, protocol, source ZIP, author source subset and `freeze.json`; do not copy `run_results`. Run `run.py run` in that fresh copy, followed by `audit.py`. Existing formal output directories are deliberately rejected. To regenerate features and masks, use a different fresh preparation copy containing the ZIP and source with no `prepared` directory, then run `prepare.py`; compare its artifacts to the frozen archive before any new training.

Python 3.12.15; NumPy1.26.4; SciPy1.13.1; scikit-learn1.5.2; pandas2.2.3; joblib1.4.2. `run_results/environment.json` records exact runtime information. The model run enforces the frozen Python/NumPy/SciPy/sklearn/joblib versions and at most two numerical threads. Source selection and full effective HGB parameters are frozen in `run.py` and `freeze.json`.

The figure was generated after results with Matplotlib3.10.6 in a separate plotting runtime; it is not part of the model environment or selection. `plot.py` writes the manuscript PNG/SVG. Numerical plotting input is the audited JSON and CSV.

## Data and code attribution

- UCI dataset: Helwig, Pignanelli and Schütze (2015), DOI https://doi.org/10.24432/C5CW21, CC BY 4.0. Official raw ZIP SHA256 `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`.
- Stempfle, Matsson, Mwai and Johansson (2025), ICML/PMLR267:56899–56919, https://proceedings.mlr.press/v267/stempfle25a.html. Healthy-AI/malearn commit `fbbdd0ca2c9d0d8072f49d82ad35f4391098863f`; original MIT license retained in the sibling vendor directory.
- Procedural assistance: Kassis et al. (2026), Scientific Agent Skills, DOI https://doi.org/10.48550/arXiv.2609.00065. This is tool/process attribution, not evidence of scientific validity.

Raw download failures are recorded under `data/download_attempts`; partial files were never used as data. Original input and all formal outputs are retained. Local storage does not imply publication or independent third-party replication.
