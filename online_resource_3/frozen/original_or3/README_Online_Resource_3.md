# Online Resource 3 — 20,000-draw conditional bootstrap reanalysis

This resource extends the primary APS C/B/M3 conditional bootstrap analysis from 1,000 to 20,000 paired, class-stratified row-weight replicates using the archived per-sample predictions in Online Resource 2 (ESM_2.zip). It does **not** retrain models, regenerate training masks, or repeat model selection.

## Files

- `reproduce_bootstrap_20k.py`: portable reproduction script.
- `r13_threeway_20k.json`: point estimates, 20,000-draw intervals, reconciliation checks and protocol metadata.
- `aps_threeway_draws_20k.npz`: bootstrap draw matrices for C-B, C-M3 and B-M3 across validation/test and S0/S1_30/S2/S3.
- `CCMS_20k_bootstrap_summary.json`: nominal, Bonferroni, sup-t and Holm summary used to update Table 5.
- `01_three_way_decomposition_20k.png` / `.pdf`: updated Figure 1 with the 20,000-draw intervals.

## Reproduction

Extract Online Resource 2 so that its root contains `08_repro_bundle/CCMS论文修订完整资料包_2026-11-01_192149/02_本轮代码与结果/results/`. Then run:

```bash
ESM_ROOT=/path/to/ESM_2 \
CCMS_OUT=/path/to/output \
CCMS_NBOOT=20000 \
python3 reproduce_bootstrap_20k.py
```

Requirements: Python 3.10+, NumPy, scikit-learn and Matplotlib (Matplotlib is only needed to regenerate Figure 1; the bootstrap script itself uses NumPy and scikit-learn). The script checks that the point estimates and the first 1,000 bootstrap draws match the archived values; the intervals are intentionally re-estimated using all 20,000 draws.

## Interpretation boundary

The additional 19,000 draws reduce Monte Carlo error in the conditional bootstrap intervals. They do not address model-training, training-mask resampling, or selection uncertainty. A3/A4 path-decomposition intervals and fixed-alert-budget decision intervals in the manuscript remain based on the archived 1,000-draw matrices and are labelled separately. Supplementary model configurations in Appendix B remain descriptive summaries and are not assigned new bootstrap intervals.
