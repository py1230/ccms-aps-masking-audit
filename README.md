# CCMS reproducibility materials

This repository accompanies the manuscript *Auditing train-time class-conditional masking under missingness shift — budget-matched evidence from industrial fault classification* by Yuchen Yang and Jian Guo.

The current resource is [Online Resource 3](online_resource_3/README.md), version 2026-11-10. Its complete payload is bound by [SHA256SUMS](online_resource_3/SHA256SUMS.txt). The [release record](RELEASE_20261110.json) identifies this dated distribution.

## Start here

```sh
cd online_resource_3
python3.12 -m venv .venv-verification
.venv-verification/bin/python -m pip install -r requirements-verification.txt
.venv-verification/bin/python scripts/verify_saved_results.py
```

Complete training commands and the three pinned environments are in [TRAINING.md](online_resource_3/TRAINING.md). The normal training entry supports all nine experiment families. Raw industrial measurements are obtained through the included download commands and exact member checks. Historical indices, resampling inputs, paired A5 references and adapted third-party code are included.

## Executed acceptance

Fresh-environment acceptance executed original full-size units from every family, with an additional initial-environment APS diagnostic. The retained acceptance covers 211 fits. Comparison against the corresponding research archives checked 747 NPZ files and 7,355 arrays with zero maximum absolute difference. Selection and metric scientific fields also agree. Exact scopes, logs and environment records are in the [acceptance summary](online_resource_3/acceptance/acceptance_summary.json).

The resource also supports complete strict-capacity replay, AP bootstrap regeneration, and the [284-cell historical-table verification](online_resource_3/historical_recovery/RECOVERY_SCOPE.md). Historical prediction and training-provenance limitations remain explicit. The separate historical model-seed training script and portions of its mask and evaluation metadata remain unavailable. Acceptance covers the listed original units, and the complete schedules for further units are provided.

## Software and data permissions

Author-owned software is released under [MIT](LICENSE). Third-party code and datasets retain their own notices, documented in [LICENSE_NOTICE.md](online_resource_3/LICENSE_NOTICE.md) and [DATA_ACCESS.md](online_resource_3/DATA_ACCESS.md).

## Earlier repository material

The existing `code/` directory is retained as the earlier historical release. Its previous documentation is preserved in [the 5 November README](legacy/README_20261105.md). Use Online Resource 3 for the current executable instructions and manuscript object mapping.

Corresponding author Jian Guo, Beijing Institute of Petrochemical Technology. Contact gj2021@bipt.edu.cn.
