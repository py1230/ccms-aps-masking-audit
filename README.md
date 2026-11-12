# CCMS reproducibility materials

**Auditing train-time class-conditional masking under missingness shift: budget-matched evidence from industrial fault classification**

Yuchen Yang and Jian Guo · Beijing Institute of Petrochemical Technology, Beijing, China. Corresponding author: Jian Guo (gj2021@bipt.edu.cn). Prepared for *Pattern Analysis and Applications*; no acceptance or submission status is implied.

The [12 November public edition](online_resource_3/PUBLIC_RELEASE_20261112.md) includes the manuscript's current evidence package, unpacked for inspection and execution. Start with [current evidence and limitations](online_resource_3/CURRENT_STATUS_20261112.md), [execution instructions](online_resource_3/README.md), [training schedules](online_resource_3/TRAINING.md), and the [result-to-evidence map](online_resource_3/RESULT_MAP.md).

New in this edition:

- Recovered 11 November historical sources and original archives, with content available unpacked.
- Correctly labelled 12 November A3 reproduction: 30 models, sample-level predictions, masks, post hoc imputers and strict independent replay.
- SECOM conditional reanalysis: all 20,000-draw arrays and resampling counts, plus deterministic verification of 400,000 fold/draw differences.
- Portable launchers, public data notices, payload fingerprints and release-check results.

Saved-model/draw agreement is computational consistency evidence, not proof of historical training provenance. Historical A3 models/runtime and historical SECOM draw/execution records remain unavailable. New intervals are conditional and do not establish equivalence.

The [data access notice](online_resource_3/DATA_ACCESS.md) provides original dataset citations and licenses. Author-owned software is MIT; third-party code and datasets retain their own notices. `code/` and `legacy/` remain earlier snapshots; use `online_resource_3/` for current reproduction. `RELEASE_20261110.json` describes the old release; `RELEASE_20261112.json` describes this edition.
