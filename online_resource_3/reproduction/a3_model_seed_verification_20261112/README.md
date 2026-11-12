# A3 complete delivery and post hoc model replay

The delivery has 119 payloads and its original manifest. This public package preserves 118 payloads plus that manifest; the 44.7 MB upstream APS CSV is acquired separately with `python scripts/prepare_a3_data.py --download` from the resource root. The downloader checks the original manifest fingerprint and restores the original received path, after which all 119 payloads can be checked. Historical records, 12 November reproduction outputs and post hoc reconstruction are separated by the supplier's three directories. The CSV remains reproduction output, following the author's accepted provenance correction. Earlier delivery and generation scripts remain in ../a3_model_seed_20261112/.

The local independent verifier reloads all 30 saved models, checks every row/label mapping in all 30 CSV groups, and reconstructs ten training masks, five evaluation-mask sets, training-only medians and five M3 matrix hashes. CSV and NPZ probabilities agree exactly. Model probabilities differ by at most 1.1102230246251565e-16; all 30 AP values equal the separate historical scalar files exactly. No classifier training is performed.

The model replay used Python 3.12.15, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 and joblib 1.6.0. Producer metadata records scikit-learn 1.9.0; version warnings are retained in independent_model_replay.json. This successful check does not establish arbitrary-version portability. Producer pandas/joblib versions are not recorded.

From online_resource_3, with those libraries available:

    python scripts/prepare_a3_data.py --download
    python reproduction/a3_model_seed_verification_20261112/verify_complete.py --out a3_replay_verification.json

The verifier reads received artifacts and recomputes deterministic masks/preprocessing in memory, without modifying received files. Failed checks return nonzero. An isolated negative test changed the expected M3 hash and updated the manifest; the verifier rejected the mismatch with exit code 1. The supplier's original verifier is preserved, but its exit code alone is insufficient: several comparison failures do not affect exit status, one matrix-hash comparison is unused, and its CSV mapping check covers only one group per split.

Evaluation masks and imputers are post hoc reconstructions. Equal predictions do not uniquely identify historical input matrices. Historical models, actual masks, original per-sample outputs, effective environment and complete input/code/execution linkage remain unrecovered. The full command transcription matches the runner bytes and the terminal's 500-character prefix, but the raw conversation projection is absent. Its nohup background invocation means terminal exit code 0 is a launcher status, not an independently captured Python training exit. SEED_DONE and the 30 result files provide separate completion evidence.

Read 核验说明.md for the detailed evidence boundary. No SECOM reanalysis or classifier training was performed in this integration. The previous 211-fit acceptance scope is unchanged. This complete delivery is included in the 12 November public edition.
