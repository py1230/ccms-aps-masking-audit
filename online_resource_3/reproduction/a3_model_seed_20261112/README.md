> Later update on 12 November: post hoc evaluation masks and imputers are now supplied and all 30 models have been replayed. Read ../a3_model_seed_verification_20261112/README.md and ../../CURRENT_STATUS_20261112.md. The following is the earlier delivery-stage report; its missing-artifact and no-replay statements are superseded. The historical exit code is now qualified as launcher-only.

# A3 model seed reproduction dated 12 November 2026

The received delivery is preserved under received/. Its 01 directory holds supplied historical runner/log/scalar records and terminal receipts; its 02 directory holds new reproduction artifacts. Its 03 directory contains the supplier's cross-reference table. The original 91-entry manifest verifies all payload files except itself.

The 270,000-row CSV is reproduction output. The authors explicitly corrected their earlier historical-original identification after checking the identical file in this run archive. See author_correction.json; earlier statements remain as dated records.

Independent checks found exact CSV/NPZ agreement for all 270,000 probabilities and matching sample indices/labels. All 30 AP values agree with historical scalars within 2.23e-16. Ten saved training masks (five splits by two arms) match their hashes and an in-memory reconstruction from seed 9107, the matching raw CSV and archived training indices. All 30 model-file hashes match; models were not deserialized and inference was not replayed. The new environment records Python 3.12.3, NumPy 2.5.1 and scikit-learn 1.9.0; these metadata are supplied records, not measurements of the historical runtime.

Historical terminal receipts record launch on 4 November 2026 at 18:30:01 UTC and exit code 0. Their command field is only 500 characters, ending mid-import. It does not contain the full invocation, runtime versions or historical input/dependency hashes. The historical script and 30 AP records are byte-identical to previously archived copies. The script saves scalar APs and does not save models, masks, predictions or environment snapshots; this does not prove that no other historical copy ever existed.

Historical fitted models, actual masks, effective environment and complete input/code/execution linkage remain unavailable. The new package also lacks separately saved evaluation masks and fitted preprocessing objects. Its environment hashes cover the runner, ccms.py, pipeline.py and raw data, but not aps_io.py or run_cb_firstlook.py. Original cloud paths are retained; no standalone portable training command or end-to-end model replay is claimed.

From the online_resource_3 directory, Python with NumPy can run:

    python reproduction/a3_model_seed_20261112/verify_artifacts.py

This verifies the supplied manifest, saved probability arrays, sample mapping, AP values, mask hashes and model hashes. It neither loads models nor regenerates masks; the raw-data-dependent mask reconstruction is recorded in independent_verification.json. No training or new bootstrap was performed in this integration.
