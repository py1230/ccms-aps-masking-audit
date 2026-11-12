#!/usr/bin/env python3
"""
Independently replay the APS alpha-selection rule from the archived candidate
validation-score table in ESM_2.zip.

This verifies the selection logic conditional on the archived candidate AP
summaries. It does NOT recompute AP for alpha=0.5 or alpha=1.0 from raw
per-sample predictions, because those candidate-level prediction arrays are
not present in the current ESM_2 archive.
"""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm-zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    member = "ESM_2/08_repro_bundle/02_本轮代码与结果/results/archive_audit.json"
    with zipfile.ZipFile(args.esm_zip) as z:
        raw = z.read(member)
    audit = json.loads(raw)
    records = []
    all_ok = True

    for split in audit["corrected_selection"]:
        seed = int(split["seed"])
        for arm in ("C", "B"):
            block = split["arms"][arm]
            candidates = block["all_candidates"]
            baseline = next(float(c["s0_ap"]) for c in candidates if float(c["alpha"]) == 0.0)
            eligible = []
            eligibility_checks = []
            for c in candidates:
                alpha = float(c["alpha"])
                observed_eligible = bool(c["eligible"])
                replay_eligible = float(c["s0_ap"]) >= baseline - 0.01 - 1e-12
                eligibility_checks.append({
                    "alpha": alpha,
                    "replayed_eligible": replay_eligible,
                    "archived_eligible": observed_eligible,
                    "matches": replay_eligible == observed_eligible,
                    "s0_ap": float(c["s0_ap"]),
                    "worst4_ap": float(c["worst4"]),
                })
                if replay_eligible:
                    eligible.append(c)

            chosen = sorted(
                eligible,
                key=lambda c: (-float(c["worst4"]), float(c["alpha"]))
            )[0]
            archived_alpha = float(block["corrected"]["alpha"])
            archived_cfg = str(block["corrected"]["cfg"])
            ok = (
                all(x["matches"] for x in eligibility_checks)
                and float(chosen["alpha"]) == archived_alpha
                and str(chosen["cfg"]) == archived_cfg
            )
            all_ok = all_ok and ok
            records.append({
                "seed": seed,
                "arm": arm,
                "replayed_alpha": float(chosen["alpha"]),
                "archived_alpha": archived_alpha,
                "replayed_cfg": str(chosen["cfg"]),
                "archived_cfg": archived_cfg,
                "eligibility_checks": eligibility_checks,
                "matches": ok,
                "selected_test_S1_30_AP": float(split["test_metrics"][arm]["S1_30"]),
            })

    selected = {
        arm: [r["replayed_alpha"] for r in records if r["arm"] == arm]
        for arm in ("C", "B")
    }
    test_s1_30 = {
        arm: float(sum(r["selected_test_S1_30_AP"] for r in records if r["arm"] == arm) /
                   sum(1 for r in records if r["arm"] == arm))
        for arm in ("C", "B")
    }
    result = {
        "title": "APS alpha-selection rule replay from archived candidate validation scores",
        "source": member,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "selection_rule": {
            "feasibility": "candidate S0 AP loss <= 0.01 relative to alpha=0 baseline",
            "objective": "maximize the worst AP across the four validation inputs",
            "tie_break": "choose smaller alpha",
        },
        "n_split_arm_checks": len(records),
        "all_replays_match_archive": all_ok,
        "selected_alpha_by_arm": selected,
        "mean_selected_test_S1_30_AP": test_s1_30,
        "records": records,
        "limitations": [
            "This replays the selection rule using archived per-candidate AP summaries.",
            "It does not recompute candidate AP values from raw per-sample predictions for alpha=0.5 and alpha=1.0.",
            "The archived per-sample APS prediction arrays contain M3, C(0.25), and B(0.25), not all alpha candidates.",
            "The official test metrics are read only after the validation-based choice and are not used by the selection rule.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_replays_match_archive": all_ok,
        "selected_alpha_by_arm": selected,
        "mean_selected_test_S1_30_AP": test_s1_30,
        "output": str(out),
    }, ensure_ascii=False, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
