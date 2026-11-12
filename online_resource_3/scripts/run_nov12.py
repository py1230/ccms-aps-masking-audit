"""Portable launchers for the preserved November 12 generators.

Default: validate inputs and show a plan. --execute explicitly starts a fresh
run. Original generator/dependency files remain byte-preserved. Only paths and
the A3 run identifier are adapted in memory; statistical/model code is intact.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time


RESOURCE = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workflow", choices=["a3", "secom"])
    p.add_argument("--output", type=Path, required=True, help="A new output directory")
    p.add_argument("--execute", action="store_true", help="Run after input validation")
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        p.error("Output directory must not already exist")
    if output.is_relative_to(RESOURCE / "reproduction") or output.is_relative_to(RESOURCE / "reanalysis"):
        p.error("Choose an output directory outside preserved evidence directories")
    sys.dont_write_bytecode = True
    if args.workflow == "a3":
        received = RESOURCE / "reproduction/a3_model_seed_verification_20261112/received"
        post = received / "03_本次事后重建与核验"
        history = received / "01_历史原件/results_seed"
        source = RESOURCE / "reproduction/a3_model_seed_20261112/received/02_复现件_模型与掩码_非原件/run_a3_repro.py"
        values = {"RUN_ID": "A3_fresh_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                  "ROOT": str(post / "data"), "OUT": str(output),
                  "SRC": str(post / "code"), "SCR": str(post / "code")}
        required = [post / "data/aps_failure_training_set.csv", *sorted(history.glob("*.json")),
                    *sorted((post / "code").glob("*.py"))]
        if len(list(history.glob("*.json"))) != 30:
            raise ValueError("Expected 30 historical AP records")
        replacements = {"新增实验_2026-11-05/results_seed": str(history)}
    else:
        inputs = RESOURCE / "historical_recovery/protocol_content_recovery_20261111/sources"
        source = RESOURCE / "reanalysis/secom_bootstrap_20261112/scripts/run_secom_boot20k.py"
        values = {"ROOT": str(output), "INPUT": str(inputs),
                  "OUT": str(output / "exports"), "LOGS": str(output / "logs")}
        required = [inputs / f"secom_fold{i}.npz" for i in range(5)] + [inputs / "secom_summary.json"]
        replacements = {}
    manifest = {}
    for line in (RESOURCE / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split(None, 1)
            manifest[rel.strip().lstrip("*")] = h
    for path in [source, *required]:
        expected = manifest.get(str(path.relative_to(RESOURCE)))
        if expected is None and path.name == "aps_failure_training_set.csv":
            from prepare_a3_data import expected_hash
            expected = expected_hash()
        if not path.is_file():
            raise ValueError("Missing input; first run python scripts/prepare_a3_data.py --download")
        if digest(path) != expected:
            raise ValueError("Input fingerprint mismatch: " + str(path))
    code = ast.parse(source.read_text(), filename=str(source))
    changed = []
    for node in code.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in values:
                value = ast.Constant(values[name])
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[value], keywords=[]) if args.workflow == "secom" else value
                changed.append(name)
    if set(changed) != set(values):
        raise ValueError("Unexpected generator assignment structure")
    replacement_count = 0
    for node in ast.walk(code):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replacements:
            node.value = replacements[node.value]
            replacement_count += 1
    if args.workflow == "a3" and replacement_count != 2:
        raise ValueError("Unexpected historical-record path structure")
    ast.fix_missing_locations(code)
    plan = {"workflow": args.workflow, "identity": "fresh execution of preserved November 12 algorithm; not a recovered historical run",
            "generator": str(source.relative_to(RESOURCE)), "generator_sha256": digest(source),
            "launcher_sha256": digest(Path(__file__)), "assignments": values,
            "path_literal_replacements": replacement_count,
            "inputs": {str(x.relative_to(RESOURCE)): digest(x) for x in required},
            "execute": args.execute,
            "scope": "30 new classifier fits" if args.workflow == "a3" else "20000 new conditional bootstrap draws; no classifier fitting",
            "notice": "Generator-native metadata retains original wording. This launcher receipt governs the identity of this fresh run; global historical-absence claims are not established."}
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        return 0
    output.mkdir(parents=True, exist_ok=False)
    (output / "portable_launch.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    exec(compile(code, str(source), "exec"), {"__name__": "__main__", "__file__": str(source)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
