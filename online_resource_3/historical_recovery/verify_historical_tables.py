"""Recalculate four historical tables from recovered archived AP records.

The check uses saved scalar AP values. It does not retrain models or recreate
per-sample scores. Run from any directory with Python 3.12 or later.
"""

import argparse
import hashlib
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
SOURCE = ROOT / "recovered" / "new_experiments_2026-11-05"
DOCUMENT = PROJECT / "paa_revision_20261109/stage6_v1.0/Online_Resource_2_Historical_Technical_Archive.docx"
OUTER = Path("/Users/bgc/Downloads/PAA 投稿包 v5(13 个文件 + 上传对照单,88 MB).zip")
NOV6_HASHES = Path("/Users/bgc/Downloads/论文实质修订_20261106/分析脚本与复算结果/补充输入哈希.json")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SPLITS = [11, 22, 33, 44, 55]
MASK_SEEDS = [9107, 9108, 9109]
MODEL_SEEDS = [2026, 2027, 2028]
HEAD = ["S0", "S1_30_101", "S2_1101", "S3_2101"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_tables():
    if not DOCUMENT.exists():
        return json.loads((ROOT / "historical_table_targets.json").read_text())["tables"]
    with zipfile.ZipFile(DOCUMENT) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    tables = {}
    last_paragraph = ""
    for child in root.find("w:body", NS):
        if child.tag == f"{{{NS['w']}}}p":
            text = "".join(item.text or "" for item in child.findall(".//w:t", NS))
            if text:
                last_paragraph = text
        if child.tag == f"{{{NS['w']}}}tbl":
            rows = []
            for row in child.findall("w:tr", NS):
                rows.append([
                    "".join(item.text or "" for item in cell.findall(".//w:t", NS))
                    for cell in row.findall("w:tc", NS)
                ])
            tables[last_paragraph.split()[1]] = rows
    return tables


def normalized_member_name(name):
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def check_archive():
    with zipfile.ZipFile(ROOT / "recovered_ESM_2.zip") as archive:
        members = {
            normalized_member_name(item.filename).removeprefix("ESM_2/"): archive.read(item)
            for item in archive.infolist() if not item.is_dir()
        }
        manifest = members["SHA256SUMS_full.txt"].decode("utf-8")
    full_checks = []
    for line in manifest.splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        actual = sha256(members[name]) if name in members else None
        full_checks.append({"member": name, "expected": expected, "actual": actual, "match": actual == expected})
    prior_checks = []
    for row in json.loads(NOV6_HASHES.read_text()):
        name = row["path_relative_to_ESM_2"]
        if not name.startswith("new_experiments_2026-11-05/"):
            continue
        actual = sha256(members[name])
        prior_checks.append({"member": name, "nov6_expected": row["sha256"], "actual": actual, "match": actual == row["sha256"]})
    source_manifest = []
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file():
            member = path.relative_to(ROOT / "recovered").as_posix()
            actual = sha256(path.read_bytes())
            source_manifest.append({"member": member, "sha256": actual, "size": path.stat().st_size, "matches_archived_member": actual == sha256(members[member])})
    return {
        "outer_archive": str(OUTER),
        "outer_sha256": sha256(OUTER.read_bytes()),
        "nested_member_suffix": "/04_Online_Resources/ESM_2.zip",
        "nested_sha256": sha256((ROOT / "recovered_ESM_2.zip").read_bytes()),
        "nested_regular_members": len(members),
        "full_manifest_checks": full_checks,
        "nov6_independent_fingerprint_checks": prior_checks,
        "recovered_member_manifest": source_manifest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-only", action="store_true", help="Verify packaged tables and saved AP records without local source ZIPs or earlier audit files.")
    args = parser.parse_args()
    tables = read_tables()
    source_paths = sorted((SOURCE / "results").glob("*.json"))
    records = [json.loads(path.read_text()) for path in source_paths]
    seed_paths = sorted((SOURCE / "results_seed").glob("*.json"))
    seed_records = [json.loads(path.read_text()) for path in seed_paths]
    expected = Counter((exp, split, arm, mask_seed)
                       for exp in "ABCD" for split in SPLITS for arm in ["none", "C", "B"]
                       for mask_seed in ([None] if arm == "none" else MASK_SEEDS))
    observed = Counter((row["exp"], row["split"], row["arm"], row["mask_seed"]) for row in records)
    assert observed == expected
    assert all(row["alpha"] == 0.25 for row in records)
    assert all(set(row["ap"]) == set(records[0]["ap"]) for row in records)
    expected_seed = Counter((split, arm, model_seed)
                            for split in SPLITS for arm in ["C", "B"] for model_seed in MODEL_SEEDS)
    observed_seed = Counter((row["split"], row["arm"], row["model_seed"]) for row in seed_records)
    assert observed_seed == expected_seed
    checks = []

    def record_value(exp, split, arm, conditions):
        rows = [row for row in records if row["exp"] == exp and row["split"] == split and row["arm"] == arm]
        return statistics.fmean(statistics.fmean(row["ap"][condition] for row in rows) for condition in conditions)

    def check(table, row, column, expected_value, source_locator):
        displayed = tables[table][row][column].strip().replace("−", "-").replace("+", "")
        actual_formatted = f"{expected_value:.4f}"
        target_formatted = f"{float(displayed):.4f}"
        checks.append({"table": table, "row": row, "column": column, "displayed": displayed,
                       "recalculated": expected_value, "formatted_match": actual_formatted == target_formatted,
                       "source_locator": source_locator})

    # A2 averages individual-model AP over masks and over three stress inputs.
    for row_index, row in enumerate(tables["A2"][1:], 1):
        exp = "ABCD"[(row_index - 1) // 10]
        split = int(row[1])
        stress = "S1_10" if "10%" in row[2] else "S1_30"
        conditions = [f"{stress}_{seed}" for seed in [101, 202, 303]]
        values = [record_value(exp, split, arm, conditions) for arm in ["none", "C", "B"]]
        values.append(values[1] - values[2])
        for column, value in enumerate(values, 3):
            check("A2", row_index, column, value, f"results/{exp}_s{split}_*.json / {stress}_{{101,202,303}}")

    for row_index, row in enumerate(tables["A3"][1:], 1):
        split, model_seed = map(int, row[:2])
        values = [next(item["ap"] for item in seed_records
                       if item["split"] == split and item["model_seed"] == model_seed and item["arm"] == arm)
                  for arm in ["C", "B"]]
        values.append(values[0] - values[1])
        for column, value in enumerate(values, 2):
            check("A3", row_index, column, value, f"results_seed/s{split}_{{C,B}}_{model_seed}.json")

    for row_index, row in enumerate(tables["B3"][1:], 1):
        exp = ["B", "A", "D", "C"][(row_index - 1) // 4]
        condition = HEAD[(row_index - 1) % 4]
        values = [statistics.fmean(record_value(exp, split, arm, [condition]) for split in SPLITS)
                  for arm in ["none", "C", "B"]]
        values.append(values[1] - values[2])
        for column, value in enumerate(values, 2):
            check("B3", row_index, column, value, f"results/{exp}_s{{11,22,33,44,55}}_*.json / {condition}")

    for row_index, row in enumerate(tables["B4"][1:], 1):
        split = int(row[0])
        for column, exp in enumerate(["B", "A", "D"], 1):
            value = record_value(exp, split, "C", ["S1_30_101"]) - record_value(exp, split, "B", ["S1_30_101"])
            check("B4", row_index, column, value, f"results/{exp}_s{split}_{{C,B}}_{{9107,9108,9109}}.json / S1_30_101")

    archive = {} if args.table_only else check_archive()
    result = {
        "scope": "Four historical table point-estimate sources; saved scalar-AP aggregation, without retraining or prediction-level AP recalculation",
        "target_document": DOCUMENT.name,
        "target_document_sha256": sha256(DOCUMENT.read_bytes()) if DOCUMENT.exists() else json.loads((ROOT / "historical_table_targets.json").read_text())["document_sha256"],
        "configuration_record_count": len(records),
        "model_seed_record_count": len(seed_records),
        "configuration_protocol_grid_complete": observed == expected,
        "model_seed_grid_complete": observed_seed == expected_seed,
        "checks_per_table": dict(Counter(row["table"] for row in checks)),
        "numeric_cell_count": len(checks),
        "numeric_cell_matches": sum(row["formatted_match"] for row in checks),
        "checks": checks,
        "archive": archive,
        "remaining_limits": [
            "Saved JSONs contain scalar AP values, without per-sample predictions or saved fitted models for these historical supplementary runs.",
            "The separately recovered model-seed runner and log specify validation and mask settings; historical runtime versions and per-sample prediction files remain unavailable. See model_seed_recovery_20261111/RECOVERY_NOTE.md.",
            "The four-configuration runner retains original absolute Linux paths and missing runtime version details.",
            "Recalculation and file fingerprints do not authenticate historical training events or establish new confirmatory evidence.",
        ],
    }
    (ROOT / "historical_table_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    assert len(checks) == 284
    assert all(row["formatted_match"] for row in checks), [row for row in checks if not row["formatted_match"]]
    assert all(row["match"] for row in archive.get("nov6_independent_fingerprint_checks", []))
    assert all(row["matches_archived_member"] for row in archive.get("recovered_member_manifest", []))
    print(json.dumps({"numeric_cells": len(checks), "matches": result["numeric_cell_matches"],
                      "per_table": result["checks_per_table"],
                      "full_manifest_matches": sum(row["match"] for row in archive.get("full_manifest_checks", [])),
                      "full_manifest_total": len(archive.get("full_manifest_checks", [])),
                      "nov6_fingerprint_matches": len(archive.get("nov6_independent_fingerprint_checks", []))}, indent=2))


if __name__ == "__main__":
    main()
