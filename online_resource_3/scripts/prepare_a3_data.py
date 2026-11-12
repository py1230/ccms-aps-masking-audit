"""Acquire the original APS training CSV and verify its preserved fingerprint.

The public package omits this large upstream data file. Use --download for the
official UCI archive, or --file for an already downloaded training CSV. The
original Scania notice remains embedded in the file. No data are transformed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reproduction/a3_model_seed_verification_20261112/received/03_本次事后重建与核验/data/aps_failure_training_set.csv"
URL = "https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"


def expected_hash():
    received = ROOT / "reproduction/a3_model_seed_verification_20261112/received"
    key = DEST.relative_to(received).as_posix()
    for line in (received / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split(None, 1)
            if rel.strip().lstrip("*./") == key:
                return h
    raise ValueError("The preserved data fingerprint is absent")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--download", action="store_true")
    g.add_argument("--file", type=Path, help="Existing, unchanged APS training CSV")
    a = p.parse_args()
    expected = expected_hash()
    plan = {"url": URL, "destination": str(DEST), "sha256": expected,
            "citation": "APS Failure at Scania Trucks (2016), DOI 10.24432/C51S51",
            "notice": "Original Scania GPL v3-or-later header retained; see DATA_ACCESS.md"}
    if DEST.exists():
        if digest(DEST) != expected:
            raise ValueError("Existing data fingerprint differs; refusing to overwrite")
        print(json.dumps({**plan, "status": "PASS_ALREADY_PRESENT"}, indent=2))
        return
    if not a.download and not a.file:
        print(json.dumps({**plan, "status": "PLAN_ONLY", "next": "Use --download or --file PATH"}, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="ccms-aps-") as tmp:
        tmp = Path(tmp)
        if a.file:
            source = a.file.resolve()
        else:
            archive = tmp / "uci_aps.zip"
            with urlopen(URL, timeout=120) as src, archive.open("wb") as out:
                shutil.copyfileobj(src, out)
            with zipfile.ZipFile(archive) as z:
                members = [n for n in z.namelist() if Path(n).name == DEST.name]
                if len(members) != 1:
                    raise ValueError("Unexpected UCI archive members")
                source = tmp / DEST.name
                with z.open(members[0]) as src, source.open("wb") as out:
                    shutil.copyfileobj(src, out)
        if digest(source) != expected:
            raise ValueError("UCI/source data fingerprint differs; expected hash is not changed")
        DEST.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, DEST.open("xb") as out:
            shutil.copyfileobj(src, out)
    if digest(DEST) != expected:
        raise ValueError("Copied data fingerprint mismatch")
    print(json.dumps({**plan, "status": "PASS_ACQUIRED", "bytes": DEST.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
