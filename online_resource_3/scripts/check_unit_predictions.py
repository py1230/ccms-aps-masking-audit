"""Compare a fresh original-unit run with the public archived score arrays."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / 'acceptance/reference_prediction_manifest.json'


def main():
    references = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unit', choices=sorted(references))
    parser.add_argument('--work-directory', required=True, type=Path)
    args = parser.parse_args()
    work = args.work_directory.resolve()
    arrays_checked = 0
    for item in references[args.unit]:
        expected_file = (PACKAGE / item['package_relative']).resolve()
        actual_file = (work / item['work_relative']).resolve()
        if not expected_file.is_relative_to(PACKAGE) or not actual_file.is_relative_to(work):
            raise ValueError('Reference path leaves its expected directory')
        if hashlib.sha256(expected_file.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Published reference fingerprint changed {expected_file}')
        with np.load(expected_file, allow_pickle=False) as expected, np.load(actual_file, allow_pickle=False) as actual:
            if not set(expected.files).issubset(actual.files):
                raise ValueError(f'Array names differ {actual_file}')
            for name in expected.files:
                x, y = expected[name], actual[name]
                if x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=True):
                    raise ValueError(f'Array differs {actual_file} {name}')
                arrays_checked += 1
    print(json.dumps({'status': 'passed', 'unit': args.unit, 'arrays_checked': arrays_checked,
                      'scope': 'Exact public archived score, label and index arrays for the named original unit.'}))


if __name__ == '__main__':
    main()
