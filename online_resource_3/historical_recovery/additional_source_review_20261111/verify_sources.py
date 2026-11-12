"""Check preserved source identity and experiment labels; execute no training."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent

def check():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    for item in provenance['source_files']:
        path = HERE / 'original_sources' / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['path']
    src = HERE / 'original_sources'
    record = json.loads((src / '01_冻结协议/protocol_v2_freeze_20261031T141021.json').read_text())
    expected = record['files']['protocol/protocol_v2_冻结_2026-10-31.md']
    actual = hashlib.sha256((src / '01_冻结协议/protocol_v2_冻结_2026-10-31.md').read_bytes()).hexdigest()
    assert expected == '95f90d409d8d9cb1899aed623f02eee5a40300b41e6320e5d62de51e51de7352'
    assert actual == '88481a04127a0ae98be9fdcab8df92feb68fd0af022e4754036cd2ab788622cb'
    assert actual != expected  # The recorded mismatch is deliberately retained.
    secom = json.loads((src / '02_SECOM_bootstrap与结果/secom_formal_20261031T145429.json').read_text())
    comparisons = [v for a in secom['comparisons'].values() for s in a.values() for v in s.values()]
    assert len(comparisons) == 27
    assert secom['n_bootstrap'] == 1000
    assert all(x['n_boot'] == 1000 and x['resample'] == 'fold_stratified_rows' for x in comparisons)
    sim = json.loads((src / '02_SECOM_bootstrap与结果/r6_simulation_uncertainty.json').read_text())
    assert sim['n_bootstrap_of_seeds'] == 20000
    assert sim['data_seeds'] == [501,502,503,504,505]
    assert len(sim['target_gap_multipliers']) == 9
    tree = ast.parse((src / '02_SECOM_bootstrap与结果/run_r6_sim.py').read_text())
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ['N_BOOT','BOOT_SEED']}
    assert constants == {'N_BOOT': 20000, 'BOOT_SEED': 20261101}
    snapshot = json.loads((src / '03_环境记录/05_环境快照.json').read_text())
    assert snapshot['captured_at'] == '2026-11-01T19:21:49.698706+08:00'
    assert '打包时只读快照' in snapshot['meaning']
    for name, expected_hash in provenance['existing_a3_files'].items():
        path = HERE.parent / 'recovered/new_experiments_2026-11-05' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    return {'status':'passed','preserved_source_files':len(provenance['source_files']),
            'freeze_protocol_matches_recorded_hash':False,
            'secom_bootstrap_replicates':1000,'secom_comparison_records':len(comparisons),
            'simulation_bootstrap_replicates':20000,'simulation_data_seeds':sim['data_seeds'],
            'snapshot_scope':'2026-11-01 packaging snapshot, not an A3 runtime receipt',
            'existing_a3_runner_and_log_identical':True,'historical_gaps_closed':False,
            'training_executed':False,'bootstrap_recomputed':False}

if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
