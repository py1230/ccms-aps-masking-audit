"""Verify restored protocol bytes and saved predictions; no fitting or bootstrap draws."""
from pathlib import Path
import hashlib,json,difflib
import numpy as np

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'sources'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ap(y,s):
    order=np.argsort(-s,kind='stable'); ys=y[order]; ss=s[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(y)-1]
    tp=np.cumsum(ys)[ends]; return float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1)))/sum(y))
def check():
    p=json.loads((ROOT/'provenance.json').read_text())
    for x in p['source_files']:assert sha(SRC/x['file'])==x['sha256'],x['file']
    protocol=SRC/'protocol_v2_冻结_2026-10-31.95f90d40.md'
    assert sha(protocol)=='95f90d409d8d9cb1899aed623f02eee5a40300b41e6320e5d62de51e51de7352'
    old=ROOT.parent/'additional_source_review_20261111/original_sources/01_冻结协议/protocol_v2_冻结_2026-10-31.md'
    record=json.loads((old.parent/'protocol_v2_freeze_20261031T141021.json').read_text())
    assert record['files']['protocol/protocol_v2_冻结_2026-10-31.md']==sha(protocol)
    a=protocol.read_text().splitlines(True);b=old.read_text().splitlines(True)
    diffs=[x for x in difflib.SequenceMatcher(a=a,b=b).get_opcodes() if x[0]!='equal']
    assert len(diffs)==1 and diffs[0][0]=='replace'
    _,i,j,k,l=diffs[0]; assert j-i==l-k==2
    assert all(x.startswith('- [ ]') for x in a[i:j]) and all(x.startswith('- [x]') for x in b[k:l])
    z=np.load(SRC/'secom_oof.npz',allow_pickle=False); y=z['y'];fold=z['fold']
    assert y.shape==fold.shape==(1567,) and int(y.sum())==104
    assert set(np.unique(y))=={0,1} and set(np.unique(fold))==set(range(5))
    copied_predictions=0
    for f in range(5):
        with np.load(SRC/f'secom_fold{f}.npz',allow_pickle=False) as q:
            idx=q['test_rows']; assert np.array_equal(np.sort(idx),np.flatnonzero(fold==f))
            assert np.array_equal(y[idx],q['y_test'])
            sets=[set(q[n].tolist()) for n in ['train_rows','val_rows','test_rows']]
            assert len(set.union(*sets))==1567
            assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            for name in z.files:
                if name in ['y','fold']:continue
                config,cond=name.split('__',1);assert np.array_equal(z[name][idx],q[f'{config}__test__{cond}'])
                copied_predictions+=1
    summary=json.loads((SRC/'secom_summary.json').read_text());assert summary['n_bootstrap']==1000
    errors=[];scalar_checks=0
    for config,groups in summary['per_config'].items():
        for group,v in groups.items():
            conds=['S0'] if group=='S0' else ['S1_30_101','S1_30_202','S1_30_303']
            calculated=[float(np.mean([ap(y[fold==f],z[f'{config}__{c}'][fold==f]) for c in conds])) for f in range(5)]
            pooled=float(np.mean([ap(y,z[f'{config}__{c}']) for c in conds]))
            actual=np.r_[calculated,np.mean(calculated),pooled]
            expected=np.r_[v['fold_AP'],v['mean_fold_AP'],v['pooled_AP_secondary']]
            errors.extend(abs(actual-expected).tolist());scalar_checks+=len(actual)
    assert max(errors)<1e-12
    with np.load(SRC/'r9_arm3_bootstrap_draws.npz',allow_pickle=False) as q:
        assert len(q.files)==50 and all(q[k].shape==(1000,5) and np.isfinite(q[k]).all() for k in q.files)
        decomposition=[]
        for k in q.files:
            if k.endswith('__C-B'):
                base=k[:-3];decomposition.append(float(np.max(abs(q[k]-q[base+'C-A3']-q[base+'A3-B']))))
        assert len(decomposition)==10 and max(decomposition)<1e-12
    r6=json.loads((ROOT.parent/'additional_source_review_20261111/original_sources/02_SECOM_bootstrap与结果/r6_simulation_uncertainty.json').read_text())
    assert sha(SRC/'simulation_continuous.json')==r6['sources'][0]['sha256']
    return {'status':'passed','source_file_checks':len(p['source_files']),
        'restored_protocol_matches_frozen_hash':True,'differing_protocol_lines':2,
        'substantive_protocol_text_identical_between_two_files':True,
        'chat_export_independently_checked':False,'SECOM_rows':1567,'SECOM_positives':104,
        'fold_to_oof_prediction_arrays_equal':copied_predictions,'SECOM_AP_scalar_checks':scalar_checks,
        'SECOM_AP_max_error':max(errors),'SECOM_summary_bootstrap_count':1000,
        'r9_draw_arrays':50,'r9_draws_per_array':1000,'r9_decomposition_max_error':max(decomposition),
        'R6_input_hash_matches':True,'new_training':0,'new_bootstrap_resamples':0,
        'remaining_gaps':['A3 runtime/prediction/model/mask/index originals','SECOM historical 20,000-draw originals','two historical target ZIPs','unprovided current author facts']}
if __name__=='__main__':print(json.dumps(check(),ensure_ascii=False,indent=2))
