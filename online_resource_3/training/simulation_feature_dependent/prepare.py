from pathlib import Path
import shutil,json,hashlib
P=Path(__file__).resolve().parent; OLD=P.parent/'simulation_expansion_20261108'
files=['run_simulation.py','launch_simulation.py','analyze_results.py','verify_results.py']
for n in files:shutil.copy2(OLD/n,P/n)
for n in ['sim_nuisance.py','ccms.py','pipeline.py','audit_metrics.py']:shutil.copy2(OLD/'source'/n,P/'source'/n)
manifest={str(p.relative_to(OLD)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(OLD/n for n in files),*(OLD/'source'/n for n in ['sim_nuisance.py','ccms.py','pipeline.py','audit_metrics.py'])]}
(P/'audit/source_provenance.json').write_text(json.dumps({'base_directory':str(OLD),'base_sha256':manifest,'copied_not_modified_in_place':True},indent=2))
cfg=json.loads((OLD/'config.json').read_text());cfg.update(data_seeds=list(range(701,731)),smoke_seed=99002,workers=1,bootstrap_seed=20261109,mechanism={'name':'label_and_observed_covariate_calibrated_logit','auxiliary_index':19,'slope':1.0,'aux_clip':[-4,4],'intercept_bisection_steps':80,'intercept_bracket':[-45.,45.],'class_mean_tolerance':1e-12,'auxiliary_always_observed':True,'preserve_class_mean_probability':True,'old_new_mechanism_contrast_identified':False})
(P/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
# Main runner uses fixed 80-iteration bisection, vectorised across 19 non-auxiliary columns.
p=P/'run_simulation.py';s=p.read_text().replace('range(601,631)','range(701,731)')
a=s.index('def natural(');b=s.index('\ndef specifications',a)
s=s[:a]+'''def natural(xx,yy,b,delta,m,U):
    pi=yy.mean();q1=b+(1-pi)*delta*m;q0=b-pi*delta*m
    assert min(q1.min(),q0.min())>=0 and max(q1.max(),q0.max())<=1
    probs=np.zeros_like(xx);intercepts=np.zeros((2,xx.shape[1]));maxerr=0.
    for cls,target in [(0,q0),(1,q1)]:
        rr=np.flatnonzero(yy==cls);zz=np.clip(xx[rr,19],-4,4)[:,None]
        use=np.flatnonzero((target>0)&(target<1));lo=np.full(len(use),-45.);hi=np.full(len(use),45.)
        for _ in range(80):
            mid=(lo+hi)/2;pred=1/(1+np.exp(-(mid[None,:]+zz)))
            below=pred.mean(0)<target[use];lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        ac=(lo+hi)/2;pred=1/(1+np.exp(-(ac[None,:]+zz)))
        probs[np.ix_(rr,use)]=pred;intercepts[cls,use]=ac
        probs[np.ix_(rr,np.flatnonzero(target==1))]=1.
        maxerr=max(maxerr,float(np.max(np.abs(probs[rr].mean(0)-target))))
    assert maxerr<=1e-12,maxerr
    assert np.all(probs[:,19]==0)
    mask=U<probs;z=xx.copy();z[mask]=np.nan
    return z,mask,{'q1':q1.tolist(),'q0':q0.tolist(),'prevalence':float(pi),
                  'expected_overall_missing':float(probs.mean()),'expected_per_column':probs.mean(0).tolist(),
                  'actual_missing':float(mask.mean()),'intercept_by_class_column':intercepts.tolist(),
                  'class_mean_max_error':maxerr,'auxiliary_always_observed':True}
''' +s[b:]
s=s.replace('b,delta=base_rates(ds);b[:6]=.4','b,delta=base_rates(ds);b[:6]=.4;b[19]=0;delta[19]=0')
s=s.replace("new=(~np.isnan(v)) & (U[f'U_stress_{ss}']<.1)","new=(~np.isnan(v)) & (U[f'U_stress_{ss}']<.1);new[:,19]=False")
s=s.replace("'data_meta':meta,'source_selection'","'data_meta':meta,'mechanism':cfg['mechanism'],'source_selection'")
s=s.replace("['run_simulation.py','launch_simulation.py','freeze_run.py','config.json']","['run_simulation.py','launch_simulation.py','config.json']")
p.write_text(s)
# Statistical analysis: same family, different fixed resampling seed.
p=P/'analyze_results.py';s=p.read_text().replace('range(601,631)','range(701,731)').replace('BOOTSTRAP_SEED = 20261108','BOOTSTRAP_SEED = 20261109');p.write_text(s)
# Separate verifier derives the logit intercept with scipy brentq, not runner bisection.
p=P/'verify_results.py';s=p.read_text().replace('range(601,631)','range(701,731)').replace('default_rng(20261108)','default_rng(20261109)')
s=s.replace('from scipy import stats','from scipy import stats\nfrom scipy.optimize import brentq\nfrom scipy.special import expit')
s=s.replace('delta=np.r_[np.full(6,.4),np.zeros(14)]','delta=np.r_[np.full(6,.4),np.zeros(14)];b[19]=0')
s=s.replace("mask=uniform<np.where(yy[:,None]==1,q1,q0);eq(ev[key],mask,key)","""prob=np.zeros((len(rows),20));ints=np.zeros((2,20))
        for cls,desired in [(0,q0),(1,q1)]:
            rr=np.flatnonzero(yy==cls);obs=np.clip(X[rows[rr],19],-4,4)
            for j,target in enumerate(desired):
                if target==0:continue
                if target==1:prob[rr,j]=1.;continue
                intercept=brentq(lambda intercept:float(expit(intercept+obs).mean()-target),-45,45,xtol=1e-13)
                ints[cls,j]=intercept;prob[rr,j]=expit(intercept+obs)
                close([prob[rr,j].mean()],[target],key+'calibration',1e-12)
        assert np.all(prob[:,19]==0)
        mask=uniform<prob;eq(ev[key],mask,key)""")
s=s.replace("close(meta['q1'],q1,key+'q1');close(meta['q0'],q0,key+'q0');close([meta['expected_overall_missing']],[b.mean()],key+'expect')", "close(meta['q1'],q1,key+'q1');close(meta['q0'],q0,key+'q0');close([meta['expected_overall_missing']],[b.mean()],key+'expect');close(meta['expected_per_column'],b,key+'expected per column');close(meta['intercept_by_class_column'],ints,key+'intercepts',1e-11);assert meta['class_mean_max_error']<=1e-12")
s=s.replace("new=(~vm)&(d[f'U_stress_{s}']<.1);eq", "new=(~vm)&(d[f'U_stress_{s}']<.1);new[:,19]=False;eq")
s=s.replace('Artifacts and arithmetic checked by a separate assistant in the same session; no historical provenance proof or independent peer review','Separate numerical implementation adapted and executed in the same session; verifier does not import runner or analysis; not an independent team replication or historical provenance proof')
p.write_text(s)
