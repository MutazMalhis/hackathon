"""Audit cached ensemble recipes against the exact 0.86380 submission.

This is a conditional OOF audit, not fully nested or independent validation.
It writes diagnostics and a byte-identical copy of a verified existing CSV.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

OUT=Path('outputs/final_review')
REFERENCE=Path('outputs/v2/submission_3seed.csv')
REFERENCE_SHA='c49a7788060dd229e1c6072b463c7f90233627f9be63cb4ad59bea4b8d0bd95a'


def rank(values):
    return rankdata(values)/len(values)


def metric(smoking,gamma_log,truth):
    auc=float(roc_auc_score(truth.Smoking,smoking))
    rmsle=float(np.sqrt(np.mean((np.log1p(truth.Gamma_GT)-gamma_log)**2)))
    return dict(auc=auc,rmsle=rmsle,score=auc-rmsle/1.0082)


def neural_gamma_stack(train,test,components,neural):
    y=np.log1p(train.Gamma_GT.to_numpy())
    def design(frame,part):
        key=lambda n: components[part+'_'+n]
        nn=neural['oof_g' if part=='oof' else 'te_g']
        cal,main,nop=key('cal'),key('main'),key('nop')
        prior=(nop+nn)/2
        upper=cal+.975*np.log(3.5/3)
        a=(upper-prior)/.26
        truncated=prior-.26*norm.pdf(a)/np.clip(norm.cdf(a),1e-6,None)
        groups={'calibrated':np.isfinite(cal)&frame.POC_GGT.gt(3).to_numpy(),
                'floor':np.isfinite(cal)&frame.POC_GGT.le(3).to_numpy(),
                'none':~np.isfinite(cal)}
        matrices={'calibrated':np.c_[main,cal,nop,nn],
                  'floor':truncated[:,None], 'none':np.c_[main,nop,nn]}
        return groups,matrices
    go,xo=design(train,'oof');gt,xt=design(test,'te')
    oof=np.zeros(len(train));prediction=np.zeros(len(test))
    for group in go:
        fit=xo[group][go[group]]
        oof[go[group]]=cross_val_predict(LinearRegression(),fit,y[go[group]],cv=5)
        prediction[gt[group]]=LinearRegression().fit(fit,y[go[group]]).predict(xt[group][gt[group]])
    return oof,prediction


def cluster_interval(truth,baseline,candidate,group,seed=44009,repeats=600):
    # Resample whole observed groups, retaining each sampled group's row multiplicity.
    keys=pd.Series(group).fillna('__MISSING__').astype(str)
    codes,unique=pd.factorize(keys)
    rng=np.random.default_rng(seed);gains=[]
    def score(pair,weights):
        auc=roc_auc_score(truth.Smoking,pair[0],sample_weight=weights)
        mse=np.average((np.log1p(truth.Gamma_GT)-pair[1])**2,weights=weights)
        return auc-np.sqrt(mse)/1.0082
    for _ in range(repeats):
        multiplicity=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
        weights=multiplicity[codes]
        gains.append(score(candidate,weights)-score(baseline,weights))
    return dict(gain_p025=float(np.quantile(gains,.025)),gain_p975=float(np.quantile(gains,.975)),
                positive_fraction=float(np.mean(np.asarray(gains)>0)),repeats=repeats,
                note='Conditional on cached predictions; excludes retraining and model-selection uncertainty.')


def main():
    OUT.mkdir(exist_ok=True)
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==REFERENCE_SHA
    train=pd.read_csv('data/train.csv');test=pd.read_csv('data/test.csv')
    sample=pd.read_csv('data/sample_submission.csv');submitted=pd.read_csv(REFERENCE)
    old=pd.read_csv('outputs/v2/oof_3seed.csv');v3=pd.read_csv('outputs/plan_v3/final_oof.csv')
    v3_test=pd.read_csv('outputs/plan_v3/submission_v3.csv')
    assert old.ID.equals(train.ID) and v3.ID.equals(train.ID)
    assert submitted.ID.equals(sample.ID) and v3_test.ID.equals(sample.ID)
    c=np.load('outputs/v2/components.npz');nn=np.load('outputs/v2/nn_preds.npz')
    gamma,gamma_test=neural_gamma_stack(train,test,c,nn)
    np.testing.assert_allclose(gamma_test,np.load('outputs/v2/nn_blend.npz')['g_test'],rtol=0,atol=1e-12)
    v2=pd.read_csv('outputs/v2/oof.csv');assert v2.ID.equals(train.ID)
    smk=rank(.9*rank(c['p_smoke_oof'])+.1*rank(nn['oof_s']))
    smk_test=rank(.9*rank(c['p_smoke_test'])+.1*rank(nn['te_s']))
    candidates={
        'submitted_3seed':(old.smoking_oof.to_numpy(),np.log1p(old.gamma_oof.to_numpy())),
        'five_seed':(v2.smoking_oof.to_numpy(),np.log1p(v2.gamma_oof.to_numpy())),
        'neural_blend':(smk,gamma),
        'neural_blend_v3':(.8*smk+.2*rank(v3.Smoking),.8*gamma+.2*np.log1p(v3.Gamma_GT)),
        'reference_smoking_neural_gamma_v3':(old.smoking_oof.to_numpy(),.8*gamma+.2*np.log1p(v3.Gamma_GT))}
    expected_smoke=.8*smk_test+.2*rank(v3_test.Smoking)
    expected_gamma=.8*gamma_test+.2*np.log1p(v3_test.Gamma_GT)
    file=pd.read_csv('outputs/v2/submission_nn_blend_v3.csv')
    assert file.ID.equals(sample.ID) and file.columns.tolist()==['ID','Smoking','Gamma_GT']
    np.testing.assert_allclose(file.Smoking,expected_smoke,rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.log1p(file.Gamma_GT),expected_gamma,rtol=0,atol=1e-12)
    assert len(file)==6399 and np.isfinite(file[['Smoking','Gamma_GT']]).all().all()
    assert file.Smoking.between(0,1).all() and file.Gamma_GT.between(1,1000).all()
    frozen=OUT/'submission_nn_v3_verified.csv'
    shutil.copyfile('outputs/v2/submission_nn_blend_v3.csv',frozen)
    rows=[dict(name=name,**metric(*pair,train)) for name,pair in candidates.items()]
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    intervals={}
    for name in ['neural_blend','neural_blend_v3','reference_smoking_neural_gamma_v3']:
        intervals[name]={}
        for group in ['Workplace_ID','POC_Batch']:
            intervals[name][group]=cluster_interval(train,candidates['submitted_3seed'],candidates[name],train[group])
        print(name,intervals[name],flush=True)
    (OUT/'conditional_intervals.json').write_text(json.dumps(intervals,indent=2))
    slices=[]
    for label,mask in [('missing',train.POC_GGT.isna()),('floor',train.POC_GGT.le(3)),('regular',train.POC_GGT.gt(3))]:
        for name,pair in candidates.items():
            residual=np.log1p(train.Gamma_GT.to_numpy())-pair[1]
            slices.append(dict(slice=label,model=name,rows=int(mask.sum()),
                 rmsle=float(np.sqrt(np.mean(residual[mask]**2))),
                 squared_error_share=float(np.sum(residual[mask]**2)/np.sum(residual**2))))
    pd.DataFrame(slices).to_csv(OUT/'gamma_error_slices.csv',index=False)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [REFERENCE,Path('outputs/v2/oof_3seed.csv'),Path('outputs/v2/oof.csv'),
         Path('data/train.csv'),Path('data/test.csv'),Path('data/sample_submission.csv'),
         Path('outputs/v2/components.npz'),Path('outputs/v2/nn_preds.npz'),
         Path('outputs/v2/submission_nn_blend_v3.csv'),Path('outputs/plan_v3/final_oof.csv'),
         Path('outputs/plan_v3/submission_v3.csv'),Path(__file__)]}
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==hashes['outputs/v2/submission_nn_blend_v3.csv']
    (OUT/'audit.json').write_text(json.dumps(dict(hashes=hashes,reference_unchanged=True,
        existing_neural_csv_reproduced=True,
        verified_copy=str(frozen),
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        limitation='Cached V2/NN predictions lack complete row-level provenance and fully nested validation; intervals are conditional diagnostics.'),indent=2))
    for name,(smoking,gamma_log) in candidates.items():
        pd.DataFrame(dict(ID=train.ID,Smoking=smoking,Gamma_GT=np.expm1(gamma_log))).to_csv(OUT/(name+'_oof.csv'),index=False)


if __name__=='__main__':main()
