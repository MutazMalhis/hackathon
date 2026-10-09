"""Conservative residual correction of the identified three-seed submission.

Validation is conditional on saved OOF predictions and inherits their nesting
limitations. It is not a new independent evaluation of the complete pipeline.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT=Path('outputs/gamma_bias_push')


def features(frame,prediction):
    z=np.log1p(prediction)
    x=pd.DataFrame({'prediction':z,'prediction_squared':z*z,'floor':frame.POC_GGT.eq(3).astype(float),
                    'missing_poc':frame.POC_GGT.isna().astype(float),'inverse_poc':1/frame.POC_GGT,
                    'low_poc':frame.POC_GGT.le(5).astype(float)},index=frame.index)
    for c in ['Screening_Center','Visit_Quarter']:
        for value in (list('ABCDEFGH') if c=='Screening_Center' else ['Q1','Q2','Q3','Q4']):
            x[c+'_'+value]=frame[c].eq(value).astype(float)
    return x.replace([np.inf,-np.inf],np.nan)


def prediction(model,x,base,weight):
    correction=np.clip(model.predict(x),-.03,.03)
    return np.clip(np.expm1(np.log1p(base)+weight*correction),1,1000)


def error(y,p):return float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2)))


def cv(frame,base,alpha,weight,seed):
    x=features(frame,base);r=np.log1p(frame.Gamma_GT.to_numpy())-np.log1p(base)
    out=np.zeros(len(frame));folds=np.zeros(len(frame),dtype=int)
    for k,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(frame,frame.Smoking)):
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=alpha))
        model.fit(x.iloc[tr],r[tr]);out[va]=prediction(model,x.iloc[va],base[va],weight);folds[va]=k
    return out,folds


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv('data/train.csv');test=pd.read_csv('data/test.csv')
    original=pd.read_csv('outputs/v2/submission_3seed.csv');old=pd.read_csv('outputs/v2/oof_3seed.csv')
    assert old.ID.equals(train.ID) and original.ID.equals(test.ID)
    split=pd.read_csv('outputs/plan_v1/split_ids.csv');ids=split.loc[split.partition.eq('development'),'ID']
    positions=train.set_index('ID').index.get_indexer(ids);dev=train.iloc[positions].reset_index(drop=True)
    base=old.gamma_oof.iloc[positions].to_numpy();baseline=error(dev.Gamma_GT,base)
    rows=[]
    for alpha in [10.,100.]:
        for weight in [.5,1.]:
            p,folds=cv(dev,base,alpha,weight,42009)
            rmsle=error(dev.Gamma_GT,p)
            rows.append(dict(alpha=alpha,weight=weight,rmsle=rmsle,score_gain=(baseline-rmsle)/1.0082))
    pd.DataFrame(rows).to_csv(OUT/'selection.csv',index=False)
    choice=max(rows,key=lambda r:r['score_gain'])
    (OUT/'frozen_config.json').write_text(json.dumps(choice,indent=2))
    p,folds=cv(dev,base,choice['alpha'],choice['weight'],42710)
    gain=(baseline-error(dev.Gamma_GT,p))/1.0082
    comparisons=[]
    for k in range(3):
        mask=folds==k
        comparisons.append(dict(fold=k+1,baseline_rmsle=error(dev.Gamma_GT[mask],base[mask]),selected_rmsle=error(dev.Gamma_GT[mask],p[mask])))
    passed=bool(choice['score_gain']>.00015 and gain>.00015 and sum(c['selected_rmsle']<c['baseline_rmsle'] for c in comparisons)>=2)
    result=dict(baseline_rmsle=baseline,selected_rmsle=error(dev.Gamma_GT,p),score_gain=gain,passed=passed,folds=comparisons,
                limitation='Conditional postprocessing validation on saved OOF predictions; inherits earlier base-model nesting limitations.')
    (OUT/'confirmation.json').write_text(json.dumps(result,indent=2))
    print('GAMMA BIAS',json.dumps(result),flush=True)
    if passed:
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=choice['alpha']))
        model.fit(features(train,old.gamma_oof),np.log1p(train.Gamma_GT)-np.log1p(old.gamma_oof))
        candidate=original.copy();candidate['Gamma_GT']=prediction(model,features(test,original.Gamma_GT),original.Gamma_GT.to_numpy(),choice['weight'])
        assert np.isfinite(candidate[['Smoking','Gamma_GT']]).all().all() and len(candidate)==6399
        candidate.to_csv(OUT/'submission_gamma_bias.csv',index=False)
        import joblib
        joblib.dump(model,OUT/'residual_model.joblib')
    identity=dict(source_sha256=hashlib.sha256(Path('gamma_bias_push.py').read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(Path('outputs/v2/submission_3seed.csv').read_bytes()).hexdigest())
    (OUT/'identity.json').write_text(json.dumps(identity,indent=2))


if __name__=='__main__':main()
