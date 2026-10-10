"""Conditional OOF check of regularized group adjustments to Smoking ranks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT=Path('outputs/smoking_group_push')
GROUPS=['Screening_Center','Visit_Quarter','Workplace_ID']


def calibrate_scale(reference_rank,reference_probability,rank):
    # Label-free quantile map onto the known model's probability distribution.
    order=np.argsort(reference_rank)
    return np.clip(np.interp(rank,np.asarray(reference_rank)[order],
                             np.sort(reference_probability)),1e-6,1-1e-6)


def fit_adjustments(frame,probability,regularization,columns):
    z=logit(np.clip(probability,1e-6,1-1e-6));y=frame.Smoking.to_numpy()
    keys={c:frame[c].fillna('__MISSING__').astype(str).to_numpy() for c in columns}
    maps={c:{v:0. for v in np.unique(keys[c])} for c in columns}
    # Coordinate Newton updates for penalized logistic offsets.
    for _ in range(20):
        for c in columns:
            for key in maps[c]:
                mask=keys[c]==key;old=maps[c][key]
                p=expit(z[mask]);g=np.sum(y[mask]-p)-regularization*old
                h=np.sum(p*(1-p))+regularization
                step=float(np.clip(g/h,-.5,.5))
                maps[c][key]+=step;z[mask]+=step
    return maps


def predict_adjustments(frame,probability,maps):
    z=logit(np.clip(probability,1e-6,1-1e-6))
    for c,values in maps.items():
        z+=frame[c].fillna('__MISSING__').astype(str).map(values).fillna(0).to_numpy()
    return expit(z)


def main():
    OUT.mkdir(exist_ok=True)
    paths=['data/train.csv','outputs/v2/oof_3seed.csv',
           'outputs/final_review/neural_blend_v3_oof.csv','outputs/plan_v1/split_ids.csv']
    identity={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths+[__file__]}
    ip=OUT/'identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity
    else:ip.write_text(json.dumps(identity,indent=2))
    train=pd.read_csv(paths[0]);old=pd.read_csv(paths[1]);candidate=pd.read_csv(paths[2])
    assert train.ID.equals(old.ID) and train.ID.equals(candidate.ID)
    split=pd.read_csv(paths[3]);dev=train.ID.isin(split.loc[split.partition.eq('development'),'ID']).to_numpy()
    reserved=~dev
    recipes=[dict(regularization=r,columns=cols) for r in [10.,50.,200.]
             for cols in [GROUPS[:2],GROUPS]]
    scores=[];predictions=[]
    d=train.loc[dev].reset_index(drop=True);r=candidate.Smoking[dev].to_numpy();b=old.smoking_oof[dev].to_numpy()
    for recipe in recipes:
        out=np.zeros(len(d))
        for tr,va in StratifiedKFold(3,shuffle=True,random_state=45109).split(d,d.Smoking):
            p=calibrate_scale(r[tr],b[tr],r[tr]);q=calibrate_scale(r[tr],b[tr],r[va])
            maps=fit_adjustments(d.iloc[tr],p,**recipe)
            out[va]=predict_adjustments(d.iloc[va],q,maps)
        # Separate fold quantile maps can change cross-fold ordering: include a
        # matched no-adjustment reference rather than comparing with raw ranks.
        base=np.zeros(len(d))
        for tr,va in StratifiedKFold(3,shuffle=True,random_state=45109).split(d,d.Smoking):
            base[va]=calibrate_scale(r[tr],b[tr],r[va])
        gain=roc_auc_score(d.Smoking,out)-roc_auc_score(d.Smoking,base)
        scores.append(dict(**recipe,auc=float(roc_auc_score(d.Smoking,out)),auc_gain=float(gain)))
        predictions.append(out)
    best=int(np.argmax([x['auc_gain'] for x in scores]));choice=recipes[best]
    pd.DataFrame(scores).to_csv(OUT/'selection.csv',index=False)
    p=calibrate_scale(r,b,r)
    q=calibrate_scale(r,b,candidate.Smoking[reserved].to_numpy())
    adjusted=predict_adjustments(train.loc[reserved],q,fit_adjustments(d,p,**choice))
    gain=float(roc_auc_score(train.Smoking[reserved],adjusted)-roc_auc_score(train.Smoking[reserved],q))
    result=dict(selected=choice,selection_gain=scores[best]['auc_gain'],reserved_gain=gain,
                passed=bool(scores[best]['auc_gain']>.0003 and gain>.0003),
                note='Conditional on reused base OOF; reserved comparison is not independent validation.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(pd.DataFrame(scores).to_string(index=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
