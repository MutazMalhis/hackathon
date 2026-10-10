"""Improve Smoking while preserving the successful three-seed Gamma_GT predictions.

Feature/model selection and confirmation use the recorded development partition.
Final CV is diagnostic; the previous Kaggle candidate is never overwritten.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from v2.common import load,base_X,plate_centered,batch_features_v2,CATS
from improvements import development_data

OUT=Path('outputs/smoking_push_consistent')
REFERENCE=Path('outputs/v2/submission_3seed.csv')
SPECS={'baseline':dict(features='baseline',depth=6,learning_rate=.025,iterations=1600,l2_leaf_reg=3),
       'clinical':dict(features='clinical',depth=6,learning_rate=.025,iterations=1600,l2_leaf_reg=3),
       'clean':dict(features='clean',depth=6,learning_rate=.025,iterations=1600,l2_leaf_reg=3),
       'clean4':dict(features='clean',depth=4,learning_rate=.035,iterations=2200,l2_leaf_reg=5)}


def data():
    train,test,sample=load()
    refined=pd.read_csv('v2/refined_cache.csv')
    assert len(refined)==len(train)+len(test)
    assert set(refined.columns)=={'rpc_cot','rpc_etg','resid_cot','resid_etg'}
    train=train.join(refined.iloc[:len(train)].reset_index(drop=True))
    test=test.join(refined.iloc[len(train):].reset_index(drop=True))
    ids=development_data().ID
    dev=train.set_index('ID').loc[ids].reset_index()
    return train,test,sample,dev


def unit_calibration(reference,frame,training=False):
    good=reference.POC_GGT.gt(3)&reference.POC_Batch.notna()
    residual=(np.log(reference.Gamma_GT)-np.log(reference.POC_GGT)).where(good)
    sums=residual.groupby(reference.POC_Batch).sum()
    counts=residual.groupby(reference.POC_Batch).count()
    total=frame.POC_Batch.map(sums);n=frame.POC_Batch.map(counts)
    if training:
        assert reference.ID.tolist()==frame.ID.tolist()
        total=total-residual.fillna(0);n=n-good.astype(int)
    offset=total/n.where(n>0)
    reading=frame.POC_GGT.where(frame.POC_GGT.gt(0))
    calibrated=np.log1p(np.exp(np.log(reading)+offset))
    upper=np.log1p(np.exp(np.log(reading.where(reading.ne(3),3.5))+offset))
    return pd.DataFrame({'unit_calibration':calibrated,'unit_upper':upper,'unit_batch_n':n.fillna(0),
                         'unit_floor':frame.POC_GGT.eq(3).astype(float)},index=frame.index)


def inputs(reference,frame,test,variant,training=False):
    # Unlabeled test features inform plate means. Use exactly the same reference
    # for fitting and evaluation; evaluation rows never redefine the transform.
    ref=pd.concat([reference.drop(columns=['Smoking','Gamma_GT']),test])
    pc=plate_centered(ref,[frame])[0]
    if training:
        legacy=batch_features_v2(reference,[reference])[0][0]
    else:
        legacy=batch_features_v2(reference,[reference,frame])[0][1]
    x=base_X(frame).join(pc).assign(cal2=legacy.cal2_log1p)
    if variant!='baseline':
        x=x.join(unit_calibration(reference,frame,training))
        x['AST_ALT']=frame.AST/frame.ALT.clip(lower=.1)
        x['TG_HDL']=frame.Triglyceride/frame.HDL.clip(lower=.1)
        x['waist_height']=frame.Waist_cm/frame.Height_cm.clip(lower=1)
        x['pulse_pressure']=frame.BP_Systolic-frame.BP_Diastolic
        for column in ['ALT','AST','Triglyceride','Fasting_Glucose','Exhaled_CO_ppm','Alcohol_Units_Week']:
            x['log_'+column]=np.log1p(frame[column].clip(lower=0))
    if variant=='clean':x=x.drop(columns=['resid_cot','resid_etg'])
    numeric=[c for c in x if c not in CATS]
    x[numeric]=x[numeric].replace([np.inf,-np.inf],np.nan)
    assert not {'ID','Smoking','Gamma_GT'}.intersection(x.columns)
    return x


def fit(reference,test,name,seed):
    spec=SPECS[name];x=inputs(reference,reference,test,spec['features'],training=True)
    model=CatBoostClassifier(cat_features=CATS,verbose=False,allow_writing_files=False,thread_count=4,
                             random_seed=seed,**{k:v for k,v in spec.items() if k!='features'})
    model.fit(x,reference.Smoking)
    return model


def cv_predictions(dev,test,names,seed,directory):
    directory.mkdir(parents=True,exist_ok=True)
    predictions={n:np.zeros(len(dev)) for n in names};assignment=np.zeros(len(dev),dtype=int)
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(dev,dev.Smoking)):
        reference,frame=dev.iloc[tr],dev.iloc[va]
        assert set(reference.ID).isdisjoint(frame.ID)
        assignment[va]=fold
        for name in names:
            path=directory/f'{name}_{fold}.csv'
            if path.exists():
                saved=pd.read_csv(path);assert saved.ID.tolist()==frame.ID.tolist();p=saved.prediction.to_numpy()
            else:
                model=fit(reference,test,name,seed*10+fold)
                x=inputs(reference,frame,test,SPECS[name]['features'])
                p=model.predict_proba(x)[:,1]
                pd.DataFrame({'ID':frame.ID,'prediction':p}).to_csv(path,index=False)
                model.save_model(str(directory/f'{name}_{fold}.cbm'))
            predictions[name][va]=p
            print(directory.name,fold+1,name,round(roc_auc_score(frame.Smoking,p),6),flush=True)
    return predictions,assignment


def select():
    train,test,sample,dev=data()
    predictions,folds=cv_predictions(dev,test,list(SPECS),41009,OUT/'selection')
    rows=[]
    for name in SPECS:
        for weight in ([1.] if name=='baseline' else [.5,1.]):
            p=weight*predictions[name]+(1-weight)*predictions['baseline']
            rows.append(dict(model=name,weight=weight,auc=float(roc_auc_score(dev.Smoking,p))))
    results=pd.DataFrame(rows);results.to_csv(OUT/'selection.csv',index=False)
    best=max(rows,key=lambda r:r['auc'])
    config=dict(selected=best,baseline_auc=rows[0]['auc'],specifications=SPECS,
                selected_before_confirmation=True,holdout_used_for_selection=False)
    (OUT/'frozen_config.json').write_text(json.dumps(config,indent=2))
    print('SELECTED',json.dumps(config),flush=True)


def confirm():
    config=json.loads((OUT/'frozen_config.json').read_text());choice=config['selected']
    if choice['model']=='baseline':
        (OUT/'confirmation.json').write_text(json.dumps({'passed':False,'reason':'Baseline wins development selection.'},indent=2));return
    _,test,_,dev=data();names=['baseline',choice['model']]
    predictions,folds=cv_predictions(dev,test,names,41710,OUT/'confirmation')
    old=predictions['baseline'];new=choice['weight']*predictions[choice['model']]+(1-choice['weight'])*old
    gain=float(roc_auc_score(dev.Smoking,new)-roc_auc_score(dev.Smoking,old));comparisons=[]
    for fold in range(3):
        mask=folds==fold
        comparisons.append(dict(fold=fold+1,baseline_auc=float(roc_auc_score(dev.Smoking[mask],old[mask])),
                                selected_auc=float(roc_auc_score(dev.Smoking[mask],new[mask]))))
    wins=sum(r['selected_auc']>r['baseline_auc'] for r in comparisons)
    result=dict(gain=gain,folds=comparisons,passed=bool(gain>.0003 and wins>=2),
                note='Reused development rows; stability check, not independent validation.')
    (OUT/'confirmation.json').write_text(json.dumps(result,indent=2));print('CONFIRMATION',json.dumps(result),flush=True)


def finalize():
    config=json.loads((OUT/'frozen_config.json').read_text());confirmation=json.loads((OUT/'confirmation.json').read_text())
    if not confirmation['passed']:
        print('No promotion; preserve submission_3seed.csv.',flush=True);return
    train,test,sample,_=data();name=config['selected']['model']
    oof=np.zeros(len(train));pred=np.zeros(len(test));directory=OUT/'final';directory.mkdir(exist_ok=True)
    for seed in [42,43,44]:
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(train,train.Smoking)):
            ref,frame=train.iloc[tr],train.iloc[va];path=directory/f'{seed}_{fold}.npz'
            if path.exists():
                saved=np.load(path);assert np.array_equal(saved['valid_ids'],frame.ID);p,q=saved['valid'],saved['test']
            else:
                model=fit(ref,test,name,seed*10+fold)
                p=model.predict_proba(inputs(ref,frame,test,SPECS[name]['features']))[:,1]
                q=model.predict_proba(inputs(ref,test,test,SPECS[name]['features']))[:,1]
                np.savez(path,valid=p,test=q,valid_ids=frame.ID,test_ids=test.ID)
                model.save_model(str(directory/f'{seed}_{fold}.cbm'))
            oof[va]+=p/3;pred+=q/15
            print('Final smoking',seed,fold+1,flush=True)
    old_oof=pd.read_csv('outputs/v2/oof_3seed.csv');original=pd.read_csv(REFERENCE)
    assert old_oof.ID.equals(train.ID) and original.ID.equals(sample.ID)
    weight=config['selected']['weight']
    # Feature-family selection determines a conservative blend with the successful model.
    new_oof=weight*oof+(1-weight)*old_oof.smoking_oof.to_numpy()
    new_test=weight*pred+(1-weight)*original.Smoking.to_numpy()
    auc_old=float(roc_auc_score(train.Smoking,old_oof.smoking_oof));auc_new=float(roc_auc_score(train.Smoking,new_oof))
    report=dict(baseline_auc=auc_old,selected_auc=auc_new,auc_gain=auc_new-auc_old,
                note='Full-data CV contains development-selection rows; diagnostic only.')
    (OUT/'final_cv.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame({'ID':train.ID,'smoking_prediction':new_oof}).to_csv(OUT/'final_oof.csv',index=False)
    if auc_new<=auc_old+.0002:
        print('No full-data diagnostic gain; no new submission promoted.',flush=True);return
    candidate=original.copy();candidate['Smoking']=new_test
    assert candidate.ID.equals(sample.ID) and len(candidate)==6399
    assert candidate.columns.tolist()==['ID','Smoking','Gamma_GT']
    assert np.isfinite(candidate[['Smoking','Gamma_GT']]).all().all()
    assert candidate.Smoking.between(0,1).all() and candidate.Gamma_GT.between(1,1000).all()
    np.testing.assert_array_equal(candidate.Gamma_GT,original.Gamma_GT)
    candidate.to_csv(OUT/'submission_smoking_push.csv',index=False)
    print('FINAL',json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['select','confirm','finalize','all'],default='all')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(code={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['smoking_push.py','v2/common.py']},
                  data={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REFERENCE,Path('outputs/v2/oof_3seed.csv'),Path('v2/refined_cache.csv'),Path('data/train.csv'),Path('data/test.csv')]},
                  versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','catboost']})
    path=OUT/'run_identity.json'
    if path.exists():assert json.loads(path.read_text())==identity,'Recipe changed: choose a new output directory.'
    else:path.write_text(json.dumps(identity,indent=2))
    (OUT/'leaderboard_reference.json').write_text(json.dumps(dict(team='XGEngineers',rank_in_user_screenshot=2,
        public_score=.86380,leader_score_in_screenshot=.86424,reference_csv=str(REFERENCE),
        sha256=identity['data'][str(REFERENCE)],public_fraction_approx=.33,source='User screenshot and explicit CSV identification'),indent=2))
    if args.phase in ['select','all']:select()
    if args.phase in ['confirm','all']:confirm()
    if args.phase in ['finalize','all']:finalize()


if __name__=='__main__':main()
