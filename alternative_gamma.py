"""Fold-isolated clinical regressors plus rapid-test calibration for Gamma_GT."""
from pathlib import Path
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

from alternative_models import AlternativeClassifier
from advanced_calibration import fit_calibration, calibrated_prediction
from smoking_push import data
from v2.common import base_X, plate_centered, CATS

OUT=Path('outputs/alternative_gamma')
CONFIG=dict(fit_slope=False,exclude_floor=True,low_weight=.3,weight=.9,
            floor_mode='censored',adaptive=True,prior_sd=.27)
NAMES=['catboost','lgb9','lgb31','extra']


def features(reference, frame):
    x=base_X(frame).join(plate_centered(reference,[frame])[0])
    x=x.drop(columns=['POC_GGT','POC_Batch'])
    x['AST_ALT']=frame.AST/frame.ALT.clip(lower=.1)
    x['TG_HDL']=frame.Triglyceride/frame.HDL.clip(lower=.1)
    x['waist_height']=frame.Waist_cm/frame.Height_cm.clip(lower=1)
    for column in ['ALT','AST','Triglyceride','Fasting_Glucose','Alcohol_Units_Week']:
        x['log_'+column]=np.log1p(frame[column].clip(lower=0))
    numeric=[c for c in x if c not in CATS]
    x[numeric]=x[numeric].replace([np.inf,-np.inf],np.nan)
    assert not {'ID','Smoking','Gamma_GT','POC_GGT','POC_Batch'}.intersection(x.columns)
    return x


def fit(reference,name,seed):
    x=features(reference,reference);y=np.log1p(reference.Gamma_GT)
    cats=[c for c in CATS if c in x]
    if name=='catboost':
        model=CatBoostRegressor(iterations=1400,depth=4,learning_rate=.03,
            l2_leaf_reg=5,cat_features=cats,random_seed=seed,thread_count=4,
            verbose=False,allow_writing_files=False)
        model.fit(x,y);return model
    if name.startswith('lgb'):
        model=LGBMRegressor(num_leaves=9 if name=='lgb9' else 31,
            n_estimators=1200,learning_rate=.025,min_child_samples=30,
            reg_lambda=5,reg_alpha=.1,verbosity=-1,n_jobs=4,
            random_state=seed,deterministic=True,force_col_wise=True)
        wrapper=AlternativeClassifier(model,x.columns.tolist(),
                                      {c:sorted(x[c].unique()) for c in cats})
    else:
        x=x.drop(columns=['Workplace_ID','Urine_Plate'])
        cats=[c for c in cats if c in x];numeric=[c for c in x if c not in cats]
        pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True),numeric),
                              ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
        model=make_pipeline(pre,ExtraTreesRegressor(n_estimators=600,min_samples_leaf=4,
                            max_features=.8,n_jobs=4,random_state=seed))
        wrapper=AlternativeClassifier(model,x.columns.tolist())
    wrapper.estimator.fit(wrapper.transform(x),y)
    return wrapper


def prediction(model,reference,frame):
    x=features(reference,frame)
    if isinstance(model,AlternativeClassifier):
        log_prediction=model.estimator.predict(model.transform(x))
    else:log_prediction=model.predict(x)
    clinical=np.clip(np.expm1(log_prediction),1,1000)
    return calibrated_prediction(fit_calibration(reference,CONFIG),frame,clinical,CONFIG)[0]


def rmsle(y,p):return float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2)))


def stage(dev,names,seed,name):
    directory=OUT/name;directory.mkdir(exist_ok=True)
    out={n:np.zeros(len(dev)) for n in names};folds=np.zeros(len(dev),int)
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(dev,dev.Smoking)):
        ref,frame=dev.iloc[tr],dev.iloc[va];folds[va]=fold
        for model_name in names:
            path=directory/f'{model_name}_{fold}.csv'
            if path.exists():
                saved=pd.read_csv(path);assert saved.ID.tolist()==frame.ID.tolist();p=saved.prediction.values
            else:
                model=fit(ref,model_name,seed+fold);p=prediction(model,ref,frame)
                pd.DataFrame(dict(ID=frame.ID,prediction=p)).to_csv(path,index=False)
                model.save_model(str(path)+'.model')
            out[model_name][va]=p
            print(name,fold+1,model_name,rmsle(frame.Gamma_GT,p),flush=True)
    return out,folds


def blend(old,new,w):return np.expm1((1-w)*np.log1p(old)+w*np.log1p(new))


def main():
    OUT.mkdir(exist_ok=True)
    identity=dict(code={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ['alternative_gamma.py','alternative_models.py','advanced_calibration.py','v2/common.py','smoking_push.py']},
        data={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ['data/train.csv','data/test.csv','v2/refined_cache.csv','outputs/plan_v1/split_ids.csv']},
        versions={p:importlib.metadata.version(p) for p in ['catboost','lightgbm','scikit-learn']},calibration=CONFIG)
    path=OUT/'identity.json'
    if path.exists():assert json.loads(path.read_text())==identity
    else:path.write_text(json.dumps(identity,indent=2))
    _,_,_,dev=data();pred,_=stage(dev,NAMES,43009,'selection')
    baseline=rmsle(dev.Gamma_GT,pred['catboost']);rows=[]
    for name in NAMES:
        for w in ([1.] if name=='catboost' else [.15,.3,.5,1.]):
            error=rmsle(dev.Gamma_GT,blend(pred['catboost'],pred[name],w))
            rows.append(dict(model=name,weight=w,rmsle=error,score_gain=(baseline-error)/1.0082))
    pd.DataFrame(rows).to_csv(OUT/'selection.csv',index=False)
    choice=min(rows,key=lambda r:r['rmsle'])
    (OUT/'frozen_config.json').write_text(json.dumps(choice,indent=2))
    print('SELECTED',choice,flush=True)
    if choice['model']=='catboost':
        result=dict(passed=False,reason='CatBoost baseline wins model selection.')
    else:
        pred,folds=stage(dev,['catboost',choice['model']],43710,'confirmation')
        p=blend(pred['catboost'],pred[choice['model']],choice['weight'])
        gain=(rmsle(dev.Gamma_GT,pred['catboost'])-rmsle(dev.Gamma_GT,p))/1.0082
        wins=sum(rmsle(dev.Gamma_GT[folds==f],p[folds==f])<
                 rmsle(dev.Gamma_GT[folds==f],pred['catboost'][folds==f]) for f in range(3))
        result=dict(passed=bool(gain>.0003 and wins>=2),score_gain=gain,fold_wins=int(wins),
                    note='Same development rows, different folds; diagnostic only.')
    (OUT/'confirmation.json').write_text(json.dumps(result,indent=2))
    print('CONFIRMATION',result,flush=True)
    # Final candidate generation is deliberately a separate stage after these checks.


if __name__=='__main__':main()
