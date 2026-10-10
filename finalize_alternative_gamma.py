"""Finalize a confirmed Gamma_GT model, gated against the submitted V2 OOF."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import alternative_gamma as gamma
from smoking_push import data, REFERENCE


def main():
    result=json.loads((gamma.OUT/'confirmation.json').read_text())
    if not result['passed']:
        print('Gamma_GT candidate did not pass confirmation; no final retraining.');return
    identity=json.loads((gamma.OUT/'identity.json').read_text())
    for p,h in {**identity['code'],**identity['data']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    source=dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest())
    path=gamma.OUT/'final_identity.json'
    if path.exists():assert json.loads(path.read_text())==source
    else:path.write_text(json.dumps(source,indent=2))
    config=json.loads((gamma.OUT/'frozen_config.json').read_text())
    train,test,sample,_=data();oof=np.zeros(len(train));pred=np.zeros(len(test))
    directory=gamma.OUT/'final';directory.mkdir(exist_ok=True)
    for seed in [42,43,44]:
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(train,train.Smoking)):
            ref,frame=train.iloc[tr],train.iloc[va];path=directory/f'{seed}_{fold}.npz'
            if path.exists():
                saved=np.load(path);assert np.array_equal(saved['ids'],frame.ID)
                assert np.array_equal(saved['test_ids'],test.ID)
                p,q=saved['valid'],saved['test']
            else:
                selected=gamma.fit(ref,config['model'],seed*10+fold)
                baseline=gamma.fit(ref,'catboost',seed*10+fold)
                p=gamma.blend(gamma.prediction(baseline,ref,frame),gamma.prediction(selected,ref,frame),config['weight'])
                q=gamma.blend(gamma.prediction(baseline,ref,test),gamma.prediction(selected,ref,test),config['weight'])
                np.savez(path,ids=frame.ID,test_ids=test.ID,valid=p,test=q)
                selected.save_model(str(path)+'.selected.model');baseline.save_model(str(path)+'.baseline.model')
            oof[va]+=np.log1p(p)/3;pred+=np.log1p(q)/15
            print('Final Gamma_GT',seed,fold+1,flush=True)
    old=pd.read_csv('outputs/v2/oof_3seed.csv');assert old.ID.equals(train.ID)
    old_error=gamma.rmsle(train.Gamma_GT,old.gamma_oof)
    error=gamma.rmsle(train.Gamma_GT,np.expm1(oof));gain=(old_error-error)/1.0082
    result=dict(baseline_rmsle=old_error,selected_rmsle=error,score_gain=gain,
                passed=bool(gain>.0002),note='Full CV reuses selection rows; diagnostic only.')
    (gamma.OUT/'final_cv.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(dict(ID=train.ID,gamma_prediction=np.expm1(oof))).to_csv(gamma.OUT/'final_oof.csv',index=False)
    if result['passed']:
        original=pd.read_csv(REFERENCE);candidate=original.copy()
        candidate.Gamma_GT=np.clip(np.expm1(pred),1,1000)
        assert candidate.ID.equals(sample.ID) and candidate.columns.tolist()==['ID','Smoking','Gamma_GT'] and len(candidate)==6399
        np.testing.assert_array_equal(candidate.Smoking,original.Smoking)
        assert np.isfinite(candidate[['Smoking','Gamma_GT']]).all().all()
        assert candidate.Gamma_GT.between(1,1000).all()
        candidate.to_csv(gamma.OUT/'submission_gamma_models.csv',index=False)
    print('FINAL',result,flush=True)


if __name__=='__main__':main()
