"""Assemble an experimental assay-calibration blend after conditional checks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT=Path('outputs/calibration_precision_push')
BASE=Path('outputs/final_review/submission_nn_v3_verified.csv')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for path,digest in manifest.items():
        if isinstance(digest,str) and len(digest)==64:
            assert sha(path)==digest, f'Input changed: {path}'
    reference=Path('outputs/v2/submission_3seed.csv')
    assert sha(reference)=='c49a7788060dd229e1c6072b463c7f90233627f9be63cb4ad59bea4b8d0bd95a'
    review=json.loads((OUT/'conditional_review.json').read_text())
    assert review['weight']==.3
    assert all(r['score_gain']>0 for r in review['metrics'])
    result=next(r for r in review['metrics'] if r['name']=='mean')
    assert result['score_gain']>.0002
    # These checks support an experiment; they do not imply independent validation.
    assert all(v['lower']>0 for v in review['conditional_bootstrap'].values())
    train=pd.read_csv('data/train.csv');test=pd.read_csv('data/test.csv')
    sample=pd.read_csv('data/sample_submission.csv')
    before=pd.read_csv(BASE,float_precision='round_trip')
    component=pd.read_csv(OUT/'test_components.csv')
    oof=pd.read_csv(OUT/'mean_oof_component.csv')
    old_oof=pd.read_csv('outputs/final_review/neural_blend_v3_oof.csv')
    assert oof.ID.equals(train.ID) and old_oof.ID.equals(train.ID)
    assert before.ID.equals(sample.ID) and component.ID.equals(test.ID) and test.ID.equals(sample.ID)
    combined=.7*np.log1p(before.Gamma_GT.to_numpy())+.3*component.posterior.to_numpy()
    submission=before.copy();submission['Gamma_GT']=np.clip(np.expm1(combined),1,1000)
    path=OUT/'submission_assay_precision.csv'
    submission.to_csv(path,index=False,float_format='%.17g')
    checked=pd.read_csv(path,float_precision='round_trip')
    assert checked.columns.tolist()==['ID','Smoking','Gamma_GT'] and len(checked)==6399
    assert checked.ID.equals(sample.ID)
    np.testing.assert_array_equal(checked.Smoking,before.Smoking)
    assert np.isfinite(checked[['Smoking','Gamma_GT']]).all().all()
    assert checked.Smoking.between(0,1).all() and checked.Gamma_GT.between(1,1000).all()
    auc=float(roc_auc_score(train.Smoking,old_oof.Smoking))
    rmsle=float(np.sqrt(np.mean((np.log1p(train.Gamma_GT)-oof.proposed_gamma_log1p)**2)))
    summary=dict(auc=auc,rmsle=rmsle,score=auc-rmsle/1.0082,
        previous_local_score=auc-review['reference_gamma_rmsle']/1.0082,
        local_score_gain=result['score_gain'],kaggle_score=None,
        submission=str(path),sha256=sha(path),smoking_unchanged=True,rows=len(checked),
        source_hashes={str(p):sha(p) for p in [BASE,reference,OUT/'test_components.csv',
                      OUT/'mean_oof_component.csv',OUT/'manifest.json',Path(__file__)]},
        limitation=review['limitations'],selection=review['selection'])
    (OUT/'submission_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
