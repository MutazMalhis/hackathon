"""Group-held-out diagnostics; never used to retune the frozen model."""
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from experiments import OUT, SEED, fit_models, predict_models, batch_correct, metrics


def main():
    cfg=json.loads((OUT/'frozen_config.json').read_text())
    data=pd.read_csv('data/train.csv')
    reports=[]
    for field in ['Workplace_ID','POC_Batch']:
        groups=data[field].fillna(-1)
        fit,hold=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(data,groups=groups))
        ref,frame=data.iloc[fit],data.iloc[hold]
        assert set(groups.iloc[fit]).isdisjoint(set(groups.iloc[hold]))
        models,iterations=fit_models(ref,cfg['variant'],SEED+500)
        s,g=predict_models(models,ref,frame,cfg['variant'])
        corrected,coverage=batch_correct(ref,frame,g,**cfg['correction'])
        result={'held_out_group':field,'fit_rows':len(ref),'holdout_rows':len(frame),
                'calibration_coverage':float(coverage.mean()),'iterations':iterations,
                **metrics(frame,s,corrected)}
        reports.append(result)
        print(result,flush=True)
    (OUT/'group_robustness.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
