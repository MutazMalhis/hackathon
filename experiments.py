"""Development-only selection, outer holdout audit, and final competition predictions."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

OUT = Path('outputs/plan_v1')
SEED = 20261009
MAX_ITERATIONS = 1600


def metrics(frame, smoking, gamma):
    auc = roc_auc_score(frame.Smoking, smoking)
    rmsle = np.sqrt(np.mean((np.log1p(frame.Gamma_GT) - np.log1p(np.clip(gamma, 1, 1000)))**2))
    return {'auc': float(auc), 'rmsle': float(rmsle), 'score': float(auc - rmsle / (2*.5041))}


def features(reference, frame, variant):
    cols = [c for c in frame if c not in ['ID', 'Smoking', 'Gamma_GT']]
    x = frame[cols].copy()
    if variant == 'urine':
        for marker in ['Urine_Cotinine_ng_mL', 'Urine_EtG_ng_mL']:
            x['log_' + marker] = np.log1p(frame[marker])
            x['log_ratio_' + marker] = np.log(frame[marker] / frame.Urine_Creatinine)
            ref_ratio = np.log(reference[marker] / reference.Urine_Creatinine)
            plate_mean = ref_ratio.groupby(reference.Urine_Plate).mean()
            # Fit the unsupervised plate reference only on the fitting partition.
            x['plate_centered_' + marker] = x['log_ratio_' + marker] - frame.Urine_Plate.map(plate_mean).fillna(ref_ratio.mean())
    cats = [c for c in x if x[c].dtype == 'object' or c in ['Workplace_ID','Urine_Plate','POC_Batch']]
    for c in x:
        if c in cats:
            x[c] = x[c].fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='raise').replace([np.inf,-np.inf],np.nan)
    return x, cats


def fit_models(reference, variant, seed):
    inner_fit, inner_stop = train_test_split(reference, test_size=.2, stratify=reference.Smoking, random_state=seed)
    assert set(inner_fit.ID).isdisjoint(inner_stop.ID)
    xi, cats = features(inner_fit, inner_fit, variant)
    xs, _ = features(inner_fit, inner_stop, variant)
    models, iterations = [], []
    for kind in ['smoking', 'gamma']:
        cls = CatBoostClassifier if kind == 'smoking' else CatBoostRegressor
        extra = {'loss_function':'Logloss','eval_metric':'AUC'} if kind == 'smoking' else {'loss_function':'RMSE'}
        params = dict(depth=6, learning_rate=.04, random_seed=seed, cat_features=cats,
                      verbose=False, allow_writing_files=False, thread_count=4, **extra)
        yi = inner_fit.Smoking if kind == 'smoking' else np.log1p(inner_fit.Gamma_GT)
        ys = inner_stop.Smoking if kind == 'smoking' else np.log1p(inner_stop.Gamma_GT)
        pilot = cls(iterations=MAX_ITERATIONS, **params)
        pilot.fit(xi, yi, eval_set=(xs, ys), early_stopping_rounds=120)
        n = max(1, pilot.get_best_iteration()+1)
        xf, _ = features(reference, reference, variant)
        yf = reference.Smoking if kind == 'smoking' else np.log1p(reference.Gamma_GT)
        model = cls(iterations=n, **params)
        model.fit(xf, yf)
        models.append(model)
        iterations.append(n)
    return models, iterations


def predict_models(models, reference, frame, variant):
    x, _ = features(reference, frame, variant)
    return models[0].predict_proba(x)[:,1], np.clip(np.expm1(models[1].predict(x)),1,1000)


def batch_correct(reference, frame, fallback, method='mean', shrink=0, weight=1):
    good = reference.POC_GGT.gt(0) & reference.Gamma_GT.gt(0) & reference.POC_Batch.notna()
    ref = reference.loc[good]
    r = np.log(ref.Gamma_GT)-np.log(ref.POC_GGT)
    grouped = r.groupby(ref.POC_Batch)
    offset = grouped.mean() if method == 'mean' else grouped.median()
    count = grouped.count()
    offset = (count*offset+shrink*r.mean())/(count+shrink)
    mapped = frame.POC_Batch.map(offset)
    usable = frame.POC_GGT.gt(0) & mapped.notna()
    p = np.asarray(fallback).copy()
    raw = np.clip(np.exp(np.log(frame.loc[usable,'POC_GGT'])+mapped[usable]),1,1000)
    p[usable] = np.expm1(weight*np.log1p(raw)+(1-weight)*np.log1p(p[usable]))
    return np.clip(p,1,1000), usable.to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv('data/train.csv'); test = pd.read_csv('data/test.csv'); sample = pd.read_csv('data/sample_submission.csv')
    dev, holdout = train_test_split(train, test_size=.2, stratify=train.Smoking, random_state=SEED)
    dev=dev.reset_index(drop=True);holdout=holdout.reset_index(drop=True)
    assert set(dev.ID).isdisjoint(holdout.ID)
    pd.concat([dev[['ID']].assign(partition='development'),holdout[['ID']].assign(partition='holdout')]).to_csv(OUT/'split_ids.csv',index=False)
    # All candidate selection happens on development folds before holdout evaluation.
    variants=['raw','urine']
    corrections=[dict(method='mean',shrink=0,weight=1),dict(method='median',shrink=0,weight=1),
                 dict(method='mean',shrink=.5,weight=1),dict(method='mean',shrink=2,weight=1),
                 dict(method='mean',shrink=0,weight=.9),dict(method='median',shrink=0,weight=.9)]
    records=[]
    predictions={}
    for variant in variants:
        smoke=np.zeros(len(dev));fallback=np.zeros(len(dev));corrected=[np.zeros(len(dev)) for _ in corrections]
        for fold,(fit,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,dev.Smoking)):
            ref=dev.iloc[fit]; frame=dev.iloc[valid]
            models,iters=fit_models(ref,variant,SEED+fold)
            s,g=predict_models(models,ref,frame,variant);smoke[valid]=s;fallback[valid]=g
            for i,c in enumerate(corrections): corrected[i][valid],_=batch_correct(ref,frame,g,**c)
            print('Development',variant,'fold',fold+1,'iterations',iters,metrics(frame,s,corrected[0][valid]),flush=True)
        records.append({'variant':variant,'correction':'none',**metrics(dev,smoke,fallback)})
        for i,c in enumerate(corrections):
            key=f'{variant}_{i}';record={'key':key,'variant':variant,**c,**metrics(dev,smoke,corrected[i])}
            records.append(record);predictions[key]=(smoke.copy(),corrected[i].copy())
    candidates=pd.DataFrame(records);candidates.to_csv(OUT/'development_comparison.csv',index=False)
    ranked=[r for r in records if 'key' in r]
    selected=max(ranked,key=lambda r:r['score'])
    variant=selected['variant']; correction={k:selected[k] for k in ['method','shrink','weight']}
    config={'variant':variant,'correction':correction,'seed':SEED,'max_iterations':MAX_ITERATIONS,
            'selection_metric':selected,'holdout_used_for_selection':False,
            'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','catboost']},
            'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data').glob('*.csv')}}
    (OUT/'frozen_config.json').write_text(json.dumps(config,indent=2))
    print('FROZEN CONFIG',selected,flush=True)
    # Evaluate selected method once; keep configuration unchanged afterward.
    models,iters=fit_models(dev,variant,SEED+100)
    hs,hg=predict_models(models,dev,holdout,variant)
    hc,covered=batch_correct(dev,holdout,hg,**correction)
    hold_metrics={'regressor_only':metrics(holdout,hs,hg),'selected':metrics(holdout,hs,hc),
                  'coverage':float(covered.mean()),'iterations':iters,
                  'note':'Prospective holdout: full data was explored earlier; not historically untouched.'}
    if variant=='raw': os,og=hs,hg
    else:
        original,_=fit_models(dev,'raw',SEED+100)
        os,og=predict_models(original,dev,holdout,'raw')
    original_g,_=batch_correct(dev,holdout,og)
    hold_metrics['original_calibrated']=metrics(holdout,os,original_g)
    print('HOLDOUT',hold_metrics,flush=True)
    (OUT/'holdout_metrics.json').write_text(json.dumps(hold_metrics,indent=2))
    audit=holdout[['ID','Smoking','Gamma_GT','Screening_Center','POC_Batch']].copy()
    audit['smoking_prediction']=hs;audit['gamma_prediction']=hc;audit['calibrated']=covered
    audit['squared_log_error']=(np.log1p(audit.Gamma_GT)-np.log1p(hc))**2
    audit['gamma_range']=pd.cut(audit.Gamma_GT,[0,15,25,50,100,np.inf])
    counts=dev.POC_Batch.value_counts();audit['batch_training_n']=holdout.POC_Batch.map(counts).fillna(0)
    audit.to_csv(OUT/'holdout_predictions.csv',index=False)
    slices=[]
    for field in ['calibrated','Screening_Center','gamma_range']:
        for group,frame in audit.groupby(field,observed=True):
            m={'slice':field,'group':str(group),'rows':len(frame),
               'rmsle':float(np.sqrt(frame.squared_log_error.mean()))}
            m['auc']=float(roc_auc_score(frame.Smoking,frame.smoking_prediction)) if frame.Smoking.nunique()==2 else None
            slices.append(m)
    pd.DataFrame(slices).to_csv(OUT/'error_slices.csv',index=False)
    # Final five-fold ensemble using the frozen development-selected configuration.
    ps=np.zeros(len(test));pg_log=np.zeros(len(test));final_oof_s=np.zeros(len(train));final_oof_g=np.zeros(len(train))
    for fold,(fit,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(train,train.Smoking)):
        ref=train.iloc[fit];frame=train.iloc[valid]
        models,iters=fit_models(ref,variant,SEED+200+fold)
        vs,vg=predict_models(models,ref,frame,variant);vg,_=batch_correct(ref,frame,vg,**correction)
        final_oof_s[valid]=vs;final_oof_g[valid]=vg
        ts,tg=predict_models(models,ref,test,variant);tg,_=batch_correct(ref,test,tg,**correction)
        ps+=ts/5;pg_log+=np.log1p(tg)/5
        for name,model in zip(['smoking','gamma'],models):model.save_model(str(OUT/f'{name}_fold{fold}.cbm'))
        print('Final fold',fold+1,metrics(frame,vs,vg),flush=True)
    submission=pd.DataFrame({'ID':test.ID,'Smoking':ps,'Gamma_GT':np.clip(np.expm1(pg_log),1,1000)})
    assert submission.ID.equals(sample.ID) and len(submission)==6399
    assert submission.columns.tolist()==['ID','Smoking','Gamma_GT']
    assert np.isfinite(submission[['Smoking','Gamma_GT']].to_numpy()).all()
    assert submission.Smoking.between(0,1).all() and submission.Gamma_GT.between(1,1000).all()
    submission.to_csv(OUT/'submission.csv',index=False)
    pd.DataFrame({'ID':train.ID,'Smoking':final_oof_s,'Gamma_GT':final_oof_g}).to_csv(OUT/'final_oof.csv',index=False)
    (OUT/'final_cv.json').write_text(json.dumps(metrics(train,final_oof_s,final_oof_g),indent=2))
    print('DONE',metrics(train,final_oof_s,final_oof_g),flush=True)


if __name__=='__main__':main()
