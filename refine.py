"""Refine the v2 candidate using calibration and nested cross-target features."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import experiments as base
import improvements as previous
from advanced_calibration import fit_calibration, calibrated_prediction, compare_cached, candidate_configs
from cross_target import fit_stacked, predict_stacked

OUT = Path('outputs/plan_v3')
SEED = previous.SEED


def calibrate_experiment():
    dev = previous.development_data()
    rows, details, predictions, smoke = compare_cached(dev, previous.OUT/'selection', SEED)
    rows.to_csv(OUT/'calibration_selection.csv', index=False)
    details.to_csv(OUT/'calibration_selection_folds.csv', index=False)
    name = rows.sort_values('score', ascending=False).iloc[0]['name']
    config = dict(name=name, config=candidate_configs()[name], selected_before_confirmation=True)
    (OUT/'calibration_frozen.json').write_text(json.dumps(config, indent=2))
    rows, details, predictions, smoke = compare_cached(dev, previous.OUT/'confirmation', SEED+701)
    rows.to_csv(OUT/'calibration_confirmation.csv', index=False)
    details.to_csv(OUT/'calibration_confirmation_folds.csv', index=False)
    a = rows.set_index('name').loc[name]; b = rows.set_index('name').loc['v2']
    wins = sum(details[details.name.eq(name)].sort_values('fold').score.to_numpy() > details[details.name.eq('v2')].sort_values('fold').score.to_numpy())
    result = dict(selected_name=name, baseline=b[['auc','rmsle','score','floor_rmsle']].to_dict(),
                  selected=a[['auc','rmsle','score','floor_rmsle']].to_dict(), fold_wins=int(wins),
                  promotion_gate_passed=bool(a.score>b.score+.001 and wins>=2),
                  note='Second fold assignment on reused development rows; not independent validation.')
    (OUT/'calibration_confirmation.json').write_text(json.dumps(result, indent=2))
    print('CALIBRATION', json.dumps(result), flush=True)


def stage_predictions(dev, stage, seed):
    cache = previous.OUT/stage
    directory = OUT/stage; directory.mkdir(exist_ok=True)
    smoking = np.zeros(len(dev)); clinical = np.zeros(len(dev)); stacked = np.zeros(len(dev))
    assignments = np.zeros(len(dev), dtype=int)
    for fold, (fit, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(dev, dev.Smoking)):
        ref, frame = dev.iloc[fit], dev.iloc[valid]
        smoke_parts = {}
        for name in ['smoke_baseline','smoke_plate4','smoke_plate6']:
            saved = pd.read_csv(cache/f'{name}_fold{fold}.csv')
            assert saved.ID.tolist() == frame.ID.tolist()
            smoke_parts[name] = saved.prediction.to_numpy()
        smoking[valid] = previous.ensemble(smoke_parts, list(smoke_parts))
        saved = pd.read_csv(cache/f'gamma_clinical4_fold{fold}.csv')
        assert saved.ID.tolist() == frame.ID.tolist()
        clinical[valid] = saved.prediction.to_numpy()
        saved_path = directory/f'stack_fold{fold}.csv'
        if saved_path.exists():
            saved = pd.read_csv(saved_path)
            assert saved.ID.tolist() == frame.ID.tolist()
            prediction = saved.prediction.to_numpy()
        else:
            model, classifier = fit_stacked(ref, directory/f'stack_fold{fold}', seed+fold,
                                            cache/f'smoke_plate4_fold{fold}.cbm')
            prediction = predict_stacked(model, classifier, ref, frame)
            pd.DataFrame({'ID':frame.ID,'prediction':prediction}).to_csv(saved_path,index=False)
        stacked[valid] = prediction
        assignments[valid] = fold
        print(stage, 'stack fold', fold+1, 'regression RMSLE', previous.rmsle(frame.Gamma_GT, prediction), flush=True)
    return smoking, clinical, stacked, assignments


def regression_choices(clinical, stacked):
    return {'clinical':clinical, 'stacked':stacked,
            'blend':np.expm1(.5*np.log1p(clinical)+.5*np.log1p(stacked))}


def evaluate_choices(dev, smoking, fallbacks, assignments, seed, config):
    predictions = {name:np.zeros(len(dev)) for name in fallbacks}
    rows = []; folds = []; floor = dev.POC_GGT.eq(3).to_numpy()
    for fold, (fit, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(dev, dev.Smoking)):
        ref, frame = dev.iloc[fit], dev.iloc[valid]
        calibration = fit_calibration(ref, config)
        for name, fallback in fallbacks.items():
            predictions[name][valid], _, _ = calibrated_prediction(calibration, frame, fallback[valid], config)
            folds.append(dict(name=name,fold=fold+1,**base.metrics(frame,smoking[valid],predictions[name][valid])))
    missing = dev.POC_GGT.isna().to_numpy()
    for name, p in predictions.items():
        rows.append(dict(name=name,**base.metrics(dev, smoking, p),
                         regression_rmsle=previous.rmsle(dev.Gamma_GT, fallbacks[name]),
                         missing_reading_rmsle=previous.rmsle(dev.Gamma_GT[missing],p[missing]),
                         floor_rmsle=previous.rmsle(dev.Gamma_GT[floor],p[floor])))
    return pd.DataFrame(rows), pd.DataFrame(folds), predictions


def select():
    calibration = json.loads((OUT/'calibration_frozen.json').read_text())
    confirmed = json.loads((OUT/'calibration_confirmation.json').read_text())
    if not confirmed['promotion_gate_passed']:
        calibration = dict(name='v2',config=candidate_configs()['v2'])
    dev = previous.development_data()
    smoke, clinical, stacked, assignments = stage_predictions(dev, 'selection', SEED)
    rows, folds, predictions = evaluate_choices(dev, smoke, regression_choices(clinical,stacked),assignments,SEED,calibration['config'])
    rows.to_csv(OUT/'stack_selection.csv',index=False);folds.to_csv(OUT/'stack_selection_folds.csv',index=False)
    best = rows.sort_values('score',ascending=False).iloc[0]
    baseline = rows.set_index('name').loc['clinical']
    name = best['name'] if best.score > baseline.score+.0005 else 'clinical'
    config = dict(calibration=calibration,regression=name,stack_threshold=.0005,
                  selection=rows.to_dict(orient='records'),seed=SEED,
                  holdout_used_for_selection=False,
                  note='Calibration and stacking selected on previously used development rows.')
    (OUT/'frozen_config.json').write_text(json.dumps(config,indent=2))
    pd.DataFrame({'ID':dev.ID,'Smoking':smoke,'Gamma_GT':predictions[name]}).to_csv(OUT/'development_oof.csv',index=False)
    print('STACK SELECTION', rows.to_json(orient='records'), 'SELECTED',name,flush=True)


def confirm():
    config = json.loads((OUT/'frozen_config.json').read_text())
    if config['regression'] == 'clinical':
        (OUT/'stack_confirmation.json').write_text(json.dumps(dict(promoted=False,selected='clinical',
            note='No development gain exceeding 0.0005; omit stacking without additional tuning.'),indent=2))
        print('Stacking not selected; retain clinical regression.',flush=True)
        return
    dev = previous.development_data()
    smoke, clinical, stacked, assignments = stage_predictions(dev,'confirmation',SEED+701)
    rows, folds, predictions = evaluate_choices(dev,smoke,regression_choices(clinical,stacked),assignments,SEED+701,config['calibration']['config'])
    rows.to_csv(OUT/'stack_confirmation.csv',index=False);folds.to_csv(OUT/'stack_confirmation_folds.csv',index=False)
    name = config['regression'];index=rows.set_index('name')
    wins = sum(folds[folds.name.eq(name)].sort_values('fold').score.to_numpy()>folds[folds.name.eq('clinical')].sort_values('fold').score.to_numpy())
    passed = bool(index.loc[name,'score']>index.loc['clinical','score']+.0005 and wins>=2)
    result=dict(promoted=passed,selected=name if passed else 'clinical',fold_wins=int(wins),
                comparisons=rows.to_dict(orient='records'),
                note='Second development fold assignment; stability check, not an independent holdout.')
    (OUT/'stack_confirmation.json').write_text(json.dumps(result,indent=2))
    print('STACK CONFIRMATION',json.dumps(result),flush=True)


def finalize():
    config = json.loads((OUT/'frozen_config.json').read_text())
    stack_confirmation = json.loads((OUT/'stack_confirmation.json').read_text())
    regression = stack_confirmation['selected']
    train = pd.read_csv('data/train.csv');test=pd.read_csv('data/test.csv');sample=pd.read_csv('data/sample_submission.csv')
    original=pd.read_csv(previous.OUT/'final_oof.csv');old_submission=pd.read_csv(previous.OUT/'submission.csv')
    assert original.ID.equals(train.ID) and old_submission.ID.equals(test.ID)
    oof=np.zeros(len(train));test_log=np.zeros(len(test));covered=np.zeros(len(train),dtype=bool);floor=np.zeros(len(train),dtype=bool)
    directory=OUT/'final';directory.mkdir(exist_ok=True)
    for fold,(fit,valid) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(train,train.Smoking)):
        ref,frame=train.iloc[fit],train.iloc[valid]
        cached=np.load(previous.OUT/'final'/f'gamma_clinical4_fold{fold}.npz')
        assert np.array_equal(cached['valid_ids'],frame.ID) and np.array_equal(cached['test_ids'],test.ID)
        vg,tg=cached['valid'],cached['test']
        if regression != 'clinical':
            path=directory/f'stack_fold{fold}.npz'
            if path.exists():
                saved=np.load(path)
                assert np.array_equal(saved['valid_ids'],frame.ID) and np.array_equal(saved['test_ids'],test.ID)
                vs,ts=saved['valid'],saved['test']
            else:
                model,classifier=fit_stacked(ref,directory/f'stack_fold{fold}',SEED+200+fold,
                    previous.OUT/'final'/f'smoke_plate4_fold{fold}.cbm')
                vs=predict_stacked(model,classifier,ref,frame);ts=predict_stacked(model,classifier,ref,test)
                np.savez(path,valid=vs,test=ts,valid_ids=frame.ID,test_ids=test.ID)
            if regression == 'stacked': vg,tg=vs,ts
            else:
                vg=np.expm1(.5*np.log1p(vg)+.5*np.log1p(vs));tg=np.expm1(.5*np.log1p(tg)+.5*np.log1p(ts))
        calibration=fit_calibration(ref,config['calibration']['config'])
        vg,covered[valid],floor[valid]=calibrated_prediction(calibration,frame,vg,config['calibration']['config'])
        tg,_,_=calibrated_prediction(calibration,test,tg,config['calibration']['config'])
        oof[valid]=vg;test_log+=np.log1p(tg)/5
        (directory/f'calibration_fold{fold}.json').write_text(json.dumps(dict(slope=calibration.slope,
            variance=calibration.variance,offsets={str(k):float(v) for k,v in calibration.offsets.items()},
            effective_counts={str(k):float(v) for k,v in calibration.effective_counts.items()}),indent=2))
        print('Final fold',fold+1,base.metrics(frame,original.Smoking.iloc[valid],vg),flush=True)
    submission=pd.DataFrame({'ID':test.ID,'Smoking':old_submission.Smoking,'Gamma_GT':np.clip(np.expm1(test_log),1,1000)})
    assert submission.ID.equals(sample.ID) and len(submission)==6399
    assert submission.columns.tolist()==sample.columns.tolist()
    assert np.isfinite(submission[['Smoking','Gamma_GT']].to_numpy()).all()
    assert submission.Smoking.between(0,1).all() and submission.Gamma_GT.between(1,1000).all()
    submission.to_csv(OUT/'submission_v3.csv',index=False)
    pd.DataFrame({'ID':train.ID,'Smoking':original.Smoking,'Gamma_GT':oof,'calibrated':covered,'floor':floor}).to_csv(OUT/'final_oof.csv',index=False)
    slices=[]
    for name,mask in [('Nonfloor calibrated',covered & ~floor),('Floor=3',train.POC_GGT.eq(3).to_numpy()),
                      ('Fallback',~covered),('Gamma_GT >100',train.Gamma_GT.gt(100).to_numpy())]:
        slices.append(dict(subset=name,rows=int(mask.sum()),baseline_rmsle=previous.rmsle(train.Gamma_GT[mask],original.Gamma_GT[mask]),
                           improved_rmsle=previous.rmsle(train.Gamma_GT[mask],oof[mask])))
    pd.DataFrame(slices).to_csv(OUT/'final_error_slices.csv',index=False)
    report=dict(baseline=base.metrics(train,original.Smoking,original.Gamma_GT),selected=base.metrics(train,original.Smoking,oof),
                regression=regression,calibration=config['calibration'],
                user_reported_previous_kaggle_score=.85329,
                note='Final CV includes model-selection rows. New public score is unknown.')
    (OUT/'final_cv.json').write_text(json.dumps(report,indent=2))
    print('FINAL',json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['calibration','select','confirm','finalize','all'],default='all')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(code={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ['refine.py','advanced_calibration.py','cross_target.py','improvements.py','experiments.py']},
        data={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data').glob('*.csv')},
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost']})
    manifest=OUT/'run_identity.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==identity,'Recipe changed: use a new output directory.'
    else: manifest.write_text(json.dumps(identity,indent=2))
    if args.phase in ['calibration','all']:calibrate_experiment()
    if args.phase in ['select','all']:select()
    if args.phase in ['confirm','all']:confirm()
    if args.phase in ['finalize','all']:finalize()


if __name__=='__main__':main()
