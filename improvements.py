"""Development-only improvements; preserve the previously evaluated holdout.

Run without arguments to select, confirm, and build a new five-fold CSV.
Cached fold results make an interrupted run resumable.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import experiments as base

OUT = Path('outputs/plan_v2')
SEED = base.SEED
SPECS = {
    'smoke_baseline': dict(kind='smoking', features='urine', depth=6, iterations=1600),
    'smoke_plate4': dict(kind='smoking', features='enriched', depth=4, iterations=2400),
    'smoke_plate6': dict(kind='smoking', features='enriched', depth=6, iterations=2400),
    'gamma_baseline': dict(kind='gamma', features='urine', depth=6, iterations=1600),
    'gamma_clinical4': dict(kind='gamma', features='clinical', depth=4, iterations=2400),
    'gamma_clinical6': dict(kind='gamma', features='clinical', depth=6, iterations=2400),
    'gamma_ridge': dict(kind='gamma', features='clinical', algorithm='ridge'),
    'gamma_hist': dict(kind='gamma', features='clinical', algorithm='hist'),
}


def engineered_features(reference, frame, variant):
    x, cats = base.features(reference, frame, 'urine')
    if variant == 'urine':
        return x, cats
    for marker in ['Urine_Cotinine_ng_mL', 'Urine_EtG_ng_mL']:
        values = np.log(reference[marker] / reference.Urine_Creatinine)
        groups = values.groupby(reference.Urine_Plate)
        for quantile in [.2, .5, .8]:
            centers = groups.quantile(quantile)
            mapped = frame.Urine_Plate.map(centers).fillna(values.quantile(quantile))
            x[f'plate_q{int(100*quantile)}_{marker}'] = x['log_ratio_' + marker] - mapped
        raw = np.log1p(reference[marker])
        centers = raw.groupby(reference.Urine_Plate).median()
        x['plate_raw_' + marker] = np.log1p(frame[marker]) - frame.Urine_Plate.map(centers).fillna(raw.median())
    for column in ['ALT', 'AST', 'Triglyceride', 'Fasting_Glucose',
                   'Alcohol_Units_Week', 'Urine_Creatinine', 'Exhaled_CO_ppm']:
        x['log1p_' + column] = np.log1p(frame[column].clip(lower=0))
    x['AST_ALT_ratio'] = frame.AST / frame.ALT.clip(lower=.1)
    x['TG_HDL_ratio'] = frame.Triglyceride / frame.HDL.clip(lower=.1)
    x['waist_height_ratio'] = frame.Waist_cm / frame.Height_cm.clip(lower=1)
    x['pulse_pressure'] = frame.BP_Systolic - frame.BP_Diastolic
    x['computed_BMI'] = frame.Weight_kg / (frame.Height_cm / 100).clip(lower=.1)**2
    x['LDL_formula_residual'] = frame.LDL - (frame.Cholesterol_Total-frame.HDL-frame.Triglyceride/5)
    if variant == 'clinical':
        # Assay scales vary by batch; the fallback should learn physiology.
        x = x.drop(columns=['POC_GGT', 'POC_Batch'])
        cats = [c for c in cats if c in x]
    numeric = [c for c in x if c not in cats]
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
    assert not {'ID', 'Smoking', 'Gamma_GT'}.intersection(x.columns)
    return x, cats


def model_input(reference, frame, spec):
    x, cats = engineered_features(reference, frame, spec['features'])
    if spec.get('algorithm') in ['ridge', 'hist']:
        # The complementary models do not learn high-cardinality group IDs.
        x = x.drop(columns=['Workplace_ID', 'Urine_Plate'], errors='ignore')
        cats = [c for c in cats if c in x]
    return x, cats


def fit_one(reference, name, seed):
    spec = SPECS[name]
    kind = spec['kind']
    algorithm = spec.get('algorithm', 'catboost')
    if algorithm != 'catboost':
        x, cats = model_input(reference, reference, spec)
        numeric = [c for c in x if c not in cats]
        preprocessing = ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
        ])
        estimator = Ridge(alpha=10) if algorithm == 'ridge' else HistGradientBoostingRegressor(
            max_iter=500, learning_rate=.04, max_leaf_nodes=15, l2_regularization=5,
            early_stopping=True, validation_fraction=.2, random_state=seed)
        model = make_pipeline(preprocessing, estimator)
        model.fit(x, np.log1p(reference.Gamma_GT))
        return model, int(getattr(estimator, 'n_iter_', 0) or 0)
    inner_fit, inner_stop = train_test_split(reference, test_size=.2,
                                           stratify=reference.Smoking, random_state=seed)
    xi, cats = model_input(inner_fit, inner_fit, spec)
    xs, _ = model_input(inner_fit, inner_stop, spec)
    cls = CatBoostClassifier if kind == 'smoking' else CatBoostRegressor
    extra = dict(loss_function='Logloss', eval_metric='AUC') if kind == 'smoking' else dict(loss_function='RMSE')
    params = dict(depth=spec['depth'], learning_rate=.04, random_seed=seed,
                  cat_features=cats, verbose=False, allow_writing_files=False,
                  thread_count=4, **extra)
    yi = inner_fit.Smoking if kind == 'smoking' else np.log1p(inner_fit.Gamma_GT)
    ys = inner_stop.Smoking if kind == 'smoking' else np.log1p(inner_stop.Gamma_GT)
    pilot = cls(iterations=spec['iterations'], **params)
    pilot.fit(xi, yi, eval_set=(xs, ys), early_stopping_rounds=120)
    iterations = max(1, pilot.get_best_iteration()+1)
    xf, _ = model_input(reference, reference, spec)
    yf = reference.Smoking if kind == 'smoking' else np.log1p(reference.Gamma_GT)
    model = cls(iterations=iterations, **params)
    model.fit(xf, yf)
    return model, iterations


def predict_one(model, reference, frame, name):
    spec = SPECS[name]
    x, _ = model_input(reference, frame, spec)
    if spec['kind'] == 'smoking':
        return model.predict_proba(x)[:, 1]
    return np.clip(np.expm1(model.predict(x)), 1, 1000)


def rmsle(actual, prediction):
    return float(np.sqrt(np.mean((np.log1p(actual)-np.log1p(prediction))**2)))


def development_data():
    train = pd.read_csv('data/train.csv')
    split = pd.read_csv(base.OUT/'split_ids.csv')
    ids = split.loc[split.partition.eq('development'), 'ID']
    dev = train.set_index('ID').loc[ids].reset_index()
    assert not set(dev.ID).intersection(split.loc[split.partition.eq('holdout'), 'ID'])
    return dev


def development_predictions(dev, seed, names, directory):
    directory.mkdir(parents=True, exist_ok=True)
    smoke = {n: np.zeros(len(dev)) for n in names if SPECS[n]['kind'] == 'smoking'}
    gamma = {n: np.zeros(len(dev)) for n in names if SPECS[n]['kind'] == 'gamma'}
    raw = np.zeros(len(dev)); covered = np.zeros(len(dev), dtype=bool)
    for fold, (fit, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(dev, dev.Smoking)):
        ref, frame = dev.iloc[fit], dev.iloc[valid]
        assert set(ref.ID).isdisjoint(frame.ID)
        raw[valid], covered[valid] = base.batch_correct(ref, frame, np.full(len(frame), 20.), weight=1)
        for name in names:
            path = directory/f'{name}_fold{fold}.csv'
            if path.exists():
                saved = pd.read_csv(path)
                assert saved.ID.tolist() == frame.ID.tolist()
                p = saved.prediction.to_numpy()
            else:
                model, trees = fit_one(ref, name, seed+fold)
                p = predict_one(model, ref, frame, name)
                pd.DataFrame({'ID': frame.ID, 'prediction': p}).to_csv(path, index=False)
                (directory/f'{name}_fold{fold}.json').write_text(json.dumps({'iterations': trees, 'seed': seed+fold}))
                if hasattr(model, 'save_model'):
                    model.save_model(str(directory/f'{name}_fold{fold}.cbm'))
            (smoke if name in smoke else gamma)[name][valid] = p
            label = dev.Smoking.iloc[valid] if name in smoke else dev.Gamma_GT.iloc[valid]
            metric = roc_auc_score(label, p) if name in smoke else rmsle(label, p)
            print(directory.name, 'fold', fold+1, name, round(metric, 6), flush=True)
    return smoke, gamma, raw, covered


def blend_gamma(raw, fallback, covered, weight):
    prediction = np.asarray(fallback).copy()
    prediction[covered] = np.expm1(weight*np.log1p(raw[covered])+(1-weight)*np.log1p(fallback[covered]))
    return np.clip(prediction, 1, 1000)


def ensemble(predictions, names, log=False):
    arrays = [np.log1p(predictions[n]) if log else predictions[n] for n in names]
    averaged = np.mean(arrays, axis=0)
    return np.expm1(averaged) if log else averaged


def evaluate_configuration(dev, predictions, config):
    smoke, gamma, raw, covered = predictions
    s = ensemble(smoke, config['smoking_models'])
    fallback = ensemble(gamma, config['gamma_models'], log=True)
    g = blend_gamma(raw, fallback, covered, config['calibration_weight'])
    result = base.metrics(dev, s, g)
    result.update(fallback_rmsle=rmsle(dev.Gamma_GT[~covered], g[~covered]),
                  high_gamma_rmsle=rmsle(dev.Gamma_GT[dev.Gamma_GT.gt(100)], g[dev.Gamma_GT.gt(100)]))
    return result, s, g


def select():
    dev = development_data()
    predictions = development_predictions(dev, SEED, list(SPECS), OUT/'selection')
    smoke, gamma, raw, covered = predictions
    smoke_candidates = [[n] for n in smoke] + [['smoke_plate4', 'smoke_plate6'], ['smoke_baseline', 'smoke_plate4', 'smoke_plate6']]
    gamma_candidates = [[n] for n in gamma] + [['gamma_clinical4', 'gamma_clinical6'],
        ['gamma_clinical4', 'gamma_clinical6', 'gamma_hist'], ['gamma_clinical4', 'gamma_hist']]
    best_smoke = max(smoke_candidates, key=lambda names: roc_auc_score(dev.Smoking, ensemble(smoke, names)))
    rows = []
    for names in gamma_candidates:
        for weight in [.8, .9, 1.]:
            config = dict(smoking_models=best_smoke, gamma_models=names, calibration_weight=weight)
            result, _, _ = evaluate_configuration(dev, predictions, config)
            rows.append({**config, **result})
    best = max(rows, key=lambda row: row['score'])
    config = {k: best[k] for k in ['smoking_models', 'gamma_models', 'calibration_weight']}
    baseline = dict(smoking_models=['smoke_baseline'], gamma_models=['gamma_baseline'], calibration_weight=.9)
    bm, _, _ = evaluate_configuration(dev, predictions, baseline)
    sm, s, g = evaluate_configuration(dev, predictions, config)
    report = dict(baseline=bm, selected=sm, improvement=sm['score']-bm['score'])
    smoke_rows = [dict(models=names, auc=float(roc_auc_score(dev.Smoking, ensemble(smoke, names)))) for names in smoke_candidates]
    pd.DataFrame(rows).to_csv(OUT/'development_comparison.csv', index=False)
    pd.DataFrame(smoke_rows).to_csv(OUT/'smoking_comparison.csv', index=False)
    pd.DataFrame({'ID': dev.ID, 'Smoking': s, 'Gamma_GT': g, 'calibrated': covered}).to_csv(OUT/'development_oof.csv', index=False)
    config.update(seed=SEED, specifications=SPECS, baseline=baseline,
                  holdout_used_for_selection=False, selection=report,
                  code_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['improvements.py', 'experiments.py']},
                  versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'catboost']},
                  data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data').glob('*.csv')})
    (OUT/'frozen_config.json').write_text(json.dumps(config, indent=2))
    print('SELECTION', json.dumps(report), 'MODELS', config['smoking_models'], config['gamma_models'], flush=True)


def confirm():
    config = json.loads((OUT/'frozen_config.json').read_text())
    dev = development_data()
    names = list(dict.fromkeys(config['smoking_models']+config['gamma_models']+['smoke_baseline', 'gamma_baseline']))
    predictions = development_predictions(dev, SEED+701, names, OUT/'confirmation')
    bm, bs, bg = evaluate_configuration(dev, predictions, config['baseline'])
    sm, s, g = evaluate_configuration(dev, predictions, config)
    folds = np.zeros(len(dev), dtype=int)
    for fold, (_, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED+701).split(dev, dev.Smoking)):
        folds[valid] = fold
    comparisons = []
    for fold in range(3):
        mask = folds == fold
        comparisons.append({'fold': fold+1, 'baseline': base.metrics(dev[mask], bs[mask], bg[mask]),
                            'selected': base.metrics(dev[mask], s[mask], g[mask])})
    # Conservative gate: improvement overall and at least two of three folds.
    passed = sm['score'] > bm['score']+.001 and sum(c['selected']['score'] > c['baseline']['score'] for c in comparisons) >= 2
    report = dict(baseline=bm, selected=sm, folds=comparisons, promotion_gate_passed=passed,
                  note='New fold assignment on reused development rows; stability check, not an independent holdout.')
    (OUT/'confirmation.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame({'ID':dev.ID, 'Smoking':s, 'Gamma_GT':g, 'calibrated': predictions[3]}).to_csv(OUT/'confirmation_oof.csv', index=False)
    print('CONFIRMATION', json.dumps(report), flush=True)


def finalize():
    config = json.loads((OUT/'frozen_config.json').read_text())
    confirmation = json.loads((OUT/'confirmation.json').read_text())
    if not confirmation['promotion_gate_passed']:
        print('No promotion: retain outputs/plan_v1/submission.csv.', flush=True)
        return
    train = pd.read_csv('data/train.csv'); test = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sample_submission.csv')
    names = list(dict.fromkeys(config['smoking_models']+config['gamma_models']))
    ps = np.zeros(len(test)); pg = np.zeros(len(test))
    os = np.zeros(len(train)); og = np.zeros(len(train))
    directory = OUT/'final'; directory.mkdir(exist_ok=True)
    for fold, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(train, train.Smoking)):
        ref, frame = train.iloc[fit], train.iloc[valid]
        vp, tp = {}, {}
        for name in names:
            path = directory/f'{name}_fold{fold}.npz'
            if path.exists():
                saved = np.load(path)
                assert np.array_equal(saved['valid_ids'], frame.ID) and np.array_equal(saved['test_ids'], test.ID)
                vp[name], tp[name] = saved['valid'], saved['test']
            else:
                model, trees = fit_one(ref, name, SEED+200+fold)
                vp[name] = predict_one(model, ref, frame, name)
                tp[name] = predict_one(model, ref, test, name)
                np.savez(path, valid=vp[name], test=tp[name], valid_ids=frame.ID, test_ids=test.ID)
                (directory/f'{name}_fold{fold}.json').write_text(json.dumps({'iterations': trees}))
                if hasattr(model, 'save_model'):
                    model.save_model(str(directory/f'{name}_fold{fold}.cbm'))
                else:
                    import joblib
                    joblib.dump(model, directory/f'{name}_fold{fold}.joblib')
            print('Final fold', fold+1, name, flush=True)
        vs = ensemble(vp, config['smoking_models']); ts = ensemble(tp, config['smoking_models'])
        vg = ensemble(vp, config['gamma_models'], log=True); tg = ensemble(tp, config['gamma_models'], log=True)
        vg, _ = base.batch_correct(ref, frame, vg, weight=config['calibration_weight'])
        tg, _ = base.batch_correct(ref, test, tg, weight=config['calibration_weight'])
        os[valid] = vs; og[valid] = vg
        ps += ts/5; pg += np.log1p(tg)/5
        print('Final metrics', fold+1, base.metrics(frame, vs, vg), flush=True)
    result = pd.DataFrame({'ID': test.ID, 'Smoking': ps, 'Gamma_GT': np.clip(np.expm1(pg), 1, 1000)})
    assert result.ID.equals(sample.ID) and len(result) == 6399
    assert result.columns.tolist() == sample.columns.tolist()
    assert np.isfinite(result[['Smoking', 'Gamma_GT']].to_numpy()).all()
    assert result.Smoking.between(0, 1).all() and result.Gamma_GT.between(1, 1000).all()
    result.to_csv(OUT/'submission.csv', index=False)
    pd.DataFrame({'ID': train.ID, 'Smoking': os, 'Gamma_GT': og}).to_csv(OUT/'final_oof.csv', index=False)
    old = pd.read_csv(base.OUT/'final_oof.csv')
    assert old.ID.equals(train.ID)
    report = dict(baseline=base.metrics(train, old.Smoking, old.Gamma_GT), selected=base.metrics(train, os, og),
                  note='Diagnostic CV on all training rows, including rows used in method selection.')
    (OUT/'final_cv.json').write_text(json.dumps(report, indent=2))
    print('FINAL', json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['select', 'confirm', 'finalize', 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # Invalidate cached runs if the training recipe or data changes.
    identity = dict(code={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['improvements.py', 'experiments.py']},
                    data={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('data').glob('*.csv')})
    manifest = OUT/'run_identity.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == identity, 'Recipe changed: choose a new output directory.'
    else:
        manifest.write_text(json.dumps(identity, indent=2))
    if args.phase in ['select', 'all']: select()
    if args.phase in ['confirm', 'all']: confirm()
    if args.phase in ['finalize', 'all']: finalize()


if __name__ == '__main__':
    main()
