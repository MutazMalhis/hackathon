"""Reproducible two-target baseline; only competition data is used."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def score(y_smoke, y_gamma, smoke, gamma):
    auc = roc_auc_score(y_smoke, smoke)
    rmsle = np.sqrt(np.mean((np.log1p(y_gamma) - np.log1p(np.clip(gamma, 1, 1000))) ** 2))
    return {"auc": float(auc), "rmsle": float(rmsle),
            "score": float(0.5 * (2 * auc - 1) + 0.5 * (1 - rmsle / 0.5041))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train = pd.read_csv(args.data / 'train.csv')
    test = pd.read_csv(args.data / 'test.csv')
    sample = pd.read_csv(args.data / 'sample_submission.csv')
    targets = ['Smoking', 'Gamma_GT']
    assert train.ID.is_unique and test.ID.is_unique and sample.ID.is_unique
    assert len(test) == 6399 and set(test.ID) == set(sample.ID)
    assert set(train.Smoking.unique()) == {0, 1}
    assert np.isfinite(train.Gamma_GT).all() and (train.Gamma_GT >= 0).all()
    features = [c for c in test.columns if c != 'ID']
    assert set(train.columns) == set(test.columns) | set(targets)
    x, xt = train[features].copy(), test[features].copy()
    # Identifier-like fields are categorical, never continuous measurements.
    categorical = [c for c in features if x[c].dtype == 'object' or c in
                   ['Urine_Plate', 'POC_Batch', 'Workplace_ID', 'Screening_Center', 'Visit_Quarter']]
    for c in features:
        if c in categorical:
            x[c] = x[c].fillna('__MISSING__').astype(str)
            xt[c] = xt[c].fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='raise').replace([np.inf, -np.inf], np.nan)
            xt[c] = pd.to_numeric(xt[c], errors='raise').replace([np.inf, -np.inf], np.nan)
    ys = train.Smoking.to_numpy()
    yg = np.log1p(train.Gamma_GT.to_numpy())
    oof_s, oof_g = np.zeros(len(x)), np.zeros(len(x))
    pred_s, pred_g_log = np.zeros(len(xt)), np.zeros(len(xt))
    args.output.mkdir(parents=True, exist_ok=True)
    fold_metrics = []
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr, va) in enumerate(cv.split(x, ys)):
        common = dict(iterations=args.iterations, depth=6, learning_rate=0.04,
                      random_seed=args.seed + fold, cat_features=categorical,
                      verbose=False, allow_writing_files=False, thread_count=4)
        classifier = CatBoostClassifier(**common, loss_function='Logloss', eval_metric='AUC')
        regressor = CatBoostRegressor(**common, loss_function='RMSE')
        classifier.fit(x.iloc[tr], ys[tr], eval_set=(x.iloc[va], ys[va]), early_stopping_rounds=150)
        regressor.fit(x.iloc[tr], yg[tr], eval_set=(x.iloc[va], yg[va]), early_stopping_rounds=150)
        oof_s[va] = classifier.predict_proba(x.iloc[va])[:, 1]
        oof_g[va] = np.clip(np.expm1(regressor.predict(x.iloc[va])), 1, 1000)
        pred_s += classifier.predict_proba(xt)[:, 1] / args.folds
        pred_g_log += regressor.predict(xt) / args.folds
        metrics = score(ys[va], train.Gamma_GT.to_numpy()[va], oof_s[va], oof_g[va])
        fold_metrics.append(metrics)
        print(f'Fold {fold + 1}: {metrics}', flush=True)
        classifier.save_model(str(args.output / f'smoking_fold{fold}.cbm'))
        regressor.save_model(str(args.output / f'gamma_fold{fold}.cbm'))
    predictions = pd.DataFrame({'ID': test.ID, 'Smoking': pred_s,
                                'Gamma_GT': np.clip(np.expm1(pred_g_log), 1, 1000)})
    submission = sample[['ID']].merge(predictions, on='ID', how='left', validate='one_to_one')
    assert submission.ID.tolist() == sample.ID.tolist()
    assert np.isfinite(submission[targets].to_numpy()).all()
    assert submission.Smoking.between(0, 1).all()
    submission.to_csv(args.output / 'submission.csv', index=False)
    pd.DataFrame({'ID': train.ID, 'Smoking': oof_s, 'Gamma_GT': oof_g}).to_csv(args.output / 'oof.csv', index=False)
    report = {'overall': score(ys, train.Gamma_GT.to_numpy(), oof_s, oof_g),
              'folds': fold_metrics, 'seed': args.seed, 'iterations': args.iterations,
              'features': features, 'categorical': categorical,
              'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in args.data.glob('*.csv')}}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(report['overall'])


if __name__ == '__main__':
    main()
