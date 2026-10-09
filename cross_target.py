"""Nested, label-isolated Smoking predictions for Gamma_GT regression."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split

from improvements import engineered_features, fit_one, predict_one


def smoking_features(reference, directory, seed, full_model_path=None):
    """Every training probability excludes that row from model and preprocessing."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    oof = np.zeros(len(reference)); audit = []
    for fold, (fit, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(reference, reference.Smoking)):
        ref, frame = reference.iloc[fit], reference.iloc[valid]
        assert set(ref.ID).isdisjoint(frame.ID)
        cache = directory/f'smoking_oof{fold}.csv'
        if cache.exists():
            saved = pd.read_csv(cache)
            assert saved.ID.tolist() == frame.ID.tolist()
            p = saved.prediction.to_numpy()
        else:
            model, trees = fit_one(ref, 'smoke_plate4', seed+fold)
            p = predict_one(model, ref, frame, 'smoke_plate4')
            pd.DataFrame({'ID':frame.ID, 'prediction':p}).to_csv(cache, index=False)
            model.save_model(str(directory/f'smoking_oof{fold}.cbm'))
        oof[valid] = p
        audit.append(dict(fold=fold, fitting_ids=ref.ID.tolist(), predicted_ids=frame.ID.tolist(), own_label_used=False))
    (directory/'crossfit_audit.json').write_text(json.dumps(audit))
    full_path = Path(full_model_path) if full_model_path else directory/'smoking_full.cbm'
    if full_path.exists():
        full = CatBoostClassifier(); full.load_model(str(full_path))
    else:
        full, trees = fit_one(reference, 'smoke_plate4', seed+20)
        full.save_model(str(full_path))
    assert np.isfinite(oof).all() and np.logical_and(oof>=0, oof<=1).all()
    return oof, full


def gamma_features(reference, frame, smoking):
    x, cats = engineered_features(reference, frame, 'clinical')
    p = np.clip(np.asarray(smoking), 1e-5, 1-1e-5)
    assert len(p) == len(frame)
    x['predicted_smoking'] = p
    x['predicted_smoking_log_odds'] = np.log(p/(1-p))
    return x, cats


def fit_stacked(reference, directory, seed, full_model_path=None):
    """Both inner early stopping and outer evaluation get isolated probabilities."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    inner_fit, inner_stop = train_test_split(reference, test_size=.2,
                                           stratify=reference.Smoking, random_state=seed)
    # Do not generate early-stopping training features using stopping-row labels.
    inner_oof, inner_classifier = smoking_features(inner_fit, directory/'pilot_smoking', seed+50)
    stop_smoke = predict_one(inner_classifier, inner_fit, inner_stop, 'smoke_plate4')
    xi, cats = gamma_features(inner_fit, inner_fit, inner_oof)
    xs, _ = gamma_features(inner_fit, inner_stop, stop_smoke)
    params = dict(depth=4, learning_rate=.04, loss_function='RMSE', random_seed=seed,
                  cat_features=cats, verbose=False, allow_writing_files=False, thread_count=4)
    pilot = CatBoostRegressor(iterations=2400, **params)
    pilot.fit(xi, np.log1p(inner_fit.Gamma_GT), eval_set=(xs, np.log1p(inner_stop.Gamma_GT)), early_stopping_rounds=120)
    trees = max(1, pilot.get_best_iteration()+1)
    oof, classifier = smoking_features(reference, directory/'refit_smoking', seed+100, full_model_path)
    xf, _ = gamma_features(reference, reference, oof)
    model = CatBoostRegressor(iterations=trees, **params)
    model.fit(xf, np.log1p(reference.Gamma_GT))
    model.save_model(str(directory/'gamma_stacked.cbm'))
    (directory/'fit.json').write_text(json.dumps({'iterations':trees, 'seed':seed,
        'inner_fit_ids':inner_fit.ID.tolist(), 'inner_stop_ids':inner_stop.ID.tolist(),
        'training_smoking_predictions':'3-fold OOF', 'evaluation_smoking_predictions':'full-fitting-partition model'}))
    return model, classifier


def predict_stacked(model, classifier, reference, frame):
    p = predict_one(classifier, reference, frame, 'smoke_plate4')
    x, _ = gamma_features(reference, frame, p)
    return np.clip(np.expm1(model.predict(x)), 1, 1000)
