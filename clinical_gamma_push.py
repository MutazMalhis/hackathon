"""Fold-fitted additive clinical regression, targeting missing rapid-test readings.

The new regressors are trained without target-derived features. Comparison to and
blending with cached NN/V3 predictions is conditional: those historic predictions
do not have fully nested provenance. No leaderboard submission is performed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from v2.common import load, plate_centered

OUT = Path('outputs/clinical_gamma_push')
REFERENCE = Path('outputs/final_review/submission_nn_v3_verified.csv')
BASE_OOF = Path('outputs/final_review/neural_blend_v3_oof.csv')
SPECS = {
    'spline4_a10': (4, 10., False),
    'spline6_a10': (6, 10., False),
    'spline4_a100': (4, 100., False),
    'spline4_interact': (4, 10., True),
}
CONTINUOUS = [
    'Triglyceride', 'Waist_cm', 'Hemoglobin', 'AST', 'Cholesterol_Total',
    'Age', 'LDL', 'Weight_kg', 'ALT', 'Serum_Creatinine', 'Fasting_Glucose',
    'Sleep_Hours_Avg', 'Alcohol_Units_Week', 'BMI', 'HDL', 'BP_Systolic',
    'BP_Diastolic', 'Height_cm', 'Exhaled_CO_ppm', 'Metabolic_Age_Gap',
]
CATEGORICAL = ['Screening_Center', 'Visit_Quarter', 'Liver_Echo_Grade',
               'Household_Smoke_Exposure', 'Dental_Caries', 'Urine_Protein',
               'Hearing_L', 'Hearing_R', 'Family_History_CVD']
LOG = ['Triglyceride', 'AST', 'ALT', 'Fasting_Glucose', 'Alcohol_Units_Week',
       'Exhaled_CO_ppm', 'Serum_Creatinine']


def features(reference, frame, interactions=False):
    """Fit only unlabeled plate means in reference; all other features row-local."""
    x = frame[CONTINUOUS + CATEGORICAL].copy()
    for c in LOG:
        x[c] = np.log1p(x[c].clip(lower=0))
    x = x.join(plate_centered(reference, [frame])[0][['pc_ratio_cot', 'pc_ratio_etg']])
    x['log_AST_ALT'] = np.log(frame.AST.clip(lower=.01) / frame.ALT.clip(lower=.01))
    x['log_TG_HDL'] = np.log(frame.Triglyceride.clip(lower=.01) / frame.HDL.clip(lower=.01))
    x['waist_height'] = frame.Waist_cm / frame.Height_cm
    x['BMI_calculated'] = frame.Weight_kg / (frame.Height_cm / 100) ** 2
    if interactions:
        for a, b in [('AST', 'ALT'), ('Alcohol_Units_Week', 'ALT'),
                     ('BMI', 'ALT'), ('Hemoglobin', 'pc_ratio_cot'),
                     ('Alcohol_Units_Week', 'pc_ratio_etg')]:
            x[a + '_times_' + b] = x[a] * x[b]
    for c in CATEGORICAL:
        x[c] = x[c].fillna('__MISSING__').astype(str)
    num = [c for c in x if c not in CATEGORICAL]
    x[num] = x[num].replace([np.inf, -np.inf], np.nan)
    assert not {'ID', 'Smoking', 'Gamma_GT', 'POC_GGT', 'POC_Batch'}.intersection(x)
    return x


def fit_predict(reference, frames, name):
    knots, alpha, interactions = SPECS[name]
    x = features(reference, reference, interactions)
    numeric = [c for c in x if c not in CATEGORICAL]
    # A separate indicator branch retains missingness without fitting splines to
    # binary indicators. All knots, medians, scales, and vocabularies fit here.
    from sklearn.impute import MissingIndicator
    pre = ColumnTransformer([
        ('smooth', make_pipeline(SimpleImputer(strategy='median'),
             SplineTransformer(n_knots=knots, degree=3, knots='quantile',
                               extrapolation='linear', include_bias=False),
             StandardScaler()), numeric),
        ('missing', MissingIndicator(features='all'), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ])
    if name.startswith('svr'):
        pre = ColumnTransformer([
            ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                      StandardScaler()), numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ])
        model = make_pipeline(pre, SVR(C=alpha, gamma=knots, epsilon=.05, cache_size=256))
    else:
        model = make_pipeline(pre, Ridge(alpha=alpha))
    model.fit(x, np.log1p(reference.Gamma_GT))
    return [np.clip(model.predict(features(reference, frame, interactions)),
                    np.log(2), np.log(1001)) for frame in frames]


def error(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def blend(base, new, mask, weight):
    p = np.array(base, copy=True)
    p[mask] = (1 - weight) * p[mask] + weight * new[mask]
    return p


def assess(y, base, new, mask):
    return dict(rmsle=error(y, new), missing_rmsle=error(y[mask], new[mask]),
                rmsle_gain=error(y, base) - error(y, new),
                combined_gain=(error(y, base) - error(y, new)) / 1.0082)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    train, test, sample = load()
    base = pd.read_csv(BASE_OOF)
    test_base = pd.read_csv(REFERENCE)
    assert base.ID.equals(train.ID) and test_base.ID.equals(sample.ID)
    base_log = np.log1p(base.Gamma_GT.to_numpy())
    y = np.log1p(train.Gamma_GT.to_numpy())
    missing = train.POC_GGT.isna().to_numpy()
    split = pd.read_csv('outputs/plan_v1/split_ids.csv')
    dev = np.flatnonzero(train.ID.isin(split.loc[split.partition.eq('development'), 'ID']))
    hold = np.setdiff1d(np.arange(len(train)), dev)
    rows, predictions = [], {}
    for name in SPECS:
        pred = np.full(len(dev), np.nan)
        for fold, (a, b) in enumerate(KFold(3, shuffle=True, random_state=47009).split(dev), 1):
            print('selection', name, fold, flush=True)
            pred[b] = fit_predict(train.iloc[dev[a]], [train.iloc[dev[b]]], name)[0]
        predictions[name] = pred
        for weight in [.15, .3, .5]:
            p = blend(base_log[dev], pred, missing[dev], weight)
            rows.append(dict(name=name, weight=weight,
                             clinical_missing_rmsle=error(y[dev][missing[dev]], pred[missing[dev]]),
                             **assess(y[dev], base_log[dev], p, missing[dev])))
    selection = pd.DataFrame(rows).sort_values('combined_gain', ascending=False)
    selection.to_csv(OUT / 'selection.csv', index=False)
    np.savez_compressed(OUT / 'selection_predictions.npz', ids=train.ID.iloc[dev], **predictions)
    best = selection.iloc[0]
    name, weight = str(best['name']), float(best.weight)
    clinical = fit_predict(train.iloc[dev], [train.iloc[hold]], name)[0]
    held = blend(base_log[hold], clinical, missing[hold], weight)
    confirmation = dict(name=name, weight=weight, rows=len(hold),
        missing_rows=int(missing[hold].sum()),
        baseline_rmsle=error(y[hold], base_log[hold]),
        baseline_missing_rmsle=error(y[hold][missing[hold]], base_log[hold][missing[hold]]),
        **assess(y[hold], base_log[hold], held, missing[hold]))
    confirmation['passed'] = bool(best.combined_gain > .0001 and confirmation['combined_gain'] > .0001)
    (OUT / 'confirmation.json').write_text(json.dumps(confirmation, indent=2))
    print(selection.to_string(index=False), flush=True)
    print('confirmation', confirmation, flush=True)
    # Finish a selected model's genuinely out-of-fold clinical predictions even
    # if the blend is rejected; this permits transparent later diagnostics.
    pred = np.full(len(train), np.nan)
    pt = np.zeros(len(test))
    for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=47710).split(train), 1):
        print('final', name, fold, flush=True)
        pv, test_fold = fit_predict(train.iloc[a], [train.iloc[b], test], name)
        pred[b] = pv
        pt += test_fold / 5
    candidate = blend(base_log, pred, missing, weight)
    test_candidate = blend(np.log1p(test_base.Gamma_GT.to_numpy()), pt,
                           test.POC_GGT.isna().to_numpy(), weight)
    result = dict(name=name, weight=weight, baseline_rmsle=error(y, base_log),
                  baseline_missing_rmsle=error(y[missing], base_log[missing]),
                  **assess(y, base_log, candidate, missing))
    result['supported'] = bool(confirmation['passed'] and result['combined_gain'] > .0001)
    (OUT / 'final_metrics.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame(dict(ID=train.ID, clinical_log=pred, candidate_gamma_log=candidate,
                     baseline_gamma_log=base_log)).to_csv(OUT / 'oof.csv', index=False)
    pd.DataFrame(dict(ID=test.ID, clinical_log=pt, candidate_gamma_log=test_candidate,
                     baseline_gamma_log=np.log1p(test_base.Gamma_GT))).to_csv(OUT / 'test.csv', index=False)
    paths = [Path(__file__), Path('v2/common.py'), Path('data/train.csv'), Path('data/test.csv'),
             Path('outputs/plan_v1/split_ids.csv'), BASE_OOF, REFERENCE]
    provenance = dict(hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn']},
        selection_seed=47009, final_seed=47710, specs=SPECS,
        limitation='Clinical models are fold-fitted with no true targets as predictors. Blends use cached NN/V3 OOF predictions, so comparison is conditional and not fully nested. The historic holdout has been inspected before.',
        all_targets_excluded_from_features=True, cpu_threads=2)
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('final', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['spline', 'svr'], default='spline')
    args = parser.parse_args()
    if args.family == 'svr':
        OUT = OUT / 'svr'
        SPECS = {'svr_smooth': (.015, 4., False), 'svr_flexible': (.03, 10., False)}
    with threadpool_limits(limits=2):
        main()
