"""Smooth-model diversity for smoking: logistic regression and MLP on engineered features (5-fold, same split as pipeline seed 42)."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
sys.path.insert(0, 'v2')
from common import load, plate_centered, batch_features_v2

d, t, _ = load()
ref = pd.read_csv('v2/refined_cache.csv').drop(columns=['resid_cot', 'resid_etg'])
d = d.join(ref.iloc[:len(d)].reset_index(drop=True)); t = t.join(ref.iloc[len(d):].reset_index(drop=True))
y = d.Smoking.values
NUM = [c for c in d if d[c].dtype != object and c not in ['ID', 'Smoking', 'Gamma_GT', 'Workplace_ID', 'Urine_Plate', 'POC_Batch']]


def design(f, med, p, g):
    x = f[NUM].join(p).assign(cal2=g.cal2_log1p)
    miss = x.isna().astype(float).add_suffix('_na')
    miss = miss.loc[:, miss.columns.isin([c + '_na' for c in ['Exhaled_CO_ppm', 'Liver_Echo_Grade', 'Alcohol_Units_Week', 'Household_Smoke_Exposure', 'cal2', 'rpc_cot', 'pc_ratio_cot']])]
    for c in ['Urine_Cotinine_ng_mL', 'Urine_EtG_ng_mL', 'Urine_Creatinine', 'Exhaled_CO_ppm', 'Triglyceride', 'ALT', 'AST', 'POC_GGT']:
        x[c] = np.log1p(x[c])
    x = x.fillna(med)
    cen = pd.get_dummies(f.Screening_Center, prefix='c').reindex(columns=[f'c_{k}' for k in 'ABCDEFGH'], fill_value=0).astype(float)
    return pd.concat([x, miss, cen], axis=1)


oof = {'logit': np.zeros(len(d)), 'mlp': np.zeros(len(d))}
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y)):
    a, b = d.iloc[tr], d.iloc[va]
    pa, pb = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b])
    (ga, gb), _ = batch_features_v2(a, [a, b])
    raw_a = a[NUM].join(pa).assign(cal2=ga.cal2_log1p)
    med = np.log1p(raw_a.clip(lower=0)).median()  # placeholder medians on transformed scale
    Xa0 = design(a, 0, pa, ga); med = Xa0.median()
    Xa = design(a, med, pa, ga); Xb = design(b, med, pb, gb)
    lg = make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3), LogisticRegression(C=0.05, max_iter=3000))
    lg.fit(Xa, y[tr]); oof['logit'][va] = lg.predict_proba(Xb)[:, 1]
    p_m = np.zeros(len(va))
    for s in range(3):
        mlp = make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=300,
                                                            early_stopping=True, validation_fraction=0.15, random_state=s))
        mlp.fit(Xa, y[tr]); p_m += mlp.predict_proba(Xb)[:, 1] / 3
    oof['mlp'][va] = p_m
cb = pd.read_csv('outputs/v2/oof.csv').smoking_oof.values
print('catboost (v2)', round(roc_auc_score(y, cb), 5))
for k, v in oof.items():
    print(k, round(roc_auc_score(y, v), 5))
    np.save(f'v2/smk_{k}.npy', v)
R = lambda v: rankdata(v) / len(v)
for w in [0.1, 0.2, 0.3]:
    print(f'blend w={w}: +logit {roc_auc_score(y, (1-w)*R(cb)+w*R(oof["logit"])):.5f}  +mlp {roc_auc_score(y, (1-w)*R(cb)+w*R(oof["mlp"])):.5f}  '
          f'+both {roc_auc_score(y, (1-w)*R(cb)+w/2*R(oof["logit"])+w/2*R(oof["mlp"])):.5f}')
