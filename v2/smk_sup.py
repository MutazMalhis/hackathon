"""Smoking CV: refined features, optionally plus label-adjusted (LOO) cotinine plate offsets."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, batch_features_v2, CATS

d, t, _ = load()
ref = pd.read_csv('v2/refined_cache.csv'); d = d.join(ref.iloc[:len(d)].reset_index(drop=True)); t = t.join(ref.iloc[len(d):].reset_index(drop=True))
y = d.Smoking.values
comp = np.load('outputs/v2/components.npz'); p_oof, p_te = comp['p_smoke_oof'], comp['p_smoke_test']
SUP = 'sup' in sys.argv


def sup_feature(a, b, gam):
    """offset_plate = mean(resid - gam*s) over the plate; s = label (fit rows, LOO) or predicted prob (others)."""
    rows = pd.concat([a.assign(s=a.Smoking.astype(float), lab=1), b.assign(s=p_oof[b.index], lab=0),
                      t.assign(s=p_te, lab=0)], ignore_index=True)
    v = rows.resid_cot - gam*rows.s
    ok = v.notna()
    S = v.where(ok, 0).groupby(rows.Urine_Plate).transform('sum'); N = ok.astype(int).groupby(rows.Urine_Plate).transform('sum')
    loo = (S - v.where(ok, 0)) / (N - ok.astype(int)).replace(0, np.nan)
    feat = rows.resid_cot - loo + rows.resid_cot*0  # resid minus other-rows offset
    return feat.iloc[:len(a)].values, feat.iloc[len(a):len(a)+len(b)].values


oof = np.zeros(len(d))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y)):
    a, b = d.iloc[tr], d.iloc[va]
    pa, pb = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b])
    (ga, gb), _ = batch_features_v2(a, [a, b])
    Xa = base_X(a).join(pa).assign(cal2=ga.cal2_log1p); Xb = base_X(b).join(pb).assign(cal2=gb.cal2_log1p)
    Xa = Xa.drop(columns=['resid_cot', 'resid_etg']); Xb = Xb.drop(columns=['resid_cot', 'resid_etg'])
    if SUP:
        c = a.resid_cot - a.resid_cot.groupby(a.Urine_Plate).transform('mean'); sc = a.Smoking - a.Smoking.groupby(a.Urine_Plate).transform('mean')
        gam = float((c*sc).sum() / (sc*sc).sum())
        fa, fb = sup_feature(a, b, gam); Xa['sup_cot'] = fa; Xb['sup_cot'] = fb
        if k == 0:
            ok = ~np.isnan(fb); print('gamma', round(gam, 3), 'fold0 single AUC sup', round(roc_auc_score(y[va][ok], fb[ok]), 4),
                                         'rpc', round(roc_auc_score(y[va][ok], b.rpc_cot.values[ok]), 4))
    m = CatBoostClassifier(iterations=1600, depth=6, learning_rate=0.025, cat_features=CATS, verbose=False,
                           allow_writing_files=False, thread_count=6, random_seed=k)
    m.fit(Xa, y[tr]); oof[va] = m.predict_proba(Xb)[:, 1]
print('SUP' if SUP else 'base', round(roc_auc_score(y, oof), 5))
