"""Semi-supervised feature-only gamma model: add test rows with a calibrated rapid reading as pseudo-labelled training rows.

Pseudo targets come from batch offsets fitted on the training fold only; validation rows never contribute.
Usage: python v2/pseudo_gamma.py <pseudo_weight>   (0 = baseline)
"""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, batch_features_v2, CATS

d, t, _ = load()
ref = pd.read_csv('v2/refined_cache.csv').drop(columns=['resid_cot', 'resid_etg'])
d = d.join(ref.iloc[:len(d)].reset_index(drop=True)); t = t.join(ref.iloc[len(d):].reset_index(drop=True))
W = float(sys.argv[1])
yg = np.log1p(d.Gamma_GT.values)
comp = np.load('outputs/v2/components.npz'); ps_oof, ps_te = comp['p_smoke_oof'], comp['p_smoke_test']
cats = [c for c in CATS if c != 'POC_Batch']
oof = np.zeros(len(d))
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(d)):
    a, b = d.iloc[tr], d.iloc[va]
    pa, pb, pt = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b, t])
    (_, gt), _ = batch_features_v2(a, [a, t])
    drop = ['POC_GGT', 'POC_Batch']
    Xa = base_X(a).drop(columns=drop).join(pa).assign(p_smoke=ps_oof[tr])
    Xb = base_X(b).drop(columns=drop).join(pb).assign(p_smoke=ps_oof[va])
    X, Y, w = Xa, yg[tr], np.ones(len(tr))
    if W > 0:
        ok = (gt.cal2_log1p.notna() & (t.POC_GGT > 3)).values  # usable, non-floor readings only
        Xt = base_X(t).drop(columns=drop).join(pt).assign(p_smoke=ps_te)[ok]
        X = pd.concat([Xa, Xt]); Y = np.r_[yg[tr], gt.cal2_log1p.values[ok]]; w = np.r_[np.ones(len(tr)), np.full(ok.sum(), W)]
    m = CatBoostRegressor(iterations=1800, depth=4, learning_rate=0.03, l2_leaf_reg=5, verbose=False, allow_writing_files=False,
                          thread_count=4, random_seed=k, cat_features=cats)
    m.fit(Pool(X, Y, cat_features=cats, weight=w)); oof[va] = m.predict(Xb)
fb = d.POC_GGT.isna().values
print(f'pseudo weight {W:.2f}: feature-only gamma RMSLE all {np.sqrt(np.mean((oof-yg)**2)):.4f}  no-reading rows {np.sqrt(np.mean((oof[fb]-yg[fb])**2)):.4f}', flush=True)
np.save(f'v2/pseudo_oof_{W:.2f}.npy', oof)
