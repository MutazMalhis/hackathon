"""Tune the feature-only gamma model (refined plate features cached)."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, CATS

d, t, _ = load()
ref = pd.read_csv('v2/refined_cache.csv'); d = d.join(ref.iloc[:len(d)].reset_index(drop=True)); t = t.join(ref.iloc[len(d):].reset_index(drop=True))
y = np.log1p(d.Gamma_GT.values)
smk = np.load('outputs/v2/components.npz')['p_smoke_oof']
depth, lr, l2, extra = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else ''
cats = [c for c in CATS if c != 'POC_Batch']
oof = np.zeros(len(d))
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(d)):
    a, b = d.iloc[tr], d.iloc[va]
    pa, pb = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b])
    Xa = base_X(a).drop(columns=['POC_GGT', 'POC_Batch']).join(pa).assign(p_smoke=smk[tr])
    Xb = base_X(b).drop(columns=['POC_GGT', 'POC_Batch']).join(pb).assign(p_smoke=smk[va])
    if 'nowp' in extra:
        Xa, Xb = Xa.drop(columns=['Workplace_ID', 'Urine_Plate']), Xb.drop(columns=['Workplace_ID', 'Urine_Plate'])
    c = [x for x in cats if x in Xa]
    kw = dict(grow_policy='Lossguide', max_leaves=int(depth)) if 'lg' in extra else dict(depth=int(depth))
    m = CatBoostRegressor(iterations=8000, learning_rate=lr, l2_leaf_reg=l2, loss_function='RMSE', cat_features=c,
                          verbose=False, allow_writing_files=False, thread_count=3, random_seed=k, **kw)
    idx = np.random.RandomState(k).rand(len(a)) < 0.85
    m.fit(Xa[idx], y[tr][idx], eval_set=(Xa[~idx], y[tr][~idx]), early_stopping_rounds=400)
    oof[va] = m.predict(Xb)
fb = d.POC_GGT.isna().values
print(sys.argv[1:], 'fallback', round(float(np.sqrt(np.mean((oof[fb]-y[fb])**2))), 4), 'all', round(float(np.sqrt(np.mean((oof-y)**2))), 4), flush=True)
