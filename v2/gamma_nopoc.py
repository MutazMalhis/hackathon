"""Feature-only gamma model (no rapid-test inputs) for fallback rows."""
import sys
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, CATS

d, t, _ = load()
y = np.log1p(d.Gamma_GT)
smk = np.load('v2/smoking_oof_plate_tt_cal.npy')
import pandas as pd
TT = 'tt' in sys.argv
depth = int(sys.argv[1])
cats = [c for c in CATS if c != 'POC_Batch']
oof = np.zeros(len(d))
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(d)):
    a, b = d.iloc[tr], d.iloc[va]
    ref = pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]) if TT else a
    pa, pb = plate_centered(ref, [a, b])
    Xa = base_X(a).drop(columns=['POC_GGT', 'POC_Batch']).join(pa); Xb = base_X(b).drop(columns=['POC_GGT', 'POC_Batch']).join(pb)
    Xa['p_smoke'] = smk[tr]; Xb['p_smoke'] = smk[va]
    m = CatBoostRegressor(iterations=5000, depth=depth, learning_rate=0.03, loss_function='RMSE', l2_leaf_reg=5,
                          cat_features=cats, verbose=False, allow_writing_files=False, thread_count=12, random_seed=k)
    idx = np.random.RandomState(k).rand(len(a)) < 0.85
    m.fit(Xa[idx], y.iloc[tr][idx], eval_set=(Xa[~idx], y.iloc[tr][~idx]), early_stopping_rounds=300)
    oof[va] = m.predict(Xb)
fb = d.POC_GGT.isna().values
r = lambda p, mask: float(np.sqrt(np.mean((p[mask]-y.values[mask])**2)))
print('depth', depth, 'tt', TT, 'no-POC model: fallback', r(oof, fb), 'all rows', r(oof, ~fb | fb))
np.save(f'v2/gamma_oof_nopoc_d{depth}{"_tt" if TT else ""}.npy', oof)
