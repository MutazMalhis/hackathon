"""Gamma-GT CV with unit fix, slope-adjusted calibration and OOF smoking probability."""
import sys
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, 'v2')
from common import load, base_X, batch_features, batch_features_v2, plate_centered, CATS

d, _, _ = load()
y = np.log1p(d.Gamma_GT)
use_smk = 'smk' in sys.argv
smk = np.load('v2/smoking_oof_plate.npy')
mod = np.zeros(len(d)); cal2 = np.full(len(d), np.nan)
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(d)):
    a, b = d.iloc[tr], d.iloc[va]
    fa, fb = batch_features(a, [a, b])
    (ga, gb), beta = batch_features_v2(a, [a, b])
    pa, pb = plate_centered(a, [a, b])
    Xa = base_X(a).join(fa).join(ga).join(pa); Xb = base_X(b).join(fb).join(gb).join(pb)
    if use_smk:
        Xa['p_smoke'] = smk[tr]; Xb['p_smoke'] = smk[va]
    m = CatBoostRegressor(iterations=4000, depth=6, learning_rate=0.03, loss_function='RMSE',
                          cat_features=CATS, verbose=False, allow_writing_files=False, thread_count=12, random_seed=k)
    idx = np.random.RandomState(k).rand(len(a)) < 0.85
    m.fit(Xa[idx], y.iloc[tr][idx], eval_set=(Xa[~idx], y.iloc[tr][~idx]), early_stopping_rounds=200)
    mod[va] = m.predict(Xb); cal2[va] = gb.cal2_log1p.values
    print('fold', k, 'beta', round(beta, 4), 'iters', m.get_best_iteration(), flush=True)
ok = ~np.isnan(cal2)
r = lambda p, mask=slice(None): float(np.sqrt(np.mean((p[mask]-y.values[mask])**2)))
print('model overall', r(mod), 'cal rows', r(mod, ok), 'fallback', r(mod, ~ok))
print('pure cal2 on cal rows', r(cal2, ok))
np.save('v2/gamma_oof_v2%s.npy' % ('_smk' if use_smk else ''), mod)
