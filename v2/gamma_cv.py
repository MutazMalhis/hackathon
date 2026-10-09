"""Compare gamma-GT approaches with 5-fold CV: pure calibration vs model using calibrated feature."""
import sys
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, 'v2')
from common import load, base_X, batch_features, plate_centered, CATS

d, _, _ = load()
y = np.log1p(d.Gamma_GT)
res = {'pure_cal': np.full(len(d), np.nan), 'model': np.zeros(len(d))}
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(d)):
    a, b = d.iloc[tr], d.iloc[va]
    fa, fb = batch_features(a, [a, b])
    pa, pb = plate_centered(a, [a, b])
    Xa = base_X(a).join(fa).join(pa); Xb = base_X(b).join(fb).join(pb)
    m = CatBoostRegressor(iterations=3000, depth=6, learning_rate=0.03, loss_function='RMSE',
                          cat_features=CATS, verbose=False, allow_writing_files=False, thread_count=12, random_seed=k)
    # early stopping on a split of the fitting partition only
    idx = np.random.RandomState(k).rand(len(a)) < 0.85
    m.fit(Xa[idx], y.iloc[tr][idx], eval_set=(Xa[~idx], y.iloc[tr][~idx]), early_stopping_rounds=200)
    res['model'][va] = m.predict(Xb)
    res['pure_cal'][va] = fb.cal_log1p.values
    print('fold', k, m.get_best_iteration(), flush=True)
cal = res['pure_cal']; mod = res['model']; ok = ~np.isnan(cal)
r = lambda p, mask=slice(None): float(np.sqrt(np.mean((p[mask]-y.values[mask])**2)))
print('coverage', ok.mean())
print('model overall', r(mod), 'model on cal rows', r(mod, ok), 'model on fallback', r(mod, ~ok))
print('pure cal on cal rows', r(cal, ok))
hyb = np.where(ok, 0.9*cal + 0.1*mod, mod); print('old 90/10 hybrid', r(hyb))
for w in [0.3, 0.5, 0.7]:
    print('blend w_cal', w, r(np.where(ok, w*cal+(1-w)*mod, mod)))
np.save('v2/gamma_oof_model.npy', mod)
