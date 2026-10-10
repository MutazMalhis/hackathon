"""How much does more training data help? Compare k-fold CV with k = 3, 5, 10 (smoking AUC + feature-only gamma)."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, batch_features_v2, CATS

d, t, _ = load()
ref = pd.read_csv('v2/refined_cache.csv').drop(columns=['resid_cot', 'resid_etg'])
d = d.join(ref.iloc[:len(d)].reset_index(drop=True)); t = t.join(ref.iloc[len(d):].reset_index(drop=True))
y = d.Smoking.values; yg = np.log1p(d.Gamma_GT.values)
K = int(sys.argv[1])
oof, oofg = np.zeros(len(d)), np.zeros(len(d))
for k, (tr, va) in enumerate(StratifiedKFold(K, shuffle=True, random_state=42).split(d, y)):
    a, b = d.iloc[tr], d.iloc[va]
    pa, pb = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b])
    (ga, gb), _ = batch_features_v2(a, [a, b])
    Xa = base_X(a).join(pa).assign(cal2=ga.cal2_log1p); Xb = base_X(b).join(pb).assign(cal2=gb.cal2_log1p)
    m = CatBoostClassifier(iterations=1600, depth=6, learning_rate=0.025, cat_features=CATS, verbose=False,
                           allow_writing_files=False, thread_count=4, random_seed=k)
    m.fit(Xa, y[tr]); oof[va] = m.predict_proba(Xb)[:, 1]
    Na = Xa.drop(columns=['POC_GGT', 'POC_Batch', 'cal2']); Nb = Xb.drop(columns=['POC_GGT', 'POC_Batch', 'cal2'])
    r = CatBoostRegressor(iterations=1800, depth=4, learning_rate=0.03, l2_leaf_reg=5, verbose=False, allow_writing_files=False,
                          thread_count=4, random_seed=k, cat_features=[c for c in CATS if c != 'POC_Batch'])
    r.fit(Na, yg[tr]); oofg[va] = r.predict(Nb)
print(f'K={K} train frac {1-1/K:.2f}  smoking AUC {roc_auc_score(y, oof):.5f}  feature-only gamma RMSLE {np.sqrt(np.mean((oofg-yg)**2)):.4f}', flush=True)
