"""Smoking: compare CatBoost settings and a HistGradientBoosting blend on fixed features."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, batch_features_v2, CATS

d, t, _ = load(); y = d.Smoking.values
cfg = sys.argv[1]
oof = np.zeros(len(d))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y)):
    a, b = d.iloc[tr], d.iloc[va]
    ref = pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t])
    pa, pb = plate_centered(ref, [a, b])
    (ga, gb), _ = batch_features_v2(a, [a, b])
    Xa = base_X(a).join(pa).assign(cal2=ga.cal2_log1p); Xb = base_X(b).join(pb).assign(cal2=gb.cal2_log1p)
    if cfg.startswith('hgb'):
        num = [c for c in Xa if c not in CATS]
        Xa2, Xb2 = Xa[num].copy(), Xb[num].copy()
        for c in ['Screening_Center', 'Visit_Quarter']:
            codes = {v: i for i, v in enumerate(sorted(Xa[c].unique()))}
            Xa2[c] = Xa[c].map(codes); Xb2[c] = Xb[c].map(codes)
        m = HistGradientBoostingClassifier(max_iter=1500, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0,
                                           categorical_features=['Screening_Center', 'Visit_Quarter'], random_state=k)
        m.fit(Xa2, y[tr]); oof[va] = m.predict_proba(Xb2)[:, 1]
    else:
        depth, lr, it, l2 = [float(x) for x in cfg.split('_')]
        m = CatBoostClassifier(iterations=int(it), depth=int(depth), learning_rate=lr, l2_leaf_reg=l2, cat_features=CATS,
                               verbose=False, allow_writing_files=False, thread_count=3, random_seed=k)
        m.fit(Xa, y[tr]); oof[va] = m.predict_proba(Xb)[:, 1]
print(cfg, round(roc_auc_score(y, oof), 5), flush=True)
np.save(f'v2/smk_tune_{cfg}.npy', oof)
