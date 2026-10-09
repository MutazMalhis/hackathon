"""Smoking CV: plate features (train+test plate means) plus calibrated gamma-GT information."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, batch_features_v2, CATS

d, t, _ = load()
y = d.Smoking
gam = np.load('v2/gamma_oof_v2.npy')  # OOF gamma predictions (log1p), no smoking labels used
variant = sys.argv[1]
oof = np.zeros(len(d))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y)):
    a, b = d.iloc[tr], d.iloc[va]
    ref = pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]) if 'tt' in variant else a
    pa, pb = plate_centered(ref, [a, b])
    Xa, Xb = base_X(a).join(pa), base_X(b).join(pb)
    if 'cal' in variant:
        (ga, gb), _ = batch_features_v2(a, [a, b])
        Xa['cal2'] = ga.cal2_log1p; Xb['cal2'] = gb.cal2_log1p
    if 'gam' in variant:
        Xa['gam'] = gam[tr]; Xb['gam'] = gam[va]
    m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, cat_features=CATS, verbose=False,
                           allow_writing_files=False, thread_count=12, random_seed=k)
    m.fit(Xa, y.iloc[tr]); oof[va] = m.predict_proba(Xb)[:, 1]
print(variant, round(roc_auc_score(y, oof), 5), flush=True)
np.save(f'v2/smoking_oof_{variant}.npy', oof)
