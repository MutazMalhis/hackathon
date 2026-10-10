"""Smoking AUC with 5-fold CV for feature-set variants."""
import sys
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, CATS

d, _, _ = load()
y = d.Smoking
variants = sys.argv[1].split(',') if len(sys.argv) > 1 else ['raw', 'plate']
for v in variants:
    oof = np.zeros(len(d))
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y)):
        a, b = d.iloc[tr], d.iloc[va]
        Xa, Xb = base_X(a), base_X(b)
        if v != 'raw':
            pa, pb = plate_centered(a, [a, b]); Xa, Xb = Xa.join(pa), Xb.join(pb)
        m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, cat_features=CATS, verbose=False,
                               allow_writing_files=False, thread_count=12, random_seed=k)
        m.fit(Xa, y.iloc[tr]); oof[va] = m.predict_proba(Xb)[:, 1]
    print(v, round(roc_auc_score(y, oof), 5), flush=True)
    np.save(f'v2/smoking_oof_{v}.npy', oof)
