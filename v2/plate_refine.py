"""Covariate-adjusted plate offsets for urine markers (unsupervised: uses features of train+test, no targets)."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, 'v2')
from common import load

DROP = ['ID', 'Smoking', 'Gamma_GT', 'Urine_Plate', 'Urine_Cotinine_ng_mL', 'Urine_EtG_ng_mL', 'POC_Batch', 'Workplace_ID']


def refined_offsets(a, n_iter=2, seed=0):
    """a: all rows (train+test features). Returns dict marker -> plate-centred refined feature (Series)."""
    X = a.drop(columns=[c for c in DROP if c in a]).copy()
    for c in ['Screening_Center', 'Visit_Quarter']:
        X[c] = X[c].astype(str)
    out = {}
    for mk, col in [('cot', 'Urine_Cotinine_ng_mL'), ('etg', 'Urine_EtG_ng_mL')]:
        r = np.log(a[col] / a.Urine_Creatinine)
        ok = r.notna().values
        off = r.groupby(a.Urine_Plate).transform('mean')
        for _ in range(n_iter):
            target = (r - off)
            pred = pd.Series(np.nan, index=a.index)
            for tr, va in KFold(5, shuffle=True, random_state=seed).split(X[ok]):
                idx_tr, idx_va = X.index[ok][tr], X.index[ok][va]
                m = CatBoostRegressor(iterations=600, depth=5, learning_rate=0.05, verbose=False, thread_count=12,
                                      cat_features=['Screening_Center', 'Visit_Quarter'], allow_writing_files=False, random_seed=seed)
                m.fit(X.loc[idx_tr], target.loc[idx_tr]); pred.loc[idx_va] = m.predict(X.loc[idx_va])
            off = (r - pred).groupby(a.Urine_Plate).transform('mean')
        out['rpc_' + mk] = r - off
        out['resid_' + mk] = r - pred  # marker minus covariate prediction (plate effect still inside)
    return pd.DataFrame(out, index=a.index)


if __name__ == '__main__':
    # Writes the per-row feature cache that v2/nn.py reads (train rows, then test rows).
    d, t, _ = load()
    refined_offsets(pd.concat([d, t], ignore_index=True)).to_csv('v2/refined_cache.csv', index=False)
    print('Wrote v2/refined_cache.csv')
