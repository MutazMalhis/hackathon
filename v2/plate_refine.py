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
    from sklearn.metrics import roc_auc_score as A
    d, t, _ = load()
    a = pd.concat([d, t], ignore_index=True)
    f = refined_offsets(a).iloc[:len(d)]
    r_c = np.log(a.Urine_Cotinine_ng_mL / a.Urine_Creatinine); r_e = np.log(a.Urine_EtG_ng_mL / a.Urine_Creatinine)
    base_c = (r_c - r_c.groupby(a.Urine_Plate).transform('mean')).iloc[:len(d)]
    base_e = (r_e - r_e.groupby(a.Urine_Plate).transform('mean')).iloc[:len(d)]
    ok = f.rpc_cot.notna()
    print('cot AUC  base', round(A(d.Smoking[ok], base_c[ok]), 4), ' refined', round(A(d.Smoking[ok], f.rpc_cot[ok]), 4))
    y = np.log1p(d.Gamma_GT)
    print('etg spearman base', round(base_e.corr(y, method='spearman'), 4), ' refined', round(f.rpc_etg.corr(y, method='spearman'), 4))
