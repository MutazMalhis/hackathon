"""Shared helpers for v4: canonical folds, metric, slices, and prediction storage.

Canonical evaluation split = StratifiedKFold(5, shuffle=True, random_state=42) on Smoking,
the same split v2/nn.py and the seed-42 CatBoost smoking stage use. Every v4 model reports
its per-fold numbers on this split. The confirmation split is the same construction with
random_state=2027 (never used for selection).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'v4'
DEV_SEED, CONF_SEED = 42, 2027


def load_raw():
    return (pd.read_csv(ROOT / 'data/train.csv'), pd.read_csv(ROOT / 'data/test.csv'),
            pd.read_csv(ROOT / 'data/sample_submission.csv'))


def folds(train, seed=DEV_SEED, k=5):
    f = np.full(len(train), -1)
    for i, (_, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(train, train.Smoking)):
        f[va] = i
    return f


def metric(y_s, p_s, gamma_true, gamma_log):
    """gamma_log = predicted log1p(Gamma_GT); clipped to the scorer's [1, 1000] range."""
    auc = float(roc_auc_score(y_s, p_s))
    lp = np.log1p(np.clip(np.expm1(gamma_log), 1, 1000))
    rmsle = float(np.sqrt(np.mean((np.log1p(gamma_true) - lp) ** 2)))
    return dict(auc=auc, rmsle=rmsle, score=auc - rmsle / 1.0082)


def per_fold(train, p_s, gamma_log, fold_ids):
    rows = []
    for k in sorted(set(fold_ids)):
        m = fold_ids == k
        rows.append(dict(fold=int(k), **metric(train.Smoking.values[m], p_s[m], train.Gamma_GT.values[m], gamma_log[m])))
    return pd.DataFrame(rows)


def reading_group(frame):
    p = frame.POC_GGT
    return np.where(p.isna() | frame.POC_Batch.isna(), 'none', np.where(p <= 3, 'floor', 'calibrated'))


def save_preds(name, oof_s=None, oof_g=None, te_s=None, te_g=None, info=None):
    """Store OOF/test predictions (gamma in log1p space) for later blending."""
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in dict(oof_s=oof_s, oof_g=oof_g, te_s=te_s, te_g=te_g).items() if v is not None}
    np.savez(OUT / f'{name}.npz', **arrays)
    if info is not None:
        (OUT / f'{name}.json').write_text(json.dumps(info, indent=1, default=float))


class Timer:
    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t
