"""Final submission: CatBoost v2 + multi-task NN, blended 80/20 with the v3 candidate.

Inputs (produced by the steps in README.md):
  outputs/v2/components.npz      v2/pipeline.py 5
  outputs/v2/nn_preds.npz        v2/nn.py 3
  outputs/plan_v3/submission_v3.csv  refine.py --phase final
Writes outputs/final/submission.csv.

Smoking: rank blend, 90% CatBoost + 10% NN, then 80% of that + 20% v3 ranks.
Gamma_GT: per-group linear stack of the CatBoost components and the NN, then
80% of that + 20% v3, all in log1p space.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LinearRegression

OUT = Path('outputs/final')
PRIOR_SD = 0.26  # spread of the feature-only prior around the truth (log scale)
FLOOR_SHIFT = 0.975 * np.log(3.5 / 3)  # a reading of 3 means "below 3.5"
W_NN_SMOKE = 0.10
W_V3 = 0.20


def rank(values):
    return rankdata(values) / len(values)


def gamma_stack(train, test, c, nn):
    """Fit one linear stack per reading group on OOF predictions, apply to test."""
    y = np.log1p(train.Gamma_GT.to_numpy())

    def design(frame, part):
        cal, main, nop = c[part + '_cal'], c[part + '_main'], c[part + '_nop']
        nn_g = nn['oof_g' if part == 'oof' else 'te_g']
        prior = (nop + nn_g) / 2
        a = (cal + FLOOR_SHIFT - prior) / PRIOR_SD
        # Floor readings: expected value of the prior truncated above at the floor.
        truncated = prior - PRIOR_SD * norm.pdf(a) / np.clip(norm.cdf(a), 1e-6, None)
        groups = {'calibrated': np.isfinite(cal) & frame.POC_GGT.gt(3).to_numpy(),
                  'floor': np.isfinite(cal) & frame.POC_GGT.le(3).to_numpy(),
                  'none': ~np.isfinite(cal)}
        X = {'calibrated': np.c_[main, cal, nop, nn_g], 'floor': truncated[:, None], 'none': np.c_[main, nop, nn_g]}
        return groups, X

    go, xo = design(train, 'oof')
    gt, xt = design(test, 'te')
    pred = np.zeros(len(test))
    for g in go:
        model = LinearRegression().fit(xo[g][go[g]], y[go[g]])
        pred[gt[g]] = model.predict(xt[g][gt[g]])
    return pred


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sample_submission.csv')
    c = np.load('outputs/v2/components.npz')
    nn = np.load('outputs/v2/nn_preds.npz')
    v3 = pd.read_csv('outputs/plan_v3/submission_v3.csv')
    assert test.ID.equals(sample.ID) and v3.ID.equals(sample.ID)

    smoke = rank((1 - W_NN_SMOKE) * rank(c['p_smoke_test']) + W_NN_SMOKE * rank(nn['te_s']))
    smoke = (1 - W_V3) * smoke + W_V3 * rank(v3.Smoking)
    gamma = (1 - W_V3) * gamma_stack(train, test, c, nn) + W_V3 * np.log1p(v3.Gamma_GT)

    sub = pd.DataFrame({'ID': sample.ID, 'Smoking': smoke, 'Gamma_GT': np.clip(np.expm1(gamma), 1, 1000)})
    assert len(sub) == len(sample) and np.isfinite(sub[['Smoking', 'Gamma_GT']]).all().all()
    assert sub.Smoking.between(0, 1).all()
    OUT.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT / 'submission.csv', index=False)
    print(f'Wrote {OUT / "submission.csv"} ({len(sub)} rows)')


if __name__ == '__main__':
    sys.exit(main())
