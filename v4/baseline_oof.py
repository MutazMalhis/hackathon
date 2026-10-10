"""Reproduce the out-of-fold score of the current final blend (README: 0.86076) and slice its errors.

Needs the outputs of the README steps (outputs/v2/components.npz, outputs/v2/nn_preds.npz,
outputs/plan_v3/final_oof.csv, outputs/plan_v3/submission_v3.csv). The OOF recipe is the one
make_submission.py applies to test: Smoking = rank(0.9 CatBoost + 0.1 NN), 80/20 with V3 ranks;
Gamma_GT = per-reading-group linear stack (5-fold cross_val_predict on OOF), 80/20 with V3 in log1p.

Writes outputs/v4/baseline.npz (+ .json), outputs/v4/baseline_folds.csv, outputs/v4/baseline_slices.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common4 import OUT, ROOT, folds, load_raw, metric, per_fold, reading_group, save_preds

PRIOR_SD, FLOOR_SHIFT, W_NN, W_V3 = 0.26, 0.975 * np.log(3.5 / 3), 0.10, 0.20
rank = lambda v: rankdata(v) / len(v)


def gamma_stack(train, test, c, nn):
    y = np.log1p(train.Gamma_GT.to_numpy())

    def design(frame, part):
        cal, main, nop = c[part + '_cal'], c[part + '_main'], c[part + '_nop']
        nn_g = nn['oof_g' if part == 'oof' else 'te_g']
        prior = (nop + nn_g) / 2
        a = (cal + FLOOR_SHIFT - prior) / PRIOR_SD
        trunc = prior - PRIOR_SD * norm.pdf(a) / np.clip(norm.cdf(a), 1e-6, None)
        g = {'calibrated': np.isfinite(cal) & frame.POC_GGT.gt(3).to_numpy(),
             'floor': np.isfinite(cal) & frame.POC_GGT.le(3).to_numpy(), 'none': ~np.isfinite(cal)}
        X = {'calibrated': np.c_[main, cal, nop, nn_g], 'floor': trunc[:, None], 'none': np.c_[main, nop, nn_g]}
        return g, X

    go, xo = design(train, 'oof'); gt, xt = design(test, 'te')
    oof, te = np.zeros(len(train)), np.zeros(len(test))
    for k in go:
        oof[go[k]] = cross_val_predict(LinearRegression(), xo[k][go[k]], y[go[k]], cv=5)
        te[gt[k]] = LinearRegression().fit(xo[k][go[k]], y[go[k]]).predict(xt[k][gt[k]])
    return oof, te


def slices(train, p_s, g_log):
    y = train.Smoking.values; err2 = (np.log1p(train.Gamma_GT.values) - g_log) ** 2
    miss_n = train.drop(columns=['ID', 'Smoking', 'Gamma_GT']).isna().sum(1)
    defs = {
        'reading_group': reading_group(train),
        'gamma_range': pd.cut(train.Gamma_GT, [0, 10, 15, 25, 50, 100, np.inf]).astype(str),
        'batch_train_n': pd.cut(train.POC_Batch.map(train.POC_Batch.value_counts()).fillna(0), [-1, 0, 3, 6, 10, 99]).astype(str),
        'center': train.Screening_Center.values,
        'age': pd.cut(train.Age, [0, 30, 40, 50, 60, 120]).astype(str),
        # No sex column exists; height is the strongest available proxy (bimodal in the data).
        'height_proxy': np.where(train.Height_cm >= 165, 'tall>=165', 'short<165'),
        'co_missing': train.Exhaled_CO_ppm.isna().map({True: 'CO missing', False: 'CO observed'}).values,
        'urine_missing': train.Urine_Cotinine_ng_mL.isna().map({True: 'urine missing', False: 'urine observed'}).values,
        'n_missing': pd.cut(miss_n, [-1, 0, 1, 2, 40]).astype(str),
    }
    rows = []
    for name, lab in defs.items():
        lab = np.asarray(lab)
        for v in pd.unique(lab):
            m = lab == v
            rows.append(dict(slice=name, group=str(v), rows=int(m.sum()), smoking_rate=float(y[m].mean()),
                             auc=float(roc_auc_score(y[m], p_s[m])) if len(set(y[m])) == 2 else np.nan,
                             rmsle=float(np.sqrt(err2[m].mean())), sq_err_share=float(err2[m].sum() / err2.sum())))
    return pd.DataFrame(rows).sort_values(['slice', 'group'])


def main():
    train, test, sample = load_raw()
    c = np.load(ROOT / 'outputs/v2/components.npz'); nn = np.load(ROOT / 'outputs/v2/nn_preds.npz')
    v3 = pd.read_csv(ROOT / 'outputs/plan_v3/final_oof.csv'); v3t = pd.read_csv(ROOT / 'outputs/plan_v3/submission_v3.csv')
    assert v3.ID.equals(train.ID) and v3t.ID.equals(sample.ID)
    g_oof, g_te = gamma_stack(train, test, c, nn)
    s_oof = 0.8 * rank(0.9 * rank(c['p_smoke_oof']) + 0.1 * rank(nn['oof_s'])) + 0.2 * rank(v3.Smoking)
    s_te = 0.8 * rank(0.9 * rank(c['p_smoke_test']) + 0.1 * rank(nn['te_s'])) + 0.2 * rank(v3t.Smoking)
    g_oof = 0.8 * g_oof + 0.2 * np.log1p(v3.Gamma_GT.values)
    g_te = 0.8 * g_te + 0.2 * np.log1p(v3t.Gamma_GT.values)

    # Cross-check against make_submission.py (same test recipe, but it fits the stack without CV).
    final = ROOT / 'outputs/final/submission.csv'
    if final.exists():
        f = pd.read_csv(final)
        print('max |Smoking - make_submission|', float(np.abs(f.Smoking - s_te).max()),
              ' max |log1p Gamma diff|', float(np.abs(np.log1p(f.Gamma_GT) - np.log1p(np.clip(np.expm1(g_te), 1, 1000))).max()))

    fid = folds(train)
    overall = metric(train.Smoking.values, s_oof, train.Gamma_GT.values, g_oof)
    pf = per_fold(train, s_oof, g_oof, fid)
    comps = {
        'catboost_smoking_only': metric(train.Smoking.values, c['p_smoke_oof'], train.Gamma_GT.values, g_oof)['auc'],
        'nn_smoking_only': metric(train.Smoking.values, nn['oof_s'], train.Gamma_GT.values, g_oof)['auc'],
        'v3_smoking_only': metric(train.Smoking.values, v3.Smoking, train.Gamma_GT.values, g_oof)['auc'],
        'v3_gamma_only_rmsle': metric(train.Smoking.values, v3.Smoking, train.Gamma_GT.values, np.log1p(v3.Gamma_GT.values))['rmsle'],
    }
    print('OVERALL', json.dumps(overall)); print(pf.to_string(index=False)); print(json.dumps(comps, indent=1))
    print('fold score mean %.5f  sd %.5f' % (pf.score.mean(), pf.score.std(ddof=1)))
    OUT.mkdir(parents=True, exist_ok=True)
    pf.to_csv(OUT / 'baseline_folds.csv', index=False)
    sl = slices(train, s_oof, g_oof); sl.to_csv(OUT / 'baseline_slices.csv', index=False)
    print(sl.to_string(index=False))
    save_preds('baseline', s_oof, g_oof, s_te, g_te, dict(overall=overall, components=comps,
               note='Reconstructed final blend; stack weights fit on the same OOF (selection-biased, see README).'))


if __name__ == '__main__':
    main()
