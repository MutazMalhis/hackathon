"""v2 pipeline: out-of-fold validation and test predictions for both targets.

Stages (each fitted inside the same 5-fold split, test predictions averaged over folds):
  1. smoking classifier  - plate-centred urine markers (creatinine unit fix) + calibrated gamma
  2. gamma main model    - all features + slope-adjusted, censoring-aware batch calibration
  3. gamma feature-only  - no rapid-test inputs; used for rows without a calibrated reading
  4. linear stack        - per group (calibrated / detection floor / no reading), fitted on OOF
Usage: python v2/pipeline.py [n_seeds] [pseudo_weight]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

sys.path.insert(0, 'v2')
from common import CATS, base_X, batch_features, batch_features_v2, load, plate_centered, score
from plate_refine import refined_offsets

OUT = Path('outputs/v2')
OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
THREADS = 12
REFIT_W = 0.5  # weight of the full-data refit in test predictions
# Test rows with a usable reading join the feature-only gamma model (target = calibrated reading).
# Second argument; 0 reproduces the best public submission (0.86397), 1.0 the later candidate.
PSEUDO_W = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SMK_CFGS = [(6, 0.025, 1600, 3), (4, 0.03, 2500, 3), (5, 0.02, 3000, 10), (6, 0.02, 2000, 5), (5, 0.03, 1800, 3)]
NOP_CFGS = [dict(depth=4, learning_rate=0.015, l2_leaf_reg=10), dict(depth=3, learning_rate=0.03, l2_leaf_reg=5),
            dict(grow_policy='Lossguide', max_leaves=16, learning_rate=0.02, l2_leaf_reg=10),
            dict(depth=5, learning_rate=0.02, l2_leaf_reg=10), dict(depth=4, learning_rate=0.03, l2_leaf_reg=5)]

d, t, sub = load()
y_s = d.Smoking.values
y_g = np.log1p(d.Gamma_GT.values)
unlabeled = t  # test features (no labels) also inform the unsupervised plate means
# Covariate-adjusted plate offsets use features of train+test only (no targets), so one global fit is fold-safe.
_ref = refined_offsets(pd.concat([d, t], ignore_index=True))
d = d.join(_ref.iloc[:len(d)].reset_index(drop=True))
t = t.join(_ref.iloc[len(d):].reset_index(drop=True))


def smoking_stage(seed):
    oof, test = np.zeros(len(d)), np.zeros(len(t))
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(d, y_s)):
        a, b = d.iloc[tr], d.iloc[va]
        ref = pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), unlabeled])
        pa, pb, pt = plate_centered(ref, [a, b, t])
        (ga, gb, gt), _ = batch_features_v2(a, [a, b, t])
        X = [base_X(f).join(p).assign(cal2=g.cal2_log1p) for f, p, g in [(a, pa, ga), (b, pb, gb), (t, pt, gt)]]
        depth, lr, it, l2 = SMK_CFGS[seed % len(SMK_CFGS)]  # vary settings across seeds for ensemble diversity
        m = CatBoostClassifier(iterations=it, depth=depth, learning_rate=lr, l2_leaf_reg=l2, cat_features=CATS,
                               verbose=False, allow_writing_files=False, thread_count=THREADS, random_seed=seed*10+k)
        m.fit(X[0], y_s[tr])
        oof[va] = m.predict_proba(X[1])[:, 1]
        test += m.predict_proba(X[2])[:, 1] / 5
    # Full-data refit: same settings on all training rows; test prediction = mean of fold average and refit.
    pd_, pt_ = plate_centered(pd.concat([d, t]), [d, t])
    (gd, gt_), _ = batch_features_v2(d, [d, t])
    Xd, Xt = base_X(d).join(pd_).assign(cal2=gd.cal2_log1p), base_X(t).join(pt_).assign(cal2=gt_.cal2_log1p)
    m = CatBoostClassifier(iterations=it, depth=depth, learning_rate=lr, l2_leaf_reg=l2, cat_features=CATS,
                           verbose=False, allow_writing_files=False, thread_count=THREADS, random_seed=seed*10+9)
    m.fit(Xd, y_s)
    test = REFIT_W*m.predict_proba(Xt)[:, 1] + (1-REFIT_W)*test
    return oof, test


def gamma_stage(seed, p_smoke_oof, p_smoke_test):
    oof_main, oof_nop, oof_cal = np.zeros(len(d)), np.zeros(len(d)), np.full(len(d), np.nan)
    te_main, te_nop = np.zeros(len(t)), np.zeros(len(t))
    best = {'main': [], 'nop': []}
    for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(d)):
        a, b = d.iloc[tr], d.iloc[va]
        fa, fb, ft = batch_features(a, [a, b, t])
        (ga, gb, gt), _ = batch_features_v2(a, [a, b, t])
        pa, pb, pt = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b, t])
        ps = [p_smoke_oof[tr], p_smoke_oof[va], p_smoke_test]
        frames = [(a, fa, ga, pa), (b, fb, gb, pb), (t, ft, gt, pt)]
        Xm = [base_X(f).join(x1).join(x2).join(x3).assign(p_smoke=p) for (f, x1, x2, x3), p in zip(frames, ps)]
        Xn = [base_X(f).drop(columns=['POC_GGT', 'POC_Batch']).join(x3).assign(p_smoke=p)
              for (f, _, _, x3), p in zip(frames, ps)]
        idx = np.random.RandomState(seed*10+k).rand(len(a)) < 0.85
        nop_kw = NOP_CFGS[seed % len(NOP_CFGS)]
        for kind, X, kw, cats in [('main', Xm, dict(depth=6, learning_rate=0.03, l2_leaf_reg=5), CATS),
                                  ('nop', Xn, nop_kw, [c for c in CATS if c != 'POC_Batch'])]:
            m = CatBoostRegressor(iterations=8000, loss_function='RMSE', cat_features=cats, verbose=False,
                                  allow_writing_files=False, thread_count=THREADS, random_seed=seed*10+k, **kw)
            if kind == 'nop' and PSEUDO_W > 0:
                ok = (gt.cal2_log1p.notna() & (t.POC_GGT > 3)).values  # offsets from this fold's training rows only
                cats_n = [c for c in CATS if c != 'POC_Batch']
                fit = Pool(pd.concat([X[0][idx], X[2][ok]]), np.r_[y_g[tr][idx], gt.cal2_log1p.values[ok]], cat_features=cats_n,
                           weight=np.r_[np.ones(idx.sum()), np.full(ok.sum(), PSEUDO_W)])
                m.fit(fit, eval_set=Pool(X[0][~idx], y_g[tr][~idx], cat_features=cats_n), early_stopping_rounds=400)
            else:
                m.fit(X[0][idx], y_g[tr][idx], eval_set=(X[0][~idx], y_g[tr][~idx]), early_stopping_rounds=400)
            best[kind].append(m.get_best_iteration() + 1)
            if kind == 'main':
                oof_main[va] = m.predict(X[1]); te_main += m.predict(X[2]) / 5
            else:
                oof_nop[va] = m.predict(X[1]); te_nop += m.predict(X[2]) / 5
        oof_cal[va] = gb.cal2_log1p.values
    # Full-data refit; fold models stopped on ~68% of rows, so scale their tree counts up for 100%.
    fd, ft = batch_features(d, [d, t])
    (gd, gt), _ = batch_features_v2(d, [d, t])
    pd_, pt_ = plate_centered(pd.concat([d, t]), [d, t])
    Xm = [base_X(f).join(x1).join(x2).join(x3).assign(p_smoke=p)
          for f, x1, x2, x3, p in [(d, fd, gd, pd_, p_smoke_oof), (t, ft, gt, pt_, p_smoke_test)]]
    Xn = [base_X(f).drop(columns=['POC_GGT', 'POC_Batch']).join(x3).assign(p_smoke=p)
          for f, x3, p in [(d, pd_, p_smoke_oof), (t, pt_, p_smoke_test)]]
    for kind, X, kw, cats in [('main', Xm, dict(depth=6, learning_rate=0.03, l2_leaf_reg=5), CATS),
                              ('nop', Xn, nop_kw, [c for c in CATS if c != 'POC_Batch'])]:
        n_it = int(np.mean(best[kind]) * 1.2)
        m = CatBoostRegressor(iterations=n_it, loss_function='RMSE', cat_features=cats, verbose=False,
                              allow_writing_files=False, thread_count=THREADS, random_seed=seed*10+9, **kw)
        if kind == 'nop' and PSEUDO_W > 0:
            ok = (gt.cal2_log1p.notna() & (t.POC_GGT > 3)).values
            m.fit(Pool(pd.concat([X[0], X[1][ok]]), np.r_[y_g, gt.cal2_log1p.values[ok]], cat_features=cats,
                       weight=np.r_[np.ones(len(d)), np.full(ok.sum(), PSEUDO_W)]))
        else:
            m.fit(X[0], y_g)
        if kind == 'main':
            te_main = REFIT_W*m.predict(X[1]) + (1-REFIT_W)*te_main
        else:
            te_nop = REFIT_W*m.predict(X[1]) + (1-REFIT_W)*te_nop
    return oof_main, oof_nop, oof_cal, te_main, te_nop


def groups(frame, cal):
    has = ~np.isnan(cal)
    floor = has & (frame.POC_GGT.values <= 3)
    return {'calibrated': has & ~floor, 'floor': floor, 'none': ~has}


def truncated_mean(mu, upper, sigma=0.26):
    """E[x | x <= upper] for x ~ N(mu, sigma): POC_GGT == 3 means the reading is at most ~3.5."""
    from scipy.stats import norm
    a = (upper - mu) / sigma
    return mu - sigma * norm.pdf(a) / np.clip(norm.cdf(a), 1e-6, None)


def stack(oof_main, oof_nop, oof_cal, te_main, te_nop, te_cal):
    oof, test, info = np.zeros(len(d)), np.zeros(len(t)), {}
    shift = 0.975 * np.log(3.5 / 3)  # floor reading -> upper bound of the censored value
    oof_tr, te_tr = truncated_mean(oof_nop, oof_cal + shift), truncated_mean(te_nop, te_cal + shift)
    go, gt = groups(d, oof_cal), groups(t, te_cal)
    for name in go:
        if name == 'none':
            Xo, Xt = np.c_[oof_main, oof_nop][go[name]], np.c_[te_main, te_nop][gt[name]]
        elif name == 'floor':
            Xo, Xt = oof_tr[go[name]][:, None], te_tr[gt[name]][:, None]
        else:
            Xo, Xt = np.c_[oof_main, oof_cal, oof_nop][go[name]], np.c_[te_main, te_cal, te_nop][gt[name]]
        lr = LinearRegression()
        oof[go[name]] = cross_val_predict(lr, Xo, y_g[go[name]], cv=5)  # honest estimate of the stack
        lr.fit(Xo, y_g[go[name]])
        test[gt[name]] = lr.predict(Xt)
        rm = float(np.sqrt(np.mean((oof[go[name]] - y_g[go[name]])**2)))
        info[name] = {'n_train': int(go[name].sum()), 'n_test': int(gt[name].sum()), 'rmsle': rm,
                      'coef': lr.coef_.round(4).tolist(), 'intercept': round(float(lr.intercept_), 4)}
    return oof, test, info


smk_oof, smk_te, g_parts = [], [], []
for s in range(N_SEEDS):
    so, st = smoking_stage(42 + s)
    smk_oof.append(so); smk_te.append(st)
    print(f'seed {s}: smoking AUC {score(y_s, so, d.Gamma_GT, d.Gamma_GT)[0]:.5f}', flush=True)
p_smoke_oof, p_smoke_test = np.mean(smk_oof, 0), np.mean(smk_te, 0)
for s in range(N_SEEDS):
    g_parts.append(gamma_stage(42 + s, p_smoke_oof, p_smoke_test))
    print(f'seed {s}: gamma main done', flush=True)
oof_main, oof_nop, oof_cal, te_main, te_nop = [np.mean([p[i] for p in g_parts], 0) for i in range(5)]
oof_cal = g_parts[0][2]
(_, te_cal_frame), _ = batch_features_v2(d, [d, t])  # test calibration from all training rows
te_cal = np.where(t.POC_GGT.notna() & te_cal_frame.cal2_log1p.notna(), te_cal_frame.cal2_log1p, np.nan)
np.savez(OUT / 'components.npz', oof_main=oof_main, oof_nop=oof_nop, oof_cal=oof_cal, te_main=te_main,
         te_nop=te_nop, te_cal=te_cal, p_smoke_oof=p_smoke_oof, p_smoke_test=p_smoke_test)
g_oof, g_test, info = stack(oof_main, oof_nop, oof_cal, te_main, te_nop, te_cal)

auc, rmsle, sc = score(y_s, p_smoke_oof, d.Gamma_GT, np.expm1(g_oof))
print(json.dumps(info, indent=1))
print(f'OOF  AUC {auc:.5f}  RMSLE {rmsle:.5f}  SCORE {sc:.5f}')

out = pd.DataFrame({'ID': t.ID, 'Smoking': np.clip(p_smoke_test, 0, 1), 'Gamma_GT': np.clip(np.expm1(g_test), 1, 1000)})
assert (out.ID.values == sub.ID.values).all() and len(out) == len(sub) and np.isfinite(out[['Smoking', 'Gamma_GT']]).all().all()
out.to_csv(OUT / 'submission.csv', index=False)
pd.DataFrame({'ID': d.ID, 'smoking_oof': p_smoke_oof, 'gamma_oof': np.expm1(g_oof)}).to_csv(OUT / 'oof.csv', index=False)
json.dump({'auc': auc, 'rmsle': rmsle, 'score': sc, 'n_seeds': N_SEEDS, 'stack': info}, open(OUT / 'cv.json', 'w'), indent=1)
print('wrote', OUT / 'submission.csv')
