"""Evaluate adding the NN to the v2 ensemble (smoking rank blend + gamma stack with NN as an extra input)."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, 'v2')
from common import load

d, t, sub = load()
y_s, y_g = d.Smoking.values, np.log1p(d.Gamma_GT.values)
c = np.load('outputs/v2/components.npz'); nnp = np.load('outputs/v2/nn_preds.npz')
R = lambda v: rankdata(v) / len(v)


def tmean(mu, up, s=0.26):
    a = (up - mu) / s
    return mu - s * norm.pdf(a) / np.clip(norm.cdf(a), 1e-6, None)


def gamma_stack(use_nn):
    oof, test = np.zeros(len(d)), np.zeros(len(t))
    shift = 0.975*np.log(3.5/3)
    o, te = dict(c), dict(c)
    # feature-only prior: optionally average CatBoost and NN feature-only models
    prior_o = (c['oof_nop'] + nnp['oof_g'])/2 if use_nn else c['oof_nop']
    prior_t = (c['te_nop'] + nnp['te_g'])/2 if use_nn else c['te_nop']
    for part, frame, cal, main, nop, nn_, pri, out in [('oof', d, c['oof_cal'], c['oof_main'], c['oof_nop'], nnp['oof_g'], prior_o, oof),
                                                        ('test', t, c['te_cal'], c['te_main'], c['te_nop'], nnp['te_g'], prior_t, test)]:
        pass
    groups = lambda f, cal: {'calibrated': ~np.isnan(cal) & (f.POC_GGT.values > 3), 'floor': ~np.isnan(cal) & (f.POC_GGT.values <= 3), 'none': np.isnan(cal)}
    go, gt = groups(d, c['oof_cal']), groups(t, c['te_cal'])
    res = {}
    for g in go:
        if g == 'none':
            Xo = np.c_[c['oof_main'], c['oof_nop']] ; Xt = np.c_[c['te_main'], c['te_nop']]
            if use_nn: Xo, Xt = np.c_[Xo, nnp['oof_g']], np.c_[Xt, nnp['te_g']]
        elif g == 'floor':
            Xo = tmean(prior_o, c['oof_cal']+shift)[:, None]; Xt = tmean(prior_t, c['te_cal']+shift)[:, None]
        else:
            Xo = np.c_[c['oof_main'], c['oof_cal'], c['oof_nop']]; Xt = np.c_[c['te_main'], c['te_cal'], c['te_nop']]
            if use_nn: Xo, Xt = np.c_[Xo, nnp['oof_g']], np.c_[Xt, nnp['te_g']]
        Xo, Xt = Xo[go[g]], Xt[gt[g]]
        oof[go[g]] = cross_val_predict(LinearRegression(), Xo, y_g[go[g]], cv=5)
        test[gt[g]] = LinearRegression().fit(Xo, y_g[go[g]]).predict(Xt)
        res[g] = round(float(np.sqrt(np.mean((oof[go[g]] - y_g[go[g]])**2))), 4)
    return oof, test, res


rm = lambda p: float(np.sqrt(np.mean((p - y_g)**2)))
g0, gt0, r0 = gamma_stack(False); g1, gt1, r1 = gamma_stack(True)
print('gamma stack without NN', round(rm(g0), 5), r0); print('gamma stack with NN   ', round(rm(g1), 5), r1)
cb = c['p_smoke_oof']
best = (0, roc_auc_score(y_s, cb))
for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    a = roc_auc_score(y_s, (1-w)*R(cb) + w*R(nnp['oof_s'])); print(f'smoking w_nn={w:.2f}: {a:.5f}')
    best = max(best, (w, a), key=lambda z: z[1])
print('best smoking', best)
for name, g in [('without NN', g0), ('with NN', g1)]:
    a = best[1] if name == 'with NN' else roc_auc_score(y_s, cb)
    print(f'SCORE {name}: {a - rm(g)/1.0082:.5f}')
np.savez('outputs/v2/nn_blend.npz', g_test=gt1, w_smoke=best[0])
