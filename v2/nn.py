"""Multi-task neural network: smoking (BCE) + feature-only log1p gamma-GT (MSE), with categorical embeddings.

Same 5-fold splits as the v2 pipeline seed 42 (StratifiedKFold for both heads), so OOF predictions can be
blended with the CatBoost components. Writes OOF + test predictions to outputs/v2/nn_preds.npz.
Usage: python v2/nn.py [n_seeds]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, 'v2')
from common import load, plate_centered

torch.set_num_threads(8)
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
d, t, _ = load()
CACHE = Path('v2/refined_cache.csv')
if not CACHE.exists():  # covariate-adjusted plate offsets (features of train+test only, no targets)
    from plate_refine import refined_offsets
    refined_offsets(pd.concat([d, t], ignore_index=True)).to_csv(CACHE, index=False)
ref = pd.read_csv(CACHE).drop(columns=['resid_cot', 'resid_etg'])
d = d.join(ref.iloc[:len(d)].reset_index(drop=True)); t = t.join(ref.iloc[len(d):].reset_index(drop=True))
y_s = d.Smoking.values.astype(np.float32)
y_g = np.log1p(d.Gamma_GT.values).astype(np.float32)
EMB = {'Workplace_ID': 8, 'Urine_Plate': 8, 'Screening_Center': 3, 'Visit_Quarter': 2}
EXCLUDE = ['ID', 'Smoking', 'Gamma_GT', 'POC_GGT', 'POC_Batch'] + list(EMB)
LOG = ['Urine_Cotinine_ng_mL', 'Urine_EtG_ng_mL', 'Urine_Creatinine', 'Exhaled_CO_ppm', 'Triglyceride', 'ALT', 'AST',
       'Alcohol_Units_Week', 'Fasting_Glucose']


def numeric(f, p):
    x = f.drop(columns=[c for c in EXCLUDE if c in f]).join(p)
    for c in LOG:
        x[c] = np.log1p(x[c].clip(lower=0))
    return x


def prepare(a, others, p_list):
    xa = numeric(a, p_list[0])
    miss_cols = [c for c in xa if xa[c].isna().mean() > 0.01]
    mu, sd = xa.mean(), xa.std().replace(0, 1)
    out = []
    for f, p in zip([a] + others, p_list):
        x = numeric(f, p)
        m = x[miss_cols].isna().astype(np.float32).values
        z = ((x - mu) / sd).clip(-5, 5).fillna(0).values.astype(np.float32)
        out.append(np.concatenate([z, m], 1))
    vocab = {c: {v: i + 1 for i, v in enumerate(pd.unique(pd.concat([a[c]] + [o[c] for o in others]).dropna()))} for c in EMB}
    cats = [np.stack([f[c].map(vocab[c]).fillna(0).astype(int).values for c in EMB], 1) for f in [a] + others]
    return out, cats, {c: len(vocab[c]) + 1 for c in EMB}


class Net(nn.Module):
    def __init__(self, n_num, sizes):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(sizes[c], EMB[c]) for c in EMB])
        width = n_num + sum(EMB.values())
        self.body = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Dropout(0.25),
                                  nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.25))
        self.head_s, self.head_g = nn.Linear(128, 1), nn.Linear(128, 1)
        self.emb_drop = nn.Dropout(0.2)

    def forward(self, x, c):
        e = torch.cat([emb(c[:, i]) for i, emb in enumerate(self.embs)], 1)
        h = self.body(torch.cat([x, self.emb_drop(e)], 1))
        return self.head_s(h).squeeze(1), self.head_g(h).squeeze(1)


def fit_predict(Xa, Ca, ys, yg, X_eval, C_eval, sizes, seed, epochs=80):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    val = rng.rand(len(Xa)) < 0.12  # inner early-stopping split of the fitting rows only
    tr = ~val
    net = Net(Xa.shape[1], sizes)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    xt, ct, yst, ygt = T(Xa[tr]), T(Ca[tr], torch.long), T(ys[tr]), T(yg[tr])
    xv, cv, ysv, ygv = T(Xa[val]), T(Ca[val], torch.long), ys[val], yg[val]
    gmu = float(ygt.mean())
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(perm), 256):
            j = perm[i:i+256]
            ps, pg = net(xt[j], ct[j])
            loss = nn.functional.binary_cross_entropy_with_logits(ps, yst[j]) + 2.0*((pg + gmu - ygt[j])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            ps, pg = net(xv, cv)
        crit = roc_auc_score(ysv, ps.numpy()) - np.sqrt(np.mean((pg.numpy() + gmu - ygv)**2)) / 1.0082
        if crit > best:
            best, bad = crit, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    net.load_state_dict(best_state); net.eval()
    outs = []
    with torch.no_grad():
        for X, C in zip(X_eval, C_eval):
            ps, pg = net(T(X), T(C, torch.long))
            outs.append((torch.sigmoid(ps).numpy(), pg.numpy() + gmu))
    return outs


oof_s, oof_g = np.zeros(len(d)), np.zeros(len(d))
te_s, te_g = np.zeros(len(t)), np.zeros(len(t))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y_s)):
    a, b = d.iloc[tr], d.iloc[va]
    pa, pb, pt = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b, t])
    (Xa, Xb, Xt), (Ca, Cb, Ct), sizes = prepare(a, [b, t], [pa, pb, pt])
    for s in range(N_SEEDS):
        (sb, gb), (st, gt) = fit_predict(Xa, Ca, y_s[tr], y_g[tr], [Xb, Xt], [Cb, Ct], sizes, seed=100*k+s)
        oof_s[va] += sb / N_SEEDS; oof_g[va] += gb / N_SEEDS
        te_s += st / (5*N_SEEDS); te_g += gt / (5*N_SEEDS)
    print('fold', k, 'done', flush=True)

fb = d.POC_GGT.isna().values
print(f'NN smoking AUC {roc_auc_score(y_s, oof_s):.5f}  feature-only gamma RMSLE all {np.sqrt(np.mean((oof_g-y_g)**2)):.4f}'
      f'  no-reading rows {np.sqrt(np.mean((oof_g[fb]-y_g[fb])**2)):.4f}')
np.savez('outputs/v2/nn_preds.npz', oof_s=oof_s, oof_g=oof_g, te_s=te_s, te_g=te_g)
