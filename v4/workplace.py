"""Workplace smoking-rate features (fold-safe), used by the v4 smoking models.

Findings (reports/experiments_v4.md): workplace smoking rates vary far beyond binomial noise
(latent SD ~0.33; ~0.32 among likely-male rows), and the rate is the strongest single
signal after the urine markers. These helpers estimate it without touching held-out labels:

* latent_sex(all_rows)      unsupervised 2-component GMM on body measurements (train+test features)
* rate_encoding(fit, frames) empirical-Bayes rates per workplace and per workplace x sex,
                             inner-fold for the fitting rows themselves
* transductive_rates(...)    adds held-out/test rows' predicted probabilities as soft counts
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SEX_COLS = ['Height_cm', 'Hemoglobin', 'Weight_kg', 'Waist_cm', 'Serum_Creatinine']


def latent_sex(rows, seed=0):
    """P(male-like cluster) per row, from features only."""
    z = StandardScaler().fit_transform(rows[SEX_COLS].fillna(rows[SEX_COLS].median()))
    gm = GaussianMixture(2, random_state=seed, n_init=5).fit(z)
    return pd.Series(gm.predict_proba(z)[:, np.argmax(gm.means_[:, 0])], index=rows.index)


def _eb(keys_fit, y_fit, keys_eval, alpha, prior, w_fit=None):
    w = np.ones(len(y_fit)) if w_fit is None else w_fit
    df = pd.DataFrame({'k': keys_fit, 'y': y_fit * w, 'w': w})
    s = df.groupby('k')[['y', 'w']].sum()
    rate = (s.y + alpha * prior) / (s.w + alpha)
    k = pd.Series(keys_eval)
    return k.map(rate).fillna(prior).to_numpy(), k.map(s.w).fillna(0).to_numpy()


def rate_features(fit, frames, sex_fit, sex_frames, alpha=4.0, inner=5, seed=0):
    """Workplace and workplace x sex smoking rates. Fitting rows get inner-fold values."""
    y = fit.Smoking.to_numpy().astype(float)
    sx_fit = (np.asarray(sex_fit) > 0.5).astype(int).astype(str)
    key_w = fit.Workplace_ID.astype(str).to_numpy()
    key_ws = np.char.add(np.char.add(key_w, '_'), sx_fit)
    prior = y.mean()
    prior_sex = {s: y[sx_fit == s].mean() for s in ['0', '1']}

    def encode(kw, kws, sx, ytr, kw_tr, kws_tr, sx_tr):
        rw, nw = _eb(kw_tr, ytr, kw, alpha, prior)
        out = {'wp_rate': rw, 'wp_n': nw}
        # workplace x sex rate shrinks toward the sex-specific prior
        rs = np.zeros(len(kw)); ns = np.zeros(len(kw))
        for s in ['0', '1']:
            m, mt = sx == s, sx_tr == s
            rs[m], ns[m] = _eb(kws_tr[mt], ytr[mt], kws[m], alpha, ytr[mt].mean() if mt.any() else prior_sex[s])
        out.update(wps_rate=rs, wps_n=ns)
        return pd.DataFrame(out)

    own = pd.DataFrame(index=range(len(fit)), columns=['wp_rate', 'wp_n', 'wps_rate', 'wps_n'], dtype=float)
    for tr, va in KFold(inner, shuffle=True, random_state=seed).split(fit):
        own.iloc[va] = encode(key_w[va], key_ws[va], sx_fit[va], y[tr], key_w[tr], key_ws[tr], sx_fit[tr]).to_numpy()
    out = [own.set_index(fit.index)]
    for f, sxf in zip(frames, sex_frames):
        sx = (np.asarray(sxf) > 0.5).astype(int).astype(str)
        kw = f.Workplace_ID.astype(str).to_numpy()
        out.append(encode(kw, np.char.add(np.char.add(kw, '_'), sx), sx, y, key_w, key_ws, sx_fit).set_index(f.index))
    return out
