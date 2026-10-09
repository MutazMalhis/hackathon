"""Shared feature construction for the v2 pipeline (fold-safe target-derived features)."""
import numpy as np
import pandas as pd

CATS = ['Screening_Center', 'Visit_Quarter', 'Workplace_ID', 'Urine_Plate', 'POC_Batch']
SCALE = 0.5041


def fix_units(f):
    # Centre E reports urine creatinine in umol/L; every other centre uses mg/dL (factor 88.4).
    f = f.copy()
    e = f.Screening_Center == 'E'
    f.loc[e, 'Urine_Creatinine'] = f.loc[e, 'Urine_Creatinine'] / 88.4
    return f


def load():
    return (fix_units(pd.read_csv('data/train.csv')), fix_units(pd.read_csv('data/test.csv')),
            pd.read_csv('data/sample_submission.csv'))


def score(y_s, p_s, y_g, p_g):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_s, p_s)
    rmsle = float(np.sqrt(np.mean((np.log1p(y_g) - np.log1p(np.clip(p_g, 1, 1000)))**2)))
    return auc, rmsle, 0.5*(2*auc-1) + 0.5*(1-rmsle/SCALE)


def plate_centered(fit, frames):
    """Urine marker features centred by fitting-partition plate means (unsupervised)."""
    base = lambda f: {
        'ratio_cot': np.log(f.Urine_Cotinine_ng_mL/f.Urine_Creatinine),
        'ratio_etg': np.log(f.Urine_EtG_ng_mL/f.Urine_Creatinine),
        'raw_cot': np.log1p(f.Urine_Cotinine_ng_mL), 'raw_etg': np.log1p(f.Urine_EtG_ng_mL),
        'creat': np.log(f.Urine_Creatinine)}
    ref = base(fit)
    out = []
    for f in frames:
        b = base(f); add = {}
        for k, s in b.items():
            pm = ref[k].groupby(fit.Urine_Plate).mean()
            add['pc_'+k] = s - f.Urine_Plate.map(pm).fillna(ref[k].mean())
        out.append(pd.DataFrame(add, index=f.index))
    return out


def batch_features(fit, frames, loo_first=True):
    """Calibrated rapid-test estimate using fitting-partition batch offsets.

    The first frame is assumed to be the fitting partition itself: its rows get
    leave-one-out offsets so no row sees its own target."""
    ok = fit.POC_GGT > 0
    off = (np.log(fit.Gamma_GT) - np.log(fit.POC_GGT))[ok]
    g = off.groupby(fit.POC_Batch[ok])
    s, n = g.sum(), g.size()
    out = []
    for i, f in enumerate(frames):
        bs = f.POC_Batch.map(s); bn = f.POC_Batch.map(n)
        if i == 0 and loo_first:
            own = (np.log(f.Gamma_GT) - np.log(f.POC_GGT)).where(f.POC_GGT > 0)
            has = own.notna()
            bs = bs - own.fillna(0); bn = bn - has.astype(int)
        mean = bs / bn.where(bn > 0)
        cal = np.log(f.POC_GGT.where(f.POC_GGT > 0)) + mean
        out.append(pd.DataFrame({'batch_off': mean, 'batch_n': bn.where(bn > 0, 0).fillna(0),
                                 'cal_log': cal, 'cal_log1p': np.log1p(np.exp(cal))}, index=f.index))
    return out


def base_X(f):
    x = f.drop(columns=[c for c in ['ID', 'Smoking', 'Gamma_GT'] if c in f]).copy()
    for c in CATS:
        x[c] = x[c].fillna('__MISSING__').astype(str)
    return x


def batch_features_v2(fit, frames, low_weight=0.0, low_cut=3):
    """Slope-adjusted batch calibration. POC_GGT == 3 is the assay detection floor (censored),
    so those readings get weight low_weight when estimating batch offsets.

    First frame = fitting partition, which receives leave-one-out batch offsets."""
    ok = (fit.POC_GGT > 0).values
    lg, lp, b = np.log(fit.Gamma_GT.values[ok]), np.log(fit.POC_GGT.values[ok]), fit.POC_Batch.values[ok]
    w = np.where(fit.POC_GGT.values[ok] > low_cut, 1.0, low_weight)
    tmp = pd.DataFrame({'lg': lg, 'lp': lp, 'b': b, 'w': w})
    dm = tmp.groupby('b')[['lg', 'lp']].transform('mean')
    x, yv = tmp.lp - dm.lp, tmp.lg - dm.lg
    beta = float((w*x*yv).sum() / (w*x*x).sum())
    tmp['r'] = tmp.lg - beta*tmp.lp
    sw = (tmp.w*tmp.r).groupby(tmp.b).sum(); ww = tmp.w.groupby(tmp.b).sum()
    out = []
    for i, f in enumerate(frames):
        num, den = f.POC_Batch.map(sw), f.POC_Batch.map(ww)
        lpf = np.log(f.POC_GGT.where(f.POC_GGT > 0))
        if i == 0:
            own_w = np.where(f.POC_GGT > low_cut, 1.0, low_weight)
            own_r = np.log(f.Gamma_GT) - beta*lpf
            has = own_r.notna()
            num = num - (own_w*own_r).where(has, 0); den = den - pd.Series(own_w, index=f.index).where(has, 0)
        off = num / den.where(den > 1e-9)
        cal = beta*lpf + off
        out.append(pd.DataFrame({'cal2_log1p': np.log1p(np.exp(cal)), 'batch_w': den.fillna(0),
                                 'poc_floor': (f.POC_GGT <= low_cut).astype(float)}, index=f.index))
    return out, beta
