"""Feature-engineering test: smoking AUC and feature-only gamma RMSLE, 5-fold, fixed settings."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'v2')
from common import load, base_X, plate_centered, batch_features_v2, CATS

d, t, _ = load()
ref = pd.read_csv('v2/refined_cache.csv').drop(columns=['resid_cot', 'resid_etg'])
d = d.join(ref.iloc[:len(d)].reset_index(drop=True)); t = t.join(ref.iloc[len(d):].reset_index(drop=True))
V = sys.argv[1]


def fe(f):
    x = pd.DataFrame(index=f.index)
    if 'ratio' in V:
        x['ast_alt'] = f.AST / f.ALT.replace(0, np.nan)
        x['tg_hdl'] = f.Triglyceride / f.HDL
        x['ldl_hdl'] = f.LDL / f.HDL
        x['chol_hdl'] = f.Cholesterol_Total / f.HDL
        x['waist_ht'] = f.Waist_cm / f.Height_cm
        x['pulse_p'] = f.BP_Systolic - f.BP_Diastolic
        x['hb_ht'] = f.Hemoglobin * f.Height_cm / 100
        x['alt_tg'] = np.log1p(f.ALT) + np.log(f.Triglyceride)
    if 'grid' in V:
        for c, gsz in [('Height_cm', 5), ('Weight_kg', 5), ('Age', 5)]:
            x[c + '_g'] = (f[c] / gsz).round() * gsz
        x['bmi_g'] = x['Weight_kg_g'] / (x['Height_cm_g'] / 100) ** 2
    if 'alc' in V:
        x['etg_x_units'] = f.rpc_etg * np.log1p(f.Alcohol_Units_Week)
        x['co_x_cot'] = np.log1p(f.Exhaled_CO_ppm) + f.rpc_cot
    return x


y = d.Smoking.values; yg = np.log1p(d.Gamma_GT.values)
oof, oofg = np.zeros(len(d)), np.zeros(len(d))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(d, y)):
    a, b = d.iloc[tr], d.iloc[va]
    pa, pb = plate_centered(pd.concat([a, b.drop(columns=['Smoking', 'Gamma_GT']), t]), [a, b])
    (ga, gb), _ = batch_features_v2(a, [a, b])
    Xa = base_X(a).join(pa).join(fe(a)).assign(cal2=ga.cal2_log1p); Xb = base_X(b).join(pb).join(fe(b)).assign(cal2=gb.cal2_log1p)
    m = CatBoostClassifier(iterations=1600, depth=6, learning_rate=0.025, cat_features=CATS, verbose=False,
                           allow_writing_files=False, thread_count=3, random_seed=k)
    m.fit(Xa, y[tr]); oof[va] = m.predict_proba(Xb)[:, 1]
    Na, Nb = Xa.drop(columns=['POC_GGT', 'POC_Batch', 'cal2']), Xb.drop(columns=['POC_GGT', 'POC_Batch', 'cal2'])
    r = CatBoostRegressor(iterations=1800, depth=4, learning_rate=0.03, l2_leaf_reg=5, verbose=False, allow_writing_files=False,
                          thread_count=3, random_seed=k, cat_features=[c for c in CATS if c != 'POC_Batch'])
    r.fit(Na, yg[tr]); oofg[va] = r.predict(Nb)
fb = d.POC_GGT.isna().values
print(f'{V:14s} smoking AUC {roc_auc_score(y, oof):.5f}   feature-only gamma RMSLE all {np.sqrt(np.mean((oofg-yg)**2)):.4f}  no-reading {np.sqrt(np.mean((oofg[fb]-yg[fb])**2)):.4f}', flush=True)
