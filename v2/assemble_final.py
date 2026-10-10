"""Assemble the final submission from the v2 pipeline, the NN, plan v3 and Codex's rapid-test posterior; report each step's OOF score."""
import sys, io, contextlib, importlib.util
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'v2')

d = pd.read_csv('data/train.csv'); ss = pd.read_csv('data/sample_submission.csv')
y, g = d.Smoking.values, np.log1p(d.Gamma_GT.values)
c = np.load('outputs/v2/components.npz'); nn = np.load('outputs/v2/nn_preds.npz')
spec = importlib.util.spec_from_file_location('b', 'v2/blend_nn.py')
with contextlib.redirect_stdout(io.StringIO()):
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
g_oof, g_te, _ = b.gamma_stack(True)
R = lambda v: rankdata(v) / len(v)
o3 = pd.read_csv('outputs/plan_v3/final_oof.csv').set_index('ID').loc[d.ID]; t3 = pd.read_csv('outputs/plan_v3/submission_v3.csv')
cx = pd.read_csv('outputs/calibration_precision_push/mean_oof_component.csv').set_index('ID').loc[d.ID]
cxt = pd.read_csv('outputs/calibration_precision_push/test_components.csv')
score = lambda s, gl: roc_auc_score(y, s) - np.sqrt(np.mean((np.clip(gl, np.log1p(1), np.log1p(1000)) - g)**2)) / 1.0082

s_oof = 0.9*R(c['p_smoke_oof']) + 0.1*R(nn['oof_s']); s_te = 0.9*R(c['p_smoke_test']) + 0.1*R(nn['te_s'])
print(f'v2 + NN                 {score(s_oof, g_oof):.5f}')
S_oof = 0.8*R(s_oof) + 0.2*R(o3.Smoking.values); S_te = 0.8*R(s_te) + 0.2*R(t3.Smoking.values)
G_oof = 0.8*g_oof + 0.2*np.log1p(o3.Gamma_GT.values); G_te = 0.8*g_te + 0.2*np.log1p(t3.Gamma_GT.values)
print(f'+ 20% plan v3           {score(S_oof, G_oof):.5f}')
print('codex test component columns:', list(cxt.columns))
for w in [0.15, 0.3]:
    print(f'+ {int(w*100)}% Codex posterior   {score(S_oof, (1-w)*G_oof + w*cx.posterior.values):.5f}')
W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
cxt = cxt.set_index('ID').loc[ss.ID]
G_fin = (1-W)*G_te + W*cxt.posterior.values
out = pd.DataFrame({'ID': ss.ID, 'Smoking': np.clip(S_te, 0, 1), 'Gamma_GT': np.clip(np.expm1(G_fin), 1, 1000)})
assert (out.ID == ss.ID).all() and out.notna().all().all() and len(out) == 6399
out.to_csv('outputs/v2/submission_final.csv', index=False)
prev = pd.read_csv('outputs/calibration_precision_push/submission_assay_precision.csv')
print(f'wrote outputs/v2/submission_final.csv (Codex weight {W}); vs current best file: smoking corr {np.corrcoef(out.Smoking, prev.Smoking)[0,1]:.4f}, '
      f'log-gamma mean abs diff {np.abs(np.log1p(out.Gamma_GT) - np.log1p(prev.Gamma_GT)).mean():.4f}')
