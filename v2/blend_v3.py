"""Write outputs/v2/submission_nn_blend_v3.csv: (v2 pipeline + NN) blended 80/20 with plan v3.

Smoking: rank average (AUC only uses ranks). Gamma-GT: weighted average in log1p space.
Needs outputs/v2/components.npz, outputs/v2/nn_preds.npz (v2/nn.py) and outputs/plan_v3/submission_v3.csv.
"""
import io, contextlib, importlib.util
import numpy as np
import pandas as pd
from scipy.stats import rankdata

W_V3 = 0.2
spec = importlib.util.spec_from_file_location('b', 'v2/blend_nn.py')
with contextlib.redirect_stdout(io.StringIO()):
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)  # also writes outputs/v2/nn_blend.npz
_, g_te, _ = b.gamma_stack(True)
c, nn = np.load('outputs/v2/components.npz'), np.load('outputs/v2/nn_preds.npz')
ss, t3 = pd.read_csv('data/sample_submission.csv'), pd.read_csv('outputs/plan_v3/submission_v3.csv')
assert (t3.ID.values == ss.ID.values).all()
R = lambda v: rankdata(v) / len(v)
s_te = 0.9*R(c['p_smoke_test']) + 0.1*R(nn['te_s'])
out = pd.DataFrame({'ID': ss.ID,
                    'Smoking': np.clip((1-W_V3)*R(s_te) + W_V3*R(t3.Smoking.values), 0, 1),
                    'Gamma_GT': np.clip(np.expm1((1-W_V3)*g_te + W_V3*np.log1p(t3.Gamma_GT.values)), 1, 1000)})
assert out.notna().all().all() and len(out) == 6399
out.to_csv('outputs/v2/submission_nn_blend_v3.csv', index=False)
print('wrote outputs/v2/submission_nn_blend_v3.csv')
