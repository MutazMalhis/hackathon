"""Run the README baseline end to end, then reproduce its OOF score.

    python v4/run_baseline.py            # run every step whose outputs are missing
    python v4/run_baseline.py --force    # rerun everything (delete outputs/plan_v* first)

The CatBoost/NN chain (v2) and the V3 chain are independent, so they run in parallel.
Logs: outputs/logs/<step>.log; timings: outputs/logs/baseline_timings.json.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'outputs' / 'logs'
PY = sys.executable
CHAIN_V2 = [('plate_refine', ['v2/plate_refine.py'], 'v2/refined_cache.csv'),
            ('pipeline5', ['v2/pipeline.py', '5'], 'outputs/v2/components.npz'),
            ('nn3', ['v2/nn.py', '3'], 'outputs/v2/nn_preds.npz')]
CHAIN_V3 = [('experiments', ['experiments.py'], 'outputs/plan_v1/final_oof.csv'),
            ('improvements', ['improvements.py', '--phase', 'all'], 'outputs/plan_v2/final_oof.csv'),
            ('refine', ['refine.py', '--phase', 'all'], 'outputs/plan_v3/submission_v3.csv')]
FINAL = [('make_submission', ['make_submission.py'], None),
         ('baseline_oof', ['v4/baseline_oof.py'], None)]
timings, failed = {}, []


def run(chain, force):
    for name, args, marker in chain:
        if marker and (ROOT / marker).exists() and not force:
            print(f'[skip] {name}: {marker} exists', flush=True); continue
        print(f'[start] {name}', flush=True); t = time.time()
        with open(LOGS / f'{name}.log', 'w') as log:
            code = subprocess.call([PY, '-u'] + args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        timings[name] = round(time.time() - t, 1)
        print(f'[done] {name} exit={code} {timings[name]}s', flush=True)
        if code != 0:
            failed.append(name); return


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    assert (ROOT / 'data/train.csv').exists(), 'put train.csv, test.csv, sample_submission.csv in data/'
    LOGS.mkdir(parents=True, exist_ok=True)
    threads = [threading.Thread(target=run, args=(c, a.force)) for c in (CHAIN_V2, CHAIN_V3)]
    [t.start() for t in threads]; [t.join() for t in threads]
    if not failed:
        run(FINAL, True)
    (LOGS / 'baseline_timings.json').write_text(json.dumps(dict(timings=timings, failed=failed), indent=1))
    print('FAILED: ' + ', '.join(failed) if failed else 'ALL DONE - see outputs/logs/baseline_oof.log', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
