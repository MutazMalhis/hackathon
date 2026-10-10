"""Remote job runner: executes queued jobs from the work branch and pushes results to GitHub.

Run this on the training machine, in a DEDICATED clone (it hard-resets tracked files):

    git clone -b claude/dsc-hackathon-objective-d50o9l https://github.com/MutazMalhis/hackathon hackathon-runner
    cd hackathon-runner   # put train.csv, test.csv, sample_submission.csv in data/
    python v4/worker.py

Protocol (both sides talk only through the private GitHub repo):
  * Jobs are JSON files jobs/<NNNN>_<name>.json on the work branch, e.g.
        {"args": ["v4/run_baseline.py"], "collect": ["outputs/v4/**"], "timeout_min": 240}
    args[0] must be a .py file inside the repo; it runs as `python -u <args>` (no shell).
  * Results go to the orphan branch `runner-results`:
        results/<job>/status.json   state, exit code, timings, host, code commit
        results/<job>/log.txt       full stdout/stderr
        results/<job>/files/...     collected outputs at their repo-relative paths
        heartbeat.json              worker state + log tail (every few minutes while running)
  * jobs/cancel/<job> (empty file) on the work branch stops a running job.
Raw competition data (data/), CatBoost model files and files over 40 MB are never collected.
"""
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / '.runner-results'
BRANCH = os.environ.get('RUNNER_BRANCH', 'claude/dsc-hackathon-objective-d50o9l')
RBRANCH = 'runner-results'
POLL = int(os.environ.get('RUNNER_POLL', 60))
BEAT_RUNNING, BEAT_IDLE = int(os.environ.get('RUNNER_BEAT', 240)), 1800
MAX_BYTES = 40 * 1024 ** 2
DEFAULT_COLLECT = ['outputs/logs/**', 'outputs/v4/**']


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def git(*args, cwd=ROOT, check=True):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} failed: {r.stderr.strip()}')
    return r.stdout.strip()


def remote_has(branch):
    return bool(git('ls-remote', '--heads', 'origin', branch))


def ensure_results():
    if RES.exists():
        if remote_has(RBRANCH):
            git('pull', '--rebase', 'origin', RBRANCH, cwd=RES, check=False)
        return
    git('fetch', 'origin')
    if remote_has(RBRANCH):
        git('worktree', 'add', '-f', '-B', RBRANCH, str(RES), f'origin/{RBRANCH}')
    else:
        git('worktree', 'add', '-f', '--detach', str(RES))
        git('checkout', '--orphan', RBRANCH, cwd=RES)
        git('rm', '-rf', '--quiet', '.', cwd=RES, check=False)
        for p in RES.iterdir():
            if p.name != '.git':
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        (RES / 'README.md').write_text('Results pushed by v4/worker.py. Do not edit by hand.\n')
        git('add', '-A', cwd=RES); git('commit', '-m', 'Initialise runner results', cwd=RES)
        git('push', '-u', 'origin', RBRANCH, cwd=RES)


def push_results(message):
    git('add', '-A', cwd=RES)
    if not git('status', '--porcelain', cwd=RES):
        return
    git('commit', '-m', message, cwd=RES)
    for _ in range(5):
        if subprocess.run(['git', 'push', 'origin', RBRANCH], cwd=RES, capture_output=True).returncode == 0:
            return
        git('pull', '--rebase', 'origin', RBRANCH, cwd=RES, check=False); time.sleep(5)
    print('WARNING: push failed; results kept locally in .runner-results', flush=True)


def sync_code():
    """Fast-forward the dedicated clone to the work branch. Returns True if worker.py changed."""
    me = Path(__file__).read_bytes()
    git('fetch', 'origin', BRANCH)
    git('reset', '--hard', f'origin/{BRANCH}')
    return Path(__file__).read_bytes() != me


def done_jobs():
    d = RES / 'results'
    return {p.parent.name for p in d.glob('*/status.json')
            if json.loads(p.read_text()).get('state') in ('done', 'failed', 'cancelled', 'rejected')} if d.exists() else set()


def pending_jobs():
    finished = done_jobs()
    return [p for p in sorted((ROOT / 'jobs').glob('*.json')) if p.stem not in finished]


def tail(path, n=40):
    try:
        return ''.join(Path(path).read_text(errors='replace').splitlines(True)[-n:])
    except FileNotFoundError:
        return ''


def heartbeat(state, job=None, log=None):
    (RES / 'heartbeat.json').write_text(json.dumps(dict(time=now(), host=platform.node(), state=state, job=job,
                                                        log_tail=tail(log) if log else ''), indent=1))
    push_results(f'heartbeat {state} {job or ""}'.strip())


def machine_info():
    info = dict(host=platform.node(), platform=platform.platform(), python=sys.version.split()[0], cpus=os.cpu_count())
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as e:  # torch missing is not fatal for CatBoost jobs
        info['torch_error'] = repr(e)
    for pkg in ['catboost', 'lightgbm', 'xgboost', 'sklearn', 'numpy', 'pandas']:
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    return info


def collect(patterns, dest):
    kept = []
    for pat in patterns:
        for f in glob.glob(str(ROOT / pat), recursive=True):
            p = Path(f)
            rel = p.relative_to(ROOT)
            if not p.is_file() or rel.parts[0] in ('data', '.git', '.runner-results') or p.suffix in ('.cbm', '.zip') \
                    or p.stat().st_size > MAX_BYTES:
                continue
            out = dest / rel; out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out)
            kept.append(str(rel).replace(os.sep, '/'))
    return sorted(set(kept))


def run_job(spec_path):
    job = spec_path.stem
    out = RES / 'results' / job
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    spec = json.loads(spec_path.read_text())
    args = [str(a) for a in spec.get('args', [])]
    script = (ROOT / args[0]).resolve() if args else None
    status = dict(job=job, spec=spec, host=platform.node(), code_commit=git('rev-parse', 'HEAD'), started=now())
    if not script or script.suffix != '.py' or ROOT not in script.parents or not script.exists():
        status.update(state='rejected', reason='args[0] must be an existing .py file inside the repo', finished=now())
        (out / 'status.json').write_text(json.dumps(status, indent=1)); push_results(f'{job}: rejected'); return
    log = ROOT / 'outputs' / 'logs' / f'job_{job}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    status['state'] = 'running'; (out / 'status.json').write_text(json.dumps(status, indent=1))
    heartbeat('running', job, log)
    t0 = time.time(); last_beat = t0; state = None
    env = dict(os.environ, PYTHONUNBUFFERED='1', RUNNER_JOB=job)
    with open(log, 'w') as fh:
        proc = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            time.sleep(15)
            if time.time() - t0 > 60 * float(spec.get('timeout_min', 480)):
                proc.kill(); state = 'failed'; status['reason'] = 'timeout'
            if time.time() - last_beat > BEAT_RUNNING:
                last_beat = time.time()
                git('fetch', 'origin', BRANCH, check=False)
                if git('ls-tree', '--name-only', f'origin/{BRANCH}', f'jobs/cancel/{job}', check=False):
                    proc.kill(); state = 'cancelled'
                (out / 'log.txt').write_text(Path(log).read_text(errors='replace'))
                heartbeat('running', job, log)
    code = proc.returncode
    status.update(state=state or ('done' if code == 0 else 'failed'), exit_code=code, finished=now(),
                  seconds=round(time.time() - t0, 1))
    shutil.copy2(log, out / 'log.txt')
    status['collected'] = collect(DEFAULT_COLLECT + list(spec.get('collect', [])), out / 'files')
    (out / 'status.json').write_text(json.dumps(status, indent=1))
    push_results(f'{job}: {status["state"]} in {status["seconds"]}s')
    print(f'[{now()}] {job}: {status["state"]} ({status["seconds"]}s)', flush=True)


def main():
    assert (ROOT / 'data/train.csv').exists(), 'put the competition CSVs in data/ first'
    ensure_results()
    (RES / 'results').mkdir(exist_ok=True)
    (RES / 'worker_info.json').write_text(json.dumps(dict(started=now(), **machine_info()), indent=1))
    heartbeat('idle')
    print(f'worker on {platform.node()} polling {BRANCH} every {POLL}s', flush=True)
    last_idle = time.time()
    while True:
        try:
            if sync_code():
                print('worker.py updated; restarting', flush=True)
                os.execv(sys.executable, [sys.executable] + sys.argv)
            ensure_results()
            jobs = pending_jobs()
            if jobs:
                run_job(jobs[0]); last_idle = time.time(); heartbeat('idle'); continue
            if time.time() - last_idle > BEAT_IDLE:
                last_idle = time.time(); heartbeat('idle')
        except Exception as e:  # keep the worker alive through network hiccups
            print(f'[{now()}] error: {e!r}', flush=True)
        time.sleep(POLL)


if __name__ == '__main__':
    main()
