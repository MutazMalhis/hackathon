"""Client side of the runner channel (see v4/worker.py for the protocol).

    python v4/channel.py submit NAME [--collect GLOB ...] [--timeout MIN] -- script.py [args...]
    python v4/channel.py status            # worker heartbeat + every job's state
    python v4/channel.py wait JOB          # block until JOB finishes (polls every 60 s)
    python v4/channel.py pull JOB          # copy JOB's collected files into this checkout
    python v4/channel.py cancel JOB

Uses the git remote named by $CHANNEL_REMOTE (default "runner").
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = os.environ.get('CHANNEL_REMOTE', 'runner')
RBRANCH = 'runner-results'


def git(*args, binary=False, check=True):
    r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=not binary)
    if check and r.returncode != 0:
        raise SystemExit(f'git {" ".join(args)} failed: {r.stderr}')
    return r.stdout if binary else r.stdout.strip()


def ref():
    return f'{REMOTE}/{RBRANCH}'


def fetch():
    git('fetch', '--quiet', REMOTE, RBRANCH, check=False)


def show(path):
    r = subprocess.run(['git', 'show', f'{ref()}:{path}'], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def job_status(job):
    s = show(f'results/{job}/status.json')
    return json.loads(s) if s else None


def push_jobs(paths, message):
    branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    git('add', *paths); git('commit', '-m', message)
    git('push', REMOTE, branch)


def submit(a):
    jobs = ROOT / 'jobs'; jobs.mkdir(exist_ok=True)
    n = max([int(p.stem.split('_')[0]) for p in jobs.glob('*.json')] + [0]) + 1
    job = f'{n:04d}_{a.name}'
    spec = dict(args=a.args, collect=a.collect or [], timeout_min=a.timeout)
    (jobs / f'{job}.json').write_text(json.dumps(spec, indent=1) + '\n')
    push_jobs([f'jobs/{job}.json'], f'Queue runner job {job}')
    print(job)


def status(_):
    fetch()
    hb = show('heartbeat.json')
    print('HEARTBEAT', hb or '(none yet)')
    names = git('ls-tree', '--name-only', f'{ref()}:results', check=False).split()
    for j in sorted(names):
        s = job_status(j) or {}
        print(f"{j:40s} {s.get('state', '?'):9s} exit={s.get('exit_code')} {s.get('seconds', '')}s")
    queued = sorted(p.stem for p in (ROOT / 'jobs').glob('*.json') if p.stem not in names)
    print('QUEUED (not started):', queued)


def wait(a):
    while True:
        fetch(); s = job_status(a.job)
        if s and s.get('state') not in ('running', None):
            print(json.dumps({k: v for k, v in s.items() if k != 'collected'}, indent=1)); return
        hb = show('heartbeat.json')
        print(time.strftime('%H:%M:%S'), (s or {}).get('state', 'queued'), (json.loads(hb).get('time') if hb else ''), flush=True)
        time.sleep(a.every)


def pull(a):
    fetch()
    prefix = f'results/{a.job}/files/'
    files = [f for f in git('ls-tree', '-r', '--name-only', ref(), prefix).splitlines() if f]
    for f in files:
        dest = ROOT / f[len(prefix):]; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(git('show', f'{ref()}:{f}', binary=True))
    log = show(f'results/{a.job}/log.txt')
    if log is not None:
        (ROOT / 'outputs' / 'logs').mkdir(parents=True, exist_ok=True)
        (ROOT / 'outputs' / 'logs' / f'job_{a.job}.log').write_text(log)
    print(f'pulled {len(files)} files for {a.job}')


def cancel(a):
    p = ROOT / 'jobs' / 'cancel' / a.job; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('')
    push_jobs([str(p.relative_to(ROOT))], f'Cancel runner job {a.job}')


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('submit'); s.add_argument('name'); s.add_argument('--collect', action='append')
    s.add_argument('--timeout', type=float, default=480); s.add_argument('args', nargs='+')
    sp.add_parser('status')
    w = sp.add_parser('wait'); w.add_argument('job'); w.add_argument('--every', type=int, default=60)
    p = sp.add_parser('pull'); p.add_argument('job')
    c = sp.add_parser('cancel'); c.add_argument('job')
    a = ap.parse_args()
    dict(submit=submit, status=status, wait=wait, pull=pull, cancel=cancel)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
