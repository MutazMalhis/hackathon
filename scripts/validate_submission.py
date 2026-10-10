"""Check a submission file against the competition format before uploading.

    python scripts/validate_submission.py SUBMISSION.csv [--sample data/sample_submission.csv] [--sha256 HEX]

Exits 0 if every check passes, 1 otherwise. Needs only numpy and pandas.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_ROWS = 6399


def check(path, sample_path, sha256=None):
    problems, notes = [], []
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    notes.append(f'sha256 {digest}')
    if sha256 and digest != sha256.lower():
        problems.append(f'sha256 differs from the expected {sha256}')
    df = pd.read_csv(path)
    if list(df.columns) != ['ID', 'Smoking', 'Gamma_GT']:
        problems.append(f'columns are {list(df.columns)}, expected [ID, Smoking, Gamma_GT]')
        return problems, notes
    if len(df) != EXPECTED_ROWS:
        problems.append(f'{len(df)} rows, expected {EXPECTED_ROWS}')
    if df.ID.duplicated().any():
        problems.append('duplicate IDs')
    for col in ('Smoking', 'Gamma_GT'):
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f'{col} is not numeric')
        elif not np.isfinite(df[col].to_numpy(dtype=float)).all():
            problems.append(f'{col} has missing or non-finite values')
    if not problems:
        if df.Smoking.lt(0).any() or df.Smoking.gt(1).any():
            problems.append(f'Smoking outside [0, 1] (min {df.Smoking.min():.4g}, max {df.Smoking.max():.4g})')
        if df.Gamma_GT.lt(1).any() or df.Gamma_GT.gt(1000).any():
            problems.append(f'Gamma_GT outside [1, 1000] (min {df.Gamma_GT.min():.4g}, max {df.Gamma_GT.max():.4g}); '
                            'the scorer clips it, but a negative value is a sign of a bug')
        if df.Smoking.nunique() < 100:
            problems.append(f'Smoking has only {df.Smoking.nunique()} distinct values (a constant would score AUC 0.5)')
        notes.append(f'Smoking  min {df.Smoking.min():.4f} mean {df.Smoking.mean():.4f} max {df.Smoking.max():.4f}')
        notes.append(f'Gamma_GT min {df.Gamma_GT.min():.2f} median {df.Gamma_GT.median():.2f} max {df.Gamma_GT.max():.2f}')
    if Path(sample_path).exists():
        sample = pd.read_csv(sample_path)
        if not df.ID.equals(sample.ID):
            problems.append('IDs differ from sample_submission.csv (set or order)')
    else:
        notes.append(f'{sample_path} not found: ID order not checked')
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('submission')
    ap.add_argument('--sample', default='data/sample_submission.csv')
    ap.add_argument('--sha256', help='expected SHA-256 of the file')
    a = ap.parse_args()
    problems, notes = check(a.submission, a.sample, a.sha256)
    for n in notes:
        print(n)
    for p in problems:
        print('FAIL:', p)
    print('OK: ready to upload' if not problems else f'{len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
