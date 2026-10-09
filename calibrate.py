"""Evaluate batch calibration with exactly the baseline's held-out folds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from train import score


def batch_predictions(reference, frame):
    valid = reference.POC_GGT.gt(0) & reference.Gamma_GT.gt(0) & reference.POC_Batch.notna()
    ref = reference.loc[valid]
    offsets = (np.log(ref.Gamma_GT) - np.log(ref.POC_GGT)).groupby(ref.POC_Batch).mean()
    correction = frame.POC_Batch.map(offsets)
    usable = frame.POC_GGT.gt(0) & correction.notna()
    pred = np.full(len(frame), np.nan)
    pred[usable] = np.clip(np.exp(np.log(frame.loc[usable, 'POC_GGT']) + correction[usable]), 1, 1000)
    return pred, offsets


def main():
    base = Path('outputs/baseline')
    output = Path('outputs/calibrated')
    output.mkdir(exist_ok=True, parents=True)
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    oof = pd.read_csv(base / 'oof.csv')
    baseline = pd.read_csv(base / 'submission.csv')
    metadata = json.loads((base / 'metrics.json').read_text())
    assert oof.ID.tolist() == train.ID.tolist()
    assert baseline.ID.tolist() == test.ID.tolist()
    corrected = oof.Gamma_GT.to_numpy().copy()
    coverage = np.zeros(len(train), dtype=bool)
    cv = StratifiedKFold(len(metadata['folds']), shuffle=True, random_state=metadata['seed'])
    for fit, hold in cv.split(train, train.Smoking):
        pred, _ = batch_predictions(train.iloc[fit], train.iloc[hold])
        ok = np.isfinite(pred)
        corrected[hold[ok]] = pred[ok]
        coverage[hold[ok]] = True
    calibrated, offsets = batch_predictions(train, test)
    ok = np.isfinite(calibrated)
    baseline.loc[ok, 'Gamma_GT'] = calibrated[ok]
    baseline.to_csv(output / 'submission.csv', index=False)
    oof.Gamma_GT = corrected
    oof.to_csv(output / 'oof.csv', index=False)
    offsets.rename('log_multiplier').to_csv(output / 'batch_offsets.csv')
    result = {'baseline': metadata['overall'],
              'calibrated': score(train.Smoking, train.Gamma_GT, oof.Smoking, corrected),
              'oof_calibration_coverage': float(coverage.mean()),
              'test_calibration_coverage': float(ok.mean()),
              'note': 'Exploratory model selection on the same CV; confirm on another split before extensive tuning.'}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
