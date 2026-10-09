"""Fold-safe batch calibration with slope, rounding, and floor-value experiments."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

import experiments as base


@dataclass
class Calibration:
    slope: float
    offsets: pd.Series
    effective_counts: pd.Series
    variance: float


def fit_calibration(reference, config):
    good = reference.POC_GGT.gt(0) & reference.POC_Batch.notna() & reference.Gamma_GT.gt(0)
    if config.get('exclude_floor', False):
        good &= reference.POC_GGT.gt(3)
    ref = reference.loc[good]
    if len(ref) == 0:
        return Calibration(1., pd.Series(dtype=float), pd.Series(dtype=float), .01)
    work = pd.DataFrame({'batch': ref.POC_Batch,
                         'x': np.log(ref.POC_GGT), 'y': np.log(ref.Gamma_GT),
                         'w': np.where(ref.POC_GGT.le(5), config.get('low_weight', 1.), 1.)})
    work['wx'] = work.w*work.x; work['wy'] = work.w*work.y
    work['ww'] = work.w**2
    grouped = work.groupby('batch')
    sums = grouped[['w', 'wx', 'wy', 'ww']].sum()
    mx = work.batch.map(sums.wx/sums.w); my = work.batch.map(sums.wy/sums.w)
    denominator = np.sum(work.w*(work.x-mx)**2)
    slope = float(np.sum(work.w*(work.x-mx)*(work.y-my))/denominator) if config.get('fit_slope', False) and denominator > 1e-9 else 1.
    # A failed or unsupported slope estimate must not create extreme predictions.
    slope = float(np.clip(slope, .5, 1.5))
    offsets = (sums.wy-slope*sums.wx)/sums.w
    residual = work.y-slope*work.x-work.batch.map(offsets)
    degrees = max(1, len(work)-len(sums)-int(config.get('fit_slope', False)))
    variance = max(1e-5, float(np.sum(work.w*residual**2)/work.w.sum()*len(work)/degrees))
    effective_counts = sums.w**2/sums.ww
    return Calibration(slope, offsets, effective_counts, variance)


def calibrated_prediction(calibration, frame, fallback, config):
    fallback = np.clip(np.asarray(fallback, dtype=float), 1, 1000)
    offset = frame.POC_Batch.map(calibration.offsets).to_numpy()
    counts = frame.POC_Batch.map(calibration.effective_counts).to_numpy()
    reading = frame.POC_GGT.to_numpy()
    known = (reading > 0) & np.isfinite(offset)
    floor = known & (reading <= 3) & config.get('exclude_floor', False)
    exact = known & ~floor
    p = fallback.copy()
    raw = np.clip(np.exp(offset[exact]+calibration.slope*np.log(reading[exact])), 1, 1000)
    if config.get('adaptive', False):
        measurement = calibration.variance*(1+1/counts[exact]) + calibration.slope**2/(12*reading[exact]**2)
        prior_variance = config.get('prior_sd', .27)**2
        weight = prior_variance/(prior_variance+measurement)
    else:
        weight = config.get('weight', .9)
    p[exact] = np.expm1(weight*np.log1p(raw)+(1-weight)*np.log1p(fallback[exact]))
    if config.get('floor_mode') == 'censored' and floor.any():
        # Hypothesis: a rounded reading of 3 represents a latent reading <=3.5.
        # Update the clinical log-Gamma prior using that one-sided observation.
        mu = np.log(fallback[floor])
        prior_variance = config.get('prior_sd', .27)**2
        measurement = calibration.variance*(1+1/counts[floor])
        sd = np.sqrt(prior_variance+measurement)
        upper = offset[floor]+calibration.slope*np.log(3.5)
        z = (upper-mu)/sd
        inverse_mills = np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z))
        p[floor] = np.exp(mu-prior_variance/sd*inverse_mills)
    return np.clip(p, 1, 1000), known, floor


def candidate_configs():
    configs = {'v2': dict(fit_slope=False, exclude_floor=False, low_weight=1., weight=.9)}
    configs['slope'] = dict(fit_slope=True, exclude_floor=False, low_weight=1., weight=.9)
    for slope in [False, True]:
        for mode in ['fallback', 'censored']:
            for adaptive in [False, True]:
                name = f"{'slope' if slope else 'unit'}_floor_{mode}_{'adaptive' if adaptive else 'fixed'}"
                configs[name] = dict(fit_slope=slope, exclude_floor=True, low_weight=.3,
                                     weight=.9, floor_mode=mode, adaptive=adaptive, prior_sd=.27)
    return configs


def compare_cached(dev, cache, seed, gamma_name='gamma_clinical4'):
    from sklearn.model_selection import StratifiedKFold
    from improvements import ensemble, rmsle
    configs = candidate_configs()
    predictions = {name: np.zeros(len(dev)) for name in configs}
    smoke = np.zeros(len(dev)); fallback = np.zeros(len(dev)); floor = np.zeros(len(dev), dtype=bool)
    details = []
    for fold, (fit, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(dev, dev.Smoking)):
        ref, frame = dev.iloc[fit], dev.iloc[valid]
        smoking = {}
        for name in ['smoke_baseline', 'smoke_plate4', 'smoke_plate6']:
            saved = pd.read_csv(cache/f'{name}_fold{fold}.csv')
            assert saved.ID.tolist() == frame.ID.tolist()
            smoking[name] = saved.prediction.to_numpy()
        smoke[valid] = ensemble(smoking, list(smoking))
        saved = pd.read_csv(cache/f'{gamma_name}_fold{fold}.csv')
        assert saved.ID.tolist() == frame.ID.tolist()
        fallback[valid] = saved.prediction.to_numpy()
        floor[valid] = frame.POC_GGT.eq(3)
        for name, config in configs.items():
            calibration = fit_calibration(ref, config)
            predictions[name][valid], _, _ = calibrated_prediction(calibration, frame, fallback[valid], config)
            details.append(dict(fold=fold+1, name=name, slope=calibration.slope, variance=calibration.variance,
                                **base.metrics(frame, smoke[valid], predictions[name][valid])))
    rows = [dict(name=name, **config, **base.metrics(dev, smoke, predictions[name]),
                 floor_rmsle=rmsle(dev.Gamma_GT[floor], predictions[name][floor])) for name, config in configs.items()]
    return pd.DataFrame(rows), pd.DataFrame(details), predictions, smoke
