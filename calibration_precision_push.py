"""Fold-fitted assay calibration audit and bounded precision experiments.

The raw calibration comparisons are label-isolated. Comparisons using saved
V2 clinical predictions are explicitly conditional, not fully nested validation.
No submissions are written by this experiment.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import truncnorm
from sklearn.model_selection import KFold
from v2.common import load, batch_features_v2

OUT = Path('outputs/calibration_precision_push')


def weighted_calibration(fit, frame):
    """Correct weighted within-batch centering; excluded rows have zero influence."""
    a = fit.loc[fit.POC_GGT > 3, ['POC_GGT', 'Gamma_GT', 'POC_Batch']].copy()
    a['x'], a['y'] = np.log(a.POC_GGT), np.log(a.Gamma_GT)
    mean = a.groupby('POC_Batch')[['x', 'y']].transform('mean')
    x, y = a.x - mean.x, a.y - mean.y
    beta = float((x*y).sum()/(x*x).sum())
    off = (a.y-beta*a.x).groupby(a.POC_Batch).mean()
    pred = beta*np.log(frame.POC_GGT)+frame.POC_Batch.map(off)
    return np.logaddexp(0, pred.to_numpy()), beta


def fit_interval_calibration(fit):
    """Fit log-normal assay noise, rounded integer readings, and left censoring.

    Model: POC=max(3,round(exp(log(Gamma)+batch_offset+epsilon))).
    Batch offsets use a weak Gaussian prior estimated from fitting batch means.
    """
    a = fit.loc[fit.POC_GGT.ge(3) & fit.POC_Batch.notna() & fit.Gamma_GT.gt(0)].copy()
    if a.empty or not a.POC_GGT.gt(3).any():
        return dict(offset=pd.Series(dtype=float), count=pd.Series(dtype=float),
                    sigma=.08, center=0., prior_sd=.3, objective=0., nit=0)
    codes, levels = pd.factorize(a.POC_Batch)
    y = np.log(a.Gamma_GT.to_numpy())
    p = a.POC_GGT.to_numpy()
    lo = np.where(p <= 3, -np.inf, np.log(np.maximum(p-.5, .01)))
    hi = np.log(p+.5)
    valid = p > 3
    initial = pd.Series(np.log(p[valid])-y[valid]).groupby(codes[valid]).mean()
    center = float(initial.mean())
    offsets = initial.reindex(np.arange(len(levels))).fillna(center).to_numpy()
    spread = float(initial.std())
    prior_sd = max(spread, .3) if np.isfinite(spread) else .3
    n = len(levels)

    def objective(params):
        off, sigma = params[:n], np.exp(params[-1])
        zl, zu = (lo-y-off[codes])/sigma, (hi-y-off[codes])/sigma
        # Use survival functions when both interval endpoints are positive.
        c_hi = np.where(zl > 0, log_ndtr(-zl), log_ndtr(zu))
        c_lo = np.where(zl > 0, log_ndtr(-zu), log_ndtr(zl))
        diff = np.minimum(c_lo-c_hi, -1e-15)
        logp = c_hi + np.log(-np.expm1(diff))
        logphi_l = -.5*zl*zl-.5*np.log(2*np.pi)
        logphi_u = -.5*zu*zu-.5*np.log(2*np.pi)
        rl = np.exp(np.minimum(logphi_l-logp, 50))
        ru = np.exp(np.minimum(logphi_u-logp, 50))
        d_off = (rl-ru)/sigma
        d_sig = np.where(np.isfinite(zl), zl, 0)*rl - zu*ru
        penalty = .5*np.sum(((off-center)/prior_sd)**2)
        grad = np.r_[-np.bincount(codes, weights=d_off, minlength=n)+(off-center)/prior_sd**2,
                     -d_sig.sum()]
        return float(-logp.sum()+penalty), grad

    result = minimize(objective, np.r_[offsets, np.log(.08)], jac=True,
                      method='L-BFGS-B', bounds=[(-5,5)]*n+[(np.log(.015),np.log(.5))],
                      options={'maxiter': 250, 'ftol': 1e-11, 'gtol': 1e-5})
    if not result.success:
        raise RuntimeError(result.message)
    count = pd.Series(codes).value_counts().reindex(np.arange(n)).to_numpy()
    return dict(offset=pd.Series(result.x[:n], index=levels), sigma=float(np.exp(result.x[-1])),
                center=center, prior_sd=prior_sd, count=pd.Series(count,index=levels),
                objective=float(result.fun), nit=int(result.nit))


def interval_predictions(model, frame, clinical_log1p=None):
    off = frame.POC_Batch.map(model['offset']).to_numpy()
    p = frame.POC_GGT.to_numpy()
    with np.errstate(invalid='ignore'):
        raw = np.logaddexp(0, np.log(p)-off)
    if clinical_log1p is None:
        return raw
    mu = np.log(np.maximum(np.expm1(clinical_log1p), 1))
    n = frame.POC_Batch.map(model['count']).fillna(0).to_numpy()
    off = np.where(np.isfinite(off), off, model['center'])
    measurement_var = model['sigma']**2*(1+1/np.maximum(n,1))
    tau2 = .26**2
    sd = np.sqrt(tau2+measurement_var)
    lo = np.where(p <= 3, -np.inf, np.log(np.maximum(p-.5,.01)))
    hi = np.log(p+.5)
    valid = np.isfinite(p) & (p >= 3) & (n > 0)
    posterior = np.array(clinical_log1p).copy()
    zl, zu = (lo[valid]-mu[valid]-off[valid])/sd[valid], (hi[valid]-mu[valid]-off[valid])/sd[valid]
    ez = mu[valid]+off[valid]+sd[valid]*truncnorm.mean(zl,zu)
    ey = mu[valid]+tau2/(tau2+measurement_var[valid])*(ez-mu[valid]-off[valid])
    posterior[valid] = np.logaddexp(0, ey)
    return raw, posterior


def legacy_posterior(cal, frame, clinical_log1p, beta):
    """Control using legacy offsets and otherwise identical posterior recipe."""
    p=frame.POC_GGT.to_numpy(); raw=cal.cal2_log1p.to_numpy()
    pred=np.array(clinical_log1p).copy(); valid=np.isfinite(raw)
    p=p[valid]
    off=np.log(np.expm1(raw[valid]))-beta*np.log(p)
    mu=np.log(np.maximum(np.expm1(clinical_log1p[valid]),1))
    count=cal.batch_w.to_numpy()[valid]
    tau2=.26**2; var=.0825**2*(1+1/np.maximum(count,1)); sd=np.sqrt(tau2+var)
    lo=np.where(p<=3,-np.inf,beta*np.log(p-.5)+off); hi=beta*np.log(p+.5)+off
    ey=mu+tau2/(tau2+var)*sd*truncnorm.mean((lo-mu)/sd,(hi-mu)/sd)
    pred[valid]=np.logaddexp(0,ey)
    return pred


def conditional_review(d, posteriors):
    """Frozen 30% component weight; grouped bootstrap is conditional only."""
    prior=pd.read_csv('outputs/final_review/neural_blend_v3_oof.csv')
    assert np.array_equal(prior.ID,d.ID)
    y=np.log1p(d.Gamma_GT.to_numpy()); ref=np.log1p(prior.Gamma_GT.to_numpy())
    rms=lambda x:float(np.sqrt(np.mean((x-y)**2)))
    metrics=[]
    for name,q in list(posteriors.items())+[('mean',np.mean(list(posteriors.values()),axis=0))]:
        pred=.7*ref+.3*q
        metrics.append(dict(name=str(name),rmsle=rms(pred),score_gain=(rms(ref)-rms(pred))/1.0082))
    q=np.mean(list(posteriors.values()),axis=0); pred=.7*ref+.3*q
    rng=np.random.default_rng(9901); bootstrap={}
    for group in ['POC_Batch','Workplace_ID']:
        code,levels=pd.factorize(d[group].fillna('missing'))
        n=np.bincount(code); old=np.bincount(code,weights=(ref-y)**2); new=np.bincount(code,weights=(pred-y)**2)
        gains=[]
        for _ in range(1000):
            sample=rng.integers(len(levels),size=len(levels)); count=n[sample].sum()
            gains.append((np.sqrt(old[sample].sum()/count)-np.sqrt(new[sample].sum()/count))/1.0082)
        bootstrap[group]=dict(lower=float(np.quantile(gains,.025)),upper=float(np.quantile(gains,.975)),positive_fraction=float(np.mean(np.array(gains)>0)))
    result=dict(reference_gamma_rmsle=rms(ref),weight=.3,metrics=metrics,conditional_bootstrap=bootstrap,
        selection='Weights 0.15 and 0.30 inspected on seeds42 and27091. Weight0.30 selected after seeing larger gains and frozen before seed90317. Seeds reuse rows; no independent validation.',
        limitations='Cached V2 clinical OOF and NN+V3 reference are not fully nested. Bootstrap is conditional on fixed reused predictions and does not adjust for model/weight selection.')
    pd.DataFrame({'ID':d.ID,'posterior':q,'proposed_gamma_log1p':pred,'poc_floor':d.POC_GGT.eq(3),'poc_missing':d.POC_GGT.isna()}).to_csv(OUT/'mean_oof_component.csv',index=False)
    (OUT/'conditional_review.json').write_text(json.dumps(result,indent=2))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    d,t,_=load()
    c=np.load('outputs/v2/components.npz')
    baseline=pd.read_csv('outputs/v2/oof_3seed.csv')
    assert np.array_equal(d.ID,baseline.ID)
    y=np.log1p(d.Gamma_GT.to_numpy())
    ref=np.log1p(baseline.gamma_oof.to_numpy())
    all_metrics=[]; posteriors={}
    for seed in [42, 27091, 90317]:
        preds={k:np.full(len(d),np.nan) for k in ['legacy','weighted','interval','posterior','legacy_posterior']}
        folds=np.zeros(len(d),int); details=[]
        for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(d)):
            a,b=d.iloc[tr],d.iloc[va]
            (_,old),beta=batch_features_v2(a,[a,b])
            preds['legacy'][va]=old.cal2_log1p
            preds['legacy_posterior'][va]=legacy_posterior(old,b,c['oof_nop'][va],beta)
            preds['weighted'][va],wb=weighted_calibration(a,b)
            model=fit_interval_calibration(a)
            preds['interval'][va],preds['posterior'][va]=interval_predictions(model,b,c['oof_nop'][va])
            folds[va]=fold
            details.append(dict(fold=fold,legacy_beta=beta,weighted_beta=wb,sigma=model['sigma'],nit=model['nit']))
            print(seed,fold,details[-1],flush=True)
        known=(d.POC_GGT.to_numpy()>3)&np.isfinite(preds['legacy'])&np.isfinite(preds['weighted'])&np.isfinite(preds['interval'])
        for name in ['legacy','weighted','interval']:
            all_metrics.append(dict(seed=seed,kind='honest_calibration_only',name=name,n=int(known.sum()),
                                     rmsle=float(np.sqrt(np.mean((preds[name][known]-y[known])**2)))))
        for w in [.15,.3]:
            blend=(1-w)*ref+w*preds['posterior']
            foldgain=[]
            for fold in range(5):
                mask=folds==fold
                foldgain.append(float((np.sqrt(np.mean((ref[mask]-y[mask])**2))-np.sqrt(np.mean((blend[mask]-y[mask])**2)))/1.0082))
            all_metrics.append(dict(seed=seed,kind='conditional_saved_components',name=f'posterior_blend_{w}',n=len(d),
                rmsle=float(np.sqrt(np.mean((blend-y)**2))),score_gain=float((np.sqrt(np.mean((ref-y)**2))-np.sqrt(np.mean((blend-y)**2)))/1.0082),fold_gains=foldgain))
        pd.DataFrame({'ID':d.ID,'fold':folds,**preds}).to_csv(OUT/f'oof_{seed}.csv',index=False)
        posteriors[seed]=preds['posterior']
        (OUT/f'fit_details_{seed}.json').write_text(json.dumps(details,indent=2))
    # Full fitting calibration creates reproducible component predictions only.
    model=fit_interval_calibration(d)
    raw,post=interval_predictions(model,t,c['te_nop'])
    pd.DataFrame({'ID':t.ID,'interval':raw,'posterior':post,'poc_floor':t.POC_GGT.eq(3),'poc_missing':t.POC_GGT.isna(),
                  'unknown_batch':~t.POC_Batch.isin(model['offset'].index)}).to_csv(OUT/'test_components.csv',index=False)
    pd.DataFrame({'batch':model['offset'].index,'offset':model['offset'].values,'n':model['count'].values}).to_csv(OUT/'full_offsets.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(all_metrics,indent=2))
    conditional_review(d,posteriors)
    identity={str(p):sha(p) for p in ['data/train.csv','data/test.csv','outputs/v2/components.npz','outputs/v2/oof_3seed.csv','outputs/final_review/neural_blend_v3_oof.csv',__file__]}
    identity.update(warning='Saved clinical components and reference OOF reuse are conditional, not fully nested. Clinical prior SD is fixed at 0.26. No submission written.', full_sigma=model['sigma'])
    (OUT/'manifest.json').write_text(json.dumps(identity,indent=2))
    print(json.dumps(all_metrics,indent=2))


if __name__=='__main__':
    run()
