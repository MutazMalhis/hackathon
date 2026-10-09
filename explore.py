"""Reproduce dataset exploration using only competition CSVs.
Run: .venv/bin/python explore.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def main():
    out = Path('outputs/exploration')
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv('data/train.csv')
    te = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sample_submission.csv')
    features = te.columns.drop('ID').tolist()
    missing = pd.DataFrame({'train_missing_pct': tr[features].isna().mean() * 100,
                            'test_missing_pct': te[features].isna().mean() * 100,
                            'train_unique': tr[features].nunique(),
                            'test_unique': te[features].nunique()})
    missing.to_csv(out / 'feature_inventory.csv')
    tr.describe(include='all').to_csv(out / 'train_descriptive_statistics.csv')
    te.describe(include='all').to_csv(out / 'test_descriptive_statistics.csv')
    numeric = tr[features].select_dtypes('number').columns
    correlations = pd.DataFrame({
        'smoking_spearman': tr[numeric].corrwith(tr.Smoking, method='spearman'),
        'gamma_spearman': tr[numeric].corrwith(tr.Gamma_GT, method='spearman'),
        'log_gamma_pearson': tr[numeric].corrwith(np.log1p(tr.Gamma_GT))})
    correlations.to_csv(out / 'target_correlations.csv')
    diagnostics = []
    for c in numeric:
        m = tr[c].notna()
        if tr.loc[m, c].nunique() < 2:
            continue
        auc = roc_auc_score(tr.loc[m, 'Smoking'], tr.loc[m, c])
        diagnostics.append({'feature': c, 'observed_rows': int(m.sum()), 'raw_auc': auc,
                            'direction_adjusted_auc': max(auc, 1-auc)})
    pd.DataFrame(diagnostics).sort_values('direction_adjusted_auc', ascending=False).to_csv(out / 'single_feature_auc.csv', index=False)
    shifts = []
    for c in numeric:
        a, b = tr[c].dropna(), te[c].dropna()
        k = ks_2samp(a, b)
        shifts.append({'feature': c, 'ks_statistic': float(k.statistic),
                       'train_median': float(a.median()), 'test_median': float(b.median())})
    pd.DataFrame(shifts).sort_values('ks_statistic', ascending=False).to_csv(out / 'numeric_distribution_shift.csv', index=False)
    groups = {}
    for c in ['Workplace_ID', 'Urine_Plate', 'POC_Batch', 'Screening_Center', 'Visit_Quarter']:
        groups[c] = {'train_unique': int(tr[c].nunique()), 'test_unique': int(te[c].nunique()),
                     'unseen_test_nonmissing_pct': float((~te[c].dropna().isin(tr[c].dropna())).mean()*100)}
    tr.groupby('Screening_Center').agg(rows=('ID','size'),smoking_rate=('Smoking','mean'),
                                     gamma_median=('Gamma_GT','median')).to_csv(out / 'centers.csv')
    pd.concat({c: tr[c].isna().groupby(tr.Screening_Center).mean()*100 for c in
               ['Exhaled_CO_ppm','Alcohol_Units_Week','POC_GGT']},axis=1).to_csv(out / 'center_missingness.csv')
    residual = np.log(tr.Gamma_GT)-np.log(tr.POC_GGT.where(tr.POC_GGT.gt(0)))
    batches = residual.groupby(tr.POC_Batch).agg(['count','mean','std'])
    batches.to_csv(out / 'rapid_test_batches.csv')
    lipid = tr.LDL - (tr.Cholesterol_Total-tr.HDL-tr.Triglyceride/5)
    train_hash = pd.util.hash_pandas_object(tr[features],index=False)
    test_hash = pd.util.hash_pandas_object(te[features],index=False)
    summary = {
        'train_shape': list(tr.shape), 'test_shape': list(te.shape), 'sample_shape':list(sample.shape),
        'features': features, 'smoking_counts':tr.Smoking.value_counts().to_dict(),
        'smoking_rate':float(tr.Smoking.mean()), 'gamma_summary':tr.Gamma_GT.describe(percentiles=[.01,.5,.9,.95,.99]).to_dict(),
        'train_feature_missing_pct':float(tr[features].isna().to_numpy().mean()*100),
        'test_feature_missing_pct':float(te[features].isna().to_numpy().mean()*100),
        'target_missing':tr[['Smoking','Gamma_GT']].isna().sum().to_dict(),
        'train_id_unique':bool(tr.ID.is_unique),'test_id_unique':bool(te.ID.is_unique),
        'train_test_id_overlap':int(len(set(tr.ID)&set(te.ID))),
        'sample_ids_match_test_order':bool(sample.ID.equals(te.ID)),
        'train_duplicate_features':int(tr[features].duplicated().sum()),
        'test_duplicate_features':int(te[features].duplicated().sum()),
        'cross_split_feature_hash_matches':int(test_hash.isin(train_hash).sum()),
        'groups':groups,
        'eyesight_9_9_counts':{c:int(tr[c].eq(9.9).sum()) for c in ['Eyesight_L','Eyesight_R']},
        'lipid_residual':{'observed_rows':int(lipid.notna().sum()),'median_absolute':float(lipid.abs().median()),'p95_absolute':float(lipid.abs().quantile(.95))},
        'batch_effect':{'between_batch_offset_std':float(batches['mean'].std()),'median_within_batch_std':float(batches['std'].median()),'median_rows':float(batches['count'].median())}
    }
    (out / 'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('TOP SINGLE FEATURE AUC')
    print(pd.read_csv(out/'single_feature_auc.csv').head(8).to_string(index=False))
    print('LARGEST NUMERIC SHIFTS')
    print(pd.read_csv(out/'numeric_distribution_shift.csv').head(5).to_string(index=False))


if __name__ == '__main__':
    main()
