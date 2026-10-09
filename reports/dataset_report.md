# DSC Modeling Hackathon — Dataset Exploration Report

All analysis and modeling were performed in Python on the competition files. This report documents completed exploration and experiments; proposed next steps are identified separately. No external dataset was used and no submission has been uploaded.

## 1. Objective and scoring

Predict `Smoking` as a probability and `Gamma_GT` as a positive continuous value. The metric published on the competition page is:

```python
score = 0.5 * (2 * auc - 1) + 0.5 * (1 - rmsle / 0.5041)
```

The regression scorer clips predictions to [1, 1000]. AUC measures ranking; RMSLE measures errors after `log1p`. An AUC gain of 0.01 increases the combined score by 0.01; an RMSLE reduction of 0.01 increases it by approximately 0.00992. Both tasks deserve attention.

Source: https://www.kaggle.com/competitions/dsc-modeling-hackathon-ml-arena/data

## 2. Files, schema, and integrity

| File | Rows | Columns | Purpose |
| --- | --- | --- | --- |
| train.csv | 9,002 | 41 | ID, 38 predictors, two targets |
| test.csv | 6,399 | 39 | ID and the same 38 predictors |
| sample_submission.csv | 6,399 | 3 | Required output structure |

There are no missing targets. IDs are unique within each split and do not overlap between train and test. Sample IDs match test IDs in the same order. No exact duplicate feature rows were found within either split; feature-row hashes also found no cross-split matches. This does not rule out near duplicates or correlated participants.

`Screening_Center` and `Visit_Quarter` are stored as text. Other features are numeric in the CSV, but `Workplace_ID`, `Urine_Plate`, and `POC_Batch` represent identifiers and are treated as categories by the baseline. Numeric storage alone does not make an identifier a continuous measurement.

Feature families include body measurements and vitals, blood chemistry, senses and dental screening, lifestyle, urine measurements, screening metadata, and the rapid gamma-GT test.

## 3. Target distributions

Smoking labels: 5,731 zeros and 3,271 ones; positive prevalence is **36.34%**. This is moderate imbalance. Stratified validation preserves its proportions; automatic class balancing is not assumed necessary for AUC.

| Gamma-GT statistic | U/L |
| --- | ---: |
| Minimum | 0.998 |
| Mean | 26.165 |
| Median | 20.843 |
| 90th percentile | 46.009 |
| 95th percentile | 57.712 |
| 99th percentile | 98.188 |
| Maximum | 485.657 |

Gamma-GT has a strong right tail. The baseline fits `log1p(Gamma_GT)` with squared-error loss, then converts predictions back with `expm1`, aligning training with RMSLE. Extreme values are not automatically deleted. The minimum is slightly below the scorer's prediction floor.

## 4. Missing values

Missing cells comprise **3.83%** of training predictors and **3.88%** of test predictors. Gaps are concentrated in particular features:

| Feature | train_missing_pct | test_missing_pct |
| --- | --- | --- |
| Exhaled_CO_ppm | 23.3726 | 23.3161 |
| Liver_Echo_Grade | 16.8074 | 17.0964 |
| Alcohol_Units_Week | 13.8636 | 13.8147 |
| Household_Smoke_Exposure | 11.2197 | 10.7829 |
| Sleep_Hours_Avg | 9.0091 | 9.4233 |
| Family_History_CVD | 8.0538 | 7.9075 |
| POC_Batch | 7.9094 | 8.2200 |
| POC_GGT | 7.9094 | 8.2200 |
| Urine_Protein | 6.0653 | 5.8134 |
| LDL | 5.0100 | 5.1414 |
| Urine_Plate | 4.7767 | 5.1883 |
| Urine_EtG_ng_mL | 4.7767 | 5.1883 |

CO missingness is **92.65% at center C** and **94.01% at center F**, versus zero at the other six centers. This is a collection pattern, not evidence that the missingness causes or identifies smoking.

The baseline retains numeric NaNs for CatBoost and uses a dedicated missing category for categorical features. It does not globally impute with the mean. If linear models are compared later, imputation must be fitted on each training fold only.

## 5. Individual feature signals

The following are descriptive, single-feature AUCs measured on rows where that feature is observed. They are not cross-validated model results and their subsets differ. Direction-adjusted AUC allows an inverse relationship to count as useful ranking information.

| feature | observed_rows | raw_auc | direction_adjusted_auc |
| --- | --- | --- | --- |
| Hemoglobin | 9002 | 0.8130 | 0.8130 |
| Height_cm | 9002 | 0.7942 | 0.7942 |
| Weight_kg | 9002 | 0.7507 | 0.7507 |
| Triglyceride | 8639 | 0.7287 | 0.7287 |
| Serum_Creatinine | 8704 | 0.7139 | 0.7139 |
| Waist_cm | 9002 | 0.6875 | 0.6875 |
| HDL | 9002 | 0.3206 | 0.6794 |
| ALT | 8742 | 0.6676 | 0.6676 |

Hemoglobin and height have stronger individual ranking signals than the raw breath or urine measurements. This motivates multivariate modeling; it does not establish a biological explanation, especially because some features are synthetic.

Raw urine cotinine AUC is **0.6356**. Cotinine divided by urine creatinine gives **0.6301** on jointly observed rows. Simple normalization did not improve this initial comparison; the two subsets need not be identical, so this is not a controlled ablation. Raw exhaled CO AUC is **0.6541** on observed rows.

The strongest positive numerical Spearman associations with gamma-GT are:

| Feature | gamma_spearman | log_gamma_pearson |
| --- | --- | --- |
| ALT | 0.6154 | 0.5775 |
| Hemoglobin | 0.5518 | 0.5103 |
| Weight_kg | 0.5231 | 0.4941 |
| Waist_cm | 0.5174 | 0.4937 |
| Triglyceride | 0.4980 | 0.4971 |
| POC_GGT | 0.4580 | 0.4336 |
| Liver_Echo_Grade | 0.4129 | 0.4230 |
| Serum_Creatinine | 0.4120 | 0.3715 |

Smoking and gamma-GT have Spearman correlation **0.4634**. The targets are connected, but neither target is available as a test-time input. A future cross-target model must use predictions generated without access to the held-out labels, with nested cross-fitting for evaluation.

## 6. Redundancy and coding checks

The data dictionary states that LDL is approximately calculated from total cholesterol, HDL, and triglycerides. On 8,209 rows with all four values, the median absolute residual from `LDL = total - HDL - triglyceride/5` is **0.7594**, and the 95th percentile is **2.2684**. The relation is approximate, not exact. It creates redundancy worth considering for linear models; trees can retain the columns initially.

The dictionary describes eyesight value 9.9 as a blindness code. There are **zero exact 9.9 values in either training eyesight column**. Thus the documented code has not been observed in training; no training recoding was performed. Wider abnormal-value inspection and equivalent test checks remain future work.

## 7. Train/test overlap and distribution checks

| Group | Training categories | Test categories | Unseen nonmissing test rows |
| --- | ---: | ---: | ---: |
| Workplace_ID | 534 | 532 | 0% |
| Urine_Plate | 366 | 366 | 0% |
| POC_Batch | 875 | 876 | 0.1703% |
| Screening_Center | 8 | 8 | 0% |
| Visit_Quarter | 4 | 4 | 0% |

The competition test split mostly contains known groups. Random stratified row validation is a reasonable initial approximation for interpolation across those groups, but it does not measure performance on new workplaces, plates, or batches. Grouped validation would answer a different generalization question and is worth checking separately.

Numerical train/test distributions were compared using the two-sample Kolmogorov–Smirnov statistic. The largest observed statistic was **0.0253 for urine creatinine**; its median is 124.8 in training and 131.2 in test. Other leading differences were CO (0.0215), total cholesterol (0.0211), workplace ID (0.0209), and triglycerides (0.0186). Workplace-ID numeric order has no substantive meaning; its categorical overlap is more informative. These are marginal checks and do not establish absence of multivariate or conditional shift.

## 8. Main discovery: rapid-test batch effects

The raw rapid-test measurement `POC_GGT` has Spearman correlation **0.4580** with gamma-GT. Its usefulness becomes much clearer after accounting for cartridge batches.

For each observed training record, calculate:

```python
log_offset = np.log(Gamma_GT) - np.log(POC_GGT)
```

The standard deviation of batch mean offsets is **0.8819**, while the median within-batch standard deviation is **0.0888**. A typical batch has eight training readings. The much larger between-batch variation suggests a substantial multiplicative batch effect.

For each validation fold, calculate mean offsets using only the other folds. Correct held-out rapid-test readings as:

```python
prediction = np.exp(np.log(POC_GGT) + training_batch_mean_offset)
```

Use the baseline regressor where the rapid test is missing, nonpositive, or the batch has no training estimate. This avoids calculating a row's correction from its own target.

Correction coverage is **91.94%** of out-of-fold records and **91.62%** of test records. On corrected validation records alone, RMSLE is **0.1105**. A second random split (seed 2026, 20% holdout) gave **0.1048 RMSLE** at **92.34% coverage** for the correction alone. This second check supports the calibration mechanism; it is not an independent evaluation of the complete fitted pipeline.

## 9. Models and measured results

The baseline trains separate CatBoost classification and regression models. It excludes ID and both targets from predictors. Parameters: five stratified shuffled folds, seed 42, depth 6, learning rate 0.04, up to 2,000 iterations, and early stopping patience 150. Test predictions are averaged across folds; baseline gamma-GT predictions are averaged in log space.

| Local out-of-fold metric | Baseline | Batch-corrected model |
| --- | ---: | ---: |
| Smoking AUC | 0.94770 | 0.94770 |
| Gamma-GT RMSLE | 0.22050 | 0.13549 |
| Combined score | 0.72899 | 0.81332 |

The combined gain is **0.08432**, entirely from regression; smoking predictions are unchanged. The corrected model uses the direct batch correction where available and the baseline elsewhere. These are local validation results, not public or private leaderboard scores.

Early stopping uses each validation fold, and the calibration method was selected after exploration of these data. Scores therefore are exploratory and can be optimistic. Extensive tuning should use an untouched holdout or nested evaluation. Single-feature ranking diagnostics also use labels descriptively and should not be mistaken for held-out performance.

## 10. Submission and reproducibility

The corrected CSV passed checks for exactly 6,399 rows, exact column order `ID,Smoking,Gamma_GT`, matching sample IDs, finite numeric predictions, smoking probabilities in [0,1], and gamma-GT in [1,1000]. Nothing has been uploaded to Kaggle.

Python files:

- `explore.py`: reproduce descriptive exploration and CSV/JSON tables.
- `train.py`: reproduce five-fold baseline models, predictions, hashes, and scores.
- `calibrate.py`: reproduce fold-safe batch correction and the final candidate CSV.

```sh
cd /Users/mac/Desktop/hackathon
.venv/bin/python explore.py
.venv/bin/python train.py --output outputs/baseline
.venv/bin/python calibrate.py
```

All software runs in `.venv`. Exact installed versions are saved in `outputs/environment.txt`. Baseline metadata records data SHA-256 hashes, feature names, categorical fields, seed, and fold scores. The full second-split calibration check is described above but is not currently a separate command in the saved scripts.

## 11. Next experiments — not completed

1. Investigate urine plate effects and their interactions with cotinine and creatinine using fold-safe transformations.
2. Test whether calibration errors at low rapid-test readings favor robust or regularized batch estimates.
3. Analyze regression error separately for calibrated records, missing readings, unseen batches, and high gamma-GT values.
4. Inspect classifier errors and test workplace-aware features within folds.
5. Compare grouped validation and a fixed independent holdout before more extensive tuning.
6. Evaluate cross-target stacking with nested cross-fitting if it improves the combined score enough to justify complexity.

This competition prohibits external data and limits these synthetic/modified records and resulting models to competition use. Keep the data and model outputs local unless an authorized competition action requires them.
