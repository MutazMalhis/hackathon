# Plan implementation — experiment record

## Validation procedure

20% stratified outer holdout, seed 20261009: 1,801 rows. Development partition: 7,201 rows. Three development folds select the configuration. Each fit uses an inner 20% split for early stopping, then refits on the full fitting partition with the selected number of trees. Outer validation labels never choose the number of trees, transformations, or batch corrections.

The entire dataset was explored previously, so the holdout is a prospective check rather than historically untouched data. The frozen configuration is selected before the outer holdout is scored; no tuning followed the holdout results. Final five-fold CV includes development-selected rows and is a diagnostic estimate.

## Experiments

Raw features versus a urine feature family: log1p cotinine and EtG; log concentration ratios to urine creatinine; ratios centered by training-partition urine plate means. The comparison tests the family together and does not establish which individual addition caused the gain.

Calibration variants: mean and median batch log offsets; mean-offset shrinkage strengths 0.5 and 2 toward the global training offset; pure calibration versus 90% calibration / 10% regression blending in log1p space. Missing readings and unknown batches use regression.

## Selected method

Urine features with unshrunk mean batch calibration blended 90/10 with regression in log1p space. Development OOF AUC 0.96032, RMSLE 0.13327, combined score 0.82814. Mean shrinkage and median alternatives did not win this development comparison. The difference between the top blends is small; selection should not be interpreted as statistical proof of superiority.

## Outer holdout

| Method | AUC | RMSLE | Score |
| --- | ---: | ---: | ---: |
| Original raw-feature calibrated approach | 0.95116 | 0.12972 | 0.82249 |
| Selected urine-feature blend | 0.96916 | 0.12394 | 0.84623 |
| Urine model with regression only | 0.96916 | 0.22222 | 0.74874 |

The original comparator is retrained on the same development rows with the same inner early-stopping procedure. These figures therefore differ from the earlier exploratory five-fold scores.

92.67% of holdout records are calibrated. Their RMSLE is 0.10103; the 132 fallback records have RMSLE 0.28381. The 20 records above gamma-GT 100 have RMSLE 0.33116; this small slice is uncertain.

## Final ensemble

Five-fold diagnostic AUC 0.96555, RMSLE 0.12701, combined score 0.83957. The final CSV has 6,399 finite prediction rows aligned with the sample IDs and passed range checks. This CV includes method-selection data and is not the outer holdout result.

## Group robustness

Unseen workplaces: AUC 0.93426, RMSLE 0.12110, score 0.81414; calibration coverage 92.31%.

Unseen cartridge batches: AUC 0.96422, RMSLE 0.27268, score 0.69376; calibration coverage 0%. This establishes that calibration depends on batch overlap. Most competition test batches are known, so this split differs from the test setting. It does not justify using this method for unseen-batch applications.

## Reproduction

Run `.venv/bin/python experiments.py`, then `.venv/bin/python robustness.py`. The same code is included in the appended notebook section. Keep notebook rerun switches false to inspect saved results; enable them to retrain. Models, split IDs, data hashes, installed versions, validation predictions, configuration and CSVs are saved under outputs/plan_v1.

## Remaining steps

Obtain a first Kaggle score from the validated candidate; keep public-score feedback separate from local model selection. Next investigation should focus on fallback regression and high gamma-GT records, using development folds rather than tuning on the evaluated holdout. Cross-target stacking has not been implemented; add it only with nested cross-fitting and a measured development gain.

## Presentation outline

1. Two-target challenge and combined AUC/RMSLE score.
2. Data quality: skewed regression target, center-specific missingness, and shared groups.
3. Baseline: separate CatBoost models with log-scale regression.
4. Discovery: large rapid-test cartridge-batch multipliers.
5. Improved method: fold-safe calibration plus urine features.
6. Evidence: same-holdout comparison and group robustness results.
7. Limitations: missing rapid readings, unseen batches, high values, and exploratory history.
8. Reproducibility: Python notebook, hashes, frozen configuration, and checked output.

No Kaggle submission has been uploaded in this run.
