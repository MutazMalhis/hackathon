# Development-only model improvements

The updated method improves the same-fold baseline in both the original development comparison and a second fold assignment. The latter improves combined score by 0.01030 and improves all three folds. This is a stability check on reused development rows, not an independent test score.

## Selected changes

Smoking probabilities average three CatBoost models: the original urine-feature depth-6 model and depth-4/depth-6 models with richer features. New features include training-only plate quantiles and median-centered urine marker measurements, clinical log transforms, and a small set of physiological ratios.

Gamma_GT uses a depth-4 CatBoost model with engineered clinical and urine features, excluding POC_GGT and POC_Batch from regression. For usable rapid readings in known batches, the original fitting-partition mean batch offset is retained, with 90% calibrated prediction and 10% regression in log1p space. Missing readings or unknown batches use regression alone.

Ridge and histogram boosting were also tested as complementary regressors. Neither was selected. The richer depth-6 clinical regressor and tested regression ensembles did not beat the depth-4 model. Removing cartridge inputs worsened overall uncalibrated regression but improved the fallback subset; calibrated measurements remain central to the final method.

## Development results

| Comparison | Method | Smoking AUC | Gamma RMSLE | Combined score | Fallback RMSLE | Gamma >100 RMSLE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Selection | baseline | 0.96032 | 0.13327 | 0.82814 | 0.27984 | 0.27280 |
| Selection | selected | 0.96806 | 0.13052 | 0.83860 | 0.26775 | 0.21849 |
| Stability | baseline | 0.96011 | 0.13472 | 0.82649 | 0.28061 | 0.25142 |
| Stability | selected | 0.96804 | 0.13232 | 0.83679 | 0.27087 | 0.19434 |

## Validation boundaries

Only the 7,201 development IDs from the earlier experiment are used for selection and stability checks. The previously scored 1,801-row holdout is excluded. Three stratified development folds select model families and calibration weights. CatBoost uses a separate inner split to choose tree count, followed by full fitting-partition refitting.

The configuration is frozen before the second fold assignment. Promotion requires a score gain exceeding 0.001 overall and gains on at least two of three folds. The selected method passes. No tuning follows the stability check. Full-dataset EDA happened earlier, and the high-value/fallback weaknesses were identified in previous validation. The current results must therefore not be described as untouched validation.

Final five-fold training uses all training rows after configuration selection. Its OOF estimate includes selection rows and is diagnostic. The competition test labels remain unavailable; no public leaderboard score is inferred from these local numbers. Unknown-batch robustness remains a limitation, as batch calibration depends on shared cartridge groups.

## Reproduction and checks

Run `.venv/bin/python improvements.py` or enable `RUN_RETRAIN` in `model_improvements.ipynb`. Earlier `experiments.py` outputs must exist to supply the development split. All implementation code is included in the notebook.

Five tests verify exclusion of targets and ID, independence from evaluation targets and reference labels in feature construction, training-only plate statistics, preservation of missing/unknown-cartridge fallback, and exclusion of the earlier holdout. The notebook runs with retraining disabled to display saved results. Model parameters, seeds, versions, data hashes, code hashes, and fold predictions are recorded under `outputs/plan_v2/`. Cached predictions are invalidated by a changed code/data identity.

## Final five-fold diagnostic result

| Method | Smoking AUC | Gamma RMSLE | Combined score |
| --- | ---: | ---: | ---: |
| baseline | 0.96555 | 0.12701 | 0.83957 |
| selected | 0.97201 | 0.12461 | 0.84841 |

Combined diagnostic score gain: **0.00884**. Original smoking fold predictions were reproduced exactly, supporting comparator consistency.

| Subset | Rows | Original RMSLE | Improved RMSLE |
| --- | ---: | ---: | ---: |
| Calibrated | 8287 | 0.10522 | 0.10424 |
| Fallback | 715 | 0.27349 | 0.26371 |
| Gamma_GT >100 | 84 | 0.25197 | 0.20633 |

`outputs/plan_v2/submission.csv` contains 6,399 finite predictions with the exact sample ID order, probability bounds [0,1], and Gamma_GT bounds [1,1000].

Submission SHA256: `05ad7f41017948f1d26ebd5e7b914f1e614d306b6be4ead3a2ea68ddd58ed400`.

No submission was uploaded to Kaggle by this workflow.
