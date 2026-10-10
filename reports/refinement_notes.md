# Calibration refinement and cross-target experiment

The user reported **0.85329** on Kaggle for the previous candidate. The updated candidate improves local five-fold diagnostic score from **0.84841 to 0.85389**, with the same Smoking predictions. Its new Kaggle score is unknown. The reported previous score was not used to select parameters.

## Observed measurement issue and hypothesis

POC_GGT readings in the development data are integers and have minimum 3; 227 development rows have that value. We tested a detection-floor hypothesis: 3 may represent a clipped/rounded measurement rather than an exact value. These observations alone do not prove the underlying measurement mechanism.

The selected method excludes readings of 3 from batch offset estimation and gives readings of 4 or 5 a weight of 0.3. For nonfloor readings, it retains a unit log slope. The fitted-slope candidates did not improve the development result.

For floor readings with a usable batch offset, the clinical log-Gamma prediction is treated as a Gaussian prior with SD 0.27. A one-sided observation, latent reading at most 3.5, updates that prior analytically. This is a modeling assumption supported by the observed development improvement, not a confirmed assay specification. If the batch offset is unavailable, the existing clinical regression remains the fallback.

Nonfloor predictions use an adaptive blend of calibration and regression. The weight accounts for training-only residual variance, effective batch sample count, and approximate integer-rounding variance. No evaluation targets enter calibration fitting.

## Development evidence

Ten predefined calibration candidates compare unit/fitted slopes, floor fallback/censoring, and fixed/adaptive blending. Development RMSLE falls from 0.13052 to 0.12563. The configuration is frozen before a second fold assignment.

| Second development comparison | AUC | Gamma RMSLE | Combined score | Floor RMSLE |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.96804 | 0.13232 | 0.83679 | 0.30158 |
| selected | 0.96804 | 0.12587 | 0.84319 | 0.22202 |

All three folds improve, passing the promotion gate (gain above 0.001 overall and at least two folds). The second split reuses development rows and is a stability check, not independent validation. The previously scored holdout is excluded from both comparisons.

A paired cartridge-cluster bootstrap estimates uncertainty conditional on these fitted predictions. Missing batches form singleton clusters. It does not account for training or model-selection uncertainty.

Estimated development score gain: 0.00640; 95% conditional bootstrap interval [0.00354, 0.00952], using 2,000 bootstrap samples.

## Cross-target experiment

Smoking probability and log odds were tested as Gamma_GT features. Training probabilities use three-fold cross-fitting within each outer fitting partition. The Gamma_GT early-stopping pilot repeats this process inside its own fitting partition, excluding stopping rows from classifier training and feature fitting. Full regression refitting generates fresh OOF probabilities. Outer prediction uses a classifier trained solely on the outer fitting partition.

The 18 nested classifier folds were audited: fitting and predicted IDs are disjoint, and pilot stopping rows are absent. All 11 unit tests pass, covering target isolation, training-only normalization, calibration recovery, floor exclusion, missing/unknown batches, numerical stability, and crossfit isolation.

| Regression choice | Gamma RMSLE after new calibration | Combined score |
| --- | ---: | ---: |
| clinical | 0.12563 | 0.84345 |
| stacked | 0.12614 | 0.84294 |
| blend | 0.12556 | 0.84352 |

The 50/50 clinical/stacked blend gains only about 0.00007 over clinical regression, below the predefined 0.0005 threshold. Pure stacking worsens calibrated predictions and missing-reading performance. Neither is promoted, and no further stacking tuning follows these results. The final candidate keeps v2 regression.

## Final five-fold diagnostic results

| Candidate | Smoking AUC | Gamma RMSLE | Combined score |
| --- | ---: | ---: | ---: |
| baseline | 0.97201 | 0.12461 | 0.84841 |
| selected | 0.97201 | 0.11909 | 0.85389 |

Final CV includes rows used to select methods. It is diagnostic, not an independent estimate of Kaggle performance. The Smoking predictions are unchanged from v2.

| Subset | Rows | v2 RMSLE | v3 RMSLE |
| --- | ---: | ---: | ---: |
| Nonfloor calibrated | 7983 | 0.09152 | 0.08931 |
| Floor=3 | 300 | 0.27483 | 0.21405 |
| Fallback | 765 | 0.27233 | 0.26428 |
| Gamma_GT >100 | 84 | 0.20633 | 0.20356 |

These slices can overlap: floor rows without usable batch offsets also appear in fallback. Excluding floor readings makes offsets unavailable for some batches supported only by floor observations; those rows now use clinical regression. The underlying regression model is unchanged, so its predictions on originally missing-reading rows are unchanged. High-Gamma results are based on only 84 rows.

## Artifacts and reproducibility

The new file is `outputs/plan_v3/submission_v3.csv`: 6,399 finite rows, exact sample ID order, Smoking within [0,1], and Gamma_GT within [1,1000]. The original submissions are retained. The workflow does not upload to Kaggle.

Run `.venv/bin/python refine.py`, or use `calibration_refinement.ipynb` with `RUN_RETRAIN=True`. Run the earlier workflows first on a fresh checkout to generate the v1 split IDs and v2 fold caches. The default notebook setting displays saved results. The notebook includes all new code and reuses earlier project helpers.

Code/data hashes, package versions, configuration, nested crossfit audits, fold models, calibration offsets, OOF predictions, bootstrap evidence, and output verification are saved in `outputs/plan_v3/`. Run `.venv/bin/python -m unittest discover -s tests` for verification.

## Remaining uncertainty

The actual measurement floor and rounding process are unconfirmed. The Gaussian prior and approximate rounding variance simplify that process. Calibration still depends on batch overlap; unknown batches use regression. Development rows have been reused across iterations, so local gains should be checked with the next Kaggle submission without treating public feedback as an unlimited tuning target.
