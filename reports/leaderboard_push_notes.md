# Improving the confirmed 0.86380 submission

The user identified `outputs/v2/submission_3seed.csv` as the file scoring **0.86380** on Kaggle. The supplied screenshot places XGEngineers second, behind 0.86424: a gap of **0.00044**. These are results from the screenshot, not a live leaderboard lookup. Approximately 33% of test rows determine the public ranking; the other 67% determine the private ranking.

The successful file is preserved. Its SHA-256 is `c49a7788060dd229e1c6072b463c7f90233627f9be63cb4ad59bea4b8d0bd95a`. Recomputing its corresponding saved OOF predictions gives Smoking AUC **0.97545948**, Gamma_GT RMSLE **0.11639699**, and combined diagnostic score **0.86000919**. That local score is distinct from the confirmed Kaggle score.

## What we tested

We focused on improving Smoking while retaining the successful Gamma_GT predictions. A separate Gamma_GT postprocessing experiment tested whether a small residual correction would help.

| Experiment | Result | Decision |
| --- | --- | --- |
| Additional clinical ratios and cleaned urine features | The existing baseline won development selection | Rejected |
| Smoking-adjusted cotinine plate normalization | Confirmation AUC gain +0.00030384; full three-seed diagnostic gain only +0.00008850 | Rejected: below the predefined final +0.0002 AUC gate |
| XGBoost, depth 3 and depth 4 | Pure-model development AUC 0.95463 and 0.95445; 50/50 blends also underperformed the 0.97203 baseline | Rejected |
| Gamma_GT residual correction | Confirmation RMSLE worsened from 0.11714188 to 0.11717911 | Rejected |
| Smoother and ordered CatBoost | Ordered model's 50/50 blend passed confirmation with +0.00038053 AUC and two of three fold wins | Rejected: final AUC gain −0.00002859 |

The cotinine feature estimates laboratory plate offsets after allowing for Smoking-related marker differences. Training features analytically exclude each row's own Smoking label from both the global coefficient and its plate offset. Three meaningful tests verify own-label invariance, agreement with physically removing a row, and held-out label isolation.

The ordered CatBoost candidate uses depth 6, 1,800 iterations, learning rate 0.025, L2 regularization 5, ordered boosting, and random strength 0.5. Development selection fixed a 50/50 blend before the confirmation comparison. Final predictions average five folds for each of seeds 42, 43, and 44, then blend with the successful submission's Smoking probabilities.

## Evaluation limits

Feature/model selection uses the original 7,201 development rows and excludes the 1,801 previously reserved rows. A second fold assignment checks stability on the same development rows; it is not independent validation. Full-data OOF evaluation includes development-selection rows and is also diagnostic.

The inherited V2 pipeline includes globally cross-fitted Smoking predictions in Gamma_GT models and non-nested final blending. The saved reference OOF score may be optimistic. This round leaves that Gamma_GT method unchanged. Cotinine preprocessing uses the existing cached covariate-adjusted plate features; its unsupervised component uses no target labels. Fold-specific plate statistics use a consistent reference-plus-unlabeled-test pool for training and prediction.

The supervised cotinine feature explicitly removes its own Smoking label, but that does not establish that the entire inherited V2 pipeline is free of every source of validation optimism. Gamma_GT residual experiments operate on saved OOF predictions and inherit their limitations.

No candidate's local gain establishes a new Kaggle result. No submission was uploaded by these workflows. The successful CSV should remain a final-selection option even if a new candidate is tested.

## Reproduction and verification

`leaderboard_push.ipynb` displays the project Python implementations, controls optional retraining, loads saved comparisons, and validates any promoted CSV. Retraining defaults to false. The five experiment scripts and commands are listed in README.md. Each family has a separate output directory, configuration, data/code identity records, and fold predictions. No external dataset was used.

The original successful CSV passed checks for 6,399 rows, sample ID order, correct columns, finite predictions, and target bounds. All 14 unit tests passed. Dependency checks passed. XGBoost 3.4.1 was installed for the comparison; macOS required the OpenMP runtime `libomp`. Native categorical inputs follow the [official XGBoost categorical documentation](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).


## Final decision

The ordered blend's full three-seed diagnostic AUC was **0.97543089**, versus
**0.97545948** for the successful submission. It did not pass the final gate.
No candidate from this round passed all promotion checks, and no new submission
CSV was promoted. The reference file remains byte-for-byte unchanged.

The next useful work is a fully nested evaluation of the V2 Gamma_GT calibration
and regression blend, with focused analysis of floor readings, missing rapid
tests, and high Gamma_GT errors. The tested classifier variations offer little
reliable improvement. Avoid selecting another tiny blend adjustment solely from
the public leaderboard; the private split is twice as large.
