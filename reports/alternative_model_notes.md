# Alternative model comparison

Reference: `outputs/v2/submission_3seed.csv`, scoring **0.86380** according to the user's screenshot and file identification. The leader in that screenshot scored 0.86424: the gap is **0.00044**. No live leaderboard lookup or new upload was made.

## Models actually trained

Smoking: LightGBM with 15 and 31 leaves, Extra Trees, and a support-vector classifier with an RBF kernel. Gamma_GT: LightGBM with 9 and 31 leaves and Extra Trees regression, alongside a CatBoost clinical baseline. Three development folds were used. We evaluated each alternative alone and 15%, 30%, and 50% contributions to the corresponding CatBoost prediction.

LightGBM tests whether a different boosting algorithm adds information. Extra Trees supplies randomized tree predictions; the SVM supplies a different, smooth nonlinear decision boundary. These were modeling hypotheses; their value was determined by the measured comparisons. Implementations follow the official [LightGBM classifier API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) and [Extra Trees API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

## Results

| Target / comparison | Initial development result | Second fold assignment | Decision |
| --- | --- | --- | --- |
| Smoking: 85% CatBoost + 15% LightGBM, 31 leaves | AUC 0.97257519, versus 0.97203186; gain +0.00054333 | AUC gain +0.00021156; two of three fold wins | Below +0.0003 confirmation gate |
| Smoking: Extra Trees alone | AUC 0.95288508; blends also underperformed | Not selected | Rejected |
| Smoking: RBF SVM alone | AUC 0.94625655; blends also underperformed | Not selected | Rejected |
| Gamma_GT: 85% calibrated CatBoost + 15% calibrated LightGBM, 31 leaves | RMSLE 0.12422420; combined-score gain +0.00011234 | Combined-score gain +0.00010201; two of three fold wins | Below +0.0003 confirmation gate |

The Gamma_GT baseline here is a clinical CatBoost model plus the existing floor-aware rapid-test calibration. It is **not** the submitted V2 ensemble; these Gamma_GT numbers do not establish an improvement over the submitted file. No candidate cleared confirmation, so the final three-seed stage was not run and no new submission was generated.

The LightGBM blends show modest diversity gains, but their gains weakened with a different split. We did not relax a predefined gate after seeing these results.

## Validation and reproducibility

Selection uses the existing 7,201 development rows. The previously reserved 1,801 rows are excluded from this round's selection and confirmation. The second fold assignment reuses development rows and checks stability; it is not an independent test.

Categorical vocabularies, numeric imputation, and scaling are fitted on each fitting partition. Unseen categories are handled without redefining the fitting vocabulary. The Smoking inputs inherit the V2 reference's calibration and cached unsupervised plate features, with previously documented validation limitations. Gamma_GT regressors exclude ID, both true targets, the raw rapid-test value, and cartridge ID; calibration is fitted using the fitting partition's Gamma_GT labels only. Gamma_GT urine plate means use fitting features only. The fixed calibration recipe was inherited from earlier work and was not searched in this round.

Cached baseline Smoking predictions were reused only after checking their source/data hashes and fold row IDs. Each new model has a separate output directory, frozen configuration, recipe identity, and saved fold predictions/model artifacts. Python dependencies include LightGBM 4.7.0. The SVM probability API emits a deprecation warning in scikit-learn 1.9; it ran successfully, but a future migration should use the recommended calibration wrapper and a new recipe/cache directory.

All **16 unit tests passed**, including model reload equivalence with missing numeric values and unseen categories, plus Gamma_GT held-out target isolation. Dependency checks passed. The successful reference CSV was verified byte-for-byte unchanged.

Open `alternative_models.ipynb` for the Python implementations, optional training controls, saved results, and reference checks. The notebook uses project scripts and does not submit files to Kaggle.
