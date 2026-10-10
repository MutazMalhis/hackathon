**DSC Modeling Hackathon — simplified report from the start**

We built a Python project that predicts Smoking and Gamma_GT from the competition dataset. We downloaded and explored the data, created Jupyter notebooks, trained a baseline, discovered measurement problems, improved the models, produced checked submission files, and reviewed a separate V2 implementation. The user reported a Kaggle score of **0.85329**. Newer local results are promising, but they are not confirmed Kaggle scores.

**1. Preparing the project**

We started by adapting the ML Engineer and skills-ml repository guidance for Codex. The ML Engineer guidance helped organize training, evaluation, and reproducibility. The skills-ml project concerns workforce/job skills and was not used as a prediction library for this competition.

We created a Python virtual environment and used pandas and NumPy for data analysis, Matplotlib for plots, scikit-learn for splitting and evaluation, and CatBoost for the main models. You requested Jupyter notebooks, so we made the exploration and modeling work available as notebook code.

**2. Understanding the challenge**

Each test row needs two predictions: the probability that the person is a smoker, and a positive numerical estimate of Gamma_GT. The output must contain the columns `ID`, `Smoking`, and `Gamma_GT`, in the same row order as the sample submission.

The competition combines Smoking AUC and Gamma_GT RMSLE. Higher AUC means better ranking of smokers above nonsmokers. Lower RMSLE means smaller Gamma_GT prediction errors measured on a logarithmic scale. Improving either task increases the combined score.

**3. Downloading and checking the data**

After you signed in to Kaggle, we downloaded the competition ZIP and extracted its three CSVs.

| File | Rows | Purpose |
| --- | ---: | --- |
| train.csv | 9,002 | Examples with both answers available |
| test.csv | 6,399 | Examples for which we must predict both answers |
| sample_submission.csv | 6,399 | Required output layout and ID order |

The training data has 38 predictors, an ID, and the two targets. Features include body measurements, blood tests, lifestyle information, urine tests, screening centers, workplaces, and laboratory batch identifiers.

We found no missing target values, no duplicate IDs, no ID overlap between training and test data, and no exact duplicate feature rows within or across the two splits. Those checks do not rule out near duplicates or related participants.

**4. Exploring the dataset**

About 36.34% of training rows are smokers. Gamma_GT values have a long right tail: most are relatively low, but a few are much higher. Its median is about 20.84, and its maximum is about 485.66. We therefore trained regression models on `log1p(Gamma_GT)` and converted predictions back afterward.

Missing values are concentrated in particular features. Exhaled CO is missing in about 23.37% of training rows, liver echo grade in 16.81%, alcohol consumption in 13.86%, and the rapid Gamma_GT reading in 7.91%. CO missingness is especially concentrated at screening centers C and F, indicating a collection pattern.

We retained numeric missing values for CatBoost and gave missing categorical values a separate category. Workplace, urine plate, and cartridge batch numbers were treated as group identifiers rather than ordinary continuous measurements.

Hemoglobin, height, weight, and triglycerides showed useful individual Smoking signals. ALT, body measurements, and other blood markers were associated with Gamma_GT. These are descriptive associations in this competition dataset, not established causes.

Most workplaces and laboratory groups appear in both training and test data. That makes learning corrections for shared groups useful, but it does not demonstrate good performance on entirely new groups.

**5. Building the first baseline**

We trained separate CatBoost models: a classifier for Smoking and a regressor for log-transformed Gamma_GT. CatBoost was useful because the dataset contains nonlinear relationships, categories, and missing values.

We initially used five folds. Each fold predicts rows excluded from its training set, giving out-of-fold, or OOF, predictions. ID and the true targets were excluded from the ordinary predictors. The initial combined local score was 0.72899, giving us a reference for later improvements.

**6. Discovering the cartridge batch problem**

The biggest early discovery was that the rapid Gamma_GT test, POC_GGT, changes scale across cartridge batches. The same underlying Gamma_GT value can produce different rapid readings in different batches.

We estimated a correction for each batch from fitting rows and applied it to validation and test readings. Rows with missing readings or unavailable batch corrections used the regression model instead. We calculated validation corrections without using the held-out row's answer.

This reduced Gamma_GT RMSLE from 0.22050 to 0.13549 and raised the combined local score from 0.72899 to 0.81332. Smoking predictions were unchanged in that experiment.

**7. Improving features and validation**

We added log-transformed urine biomarkers, concentration ratios relative to urine creatinine, and features centered by urine plate. These transformations help the model distinguish participant differences from dilution and laboratory plate differences.

We also made validation more disciplined: 7,201 rows were used for development, and 1,801 were set aside for an outer check. An additional inner split selected CatBoost's tree count before refitting on each fitting partition.

The selected method combined calibrated rapid readings with regression: 90% calibration and 10% regression in log space. Its outer-check score was 0.84623, and its final five-fold diagnostic score was 0.83957. These are different evaluations and should not be mixed.

The whole dataset had already been explored before the outer split was created. It was therefore not historically untouched. Later comparisons excluded these outer-check rows from method selection, but development data were reused across iterations.

We also checked new workplaces and new cartridge batches. Performance dropped sharply when all evaluation batches were unseen, confirming that batch calibration depends on shared batches.

**8. Building the plan_v2 candidate**

We tested richer urine plate normalization, clinical ratios and log transforms, different CatBoost depths, Ridge regression, and histogram gradient boosting. The selected Smoking predictor averaged three CatBoost models. Gamma_GT used a shallower clinical regressor without raw cartridge inputs, together with the existing batch calibration.

Ridge and histogram boosting did not improve the selected final method. Removing cartridge inputs worsened overall uncalibrated regression but helped the fallback cases. A second development fold assignment showed combined-score gains in all three folds.

The final diagnostic score improved from 0.83957 to 0.84841. Fallback errors and errors on high Gamma_GT values also decreased.

**9. Building the plan_v3 candidate**

We noticed that POC_GGT readings are integers with a minimum of 3. We tested the hypothesis that a reading of 3 may represent a detection floor, rather than an exact measurement.

The selected correction excludes these floor readings from estimating batch offsets. It treats a floor observation as an upper-bound clue and combines it with the clinical prediction. Other readings use a blend adjusted for rounding uncertainty and the amount of available batch information. This is a useful modeling hypothesis supported by development results; the actual assay mechanism has not been confirmed.

We also tested predicted Smoking probability as a Gamma_GT feature. We used nested cross-fitting so each training probability excluded its own row's labels. Pure stacking worsened the final calibrated predictions; a 50/50 blend improved the development score by only about 0.00007, below our promotion threshold. We did not include it in v3.

V3 kept the Smoking predictions and improved Gamma_GT RMSLE from 0.12461 to 0.11909. Its combined diagnostic score became 0.85389. On the 300 training rows with readings of 3, RMSLE fell from 0.27483 to 0.21405.

All 11 unit tests passed, and audits checked 18 nested classifier folds for label isolation. The new submission file passed checks for 6,399 rows, correct columns and ID order, finite predictions, Smoking bounds, and Gamma_GT bounds.

**10. Reviewing the separate v2 folder**

The folder `v2/` is a separate implementation from our `outputs/plan_v2/` workflow. We reviewed it and independently reproduced its saved local score of 0.86001.

It adds calibrated Gamma_GT information to the Smoking model, more advanced urine plate normalization, a screening-center urine creatinine scaling correction, separate learned regression blends for different reading groups, and three random seeds. Its submission CSV passes the format and range checks.

Its validation needs further work: globally generated Smoking predictions feed regression models using different folds, and the final blending evaluation is not fully nested. That can make the local score optimistic. Its features are promising, but we should test them under the stricter evaluation procedure before treating it as the better model.

**11. Results recorded so far**

| Method | Local Smoking AUC | Local Gamma_GT RMSLE | Local combined score |
| --- | ---: | ---: | ---: |
| Initial baseline | 0.94770 | 0.22050 | 0.72899 |
| Initial batch calibration | 0.94770 | 0.13549 | 0.81332 |
| plan_v1 feature and calibration workflow | 0.96555 | 0.12701 | 0.83957 |
| plan_v2 richer features and ensemble | 0.97201 | 0.12461 | 0.84841 |
| plan_v3 refined calibration | 0.97201 | 0.11909 | 0.85389 |
| Separate v2 folder, validation caveats | 0.97546 | 0.11640 | 0.86001 |

The early exploratory scores, later diagnostic scores, and separate V2 score use different procedures. This table records project history; it is not a comparison on one untouched test set.

The user reported a real Kaggle score of **0.85329**. The exact submitted CSV associated with that score remains unconfirmed in this conversation. None of the newer local scores should be presented as a confirmed Kaggle score.

**12. What we delivered and what remains**

We produced the Python scripts, exploration report, experiment reports, plots, three Jupyter notebooks, trained models, saved fold predictions, and checked submission CSVs. The initial project was published at https://github.com/MutazMalhis/hackathon. Competition data, models, and prediction outputs were excluded from Git. Some later improvements remain local and need synchronization if they are to be published.

The notebooks are `hackathon_analysis.ipynb`, `model_improvements.ipynb`, and `calibration_refinement.ipynb`. The latest candidate from our refinement workflow is `outputs/plan_v3/submission_v3.csv`; the separate V2 candidate is `outputs/v2/submission.csv`.

The next priorities are to identify which file produced the reported Kaggle score, evaluate the separate V2 features with nested validation, compare promising candidates fairly, and obtain a Kaggle result for the next selected submission. We also need to prepare the final presentation and verify that the chosen solution can be reproduced from downloaded competition files.

The main lesson from the work is that understanding the measurements mattered as much as choosing a model. Correcting laboratory batch effects, normalizing urine measurements, and handling low rapid-test readings produced the strongest gains.


**13. Confirmed leaderboard update**

You identified `outputs/v2/submission_3seed.csv` as the submitted file that scored
**0.86380**, placing XGEngineers second in your screenshot. The leader was at
0.86424, a gap of 0.00044. This confirms the result of the separate V2 three-seed
pipeline. The earlier 0.85329 remains a historical result whose file association
was not confirmed.

The public leaderboard uses approximately 33% of the test rows; the final ranking
uses the other 67%. We preserved the successful CSV and tested additional
feature corrections, cotinine normalization, XGBoost, CatBoost variants, and
Gamma_GT bias correction. The detailed results are recorded in
`reports/leaderboard_push_notes.md`; `leaderboard_push.ipynb` contains the new
Python workflow. A local improvement is a reason to test a candidate, not a
confirmed increase in the Kaggle score.
