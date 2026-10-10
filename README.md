# DSC ML Arena

**Best public submission (0.86397): see [REPRODUCE_BEST_PUBLIC.md](REPRODUCE_BEST_PUBLIC.md), or run `bash scripts/reproduce_best_public.sh`.**

Competition: https://www.kaggle.com/competitions/dsc-modeling-hackathon-ml-arena

Place the competition's `train.csv`, `test.csv`, and `sample_submission.csv` in `data/` after joining and accepting the rules in Kaggle. External data, including the original source dataset, is prohibited.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
```

The baseline trains separate CatBoost models for smoking and log1p gamma-GT with five stratified folds. Predictions are averaged across folds (gamma-GT in log space). Outputs include models, out-of-fold predictions, metric details, data hashes, and `outputs/submission.csv` aligned to sample IDs.

Score: `0.5 * (2 * AUC - 1) + 0.5 * (1 - RMSLE / 0.5041)`. Gamma-GT predictions are clipped to [1, 1000]. Neither target nor ID is used as a predictor.

Random stratification is an initial assumption. Once data is available, inspect duplicate records and overlap of workplaces, assay plates, and cartridge batches between splits. Compare grouped validation where the intended generalization requires it.

Next experiments: urine concentration normalization, batch effects, rapid-test calibration, and cross-target stacking. Cross-target stacking requires predictions generated without access to the predicted row's labels, including within each outer validation fold. No in-sample target predictions should enter validation features.

The listed Kaggle close time is October 10, 2026 at 14:00 Europe/Paris. Some overview judging and timeline text contains placeholders; confirm presentation requirements with the organizers.

## Jupyter notebook

Open `hackathon_analysis.ipynb` for the complete exploration, plots, baseline training, batch calibration, and submission checks. Select the project `.venv` Python interpreter and run cells in order. Notebook outputs are saved under `outputs/notebook/`; the notebook contains all modeling code and does not depend on importing the project scripts.

For a JupyterLab interface, install `jupyterlab` in `.venv` and run `.venv/bin/jupyter lab hackathon_analysis.ipynb`. The notebook itself does not upload any submission.

## Stronger validation and selected candidate

Run `.venv/bin/python experiments.py` for development-only selection, an outer holdout check with inner early stopping, error analysis, and final five-fold predictions. Run `.venv/bin/python robustness.py` for group-held-out diagnostics. Results are under `outputs/plan_v1/`; see `experiment_notes.md` there. The appended notebook contains the same workflow and saved result displays. Its retraining switches default to false.

## Repository contents and notebook outputs

The competition CSVs, trained models, predictions, and local outputs are excluded from Git. Obtain data through Kaggle after accepting the competition rules. Aggregate reports are under `reports/`.

The local notebook can retain its executed outputs, but the Git clean filter removes outputs from committed notebook content. After cloning, enable the filter before committing executed notebooks:

```sh
git config filter.notebook-clean.clean 'python3 scripts/clean_notebook.py'
git config filter.notebook-clean.required true
```

To inspect previous saved experiment results, run the workflows first. The appended notebook's `RUN_EXPERIMENTS` and `RUN_GROUP_DIAGNOSTICS` switches can be enabled for reproduction when `outputs/plan_v1/` is absent.

## Development-only model improvements

Run `.venv/bin/python improvements.py` to compare richer urine plate features,
clinical Gamma_GT regressors, and complementary models. The workflow selects on
the original development rows, checks stability with another fold assignment,
and generates `outputs/plan_v2/submission.csv` only if its promotion gate passes.
The previously evaluated holdout is excluded from these comparisons. Existing
fold predictions are cached; a changed training recipe requires a new output
directory. Selection results and final CV remain diagnostic estimates, not
independent test scores.

Open `model_improvements.ipynb` for the full Python implementation and saved
comparisons. Its `RUN_RETRAIN` switch defaults to false. Run
`.venv/bin/python -m unittest discover -s tests` for target isolation,
training-only plate normalization, and fallback checks. These checks require the
downloaded competition data.

## Calibration refinement and cross-target experiment

The user reported a Kaggle score of **0.85329** for the v2 candidate. Run
`.venv/bin/python refine.py` after the earlier workflows to reproduce the
calibration and nested cross-target comparison. The selected calibration treats
minimum rapid-test readings as a detection-floor hypothesis, excludes them from
batch-offset estimation, and blends other calibrated readings according to
estimated precision. Cross-target stacking was tested but did not pass the
promotion threshold; the clinical regressor and Smoking predictions are retained.

The new candidate is `outputs/plan_v3/submission_v3.csv`. Its local diagnostic
score is 0.85389, versus 0.84841 for v2; the new Kaggle score is unknown. Open
`calibration_refinement.ipynb` for the new implementation and saved results, and
see `reports/refinement_notes.md` for assumptions, validation, and limitations.
Notebook retraining is disabled by default. No workflow uploads submissions.


## Push beyond the confirmed leaderboard result

The user identified `outputs/v2/submission_3seed.csv` as the file scoring
**0.86380** on Kaggle (second place in the supplied screenshot). Keep that CSV
as the reference. The V2 OOF diagnostic score is **0.86001**; it is a different
measurement from the public leaderboard score.

Open `leaderboard_push.ipynb` for the Python experiment controls, implementations,
saved comparisons, and submission checks. This notebook uses the project scripts;
`RUN_RETRAIN` defaults to false. The experiments are:

```sh
.venv/bin/python smoking_push.py --phase all
.venv/bin/python supervised_smoking.py --phase all
.venv/bin/python xgboost_push.py --phase all
.venv/bin/python ordered_smoking_push.py --phase all
.venv/bin/python gamma_bias_push.py
```

Each family has its own output directory and cached predictions. Smoking
candidates preserve the reference Gamma_GT predictions. Development selection,
a second fold assignment, and a full-data diagnostic comparison must pass before
a new submission is written. These checks reuse development rows and do not
establish a new Kaggle score. See `reports/leaderboard_push_notes.md` for results.

XGBoost is an optional comparison dependency. On Apple Silicon, its OpenMP
runtime may require `brew install libomp`. No workflow uploads submissions.


## Alternative model families

Open `alternative_models.ipynb` to review LightGBM, Extra Trees, and RBF SVM
comparisons. Run `.venv/bin/python alternative_models.py --phase all` for
Smoking, then `.venv/bin/python alternative_gamma.py` for Gamma_GT.
`.venv/bin/python finalize_alternative_gamma.py` performs final evaluation only
when Gamma_GT confirmation passes, and compares it with the submitted V2 OOF
before creating a candidate. The current alternatives did not clear confirmation;
no new submission was generated. Results are in
`reports/alternative_model_notes.md`. These are local diagnostics.


## Final review of the newer neural-network blend

`final_ensemble_review.ipynb` audits the existing neural-network + V3 submission.
Run `.venv/bin/python final_ensemble_review.py` to reproduce its CSV, compare
saved OOF predictions, and run conditional grouped bootstrap diagnostics.
The verified copy is `outputs/final_review/submission_nn_v3_verified.csv`.
Its local diagnostic score is 0.860764 versus 0.860009 for the submitted
three-seed reference; its Kaggle score is unknown. The comparison inherits
V2's nesting and model-selection limitations. See
`reports/final_enhancement_review.md` for the evidence and interpretation.


## Rounded and censored assay candidate

`assay_precision.ipynb` contains the newest calibration experiment and its saved
results. Run `.venv/bin/python calibration_precision_push.py` and then
`.venv/bin/python assemble_precision_candidate.py` to reproduce the candidate
`outputs/calibration_precision_push/submission_assay_precision.csv`.
Its local diagnostic score is 0.861158, versus 0.860764 for the previous NN + V3
blend. Its public score is unknown, and the evaluation inherits cached V2
validation limitations. See `reports/assay_precision_notes.md`.
