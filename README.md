# DSC ML Arena — final solution

**Best public submission (0.86397): see [REPRODUCE_BEST_PUBLIC.md](REPRODUCE_BEST_PUBLIC.md), or run `bash scripts/reproduce_best_public.sh`.**

Competition: https://www.kaggle.com/competitions/dsc-modeling-hackathon-ml-arena

Two targets are predicted for each person: `Smoking` (probability) and `Gamma_GT`
(liver enzyme, scored by RMSLE). Score: `0.5 * (2 * AUC - 1) + 0.5 * (1 - RMSLE / 0.5041)`.

## Result

| | Smoking AUC | Gamma_GT RMSLE | Score |
|---|---:|---:|---:|
| Final submission (CatBoost + NN, 20% V3) — local out-of-fold | 0.97582 | 0.11600 | **0.86076** |
| Best public submission (above + 30% rapid-test calibration) — local out-of-fold; public 0.86397 | 0.97582 | 0.11561 | **0.86116** |

Local scores are out-of-fold estimates on the training data, not leaderboard scores.

## How it works

1. **Feature cache** (`v2/plate_refine.py`) — urine cotinine and EtG are divided by
   creatinine and corrected for assay-plate offsets, adjusted for covariates.
   Uses features only, never targets.
2. **CatBoost pipeline** (`v2/pipeline.py`, 5 seeds) — inside the same 5-fold split:
   - a Smoking classifier on plate-centred urine markers and calibrated Gamma_GT;
   - a Gamma_GT model with all features, including a censoring-aware calibration of
     the rapid test (`POC_GGT`) per cartridge batch;
   - a Gamma_GT model without rapid-test inputs, for rows with no usable reading.
3. **Neural network** (`v2/nn.py`, 3 seeds) — one multi-task network with
   categorical embeddings predicts Smoking and feature-only Gamma_GT.
4. **V3 candidate** (`experiments.py` → `improvements.py` → `refine.py`) — an
   independent pipeline with engineered plate features, clinical Gamma_GT
   regressors, and detection-floor calibration. Each step selects on development
   folds and confirms on a second split before producing output.
5. **Final blend** (`make_submission.py`):
   - Smoking: rank average, 90% CatBoost + 10% NN, then 80% of that + 20% V3.
   - Gamma_GT: a linear stack per reading group (calibrated reading / reading at the
     detection floor of 3 / no reading) over the CatBoost and NN predictions,
     then 80% of that + 20% V3, in log1p space.

## Reproduce

Place the competition's `train.csv`, `test.csv` and `sample_submission.csv` in `data/`
(download from Kaggle after accepting the rules; external data is not allowed).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the steps in order. CatBoost and PyTorch training take a while; outputs go to `outputs/`.

```sh
.venv/bin/python v2/plate_refine.py
.venv/bin/python v2/pipeline.py 5
.venv/bin/python v2/nn.py 3
.venv/bin/python experiments.py
.venv/bin/python improvements.py --phase all
.venv/bin/python refine.py --phase all
.venv/bin/python make_submission.py
```

The submission is written to `outputs/final/submission.csv`.

```sh
.venv/bin/python -m unittest discover -s tests
```

runs the unit tests (they need the competition data).

## Exploration

`hackathon_analysis.ipynb` and `reports/dataset_report.md` contain the data
exploration, baseline and batch-calibration analysis. The notebook is
self-contained and does not import the scripts above.

## Notebook outputs in Git

A clean filter strips notebook outputs from commits. After cloning, enable it:

```sh
git config filter.notebook-clean.clean 'python3 scripts/clean_notebook.py'
git config filter.notebook-clean.required true
```
## Experiment history

The sections below record the development experiments behind the final files, in the order they were run.

### Development-only model improvements

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

### Calibration refinement and cross-target experiment

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


### Push beyond the confirmed leaderboard result

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


### Alternative model families

Open `alternative_models.ipynb` to review LightGBM, Extra Trees, and RBF SVM
comparisons. Run `.venv/bin/python alternative_models.py --phase all` for
Smoking, then `.venv/bin/python alternative_gamma.py` for Gamma_GT.
`.venv/bin/python finalize_alternative_gamma.py` performs final evaluation only
when Gamma_GT confirmation passes, and compares it with the submitted V2 OOF
before creating a candidate. The current alternatives did not clear confirmation;
no new submission was generated. Results are in
`reports/alternative_model_notes.md`. These are local diagnostics.


### Final review of the newer neural-network blend

`final_ensemble_review.ipynb` audits the existing neural-network + V3 submission.
Run `.venv/bin/python final_ensemble_review.py` to reproduce its CSV, compare
saved OOF predictions, and run conditional grouped bootstrap diagnostics.
The verified copy is `outputs/final_review/submission_nn_v3_verified.csv`.
Its local diagnostic score is 0.860764 versus 0.860009 for the submitted
three-seed reference; its Kaggle score is unknown. The comparison inherits
V2's nesting and model-selection limitations. See
`reports/final_enhancement_review.md` for the evidence and interpretation.


### Rounded and censored assay candidate

`assay_precision.ipynb` contains the newest calibration experiment and its saved
results. Run `.venv/bin/python calibration_precision_push.py` and then
`.venv/bin/python assemble_precision_candidate.py` to reproduce the candidate
`outputs/calibration_precision_push/submission_assay_precision.csv`.
Its local diagnostic score is 0.861158, versus 0.860764 for the previous NN + V3
blend. Its public score is 0.86397, our best; the evaluation inherits cached V2
validation limitations. Full rebuild: [REPRODUCE_BEST_PUBLIC.md](REPRODUCE_BEST_PUBLIC.md). See `reports/assay_precision_notes.md`.
