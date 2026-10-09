# DSC ML Arena — final solution

Competition: https://www.kaggle.com/competitions/dsc-modeling-hackathon-ml-arena

Two targets are predicted for each person: `Smoking` (probability) and `Gamma_GT`
(liver enzyme, scored by RMSLE). Score: `0.5 * (2 * AUC - 1) + 0.5 * (1 - RMSLE / 0.5041)`.

## Result

| | Smoking AUC | Gamma_GT RMSLE | Score |
|---|---:|---:|---:|
| Final submission (CatBoost + NN, 20% V3) — local out-of-fold | 0.97582 | 0.11600 | **0.86076** |

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
