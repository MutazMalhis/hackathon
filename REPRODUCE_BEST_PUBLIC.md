# Reproducing the best public submission (0.86397)

This guide rebuilds `outputs/calibration_precision_push/submission_assay_precision.csv`, our best public-leaderboard submission for the [DSC Modeling Hackathon – ML Arena](https://www.kaggle.com/competitions/dsc-modeling-hackathon-ml-arena) (public score **0.86397**; local cross-validation score **0.8612**).

The fast path is one command:

```sh
bash scripts/reproduce_best_public.sh
```

The script stops at the first error, and at the end it prints the file's SHA-256 next to the submitted file's: `ad421dcf185ccdaa82aad524efde61deb7bb0a29181fd332077c3b003c56ca0e`.

## 1. Setup

1. Join the competition on Kaggle and accept its rules. External data is not allowed.
2. Download `train.csv`, `test.csv` and `sample_submission.csv` into `data/`. Competition data, models and predictions are kept out of Git under the competition rules (see `.gitignore`).
3. Create the environment from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The submitted file was built with Python 3.14.7, catboost 1.2.10, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1 and torch 2.14.1 on macOS (Apple silicon, CPU). Run every command from the repository root: the scripts use paths relative to it.

## 2. What the submission is

It blends four model lines, each correcting the measurement problems we found in the data before modeling:

| Part | What it does | Weight in the final file |
| --- | --- | --- |
| Final pipeline (`v2/pipeline.py`) | Fixes center E's creatinine units (÷ 88.4), corrects rapid-test cartridge batches and urine plates, handles the rapid-test detection floor (a reading of 3 means "3 or less"), then trains CatBoost models and a per-group stack | Core of both targets |
| Neural network (`v2/nn.py`) | Multi-task network with embeddings for workplace, plate, center and quarter | 10% of smoking ranks; one input of the gamma-GT stack |
| Plan v3 (`refine.py`) | The earlier model line, kept for diversity | 20% of both targets |
| Rapid-test calibration (`calibration_precision_push.py`) | Treats each rapid reading as an interval (a 10 means 9.5–10.5) and the floor as an upper bound | 30% of gamma-GT only |

Smoking is blended by rank (AUC only uses ranks); gamma-GT is blended on the log1p scale.

## 3. Step by step

| Step | Command | Writes | Time on our machine |
| --- | --- | --- | --- |
| 1. Plans v1–v3 | `.venv/bin/python experiments.py`<br>`.venv/bin/python improvements.py --phase all`<br>`.venv/bin/python refine.py --phase all` | `outputs/plan_v1/`, `outputs/plan_v2/`, `outputs/plan_v3/` | longest step, tens of minutes |
| 2. First final-pipeline version (3 seeds) | `git show c670e5d:v2/pipeline.py > build/pipeline_3seed.py`<br>`.venv/bin/python build/pipeline_3seed.py 3`<br>then copy `outputs/v2/submission.csv` → `submission_3seed.csv` and `oof.csv` → `oof_3seed.csv` | `outputs/v2/*_3seed.csv` | about 4 minutes |
| 3. Final pipeline (5 seeds + refit) | `.venv/bin/python v2/pipeline.py 5 0` | `outputs/v2/components.npz`, `oof.csv`, `cv.json` | about 10 minutes |
| 4. Neural network | `.venv/bin/python v2/nn.py 10` | `outputs/v2/nn_preds.npz` (and `v2/refined_cache.csv`) | about 1 minute |
| 5. Pipeline + network + plan v3 blend | `.venv/bin/python v2/blend_v3.py` | `outputs/v2/submission_nn_blend_v3.csv` | seconds |
| 6. Audit of that blend | `.venv/bin/python final_ensemble_review.py` | `outputs/final_review/` | about 1 minute |
| 7. Rapid-test calibration | `.venv/bin/python calibration_precision_push.py` | `outputs/calibration_precision_push/` components | a few minutes |
| 8. Final assembly | `.venv/bin/python assemble_precision_candidate.py` | `outputs/calibration_precision_push/submission_assay_precision.csv` | seconds |

Two details matter:

- **Step 3 must pass `0` as the second argument.** It turns off the later test-row training for the feature-only gamma-GT model. That change came after this submission and produces `outputs/v2/submission_final.csv` instead (local 0.8615, public 0.86378).
- **Step 2 uses the code as it was at commit `c670e5d`.** Later steps compare against that first version, so it is rebuilt from that commit. The shared modules it imports (`v2/common.py`, `v2/plate_refine.py`) have not changed since.

## 4. Checks built into the scripts

- Every submission is checked for 6,399 rows, the sample's ID order, the columns `ID,Smoking,Gamma_GT`, no missing values, smoking in [0, 1] and gamma-GT in [1, 1000].
- `final_ensemble_review.py` and `calibration_precision_push.py` record SHA-256 fingerprints of their inputs. `improvements.py` and `refine.py` refuse to overwrite an output folder made with a different recipe ("Recipe changed"). Delete that folder to rebuild it.
- Unit tests for the earlier lines are in `tests/`: `.venv/bin/python -m pytest tests`.

**Exactness.** On our machine, steps 5–8 rebuild the submitted file byte for byte from the saved model outputs (verified 10 October 2026). A full retrain on another machine or library version can differ in the last decimal places. `final_ensemble_review.py` then stops with a fingerprint mismatch on `outputs/v2/submission_3seed.csv`, and the final file will match the submitted one closely but not byte for byte.

## 5. Local scores (5-fold cross-validation on 9,002 training rows)

| Stage | Smoking AUC | Gamma-GT RMSLE | Score |
| --- | --- | --- | --- |
| Final pipeline (step 3) | 0.9754 | 0.1164 | 0.8599 |
| + neural network + 20% plan v3 (step 5) | 0.9758 | 0.1160 | 0.8608 |
| + 30% rapid-test calibration (step 8) | 0.9758 | 0.1156 | 0.8612 |

Score = 0.5 × (2 × AUC − 1) + 0.5 × (1 − RMSLE / 0.5041). The public leaderboard uses about 33% of the test set and moves about ±0.004 by chance; the final ranking uses the other 67%.
