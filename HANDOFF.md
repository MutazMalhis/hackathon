# Handoff: v4 work in progress (2026-10-10)

Read this first if you are a new Claude Code session continuing the v4 task.
The full task brief is in the user's original message. In short: raise the
competition score from the current OOF baseline (AUC 0.97582, RMSLE 0.11600,
score 0.86076), with honest dev/confirmation evaluation, new code in `v4/`,
predictions in `outputs/v4/`, a report in `reports/experiments_v4.md`, and commits to
branch `claude/dsc-hackathon-objective-d50o9l`. That branch is pushed to `origin`
(MutazMalhis/hackathon) and to `runner` (harvrennix/dsc-hackathon-v4).
Kaggle close time per the old README: **2026-10-10 14:00 Europe/Paris**.

## Done so far

* `v4/run_baseline.py` runs every README step (v2 chain and V3 chain in parallel), then
  `make_submission.py` and `v4/baseline_oof.py`. **The baseline has not been run yet.**
  On a fast machine just run `python v4/run_baseline.py` (it also writes
  `outputs/v4/baseline.npz`, `baseline_folds.csv` and `baseline_slices.csv`).
* `v4/common4.py`: canonical dev split = `StratifiedKFold(5, shuffle=True, random_state=42)`
  on Smoking; confirmation split = same with `random_state=2027`. Metric helper, slices, saving.
* `v4/worker.py` + `v4/channel.py` + `RUNNER.md`: a GitHub job queue for a remote GPU box.
  It isn't needed if you work directly on the GPU machine.
* `v4/workplace.py`: latent-sex GMM and fold-safe workplace rate encodings.

## Findings (local, LightGBM quick tests on the dev split; selection-biased, small grids)

1. **No leaks found.** IDs: train 0–9001, test 9002–15400, with no signal (AUC(ID)=0.519).
   Row order is random (lag-1 correlations ~0). No duplicates (dataset_report.md).
2. **Gamma_GT = integer × exp(ε)**, ε SD 0.004: an irreducible but negligible floor.
3. **Rapid-test (POC_GGT) noise**: integer readings; within-batch log residual SD ≈ 0.091
   (0.108 at readings 4–5, 0.091 above 100). Residuals are **not predictable** from any
   feature (|Spearman| < 0.035; no center/quarter/smoking effect). Batch offsets are random
   (SD 0.87), one center per batch, uncorrelated with batch ID.
   ⇒ calibrated rows (84%) have an RMSLE floor ≈ 0.083 (now ≈ 0.088).
4. **Workplace_ID is the big smoking signal.** The latent SD of workplace smoking rates is 0.33
   (binomial noise would give 0.14). A workplace's rate correlates 0.81 with its mean height and
   0.85 with its mean hemoglobin (there is no sex column, so this proxies sex). A GMM on body
   measurements gives "male-like" (59% smoke) vs "female-like" (4.5% smoke). Even among
   male-like rows the workplace-rate SD is 0.32, so it is a real workplace effect.
   * LightGBM smoking: base 0.95914 → + in-fold workplace target encoding 0.97495.
     A workplace × sex encoding did not help (0.97455).
   * **Transductive workplace rate** (add held-out + test rows' stage-1 predicted
     probabilities as soft counts, weight W): W=0 0.97486, W=0.5 0.97532, W=1.0 0.97536.
5. Gamma_GT information for smoking (oracle): + true log Gamma_GT adds +0.0025 AUC to LightGBM;
   with noise SD 0.09 it adds +0.0021, with SD 0.26 +0.0007.
6. **Workplace effect on Gamma_GT**: the SD of workplace mean log Gamma_GT is 0.33 vs 0.40 within.
   Feature-only LightGBM (no POC inputs), RMSLE on all rows / no-reading rows:
   * plain 0.2784 / 0.2686
   * + in-fold workplace mean of log Gamma_GT: 0.2678 / 0.2603
   * + transductive (pool includes calibrated readings of held-out and test rows, weight 0.8,
     own row excluded): 0.2659 / 0.2573
   * + also train on test rows whose reading is calibrated (pseudo-label = calibrated
     reading, weight 0.5): **0.2614 / 0.2507**. The current blend has ~0.255 on no-reading rows.

## Next steps (priority order)

1. Run `python v4/run_baseline.py`; confirm ≈0.86076 and record per-fold numbers.
2. Gamma_GT: build the feature-only prior with workplace pooling + pseudo-labelled test rows
   (finding 6) in CatBoost/LightGBM/NN; feed it into the per-group stack in place of `nop`/NN
   (this affects the "none" and "floor" groups and the calibrated posterior).
3. Smoking: add transductive workplace rates (finding 4) to the v2 CatBoost smoking features
   and to LightGBM/XGBoost GPU models; rank-blend with the baseline.
4. Assay likelihood (rounded/censored readings), as in the `improve-score` branch:
   +0.0004 there.
5. Confirm every candidate on the confirmation split (seed 2027) before promoting; write
   `reports/experiments_v4.md`; only write `outputs/final/submission.csv` if the gain clears
   the noise margin (≥ +0.002 on confirmation, with most folds winning).

## Realistic ceiling (evidence so far)

Gamma_GT: floor ≈ 0.083 on calibrated rows, ~0.2 on floor-3 rows, ~0.24 on no-reading rows
⇒ overall RMSLE ≈ 0.110–0.112 at best (now 0.116). Smoking AUC ~0.977–0.978 looks plausible.
Best total ≈ 0.865–0.868. **0.99 is not reachable**: it would need AUC ≈ 1 and RMSLE ≈ 0.01,
but the assay noise alone keeps RMSLE above ~0.08.
