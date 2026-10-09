# Final enhancement review

There is one candidate worth a Kaggle test: **the existing neural-network + V3 blend**. This review reproduced its predictions and saved a byte-identical verified copy at `outputs/final_review/submission_nn_v3_verified.csv`. The original candidate is `outputs/v2/submission_nn_blend_v3.csv`. No new model was trained during this audit; it reviews the newer neural-network work already present in the repository.

The user-identified reference remains `outputs/v2/submission_3seed.csv`, with a confirmed public score of **0.86380**. The candidate's Kaggle score is unknown. Its local diagnostic gain cannot be added to the public score to predict a new rank.

## Comparison

| Recipe | Smoking AUC | Gamma_GT RMSLE | Combined local score |
| --- | ---: | ---: | ---: |
| Submitted three-seed reference | 0.975459 | 0.116397 | 0.860009 |
| Five-seed V2 | 0.975424 | 0.116436 | 0.859936 |
| Neural-network blend | 0.975580 | 0.116247 | 0.860278 |
| Neural-network blend + 20% V3 | 0.975825 | 0.116004 | 0.860764 |
| Reference Smoking + blended Gamma_GT | 0.975459 | 0.116004 | 0.860399 |

The full neural-network + V3 blend improves the local combined score by **0.00075499**. Both overall target metrics improve. Adding seeds alone did not help. Keeping the original Smoking predictions loses some of the useful diversity gain, so the complete existing blend is the stronger candidate in this comparison.

## What the candidate contains

Smoking combines the five-seed V2 and neural-network ranks at 90/10, ranks that result, then blends it 80/20 with V3 ranks. These values are ranking scores within [0, 1]; they are not calibrated smoking probabilities. Ranking is appropriate for the competition's AUC metric.

Gamma_GT adds the neural network's clinical prediction to the existing group-specific regression blend. Floor readings use a truncated prior, and the final prediction is combined 80/20 with V3 in log1p space. The neural network has separate Smoking and Gamma_GT output heads and excludes both true targets from its inputs.

## Where the error remains

Missing rapid-test readings account for **712 of 9,002 rows (7.9%)**, but **38.4%** of the reference Gamma_GT squared log error. Their RMSLE falls from **0.256424** to **0.255020** in the candidate. Regular readings improve from **0.088444** to **0.088173**. Floor readings slightly worsen, from **0.205303** to **0.205889**; the candidate does not improve every subgroup.

## Stability checks and limits

We ran 600 paired bootstrap resamples of whole workplaces and another 600 of whole cartridge batches. For the complete candidate, the conditional 95% intervals for local combined-score gain were:

- Workplaces: **+0.000036 to +0.001419**.
- Cartridge batches: **+0.000215 to +0.001217**.

Both intervals are positive, supporting a trial of this candidate. They are conditional on cached predictions and omit model-selection and retraining uncertainty. They do not establish a 95% probability of beating the public or private leaderboard score.

The V2 regression stack is not fully nested: globally cross-fitted Smoking predictions enter regression folds, and second-stage cross-validation uses cached first-stage predictions. The neural-network weight was also selected using existing OOF results. The cached NPZ files do not contain full row-level training provenance. Code review, shape/order assumptions, source hashes, and reproduction of the existing CSV improve traceability but do not retroactively make the validation independent.

## File verification

The candidate passed checks for 6,399 rows, exact sample ID order, the three required columns, finite values, and bounds. Its Smoking and log-Gamma_GT values matched the reconstructed recipe to a maximum tolerance of 1e-12. The verified copy is byte-identical to the existing candidate. The successful three-seed file's SHA-256 remains `c49a7788060dd229e1c6072b463c7f90233627f9be63cb4ad59bea4b8d0bd95a`.

`outputs/final_review/audit.json` records input hashes and runtime versions. Reproduce this review with `.venv/bin/python final_ensemble_review.py`, or open `final_ensemble_review.ipynb`. No code in this review uploads a Kaggle submission.

For the last competition test, use the verified neural-network + V3 CSV and retain the confirmed three-seed submission as the fallback. Avoid further small weight searches on this reused validation set. A stronger claim would require fully nested training and a genuinely independent assessment.
