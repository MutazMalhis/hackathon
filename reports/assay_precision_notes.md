# Assay precision improvement

The new experimental candidate is `outputs/calibration_precision_push/submission_assay_precision.csv`. It combines 70% of the previous neural-network + V3 Gamma_GT prediction with 30% of a newly fitted assay model, in log1p space. Smoking predictions are exactly unchanged from that previous candidate.

## Measured result

| Local diagnostic | Previous NN + V3 candidate | New assay candidate |
| --- | ---: | ---: |
| Smoking AUC | 0.97582479 | 0.97582479 |
| Gamma_GT RMSLE | 0.11600413 | 0.11560674 |
| Combined score | 0.86076415 | 0.86115831 |

The local gain is **+0.00039416**. The user-confirmed three-seed submission has a different local score, 0.86000919, and a confirmed Kaggle score of **0.86380**. The latest screenshot shows the leader at **0.86430**. Neither the NN + V3 candidate nor this new assay candidate has a confirmed Kaggle score in this chat. Local gains cannot simply be added to a public score.

## What changed

The previous calibration treated most integer rapid-test readings approximately as exact measurements. The new model treats a reading of 10 as evidence for an underlying measurement between 9.5 and 10.5. A minimum reading of 3 is treated as left-censored below 3.5. This is a modeling hypothesis inferred from the dataset, not a confirmed instrument specification.

The model fits laboratory batch offsets and assay noise by maximizing a rounded/censored log-normal likelihood. Batch offsets receive weak regularization, and the posterior combines the assay information with an existing clinical Gamma_GT prior. It accounts approximately for batch-offset uncertainty using batch counts. Missing or unknown assay information falls back to the clinical prior. The clinical prior standard deviation is fixed at 0.26; the fitted assay standard deviation is approximately 0.0839 on the logarithmic scale.

We also found an inconsistency in V2's weighted slope calculation: zero-weight floor readings still affect unweighted batch centering. Correcting that calculation alone did not improve held-out calibration error, so it was not promoted as an isolated change. The useful gain comes from the broader assay likelihood and posterior blend.

## Validation procedure and limits

Assay calibration was fitted independently in five folds for each of three seeds: 42, 27091, and 90317. Evaluation rows' Gamma_GT labels did not enter that fold's assay calibration. All three split assignments produced a positive conditional gain over the existing NN + V3 candidate. The final OOF diagnostic averages the three posterior predictions. Test calibration is fitted on all training rows.

Weights 0.15 and 0.30 were inspected on the first two split assignments. The 0.30 weight was selected after observing those results and fixed before checking the third assignment. These assignments reuse the same dataset; the third is a stability check, not an independent test. We did not increase the weight after its result.

The clinical prior and comparison predictions are cached V2 outputs. Their original stacking is not fully nested. Consequently the overall blend metrics are conditional diagnostics, even though the new assay calibration itself excludes evaluation labels. The 1,000 workplace and 1,000 cartridge-batch bootstrap resamples also condition on reused predictions and exclude model-selection uncertainty.

Conditional 95% gain intervals were **+0.000149 to +0.000640** by workplace and **+0.000212 to +0.000684** by cartridge batch. These support a Kaggle trial; they do not guarantee a public or private leaderboard improvement.

## Other experiments in this round

- Additive spline/Ridge clinical regression for missing rapid readings: combined local score change −0.00004682; rejected.
- RBF support-vector clinical regression for missing readings: −0.00020382; rejected.
- Regularized Smoking adjustments by center, quarter, and workplace: no useful development gain, with the selected recipe losing AUC on reserved rows; rejected.

## Verification and reproduction

All **20 unit tests passed**. New tests cover synthetic recovery of batch offsets, independence from held-out labels, unchanged fallback predictions for unknown/missing readings, extreme interval predictions, and exclusion of floor labels from the corrected weighted slope.

The submission has **6,399 rows**, exact sample ID order, required columns, finite values, Smoking bounds [0,1], and Gamma_GT bounds [1,1000]. Smoking values match the previous candidate exactly after CSV round-trip. The successful three-seed file was verified unchanged. Source, data, component, and submission hashes are recorded in the output manifests.

Run `.venv/bin/python calibration_precision_push.py`, followed by `.venv/bin/python assemble_precision_candidate.py`, or use `assay_precision.ipynb`. Notebook retraining is disabled by default. No Kaggle submission has been uploaded by this workflow.
