# DSC ML Arena

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
