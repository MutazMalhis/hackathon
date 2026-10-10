#!/usr/bin/env bash
# Rebuild outputs/calibration_precision_push/submission_assay_precision.csv,
# the best public-leaderboard submission (0.86397), from the three competition CSVs in data/.
# Run from the repository root:  bash scripts/reproduce_best_public.sh
# See REPRODUCE_BEST_PUBLIC.md for what each step does and how long it takes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=".venv/bin/python"

for f in data/train.csv data/test.csv data/sample_submission.csv; do
  [[ -f $f ]] || { echo "Missing $f: download the competition data into data/ first." >&2; exit 1; }
done
mkdir -p build

echo "== 1/8 Plans v1, v2 and v3 (earlier model line)"
$PY experiments.py
$PY improvements.py --phase all
$PY refine.py --phase all

echo "== 2/8 First final-pipeline version: 3 seeds, code as of commit c670e5d"
git show c670e5d:v2/pipeline.py > build/pipeline_3seed.py
$PY build/pipeline_3seed.py 3
cp outputs/v2/submission.csv outputs/v2/submission_3seed.csv
cp outputs/v2/oof.csv outputs/v2/oof_3seed.csv

echo "== 3/8 Final pipeline: 5 seeds + full-data refit, test-row training OFF (second argument 0)"
$PY v2/pipeline.py 5 0

echo "== 4/8 Multi-task neural network, 10 seeds"
$PY v2/nn.py 10

echo "== 5/8 Blend: pipeline + neural network, then 80/20 with plan v3"
$PY v2/blend_v3.py

echo "== 6/8 Audit of that blend (frozen copy in outputs/final_review/)"
$PY final_ensemble_review.py

echo "== 7/8 Rounding- and floor-aware rapid-test calibration"
$PY calibration_precision_push.py

echo "== 8/8 Final assembly: 70% blend + 30% calibration estimate (gamma-GT only)"
$PY assemble_precision_candidate.py

echo
echo "Done: outputs/calibration_precision_push/submission_assay_precision.csv"
shasum -a 256 outputs/calibration_precision_push/submission_assay_precision.csv
echo "Reference SHA-256 of the submitted file: ad421dcf185ccdaa82aad524efde61deb7bb0a29181fd332077c3b003c56ca0e"
