#!/usr/bin/env bash
# Rebuild everything. Argument: the pitch table from the abs-challenge-analysis pipeline.
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-../abs-challenge-analysis/data/pitches.pkl.gz}"
python src/build_pitches.py "$SRC"   # features and denoised target (~3 min)
python src/models.py                 # Stuff+, Location+, Pitching+, cross-fit by season (~12 min)
python src/grades.py                 # 100-scale grades and validation
python src/lab.py                    # design lab (~2 min)
python src/figures.py                # figures and case studies
python src/backtest_preds.py         # leak-free predictions for the backtest (~2 min)
python src/backtest.py               # did predicted changes match real ones?
python src/export.py                 # data/processed/
