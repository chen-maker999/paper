#!/usr/bin/env bash
# Reproduce every number and figure in the paper (CPU only, ~1 hour on 4 cores).
set -euo pipefail
DATA=${DATA:-data}
python experiments/download_trajs.py "$DATA/trajs"          # ~6 GB of public trajectories
python experiments/corpus.py "$DATA/trajs" "$DATA/cache"     # parse + cache
python experiments/stats.py "$DATA/cache" results            # corpus statistics
python experiments/run_swe.py --cache "$DATA/cache" --suite main
python experiments/run_swe.py --cache "$DATA/cache" --suite ablation --budgets 4000,8000
python experiments/run_synthetic.py                          # StateTrack S1-S5
python experiments/plots.py results paper/figures
(cd paper && latexmk -pdf -quiet main.tex)
