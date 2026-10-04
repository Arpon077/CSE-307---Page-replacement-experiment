#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 src/run_experiment.py
printf '\nDone. See results/policy_metrics.csv and results/hit_ratio.svg\n'
