#!/bin/bash
# Three 1000-live DESI runs, sequential (each Julia server takes all cores).
#   1. no-lambda baseline, post stigma-wall fix
#   2. lambda + OmL   (5D)
#   3. lambda, OmL=0  (4D)
set -u
cd "$(dirname "$0")/.."
PY=/Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/python
mkdir -p logs

echo "=== [1/3] no-lambda baseline (post-fix) ==="
TORC_TAG=_nostigmawall $PY scripts/run_ns_julia_1000.py 2>&1 | tee logs/run_baseline.log
echo "=== [2/3] lambda + OmL (5D) ==="
$PY scripts/run_ns_julia_1000_lam.py 2>&1 | tee logs/run_lam_OmL.log
echo "=== [3/3] lambda only, OmL=0 (4D) ==="
TORC_SAMPLE_OML=0 $PY scripts/run_ns_julia_1000_lam.py 2>&1 | tee logs/run_lam_noOmL.log
echo "=== ALL THREE DONE ==="
