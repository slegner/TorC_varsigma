#!/bin/bash
# DESI DR2 BAO + supernovae, for both SN datasets and all three models.
#
# Six runs, 1000 live each, sequential (the Julia server takes every core).
# Settings identical to run_all_lam.sh, so every ln Z in chains/ is comparable.
set -u
cd "$(dirname "$0")/.."
PY=/Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/python
mkdir -p logs

for SN in pantheonplus desy5; do
  echo "=== ${SN}: [1/3] no lambda ==="
  TORC_SN=$SN TORC_SAMPLE_LAM=0 $PY scripts/run_ns_julia_1000_sn.py 2>&1 \
      | tee logs/run_${SN}_nolam.log | grep -vE "pts/s|it/s"
  echo "=== ${SN}: [2/3] lambda + OmL ==="
  TORC_SN=$SN $PY scripts/run_ns_julia_1000_sn.py 2>&1 \
      | tee logs/run_${SN}_lam_OmL.log | grep -vE "pts/s|it/s"
  echo "=== ${SN}: [3/3] lambda only, OmL=0 ==="
  TORC_SN=$SN TORC_SAMPLE_OML=0 $PY scripts/run_ns_julia_1000_sn.py 2>&1 \
      | tee logs/run_${SN}_lam_noOmL.log | grep -vE "pts/s|it/s"
done
echo "=== ALL SN RUNS DONE ==="
