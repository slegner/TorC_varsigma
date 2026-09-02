#!/bin/bash
# Individual fgivenx posterior-predictive plots, one per chain.
set -u
cd "$(dirname "$0")/.."
PY=/Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/python
mkdir -p logs
for c in torc_ns_julia_1000_powerlaw_nostigmawall \
         torc_ns_julia_1000_lam_noOmL_powerlaw; do
    echo "=== fgivenx: $c ==="
    $PY scripts/plot_fgivenx_lam.py "chains/$c.npz" 2>&1 | tee "logs/fgivenx_$c.log" | grep -v "it/s"
done
echo "=== FGIVENX ALL DONE ==="
