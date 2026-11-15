#!/bin/bash
# Prevalence sweep (seed 0). Usage: bash run_frac.sh "0.01 0.02 0.08" [DATA_DIR]
DATA=${2:-../data/yelp}; CODE=$(cd "$(dirname "$0")/../code" && pwd); cd $DATA
for fr in $1; do for sc in I0_baseline I2_jitter; do INJ_FRAC=$fr PYTHONPATH=$CODE python3 $CODE/yelp_inject.py . $sc 0 > inj_frac_${fr}_$sc.log 2>&1; done; done
