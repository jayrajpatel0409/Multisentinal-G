#!/bin/bash
# Corrected-protocol ablations, sensitivity sweeps and validated-edge GNN. Usage: bash run_st2.sh <DATA_DIR>
DATA=${1:-../data/yelp}; CODE=$(cd "$(dirname "$0")/../code" && pwd); cd $DATA; export PYTHONPATH=$CODE
for s in 0 1 2; do python3 $CODE/yelp_abl_nested.py . $s > abl_nested_seed$s.log 2>&1; done
for s in 0 1 2; do python3 $CODE/yelp_sens_nested.py . $s > sens_nested_seed$s.log 2>&1; done
for s in 0 1 2; do HYBRID=1 python3 $CODE/yelp_nested.py . $s > nested_hyb_seed$s.log 2>&1; done
