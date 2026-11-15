#!/bin/bash
# GraphSAGE / MLP account-context control, 3 seeds. Usage: bash run_gctrl.sh <DATA_DIR>   (YelpZip or YelpNYC folder)
DATA=${1:-../data/yelp}; CODE=$(cd "$(dirname "$0")/../code" && pwd); cd $DATA
for s in 0 1 2; do GCTRL=1 PYTHONPATH=$CODE python3 $CODE/yelp_nested.py . $s > gctrl_seed$s.log 2>&1; done
