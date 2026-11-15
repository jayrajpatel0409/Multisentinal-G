#!/bin/bash
# Injection benchmark on the YelpZip genuine background. Usage: bash run_inject_all.sh <lane 0|1> [DATA_DIR]
# DATA_DIR holds graph_layers.pkl and reviews.csv produced by yelp_graph_probe.py (default ../data/yelp). Resumable.
lane=$1; DATA=${2:-../data/yelp}; CODE=$(cd "$(dirname "$0")/../code" && pwd); i=0
for seed in 0 1 2; do for sc in I0_baseline I1_paraphrase I2_jitter I3_fragmented I4_camouflage I5_adaptive; do
  if [ $((i % 2)) -eq $lane ] && ! grep -q "\"_scenario\": \"$sc\", \"_seed\": $seed" $DATA/yelp_inject.jsonl 2>/dev/null; then
    PYTHONPATH=$CODE python3 $CODE/yelp_inject.py $DATA $sc $seed > $DATA/inj_${sc}_${seed}.log 2>&1; fi
  i=$((i+1)); done; done
