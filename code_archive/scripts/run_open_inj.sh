#!/bin/bash
# Injection on the Yelp Open background. Usage: bash run_open_inj.sh base|sup [OPEN_DIR] [YELPZIP_DIR]
# 'base' keeps friend edges, 'sup' removes validated edges between declared friends (FRSUP=1).
OPEN=${2:-../data/open}; export YELPZIP_DIR=${3:-../data/yelp}; CODE=$(cd "$(dirname "$0")/../code" && pwd); cd $OPEN
for s in 0 1 2; do for sc in I0_baseline I1_paraphrase I2_jitter I3_fragmented I4_camouflage I5_adaptive; do
 if [ "$1" = "sup" ]; then FRSUP=1 PYTHONPATH=$CODE python3 $CODE/yelp_inject_open.py . $sc $s > injS_${sc}_$s.log 2>&1
 else PYTHONPATH=$CODE python3 $CODE/yelp_inject_open.py . $sc $s > inj_${sc}_$s.log 2>&1; fi
done; done
