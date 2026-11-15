# MultiSentinel-G: code and result files

Reproduces the experiments of "How Much Do Coordination Graphs Add to Opinion-Spam Detection Beyond Reviewer Context?".
No Yelp data are included. Obtain them yourself under their own terms (see Section 1).

Layout: `code/` Python scripts, `scripts/` shell drivers, `results/` result files (JSON/JSONL/CSV), `logs/` run logs, `requirements.txt`.
All shell scripts take the data folder as an argument; nothing is hard-coded to one computer. Run them from `scripts/`.
Install: `python -m pip install -r requirements.txt` (Python 3.13; torch is only needed for GraphSAGE/MLP runs).
Licence: to be added by the authors (see LICENSE).

## 1. Data (not included)
| Dataset | Source | Put in |
|---|---|---|
| YelpZip | Rayana and Akoglu (2015), authors' release (metadata + reviewContent files) | `data/raw_yelpzip/` |
| YelpNYC | same release (Kaggle mirror of the YelpNYC files) | `data/raw_yelpnyc/` |
| Yelp Open Dataset (incl. friend lists) | yelp.com/dataset, under the Yelp Dataset Terms | `data/raw_yelp_open/` |

Convert each raw release to `reviews.csv` + `accounts.csv` (+ `friends.csv`, `users_meta.csv` for Open) with
`code/prepare_yelp_upload.py` (YelpZip), `code/prepare_yelpnyc_kaggle.py` (YelpNYC), `code/prepare_yelp_open.py` (Open).
Each script prints its usage with `--help`. Then build the graph layers once per dataset:
`python code/yelp_graph_probe.py data/yelp` (Open: `yelp_graph_probe_open.py data/open`), about 5-10 minutes.
(`data/yelp`, `data/nyc`, `data/open` are the folders holding the converted files.)

## 2. One command per result
| Result in paper | Command (from `scripts/` unless stated) | Time (2 CPUs) |
|---|---|---|
| Simulation, Tables 12-18, Figures 2-4 | `cd code && python run_all.py ../results/results.jsonl && python analyze.py ../results/results.jsonl` | 1-2 h |
| Simulation spot check | see `logs/simulation_verification_seeds0-1.txt` | 10 min |
| Table 3 and 7 main results (3 seeds) | `cd ../data/yelp && for s in 0 1 2; do python ../../code/yelp_nested.py . $s; done` (same in `data/nyc`) | 6 min/seed |
| GraphSAGE / MLP, account-context control | `bash run_gctrl.sh ../data/yelp` (and `../data/nyc`) | ~80 min/seed |
| Account-context control (Table 6) | `CTRL=1 python ../code/yelp_nested.py . <seed>` then `python ../code/yelp_ctrl_table.py .` | 30 min/seed |
| Ablations, sensitivity, validated-edge GNN (Tables 4, 5, 8) | `bash run_st2.sh ../data/yelp` | ~6 h |
| k_min permutation diagnostic | `python ../code/yelp_kmin_rule.py .` | 5 min |
| Injection, YelpZip background (Table 11) | `bash run_inject_all.sh 0 & bash run_inject_all.sh 1` | ~3 h |
| Prevalence sweep (Table 9) | `bash run_frac.sh "0.01 0.02 0.08"` | 1 h |
| Friend-group test (Section 7.5) | `python ../code/open_friends.py` and `open_friends2.py` in `data/open` | 5 min |
| Injection on Open background, with/without friend edges (Table 10) | `bash run_open_inj.sh base` and `bash run_open_inj.sh sup` | ~4 h |
| Numbers and tables for the manuscript | `YELP_DIR=../data/yelp python ../code/make_real_numbers.py` | 1 min |

Note: `code/yelp_eval.py` and `code/yelp_sens.py` are the scripts of the earlier (leaky) text-score protocol. They are kept only to reproduce the leak comparison described in Section 7.1 and must not be used for the reported tables; use `yelp_nested.py`, `yelp_abl_nested.py` and `yelp_sens_nested.py`.

## 3. Checking a result against the paper
Result files are in `results/` (JSON per run, `*.csv` tidy tables). Mean AP over three seeds on YelpZip must match Table 3
(text only 0.293, stage fusion 0.465, account context 0.478, MultiSentinel-G 0.479, MLP 0.482, GraphSAGE 0.491) to about 0.002.
Simulation: a single seed differs from the 14-run mean but must keep the order of methods (stage fusion < raw graph < MultiSentinel-G).
The seed-0/1 check we ran is in `logs/simulation_verification_seeds0-1.txt`.
