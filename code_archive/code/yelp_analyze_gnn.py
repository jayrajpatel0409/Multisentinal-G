"""Aggregate GNN baseline + paired cluster bootstrap vs MultiSentinel-G. Usage: python yelp_analyze_gnn.py <dir> <seeds> <B>"""
import sys, json, pickle, numpy as np
from sklearn.metrics import average_precision_score as aps, roc_auc_score as auc
D = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]; B = int(sys.argv[3])
res = [json.load(open(f"{D}/yelp_gnn_seed{s}.json")) for s in seeds]; out = {"seeds": seeds}
for m in ("gnn", "mlp_nograph"):
    out[m] = {met: [float(np.mean([r[m][met] for r in res])), float(np.std([r[m][met] for r in res], ddof=1))] for met in ("ap", "auroc", "f1", "rec_fpr1", "ece")}
P = pickle.load(open(f"{D}/graph_layers.pkl", "rb")); user = P["R"]["user"].values; n = P["n"]
Z = np.load(f"{D}/yelp_preds_seed{seeds[0]}.npz"); G = np.load(f"{D}/yelp_gnn_preds_seed{seeds[0]}.npz")
idx = Z["full__idx"]; y = Z["full__y"]; assert (G["gnn__idx"] == idx).all() and (G["mlp_nograph__idx"] == idx).all()
p = {"full": Z["full__p"], "acct_context": Z["acct_context__p"], "gnn": G["gnn__p"], "mlp_nograph": G["mlp_nograph__p"]}
u = user[idx]; rng = np.random.default_rng(1); ab = {m: [] for m in p}; ub = {m: [] for m in p}
for b in range(B):
    w = np.bincount(rng.integers(0, n, n), minlength=n)[u].astype(float)
    for m in p: ab[m].append(aps(y, p[m], sample_weight=w)); ub[m].append(auc(y, p[m], sample_weight=w))
out["bootstrap"] = {}
for a, c in [("gnn", "full"), ("gnn", "mlp_nograph"), ("mlp_nograph", "full"), ("gnn", "acct_context")]:
    d1 = np.array(ab[a]) - np.array(ab[c]); d2 = np.array(ub[a]) - np.array(ub[c])
    out["bootstrap"][f"{a}-{c}"] = dict(d_ap=float(aps(y, p[a]) - aps(y, p[c])), ap_ci=[float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))], d_auroc=float(auc(y, p[a]) - auc(y, p[c])), auroc_ci=[float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))])
json.dump(out, open(f"{D}/analysis_gnn.json", "w"), indent=1); print(json.dumps(out, indent=1))
