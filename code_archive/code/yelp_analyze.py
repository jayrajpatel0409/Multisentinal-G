"""Aggregate YelpZip review-level results over seeds + cluster-bootstrap CIs. Usage: python yelp_analyze.py <dir> <seeds comma-separated> <bootstrap B>"""
import sys, json, pickle, numpy as np
from sklearn.metrics import average_precision_score as aps, roc_auc_score as auc
D = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]; B = int(sys.argv[3])
res = [json.load(open(f"{D}/yelp_seed{s}.json")) for s in seeds]; out = {"seeds": seeds, "meta": res[0]["_meta"]}
models = [k for k in res[0] if not k.startswith("_")]
for m in models:
    out[m] = {}
    for met in ("ap", "auroc", "f1", "rec_fpr1", "ece", "brier"):
        v = np.array([r[m][met] for r in res]); out[m][met] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
    for extra in ("covered_share", "ap_covered", "ap_uncovered"):
        if extra in res[0][m]: v = np.array([r[m][extra] for r in res]); out[m][extra] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]
# paired cluster bootstrap over reviewers (seed-0 out-of-fold predictions)
P = pickle.load(open(f"{D}/graph_layers.pkl", "rb")); user = P["R"]["user"].values; n = P["n"]
Z = np.load(f"{D}/yelp_preds_seed{seeds[0]}.npz"); cmp_models = ["text_only", "stage_fusion", "acct_context", "sign_rawgraph", "full"]
idx = Z["full__idx"]; y = Z["full__y"]; u = user[idx]; p = {m: Z[f"{m}__p"] for m in cmp_models}
assert all((Z[f"{m}__idx"] == idx).all() for m in cmp_models)
rng = np.random.default_rng(0); ap_b = {m: [] for m in cmp_models}; au_b = {m: [] for m in cmp_models}
for b in range(B):
    w = np.bincount(rng.integers(0, n, n), minlength=n)[u].astype(float)
    for m in cmp_models: ap_b[m].append(aps(y, p[m], sample_weight=w)); au_b[m].append(auc(y, p[m], sample_weight=w))
    if b % 20 == 0: print("bootstrap", b, flush=True)
pairs = [("full", "stage_fusion"), ("full", "acct_context"), ("full", "sign_rawgraph"), ("acct_context", "stage_fusion"), ("full", "text_only")]
out["bootstrap"] = {}
for a, c in pairs:
    d1 = np.array(ap_b[a]) - np.array(ap_b[c]); d2 = np.array(au_b[a]) - np.array(au_b[c])
    out["bootstrap"][f"{a}-{c}"] = dict(d_ap=float(aps(y, p[a]) - aps(y, p[c])), ap_ci=[float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))],
                                        d_auroc=float(auc(y, p[a]) - auc(y, p[c])), auroc_ci=[float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))])
json.dump(out, open(f"{D}/analysis.json", "w"), indent=1); print(json.dumps(out["bootstrap"], indent=1))
