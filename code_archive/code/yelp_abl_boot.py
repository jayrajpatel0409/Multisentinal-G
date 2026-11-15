"""Build analysis_corrected.json from the corrected-protocol runs: yelp_abl_nested_seed{0,1,2}.json (+ gctrl for GNN/MLP),
paired cluster bootstrap on seed-0 predictions, and graph coverage. Usage: python yelp_abl_boot.py <DATA_DIR> [B]"""
import sys, json, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import average_precision_score as aps, roc_auc_score as auc
import coordgraph as cg
D = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 500
P = pickle.load(open(f"{D}/graph_layers.pkl", "rb")); R, n, (pi, pj), L = P["R"], P["n"], P["pairs"], P["L"]; user = R["user"].values; N = len(R); T = P["T"]
RES = [json.load(open(f"{D}/yelp_abl_nested_seed{s}.json")) for s in (0, 1, 2)]
GC = [json.load(open(f"{D}/yelp_nested_gctrl_seed{s}.json")) for s in (0, 1, 2)]
out = {}
for m in RES[0]:
    out[m] = {k: [float(np.mean([r[m][k] for r in RES])), float(np.std([r[m][k] for r in RES], ddof=1))] for k in ("ap", "auroc", "f1", "rec_fpr1")}
for m, g in (("gnn", "gnn_ctx"), ("mlp_nograph", "mlp_nograph_ctx")):
    out[m] = {k: [float(np.mean([r[g][k] for r in GC])), float(np.std([r[g][k] for r in GC], ddof=1))] for k in ("ap", "auroc", "f1", "rec_fpr1")}
Z = np.load(f"{D}/yelp_abl_nested_preds_seed0.npz"); idx = Z["full__idx"]; y = Z["full__y"]
for m in Z.files:
    if m.endswith("__idx"): assert (Z[m] == idx).all()
p = {m[:-3]: Z[m] for m in Z.files if m.endswith("__p")}
# graph coverage (exploratory k=1 graph, as in the ablations)
a_, b_ = np.minimum(user[pi], user[pj]), np.maximum(user[pi], user[pj])
df = pd.DataFrame({"u": a_, "v": b_}).groupby(["u", "v"], as_index=False).size().rename(columns={"size": "k"}); nt = len(df)
deg = np.bincount(pi, minlength=N) + np.bincount(pj, minlength=N); Dg = np.bincount(user, weights=deg, minlength=n)
def cl(k):
    dd = df[df.k >= k]; mu = Dg[dd.u.values] * Dg[dd.v.values] / max(deg.sum(), 1); keep, a = cg.poisson_validate(dd.k.values, mu, nt, 0.01, 10.0); return dd.u.values[keep], dd.v.values[keep], a
cov = {}
for tag, k in (("k1", 1), ("k2", 2)):
    W, mats = cg.fuse_layers({"sync": L["sync"], "content": cl(k)}, n); cov[tag] = np.asarray((W > 0).sum(1)).ravel() > 0
for tag in ("k1", "k2"):
    c = cov[tag][user[idx]]; nm = "full" if tag == "k1" else "full_k2"
    out[f"cov_{tag}"] = dict(covered_accounts=int(cov[tag].sum()), covered_share_reviews=float(c.mean()), ap_cov_full=float(aps(y[c], p[nm][c])), ap_cov_acct_context=float(aps(y[c], p["acct_context"][c])), ap_cov_stage_fusion=float(aps(y[c], p["stage_fusion"][c])),
                              ap_unc_full=float(aps(y[~c], p[nm][~c])), ap_unc_acct_context=float(aps(y[~c], p["acct_context"][~c])), ap_unc_stage_fusion=float(aps(y[~c], p["stage_fusion"][~c])))
# paired cluster bootstrap over reviewers (seed-0 predictions)
u = user[idx]; rng = np.random.default_rng(5); pairs = [("full_k2", "stage_fusion"), ("full_k2", "acct_context"), ("full_k2", "sign_rawgraph"), ("acct_context", "stage_fusion"), ("full_k2", "text_only"), ("full", "full_k2")]
ab = {m: [] for m in p}; au = {m: [] for m in p}
for b in range(B):
    w = np.bincount(rng.integers(0, n, n), minlength=n)[u].astype(float)
    for m in p: ab[m].append(aps(y, p[m], sample_weight=w)); au[m].append(auc(y, p[m], sample_weight=w))
out["bootstrap"] = {}
for a, c in pairs:
    d = np.array(ab[a]) - np.array(ab[c]); du = np.array(au[a]) - np.array(au[c])
    out["bootstrap"][f"{a}-{c}"] = dict(d_ap=float(aps(y, p[a]) - aps(y, p[c])), ap_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], d_auroc=float(auc(y, p[a]) - auc(y, p[c])), auroc_ci=[float(np.percentile(du, 2.5)), float(np.percentile(du, 97.5))])
json.dump(out, open(f"{D}/analysis_corrected.json", "w"), indent=1); print(json.dumps(out["bootstrap"], indent=1)[:1500])
