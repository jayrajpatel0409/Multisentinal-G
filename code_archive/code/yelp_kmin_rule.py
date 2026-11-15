"""Label-free rule for the content layer's minimum coincidence count k_min on real text.
Permute authorship (global, and within product) so that real near-duplicate texts stay but account links are destroyed;
count validated content edges at q=0.01 for k_min in {1,2}. Rule: smallest k_min with at most one validated edge on every permutation.
Usage: python yelp_kmin_rule.py <dir> <n_perm>"""
import sys, json, pickle, numpy as np, pandas as pd
import coordgraph as cg
D, B = sys.argv[1], int(sys.argv[2])
P = pickle.load(open(f"{D}/graph_layers.pkl", "rb")); R, n, (pi, pj) = P["R"], P["n"], P["pairs"]; user0 = R["user"].values; N = len(R); prod = R["product"].values
def layer(user, min_k, q=0.01):
    a_, b_ = np.minimum(user[pi], user[pj]), np.maximum(user[pi], user[pj]); m = a_ != b_
    df = pd.DataFrame({"u": a_[m], "v": b_[m]}).groupby(["u", "v"], as_index=False).size().rename(columns={"size": "k"}); n_tests = len(df)
    deg = np.bincount(pi[m], minlength=N) + np.bincount(pj[m], minlength=N); Dg = np.bincount(user, weights=deg, minlength=n)
    dd = df[df.k >= min_k]; mu = Dg[dd.u.values] * Dg[dd.v.values] / max(deg.sum(), 1)
    keep, a = cg.poisson_validate(dd.k.values, mu, n_tests, q, 10.0); return int(keep.sum()), n_tests
out = {"real": {k: layer(user0, k) for k in (1, 2)}, "perm_global": {1: [], 2: []}, "perm_product": {1: [], 2: []}}
rng = np.random.default_rng(2024)
for b in range(B):
    ug = rng.permutation(user0)
    up = user0.copy()
    for g in pd.Series(np.arange(N)).groupby(prod).groups.values():
        g = np.asarray(g); up[g] = rng.permutation(user0[g])
    for k in (1, 2):
        out["perm_global"][k].append(layer(ug, k)[0]); out["perm_product"][k].append(layer(up, k)[0])
    print(b, {s: {k: out[s][k][-1] for k in (1, 2)} for s in ("perm_global", "perm_product")}, flush=True)
chosen = next((k for k in (1, 2) if max(out["perm_global"][k] + out["perm_product"][k]) <= 1), 2)
out["rule_choice"] = chosen; json.dump(out, open(f"{D}/kmin_rule.json", "w"), indent=1); print("real", out["real"], "chosen k_min:", chosen)
