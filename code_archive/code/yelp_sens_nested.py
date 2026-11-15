"""[sensitivity variant] Review-level evaluation of MultiSentinel-G on YelpZip (platform-filter labels), reviewer-grouped 5-fold CV.
Usage: python yelp_eval.py <dir with graph_layers.pkl> <seed> [max_folds]"""
import os, sys, time, json, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold
import coordgraph as cg
from pipeline import gbm, fit_fold, Pool
D, seed = sys.argv[1], int(sys.argv[2]); max_folds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
t0 = time.time(); tick = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
P = pickle.load(open(f"{D}/graph_layers.pkl", "rb")); R, A, T, y, L, n, (pi, pj) = P["R"], P["A"], P["T"], P["y_rev"], P["L"], P["n"], P["pairs"]
user = R["user"].values; N = len(R)

# ---------------------------------------------------------------- review-level features (all label-free)
def review_features(R):
    F = pd.DataFrame(index=R.index); u = R["user"]; pr = R["product"]
    F["s_text"] = R["s_text"].values; F["rating"] = R["rating"].values; F["t"] = R["t"].values; F["txtlen"] = R["txtlen"].values
    F["r_dev"] = (R["rating"] - R.groupby("product")["rating"].transform("mean")).abs()
    g = R.groupby("user")
    F["n_rev"] = g["rating"].transform("size"); F["n_prod"] = g["product"].transform("nunique")
    t1, t2 = g["t"].transform("min"), g["t"].transform("max"); F["span"] = t2 - t1; F["age_at_rev"] = R["t"] - t1
    F["r_mean"] = g["rating"].transform("mean"); F["r_std"] = g["rating"].transform("std").fillna(0.0)
    F["r_five"] = (R["rating"] == 5).groupby(u).transform("mean"); F["r_ext"] = R["rating"].isin([1, 5]).groupby(u).transform("mean")
    day = np.floor(R["t"]); cnt = R.assign(day=day).groupby(["user", "day"])["rating"].transform("size"); F["same_day_max"] = cnt.groupby(u).transform("max")
    K = u.values.astype(np.int64) * 10000 + R["t"].values; order = np.argsort(K); Ks = K[order]
    win = np.empty(len(R)); win[order] = np.searchsorted(Ks, Ks + 30.0, side="right") - np.arange(len(R)); F["burst30"] = pd.Series(win).groupby(u.values).transform("max").values
    rk = g["t"].rank(method="first"); F["rank"] = rk; F["is_first"] = (rk == 1).astype(float)
    F["p_n"] = R.groupby("product")["rating"].transform("size"); F["p_std"] = R.groupby("product")["rating"].transform("std").fillna(0.0)
    wk = np.floor(R["t"] / 7)
    F["pw_n"] = R.groupby(["product", wk])["rating"].transform("size"); F["pw_five"] = (R["rating"] == 5).groupby([pr, wk]).transform("mean")
    return F
F = review_features(R); BASE = list(F.columns); BEHAV = [c for c in BASE if c != "s_text"]
tick(f"review features: {len(BASE)} columns")

# ---------------------------------------------------------------- layers
def content_layer_from_pairs(min_k, q=0.01):
    a_, b_ = np.minimum(user[pi], user[pj]), np.maximum(user[pi], user[pj])
    df = pd.DataFrame({"u": a_, "v": b_}).groupby(["u", "v"], as_index=False).size().rename(columns={"size": "k"}); n_tests = len(df)
    deg = np.bincount(pi, minlength=N) + np.bincount(pj, minlength=N); Dg = np.bincount(user, weights=deg, minlength=n)
    dd = df[df.k >= min_k]; mu = Dg[dd.u.values] * Dg[dd.v.values] / max(deg.sum(), 1)
    keep, a = cg.poisson_validate(dd.k.values, mu, n_tests, q, 10.0)
    return dd.u.values[keep], dd.v.values[keep], a, (dd.u.values, dd.v.values, np.minimum(1.0, dd.k.values / 4.0))
cu2 = content_layer_from_pairs(2)
# configurations: (name, window W, q, alpha)
CFG = [("base", 7.0, 0.01, 0.8), ("W3", 3.0, 0.01, 0.8), ("W14", 14.0, 0.01, 0.8), ("W30", 30.0, 0.01, 0.8),
       ("q0.001", 7.0, 0.001, 0.8), ("q0.05", 7.0, 0.05, 0.8), ("a0.5", 7.0, 0.01, 0.5), ("a0.95", 7.0, 0.01, 0.95)]
import os
sel = os.environ.get("SENS_CFG"); CFG = [c for c in CFG if (sel is None or c[0] in sel.split(","))]
GR = {}
for name, W_, q_, a_ in CFG:
    sy = L["sync"] if (W_ == 7.0 and q_ == 0.01) else cg.sync_layer(R, n, W=W_, q=q_)
    cu = cu2 if q_ == 0.01 else content_layer_from_pairs(2, q_)
    W, mats = cg.fuse_layers({"sync": sy, "content": cu[:3]}, n); feats, cid, W = cg.graph_features(W, mats, R, n, T, seed=seed)
    GR[name] = (W, feats.drop(columns=["comm"]), a_); tick(f"graph {name}: {W.nnz // 2} edges, {len(sy[0])} sync edges")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
txt = pd.read_csv(f"{D}/reviews.csv", usecols=["text"], keep_default_na=False)["text"].values; assert len(txt) == N
Xt = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True, dtype=np.float32).fit_transform(txt).tocsr(); tick("tfidf matrix (unsupervised fit)")
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=seed); fold = np.zeros(N, int)
for k, (_, te) in enumerate(sgk.split(np.zeros(N), y, user)): fold[te] = k
acc_mean = lambda v: np.bincount(user, weights=v, minlength=n) / np.maximum(np.bincount(user, minlength=n), 1)
pools = {}
for k in range(5):
    te, va = fold == k, fold == (k + 1) % 5; tr = ~(te | va); trva = ~te
    def run(name, X):
        pt, th = fit_fold(X[tr], y[tr], X[va], y[va], X[te], seed); pools.setdefault(name, Pool()).add(y[te], pt, th, np.where(te)[0])
    idx = np.where(trva)[0]; tei = np.where(te)[0]
    st = np.zeros(N); rng = np.random.default_rng(seed + 11 + k); ug = np.unique(user[idx]); rng.shuffle(ug)
    for f in np.array_split(ug, 5):
        ti = idx[np.isin(user[idx], f)]; ri = np.setdiff1d(idx, ti); st[ti] = LogisticRegression(C=1.0, max_iter=200).fit(Xt[ri], y[ri]).predict_proba(Xt[ti])[:, 1]
    st[tei] = LogisticRegression(C=1.0, max_iter=200).fit(Xt[idx], y[idx]).predict_proba(Xt[tei])[:, 1]
    Fk = F.copy(); Fk["s_text"] = st; Xb = Fk[BASE].values
    s0 = np.zeros(N); idx = np.where(trva)[0]; rng = np.random.default_rng(seed + 5 + k); ug = np.unique(user[idx]); rng.shuffle(ug)
    for f in np.array_split(ug, 5):
        ti = idx[np.isin(user[idx], f)]; ri = np.setdiff1d(idx, ti); s0[ti] = gbm(seed).fit(Xb[ri], y[ri]).predict_proba(Xb[ti])[:, 1]
    tei = np.where(te)[0]; s0[tei] = gbm(seed).fit(Xb[idx], y[idx]).predict_proba(Xb[tei])[:, 1]
    s0a = acc_mean(s0); X0 = np.hstack([Xb, s0a[user][:, None]])
    run("acct_context", X0)
    for name, (W, G, a_) in GR.items():
        Gp = G.assign(g_prop=cg.propagate(W, s0a, alpha=a_), g_nbr_s0=cg.neighbour_mean(W, s0a)); run(name, np.hstack([X0, Gp.values[user]]))
    tick(f"fold {k + 1}/5 done")
res = {k: p.metrics() for k, p in pools.items()}
json.dump(res, open(f"{D}/yelp_sens_nested_seed{seed}{'_' + sel.replace(',', '') if sel else ''}.json", "w"), indent=1)
for k, v in res.items(): print(f"{k:14s} AP {v['ap']:.4f} AUROC {v['auroc']:.4f}")
