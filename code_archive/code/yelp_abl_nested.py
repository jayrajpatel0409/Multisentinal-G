"""[fully nested text score] Review-level evaluation of MultiSentinel-G on YelpZip (platform-filter labels), reviewer-grouped 5-fold CV.
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
W, mats = cg.fuse_layers({"sync": L["sync"], "content": cu2[:3]}, n); feats, cid, W = cg.graph_features(W, mats, R, n, T, seed=seed); G = feats.drop(columns=["comm"]); tick("graph built")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
txt = pd.read_csv(f"{D}/reviews.csv", usecols=["text"], keep_default_na=False)["text"].values; assert len(txt) == N
Xt = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True, dtype=np.float32).fit_transform(txt).tocsr(); tick("tfidf matrix (unsupervised fit)")
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=seed); fold = np.zeros(N, int)
for k, (_, te) in enumerate(sgk.split(np.zeros(N), y, user)): fold[te] = k
acc_mean = lambda v: np.bincount(user, weights=v, minlength=n) / np.maximum(np.bincount(user, minlength=n), 1)
import os
def sage_scores(Wg, Xn, tgt, wgt, train_mask, seed, epochs=120):
    """2-layer GraphSAGE (mean aggregator) trained on account nodes in train_mask; returns logits for all nodes."""
    import torch; torch.manual_seed(seed); torch.set_num_threads(2)
    Wc = Wg.tocoo(); deg = np.asarray(Wg.sum(1)).ravel(); inv = 1.0 / np.maximum(deg, 1)
    Ai = torch.sparse_coo_tensor(np.vstack([Wc.row, Wc.col]), (Wc.data * inv[Wc.row]).astype(np.float32), Wg.shape).coalesce()
    X = torch.tensor(Xn, dtype=torch.float32); t = torch.tensor(tgt, dtype=torch.float32); w = torch.tensor(wgt * train_mask, dtype=torch.float32)
    l1 = torch.nn.Linear(2 * X.shape[1], 64); l2 = torch.nn.Linear(128, 64); out = torch.nn.Linear(64, 1)
    ps = list(l1.parameters()) + list(l2.parameters()) + list(out.parameters()); opt = torch.optim.Adam(ps, lr=0.01, weight_decay=1e-5)
    def fwd():
        h = torch.relu(l1(torch.cat([X, torch.sparse.mm(Ai, X)], 1))); h = torch.relu(l2(torch.cat([h, torch.sparse.mm(Ai, h)], 1))); return out(h).squeeze(1)
    for _ in range(epochs):
        opt.zero_grad(); lo = fwd(); loss = (torch.nn.functional.binary_cross_entropy_with_logits(lo, t, reduction="none") * w).sum() / w.sum(); loss.backward(); opt.step()
    with torch.no_grad(): return fwd().numpy()

NG = os.environ.get("NESTED_GNN") == "1"; HYB = os.environ.get("HYBRID") == "1"; CTRL = os.environ.get("CTRL") == "1"; GC = os.environ.get("GCTRL") == "1"; NG = NG or HYB or GC
if NG:
    SYraw = L["sync_raw"]; Wraw = sp.coo_matrix((np.ones(len(SYraw[0])), (SYraw[0], SYraw[1])), shape=(n, n)).tocsr(); Wraw = Wraw.maximum(Wraw.T)
    cu1r = content_layer_from_pairs(1)[3]; Cc = sp.coo_matrix((np.ones(len(cu1r[0])), (cu1r[0], cu1r[1])), shape=(n, n)).tocsr(); Wg = (Wraw + Cc + Cc.T).tocsr(); Wg.data[:] = 1.0

SYraw = L["sync_raw"]; SY = L["sync"]
cu1 = content_layer_from_pairs(1); cu2 = content_layer_from_pairs(2)

configs = {  # name -> dict of layers
    "full": {"sync": SY, "content": cu1[:3]}, "full_k2": {"sync": SY, "content": cu2[:3]},
    "abl_no_sync": {"content": cu1[:3]}, "abl_no_content": {"sync": SY},
    "abl_no_validation": {"sync": SYraw, "content": cu1[3]}}

GR = {}
for name, lay in configs.items():
    W_, mats_ = cg.fuse_layers(lay, n); feats_, cid_, W_ = cg.graph_features(W_, mats_, R, n, T, seed=seed)
    GR[name] = (W_, feats_.drop(columns=["comm"])); tick(f"graph {name}: {W_.nnz // 2} fused edges")
Wraw = sp.coo_matrix((np.ones(len(SYraw[0])), (SYraw[0], SYraw[1])), shape=(n, n)).tocsr(); Wraw = Wraw.maximum(Wraw.T)
dense = cg.dense_block_scores(R, n); tick("dense-block baseline scores done")
pools = {}
for k in range(5):
    te, va = fold == k, fold == (k + 1) % 5; tr = ~(te | va); trva = ~te
    idx = np.where(trva)[0]; tei = np.where(te)[0]
    st = np.zeros(N); rng = np.random.default_rng(seed + 11 + k); ug = np.unique(user[idx]); rng.shuffle(ug)
    for f in np.array_split(ug, 5):
        ti = idx[np.isin(user[idx], f)]; ri = np.setdiff1d(idx, ti); st[ti] = LogisticRegression(C=1.0, max_iter=200).fit(Xt[ri], y[ri]).predict_proba(Xt[ti])[:, 1]
    st[tei] = LogisticRegression(C=1.0, max_iter=200).fit(Xt[idx], y[idx]).predict_proba(Xt[tei])[:, 1]; tick(f"fold {k + 1} nested text score")
    Fk = F.copy(); Fk["s_text"] = st; Xb = Fk[BASE].values
    def run(name, X):
        pt, th = fit_fold(X[tr], y[tr], X[va], y[va], X[te], seed); pools.setdefault(name, Pool()).add(y[te], pt, th, np.where(te)[0])
    s0 = np.zeros(N); rng = np.random.default_rng(seed + 5 + k); ug = np.unique(user[idx]); rng.shuffle(ug)
    for f in np.array_split(ug, 5):
        ti = idx[np.isin(user[idx], f)]; ri = np.setdiff1d(idx, ti); s0[ti] = gbm(seed).fit(Xb[ri], y[ri]).predict_proba(Xb[ti])[:, 1]
    s0[tei] = gbm(seed).fit(Xb[idx], y[idx]).predict_proba(Xb[tei])[:, 1]
    s0a = acc_mean(s0); X0 = np.hstack([Xb, s0a[user][:, None]])
    asx = acc_mean(st); ard = acc_mean(Fk["r_dev"].values)
    sign = np.column_stack([cg.neighbour_mean(Wraw, asx), cg.neighbour_mean(Wraw, cg.neighbour_mean(Wraw, asx)), cg.neighbour_mean(Wraw, ard), np.asarray(Wraw.sum(1)).ravel()])[user]
    nm1 = cg.neighbour_mean(Wraw, s0a); nm2 = cg.neighbour_mean(Wraw, nm1)
    run("text_only", st[:, None]); run("behav_only", Fk[BEHAV].values); run("stage_fusion", Xb); run("acct_context", X0)
    run("sign_rawgraph", np.hstack([X0, sign, nm1[user][:, None], nm2[user][:, None]]))
    pools.setdefault("dense_block", Pool()).add(y[te], dense[user][te], 0.5, tei)
    for name, (W_, G_) in GR.items():
        Gp = G_.assign(g_prop=cg.propagate(W_, s0a), g_nbr_s0=cg.neighbour_mean(W_, s0a)); run(name, np.hstack([X0, Gp.values[user]]))
        if name == "full":
            run("graph_only", G_.values[user]); run("abl_no_propagation", np.hstack([X0, G_.values[user]]))
            run("abl_no_community", np.hstack([X0, Gp.drop(columns=[c for c in Gp.columns if c.startswith("c_")]).values[user]]))
    tick(f"fold {k + 1}/5 done")
res = {k: p.metrics() for k, p in pools.items()}
json.dump(res, open(f"{D}/yelp_abl_nested_seed{seed}.json", "w"), indent=1)
np.savez_compressed(f"{D}/yelp_abl_nested_preds_seed{seed}.npz", **{f"{k}__{w}": np.concatenate(getattr(p, a)) for k, p in pools.items() for w, a in (("idx", "idx"), ("y", "y"), ("p", "p"))})
for k, v in res.items(): print(f"{k:20s} AP {v['ap']:.4f} AUROC {v['auroc']:.4f} F1 {v['f1']:.4f} Rec@1%FPR {v['rec_fpr1']:.4f}")
tick("done")
