"""End-to-end pipeline: simulate -> account features -> multiplex graph -> models -> metrics."""
import time, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from campaignsim import simulate, SimConfig
import coordgraph as cg

warnings.filterwarnings("ignore")
STAGE = ["t_max", "t_mean", "t_pos", "i_max", "i_n", "gap_mean"]
META = ["age", "n_rev", "burst30", "r_mean", "r_five", "r_dev", "n_prod"]


def account_features(R, A, T):
    n = len(A)
    pm = R.groupby("product")["rating"].transform("mean")
    R = R.assign(dev=np.abs(R["rating"] - pm), pos=(R["s_text"] > 1.5).astype(float))
    g = R.groupby("user")
    F = pd.DataFrame(index=np.arange(n))
    F["t_max"] = g["s_text"].max(); F["t_mean"] = g["s_text"].mean(); F["t_pos"] = g["pos"].mean()
    F["i_max"] = g["s_img"].max(); F["i_n"] = g["has_img"].sum(); F["gap_mean"] = g["s_gap"].mean()
    F["age"] = A["age"].values; F["n_rev"] = g.size(); F["r_mean"] = g["rating"].mean()
    F["r_five"] = g["rating"].apply(lambda s: (s == 5).mean()); F["r_dev"] = g["dev"].mean()
    F["n_prod"] = g["product"].nunique()
    def burst(s):
        t = np.sort(s.values)
        if len(t) < 2: return 1
        j = np.searchsorted(t, t + 30.0, side="right"); return int((j - np.arange(len(t))).max())
    F["burst30"] = g["t"].apply(burst)
    return F.fillna(0.0)


def ece_score(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def gbm(seed):
    return xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=2.0, n_jobs=1, random_state=seed, verbosity=0, eval_metric="logloss")


def split_groups(groups, seed, test_frac=0.4, val_frac=0.25):
    rng = np.random.default_rng(seed + 991)
    ug = np.unique(groups); rng.shuffle(ug)
    nt = int(len(ug) * test_frac); test = set(ug[:nt]); rest = ug[nt:]
    nv = int(len(rest) * val_frac); val = set(rest[:nv])
    role = np.array(["train"] * len(groups), dtype=object)
    role[np.isin(groups, list(test))] = "test"; role[np.isin(groups, list(val))] = "val"
    return role


def fit_eval(F, y, role, cols, seed, R=None, A=None):
    tr, va, te = role == "train", role == "val", role == "test"
    m = gbm(seed).fit(F.loc[tr, cols].values, y[tr])
    pv_raw, pt_raw = m.predict_proba(F.loc[va, cols].values)[:, 1], m.predict_proba(F.loc[te, cols].values)[:, 1]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(C=1e6).fit(lg(pv_raw)[:, None], y[va])             # Platt scaling on validation
    pv, pt = cal.predict_proba(lg(pv_raw)[:, None])[:, 1], cal.predict_proba(lg(pt_raw)[:, None])[:, 1]
    return finish(y[va], pv, y[te], pt, pt_raw)


def finish(yv, pv, yt, pt, raw=None):
    ths = np.linspace(0.05, 0.95, 91)
    th = ths[int(np.argmax([f1_score(yv, pv >= t, zero_division=0) for t in ths]))]   # threshold fixed on validation
    out = dict(auroc=roc_auc_score(yt, pt), ap=average_precision_score(yt, pt), f1=f1_score(yt, pt >= th, zero_division=0),
               ece=ece_score(yt, pt), brier=brier_score_loss(yt, pt))
    neg = np.sort(pt[yt == 0])[::-1]
    thr1 = neg[max(0, int(0.01 * len(neg)) - 1)]
    out["rec_fpr1"] = float((pt[yt == 1] > thr1).mean())
    out["_th"] = th
    return out, pt


def unsup_eval(score, y, role):
    te = role == "test"
    return dict(auroc=roc_auc_score(y[te], score[te]), ap=average_precision_score(y[te], score[te]))


def base_oof(F, y, role, cols, seed, groups):
    """Out-of-fold base-detector probabilities on train accounts; train-fit predictions for the rest (label-free for test)."""
    s0 = np.zeros(len(F)); tr = np.where((role != "test") & (role != "drop"))[0]
    rng = np.random.default_rng(seed + 5); ug = np.unique(groups[tr]); rng.shuffle(ug)
    folds = np.array_split(ug, 5)
    for f in folds:
        te_idx = tr[np.isin(groups[tr], f)]; tr_idx = np.setdiff1d(tr, te_idx)
        m = gbm(seed).fit(F.iloc[tr_idx][cols].values, y[tr_idx]); s0[te_idx] = m.predict_proba(F.iloc[te_idx][cols].values)[:, 1]
    m = gbm(seed).fit(F.iloc[tr][cols].values, y[tr]); te_idx = np.where(role == "test")[0]
    s0[te_idx] = m.predict_proba(F.iloc[te_idx][cols].values)[:, 1]
    return s0


def build_graph(R, X, n_users, T, layers=("sync", "content", "media"), validated=True, discount=True, mode="noisyor", seed=0, a_min=0.3):
    L = {}
    if "sync" in layers: L["sync"] = cg.sync_layer(R, n_users, validated=validated)
    if "content" in layers: L["content"] = cg.content_layer(R, X, discount=discount)
    if "media" in layers: L["media"] = cg.media_layer(R, discount=discount)
    W, mats = cg.fuse_layers(L, n_users, mode=mode, a_min=a_min)
    feats, cid, W = cg.graph_features(W, mats, R, n_users, T, seed=seed)
    return W, mats, feats, cid, L


def cell_recovery(cid, A, thr=0.5):
    hits, tot = 0, 0
    for c, g in A[A.cell != ""].groupby("cell"):
        mem = g["user"].values; tot += 1
        comm = cid[mem]; comm = comm[comm >= 0]
        if len(comm) == 0: continue
        best = np.bincount(comm).argmax(); size = (cid == best).sum()
        inter = (comm == best).sum(); jac = inter / (len(mem) + size - inter)
        hits += jac >= thr
    return hits / max(tot, 1)


def rating_distortion(R, A, flagged, y):
    """MAE of product mean rating vs genuine-only mean on products touched by campaigns, after removing flagged accounts."""
    fake_prod = R[R["user"].isin(A.user[A.fake])]["product"].unique()
    sub = R[R["product"].isin(fake_prod)]
    truth = sub[~sub["user"].isin(A.user[A.fake])].groupby("product")["rating"].mean()
    def mae(mask):
        est = sub[mask].groupby("product")["rating"].mean().reindex(truth.index)
        return float(np.nanmean(np.abs(est - truth)))
    return mae(np.ones(len(sub), bool)), mae(~sub["user"].isin(flagged).values)


class Pool:
    """Accumulates out-of-fold test predictions across CV folds."""
    def __init__(self): self.y, self.p, self.pred, self.idx = [], [], [], []
    def add(self, yt, pt, th, idx=None): self.y.append(yt); self.p.append(pt); self.pred.append(pt >= th); self.idx.append(idx)
    def metrics(self):
        y, p, pr = map(np.concatenate, (self.y, self.p, self.pred))
        neg = np.sort(p[y == 0])[::-1]; thr1 = neg[max(0, int(0.01 * len(neg)) - 1)]
        return dict(auroc=roc_auc_score(y, p), ap=average_precision_score(y, p), f1=f1_score(y, pr, zero_division=0),
                    ece=float(ece_score(y, p)), brier=brier_score_loss(y, p), rec_fpr1=float((p[y == 1] > thr1).mean()))


def assign_folds(groups, fake, seed, k=5):
    rng = np.random.default_rng(seed + 991)
    fold = np.zeros(len(groups), int)
    for is_fake in (True, False):
        ug = np.unique(groups[fake == is_fake]); rng.shuffle(ug)
        gf = {g: i % k for i, g in enumerate(ug)}
        m = fake == is_fake
        fold[m] = [gf[g] for g in groups[m]]
    return fold


def fit_fold(Ftr, ytr, Fva, yva, Fte, seed):
    m = gbm(seed).fit(Ftr, ytr)
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    pv_raw, pt_raw = m.predict_proba(Fva)[:, 1], m.predict_proba(Fte)[:, 1]
    cal = LogisticRegression(C=1e6).fit(lg(pv_raw)[:, None], yva)
    pv, pt = cal.predict_proba(lg(pv_raw)[:, None])[:, 1], cal.predict_proba(lg(pt_raw)[:, None])[:, 1]
    ths = np.linspace(0.05, 0.95, 91)
    th = ths[int(np.argmax([f1_score(yva, pv >= t, zero_division=0) for t in ths]))]
    return pt, th


def run_once(cfg: SimConfig, seed: int, which="main"):
    t0 = time.time()
    R, X, A, info = simulate(cfg, seed)
    return evaluate_market(R, X, A, info["T"], seed, which, t0)


def evaluate_market(R, X, A, T, seed, which="main", t0=None):
    """Evaluate on any market given as review table R, unit-norm text embeddings X and account table A
    (columns: user, fake, group, cell, age, terse, family, flash, ai_assist)."""
    t0 = t0 or time.time()
    n = len(A)
    present = np.zeros(n, bool); present[R["user"].unique()] = True
    F = account_features(R, A, T)
    y = A["fake"].values.astype(int)
    fold = assign_folds(A["group"].values, A["fake"].values, seed)
    fold[~present] = -1                                  # accounts with no observed review are excluded
    out, pools = {}, {}
    from scipy import sparse as sp
    # ---- label-free graph structures (built once)
    u, v, a = cg.sync_layer(R, n, validated=False)
    Wraw = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr(); Wraw = Wraw.maximum(Wraw.T)
    sign = pd.DataFrame({f"sg_{c}_{h}": (cg.neighbour_mean(Wraw, F[c].values) if h == 1 else cg.neighbour_mean(Wraw, cg.neighbour_mean(Wraw, F[c].values)))
                         for c in ["t_max", "i_max", "gap_mean", "r_five", "burst30"] for h in (1, 2)})
    sign["sg_deg"] = np.asarray(Wraw.sum(1)).ravel()
    db = cg.dense_block_scores(R, n)
    m_ = fold >= 0
    out["dense_block"] = dict(auroc=roc_auc_score(y[m_], db[m_]), ap=average_precision_score(y[m_], db[m_]))

    gcache = {}
    def graph(layers=("sync", "content", "media"), validated=True, discount=True, mode="noisyor", a_min=0.3):
        key = (layers, validated, discount, mode, a_min)
        if key not in gcache: gcache[key] = build_graph(R, X, n, T, layers, validated, discount, mode, seed, a_min)
        return gcache[key]

    base_cols = STAGE + META
    methods = {  # name -> (kwargs for graph, flags)
        "full": dict(), "graph_only": dict(graph_only=True, prop=False),
        "abl_no_sync": dict(layers=("content", "media")), "abl_no_content": dict(layers=("sync", "media")),
        "abl_no_media": dict(layers=("sync", "content")), "abl_no_validation": dict(validated=False),
        "abl_no_discount": dict(discount=False), "abl_sum_fusion": dict(mode="sum"),
        "abl_no_propagation": dict(prop=False), "abl_no_community": dict(comm=False)}
    if which != "main": methods = {"full": dict()}
    simple = {"text_only": ["t_max", "t_mean", "t_pos"], "image_only": ["i_max", "i_n"], "meta_only": META, "stage_fusion": base_cols}
    for name in list(simple) + ["sign_rawgraph"] + list(methods): pools[name] = Pool()

    for k in range(5):
        role = np.array(["train"] * n, dtype=object); role[fold == k] = "test"; role[fold == (k + 1) % 5] = "val"; role[fold < 0] = "drop"
        tr, va, te = (role == "train"), (role == "val"), (role == "test")
        s0 = base_oof(F, y, role, base_cols, seed, A["group"].values)

        def run(name, FF, cols):
            pt, th = fit_fold(FF.loc[tr, cols].values, y[tr], FF.loc[va, cols].values, y[va], FF.loc[te, cols].values, seed)
            pools[name].add(y[te], pt, th, np.where(te)[0])
        for name, cols in simple.items(): run(name, F, cols)
        FS = pd.concat([F, sign], axis=1); run("sign_rawgraph", FS, base_cols + list(sign.columns))
        for name, kw in methods.items():
            kw = dict(kw); graph_only = kw.pop("graph_only", False); prop = kw.pop("prop", True); comm = kw.pop("comm", True)
            W, mats, gf, cid, L = graph(**kw)
            G = gf.drop(columns=["comm"])
            if not comm: G = G.drop(columns=[c for c in G.columns if c.startswith("c_")])
            if prop: G = G.assign(g_prop=cg.propagate(W, s0), g_nbr_s0=cg.neighbour_mean(W, s0))
            FF = pd.concat([F, G], axis=1)
            run(name, FF, list(G.columns) + ([] if graph_only else base_cols))
            if name == "full" and k == 0:
                out["_cell_recovery"] = cell_recovery(cid, A)
                out["_edges"] = {kk: int(len(vv[0])) for kk, vv in L.items()}
                out["_graph_nnz"] = int(W.nnz // 2)
                # edge-level precision of the fused graph (are linked accounts truly in the same campaign?)
                Wc = W.tocoo(); mk = Wc.row < Wc.col
                same = (A["cell"].values[Wc.row[mk]] == A["cell"].values[Wc.col[mk]]) & (A["cell"].values[Wc.row[mk]] != "")
                both_fake = A["fake"].values[Wc.row[mk]] & A["fake"].values[Wc.col[mk]]
                fam = (A["group"].values[Wc.row[mk]] == A["group"].values[Wc.col[mk]]) & ~A["fake"].values[Wc.row[mk]]
                supp = sum(((M + M.T) > 0).astype(float) for M in mats.values()).tocsr()
                sup = supp[Wc.row[mk], Wc.col[mk]]; sup = np.asarray(sup.todense() if hasattr(sup, 'todense') else sup).ravel()
                out["_edge_by_layers"] = {int(c): dict(n=int((sup == c).sum()), same_cell=float(same[sup == c].mean()) if (sup == c).any() else None) for c in (1, 2, 3)}
                out["_edge_quality"] = dict(same_cell=float(same.mean()) if mk.any() else 0.0, fake_fake=float(both_fake.mean()) if mk.any() else 0.0,
                                            honest_family=float(fam.mean()) if mk.any() else 0.0, n=int(mk.sum()))
    for name, p in pools.items(): out[name] = p.metrics()
    # false-positive rate on honest sub-populations (threshold fixed on validation folds)
    fpr = {}
    for name in ("stage_fusion", "sign_rawgraph", "full"):
        pl = pools[name]; idx = np.concatenate(pl.idx); pr = np.concatenate(pl.pred); yy = np.concatenate(pl.y)
        sub = A.iloc[idx]; reg = ~(sub["terse"] | sub["family"] | sub["flash"] | sub["ai_assist"]).values
        d = {"regular": reg, "terse": sub["terse"].values, "family": sub["family"].values, "flash": sub["flash"].values, "ai_assist": sub["ai_assist"].values}
        fpr[name] = {g: (float(pr[(yy == 0) & m].mean()) if ((yy == 0) & m).sum() >= 5 else None) for g, m in d.items()}
        fpr[name]["_n"] = {g: int(((yy == 0) & m).sum()) for g, m in d.items()}
    out["_fpr_groups"] = fpr
    out["_n"] = dict(users=int(n), reviews=int(len(R)), fake=int(y.sum()), prevalence=float(y.mean()))
    out["_secs"] = time.time() - t0
    return out
