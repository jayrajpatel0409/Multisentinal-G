"""Coordination-evidence multiplex graph, account features, and score propagation.

Layers (accounts are nodes):
  SYNC   statistically validated co-review synchrony   (Chung-Lu / Poisson null + BH-FDR)
  CONTENT near-duplicate text with generic-phrase discount (collision-rate surprisal)
  MEDIA  shared image hashes with popularity discount
Layers are fused by noisy-OR into edge confidence a_uv in [0,1].
"""
import numpy as np, pandas as pd, scipy.sparse as sp, networkx as nx, heapq
from scipy.stats import poisson

# ------------------------------------------------------------------ shared validation machinery
def poisson_validate(k, mu, n_tests, q, smax):
    """Chung-Lu/Poisson null for pair co-occurrence counts, Benjamini-Hochberg FDR over the n_tests testable pairs.
    Returns (keep_mask, edge_confidence for kept pairs)."""
    p = np.maximum(poisson.sf(k - 1, mu), 1e-300)
    order = np.argsort(p)
    ok = np.where(p[order] <= q * np.arange(1, len(p) + 1) / max(n_tests, 1))[0]
    cutoff = ok.max() if len(ok) else -1
    keep = np.zeros(len(p), bool); keep[order[: cutoff + 1]] = True
    return keep, np.minimum(1.0, -np.log10(p[keep]) / smax)


# ------------------------------------------------------------------ SYNC layer
def sync_layer(R, n_users, W=7.0, q=0.01, smax=10.0, validated=True, min_k=2):
    ev_key = R["product"].values.astype(np.int64) * 100000 + np.floor(R["t"].values / W).astype(np.int64)
    _, ev = np.unique(ev_key, return_inverse=True)
    ui = R["user"].values
    B = sp.csr_matrix((np.ones(len(R)), (ui, ev)), shape=(n_users, ev.max() + 1))
    B.sum_duplicates(); B.data[:] = 1.0
    nu = np.asarray(B.sum(1)).ravel(); se = np.asarray(B.sum(0)).ravel(); S = se.sum()
    K = sp.triu(B @ B.T, k=1).tocoo()
    n_tests = len(K.data)                       # testable pairs: at least one shared product-week event
    m = K.data >= min_k
    u, v, k = K.row[m], K.col[m], K.data[m]
    if not validated:                           # ablation: raw co-occurrence counts
        return u, v, np.minimum(1.0, (k - 1) / 3.0)
    mu = nu[u] * nu[v] * (se ** 2).sum() / S ** 2
    keep, a = poisson_validate(k, mu, n_tests, q, smax)
    return u[keep], v[keep], a


# ------------------------------------------------------------------ CONTENT layer
def content_layer(R, X, theta=0.80, q=0.01, smax=10.0, discount=True, chunk=None, min_k=2):
    N = len(R); users = R["user"].values
    chunk = chunk or int(min(2500, max(100, 3e7 / N)))
    pi, pj = [], []
    for i0 in range(0, N, chunk):
        S = X[i0:i0 + chunk] @ X.T
        S[np.arange(S.shape[0]), np.arange(i0, i0 + S.shape[0])] = -1
        r, c = np.where(S >= theta)
        keep = (c > (r + i0)) & (users[c] != users[r + i0])
        pi.append(r[keep] + i0); pj.append(c[keep])
    pi, pj = np.concatenate(pi), np.concatenate(pj)
    if len(pi) == 0: return np.array([], int), np.array([], int), np.array([])
    deg = np.bincount(pi, minlength=N) + np.bincount(pj, minlength=N)          # cross-account near-duplicate degree
    a_, b_ = np.minimum(users[pi], users[pj]), np.maximum(users[pi], users[pj])
    df = pd.DataFrame({"u": a_, "v": b_}).groupby(["u", "v"], as_index=False).size().rename(columns={"size": "k"})
    n_tests = len(df); df = df[df.k >= min_k]
    if not discount:
        return df.u.values, df.v.values, np.minimum(1.0, df.k.values / 4.0)
    D = np.bincount(users, weights=deg, minlength=users.max() + 1)             # account-level "genericness" mass
    mu = D[df.u.values] * D[df.v.values] / deg.sum()
    keep, a = poisson_validate(df.k.values, mu, n_tests, q, smax)
    return df.u.values[keep], df.v.values[keep], a


# ------------------------------------------------------------------ MEDIA layer
def media_layer(R, q=0.01, smax=10.0, discount=True, min_k=1):
    I = R[R["img"] >= 0][["user", "img"]].drop_duplicates()
    sizes = I.groupby("img")["user"].transform("size")
    I = I[sizes >= 2]
    if I.empty: return np.array([], int), np.array([], int), np.array([])
    us, vs = [], []
    for h, grp in I.groupby("img"):
        gg = np.sort(grp["user"].values)[:300]
        iu, iv = np.triu_indices(len(gg), 1); us.append(gg[iu]); vs.append(gg[iv])
    df = pd.DataFrame({"u": np.concatenate(us), "v": np.concatenate(vs)}).groupby(["u", "v"], as_index=False).size().rename(columns={"size": "k"})
    n_tests = len(df); df = df[df.k >= min_k] if min_k > 1 else df
    if not discount:
        return df.u.values, df.v.values, np.minimum(1.0, df.k.values / 2.0)
    gsz = I.groupby("img")["user"].transform("size").values
    D = np.bincount(I["user"].values, weights=gsz - 1, minlength=R["user"].max() + 1)
    mu = D[df.u.values] * D[df.v.values] / D.sum()
    keep, a = poisson_validate(df.k.values, mu, n_tests, q, smax)
    return df.u.values[keep], df.v.values[keep], a


# ------------------------------------------------------------------ fusion
def fuse_layers(layers, n_users, mode="noisyor", a_min=0.3):
    """layers: dict name -> (u,v,a).  Returns sparse symmetric weight matrix + per-layer matrices."""
    mats = {}
    for name, (u, v, a) in layers.items():
        M = sp.coo_matrix((a, (u, v)), shape=(n_users, n_users)).tocsr() if len(a) else sp.csr_matrix((n_users, n_users))
        mats[name] = M
    if not mats: return sp.csr_matrix((n_users, n_users)), mats
    if mode == "noisyor":
        comp = None
        for M in mats.values():
            one_minus = M.copy(); one_minus.data = 1 - one_minus.data
            # prod over layers of (1-a) restricted to union support -> compute via log
            L = M.copy(); L.data = -np.log(np.maximum(1 - M.data, 1e-9))
            comp = L if comp is None else comp + L
        comp = comp.tocoo()
        W = sp.coo_matrix((1 - np.exp(-comp.data), (comp.row, comp.col)), shape=comp.shape).tocsr()
    else:                                                # sum then clip
        tot = sum(mats.values()).tocoo()
        W = sp.coo_matrix((np.minimum(1.0, tot.data), (tot.row, tot.col)), shape=tot.shape).tocsr()
    W = W.tocoo(); keep = W.data >= a_min
    W = sp.coo_matrix((W.data[keep], (W.row[keep], W.col[keep])), shape=W.shape).tocsr()
    W = W.maximum(W.T)
    return W, mats


# ------------------------------------------------------------------ graph features
def graph_features(W, mats, R, n_users, T, seed=0, resolution=1.0):
    G = nx.from_scipy_sparse_array(W)
    feats = pd.DataFrame(index=np.arange(n_users))
    strength = np.asarray(W.sum(1)).ravel(); deg = np.asarray((W > 0).sum(1)).ravel()
    feats["g_strength"] = strength; feats["g_degree"] = deg
    for name, M in mats.items():
        Ms = (M + M.T).tocsr()
        feats[f"g_str_{name}"] = np.asarray(Ms.sum(1)).ravel()
    feats["g_nlayers"] = sum((feats[f"g_str_{n}"] > 0).astype(int) for n in mats)
    # layer-agreement subgraph: edges supported by >= 2 independent evidence layers
    supp = None
    for M in mats.values():
        Bm = ((M + M.T) > 0).astype(np.float64)
        supp = Bm if supp is None else supp + Bm
    multi = (supp >= 2).astype(np.float64).multiply(W) if supp is not None else W * 0
    feats["g_str_multi"] = np.asarray(multi.sum(1)).ravel()
    feats["g_deg_multi"] = np.asarray((multi > 0).sum(1)).ravel()
    # local structure
    sub = G.subgraph([n for n in G.nodes if G.degree(n) > 0])
    Ab = (W > 0).astype(np.float64).tocsr()
    tri = np.asarray((Ab @ Ab).multiply(Ab).sum(1)).ravel() / 2.0
    bd = np.asarray(Ab.sum(1)).ravel()
    feats["g_clust"] = np.where(bd > 1, tri / np.maximum(bd * (bd - 1) / 2.0, 1), 0.0)
    core = nx.core_number(nx.Graph(sub)) if sub.number_of_nodes() else {}
    feats["g_core"] = pd.Series(core).reindex(feats.index).fillna(0.0)
    # communities
    comms = nx.community.louvain_communities(sub, weight="weight", resolution=resolution, seed=seed) if sub.number_of_nodes() else []
    cid = -np.ones(n_users, int)
    for k, c in enumerate(comms): cid[list(c)] = k
    feats["comm"] = cid
    # per-community statistics
    Rc = R.assign(comm=cid[R["user"].values])
    Rc = Rc[Rc.comm >= 0]
    size = np.bincount(cid[cid >= 0], minlength=len(comms)) if len(comms) else np.array([])
    dens = np.zeros(len(comms)); focus = np.zeros(len(comms)); compact = np.zeros(len(comms))
    Wc = W.tocoo()
    same = (cid[Wc.row] == cid[Wc.col]) & (cid[Wc.row] >= 0) & (Wc.row < Wc.col)
    wsum = np.bincount(cid[Wc.row][same], weights=Wc.data[same], minlength=len(comms)) if len(comms) else np.array([])
    pairs = size * (size - 1) / 2
    dens = np.where(pairs > 0, wsum / np.maximum(pairs, 1), 0.0)
    if len(Rc):
        cnt = Rc.groupby(["comm", "product"]).size().rename("n").reset_index()
        tot = cnt.groupby("comm")["n"].transform("sum")
        cnt["share"] = cnt["n"] / tot
        # Herfindahl index of the community's review mass over products ("campaign focus")
        hhi = cnt.assign(h=cnt.share ** 2).groupby("comm")["h"].sum()
        # share of the community's reviews that land on its top-3 products
        top3 = cnt.sort_values(["comm", "n"], ascending=[True, False]).groupby("comm").head(3).groupby("comm")["n"].sum() / cnt.groupby("comm")["n"].sum()
        # temporal compactness on the community's top product: IQR(t)/T
        topp = cnt.sort_values(["comm", "n"], ascending=[True, False]).groupby("comm").head(1).set_index("comm")["product"]
        tt = Rc.merge(topp.rename("tp"), left_on="comm", right_index=True)
        tt = tt[tt["product"] == tt["tp"]]
        iqr = tt.groupby("comm")["t"].apply(lambda s: np.subtract(*np.percentile(s, [75, 25])) / T if len(s) > 1 else 1.0)
        for k in range(len(comms)):
            focus[k] = top3.get(k, 0.0); compact[k] = 1 - min(1.0, iqr.get(k, 1.0) * 10)
        hh = hhi.reindex(range(len(comms))).fillna(0).values
    else:
        hh = np.zeros(len(comms))
    ok = cid >= 0
    for name, arr in [("c_size", size), ("c_density", dens), ("c_focus", focus), ("c_compact", compact), ("c_hhi", hh)]:
        col = np.zeros(n_users)
        if len(arr): col[ok] = arr[cid[ok]]
        feats[name] = col
    return feats, cid, W


def propagate(W, s0, alpha=0.8, iters=60):
    """Coordination-aware propagation: s = (1-a) s0 + a * P s, P row-normalised weighted adjacency."""
    d = np.asarray(W.sum(1)).ravel(); iso = d == 0; d[iso] = 1
    P = (sp.diags(1.0 / d) @ W + sp.diags(iso.astype(float))).tocsr()      # isolated nodes keep their own score (self-loop)
    s = s0.copy()
    for _ in range(iters):
        s = (1 - alpha) * s0 + alpha * (P @ s)
    return s


def neighbour_mean(W, x):
    d = np.asarray(W.sum(1)).ravel(); d[d == 0] = 1
    return (sp.diags(1.0 / d) @ W) @ x


# ------------------------------------------------------------------ dense-block baseline (unsupervised, FRAUDAR-style)
def dense_block_scores(R, n_users, W=7.0, n_blocks=15):
    ev_key = R["product"].values.astype(np.int64) * 100000 + np.floor(R["t"].values / W).astype(np.int64)
    _, ev = np.unique(ev_key, return_inverse=True)
    E = pd.DataFrame({"u": R["user"].values, "e": ev}).drop_duplicates()
    ecount = E.groupby("e")["u"].transform("size").values
    E["w"] = 1.0 / np.log(ecount + 5.0)
    scores = np.zeros(n_users)
    for _ in range(n_blocks):
        if len(E) < 5: break
        adj_u, adj_e = {}, {}
        for u, e, w in zip(E.u.values, E.e.values, E.w.values):
            adj_u.setdefault(("u", u), []).append((("e", e), w)); adj_e.setdefault(("e", e), []).append((("u", u), w))
        adj = {**adj_u, **adj_e}
        wdeg = {n: sum(w for _, w in nb) for n, nb in adj.items()}
        total = sum(wdeg.values()) / 2.0
        heap = [(d, n) for n, d in wdeg.items()]; heapq.heapify(heap)
        alive = set(adj); order = []; best, best_i = -1, 0
        nodes = len(alive)
        removed = []
        while heap:
            d, n = heapq.heappop(heap)
            if n not in alive or abs(d - wdeg[n]) > 1e-12: continue
            dens = total / nodes
            if dens > best: best, best_i = dens, len(removed)
            alive.remove(n); removed.append(n); total -= wdeg[n]; nodes -= 1
            for nb, w in adj[n]:
                if nb in alive:
                    wdeg[nb] -= w; heapq.heappush(heap, (wdeg[nb], nb))
        block = set(removed[best_i:])
        bu = [n[1] for n in block if n[0] == "u"]; be = [n[1] for n in block if n[0] == "e"]
        if len(bu) < 3: break
        for u in bu: scores[u] = max(scores[u], best)
        E = E[~E.u.isin(bu)]
    return scores
