"""Build the YelpZip coordination layers once and cache them (label-free). Usage: python yelp_graph_probe.py <dir with reviews.csv/accounts.csv>"""
import sys, time, pickle, numpy as np, pandas as pd, scipy.sparse as sp
import coordgraph as cg
from sklearn.feature_extraction.text import TfidfVectorizer
D = sys.argv[1]; t0 = time.time(); tick = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
rv = pd.read_csv(f"{D}/reviews.csv", dtype={"user_id": str, "product_id": str}, keep_default_na=False)
ac = pd.read_csv(f"{D}/accounts.csv", dtype={"user_id": str})
uid = {u: i for i, u in enumerate(ac["user_id"])}; n = len(uid)
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
z = lg(rv["text_ai_score"].values.astype(float)); z = (z - z.mean()) / z.std()
R = pd.DataFrame({"user": rv["user_id"].map(uid).values, "product": pd.factorize(rv["product_id"])[0], "t": rv["timestamp"].values,
                  "rating": rv["rating"].values, "s_text": z, "s_img": 0.0, "s_gap": 0.0, "img": -1, "has_img": False, "rid": np.arange(len(rv))})
R["review_id"] = R["rid"]; T = float(R["t"].max() + 1); y_rev = rv["label"].values.astype(int)
A = pd.DataFrame({"user": np.arange(n), "fake": ac["label"].values.astype(bool)}); created = ac["account_created"].values
A["age"] = np.clip(R["t"].max() - created, 0, None)
tick(f"loaded {len(R)} reviews, {n} accounts")

# ---- SYNC layer (validated and raw)
L = {}
L["sync"] = cg.sync_layer(R, n); tick(f"sync validated edges: {len(L['sync'][0])}")
L["sync_raw"] = cg.sync_layer(R, n, validated=False); tick(f"sync raw (k>=2) edges: {len(L['sync_raw'][0])}")

# ---- CONTENT layer: near-duplicate search with hashed word 4-gram shingles (copy-paste / lightly edited text), exact sparse cosine
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
hv = HashingVectorizer(ngram_range=(4, 4), n_features=2 ** 24, alternate_sign=False, norm=None, binary=True, lowercase=True, dtype=np.float32,
                       token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9']*\b")
H = hv.transform(rv["text"].fillna("").values).tocsc(); dfc = np.diff(H.indptr)
H = H[:, np.where((dfc >= 2) & (dfc <= 200))[0]].tocsr()          # drop singletons (cannot match) and boilerplate 4-grams
Xs = normalize(H); nnz = np.diff(Xs.indptr); tick(f"shingle matrix {Xs.shape}, docs with >=5 informative 4-grams: {(nnz >= 5).sum()}")
elig = np.where(nnz >= 5)[0]; Xe = Xs[elig]
def find_pairs(Xe, tag):
    XT = Xe.T.tocsr(); pi, pj = [], []
    for i0 in range(0, len(elig), 2000):
        S = (Xe[i0:i0 + 2000] @ XT).tocoo(); m = (S.data >= 0.6) & (S.col > S.row + i0)
        pi.append(elig[S.row[m] + i0]); pj.append(elig[S.col[m]])
    tick(f"  {tag}: pair search done")
    return np.concatenate(pi), np.concatenate(pj)
pi, pj = find_pairs(Xe, "real")
users = R["user"].values; cross = users[pi] != users[pj]; pi, pj = pi[cross], pj[cross]
tick(f"cross-account near-duplicate review pairs (shingle cos>=0.6): {len(pi)}")
prod = R["product"].values; tt = R["t"].values; same_p = (prod[pi] == prod[pj]).mean(); same_w = ((prod[pi] == prod[pj]) & (np.abs(tt[pi] - tt[pj]) <= 7)).mean()
tick(f"near-dup pairs: same restaurant {same_p:.3f}, same restaurant within 7 days {same_w:.3f}  (chance for random review pairs: ~{(np.bincount(prod) ** 2).sum() / len(prod) ** 2:.4f})")
N = len(R); deg = np.bincount(pi, minlength=N) + np.bincount(pj, minlength=N)
a_, b_ = np.minimum(users[pi], users[pj]), np.maximum(users[pi], users[pj])
df = pd.DataFrame({"u": a_, "v": b_}).groupby(["u", "v"], as_index=False).size().rename(columns={"size": "k"}); n_tests = len(df)
tick(f"account pairs with >=1 near-duplicate pair: {n_tests}; with >=2: {(df.k >= 2).sum()}")
dd = df[df.k >= 2]; Dg = np.bincount(users, weights=deg, minlength=n)
mu = Dg[dd.u.values] * Dg[dd.v.values] / max(deg.sum(), 1); keep, a = cg.poisson_validate(dd.k.values, mu, n_tests, 0.01, 10.0)
L["content"] = (dd.u.values[keep], dd.v.values[keep], a); L["content_raw"] = (dd.u.values, dd.v.values, np.minimum(1.0, dd.k.values / 4.0))
L["content_k1"] = (df.u.values, df.v.values, np.minimum(1.0, df.k.values / 4.0))
tick(f"content validated edges: {len(L['content'][0])}")
R['txtlen'] = rv['text'].str.len().values
pickle.dump(dict(R=R, A=A, T=T, y_rev=y_rev, L=L, n=n, pairs=(pi, pj)), open(f"{D}/graph_layers.pkl", "wb"))

# ---- diagnostics: are linked accounts more often both-fake than chance?
fake = A["fake"].values; base = fake.mean()
for name, (u, v, a) in L.items():
    if len(u) == 0: print(f"{name:12s} edges 0"); continue
    bf = (fake[u] & fake[v]).mean(); anyf = (fake[u] | fake[v]).mean(); ex = fake[u].mean()
    print(f"{name:12s} edges {len(u):8d}  P(both fake) {bf:.3f}  (chance {base ** 2:.3f}, lift {bf / base ** 2:.1f}x)  P(v fake | u fake) {(fake[v][fake[u]].mean() if fake[u].any() else float('nan')):.3f}  nodes {len(np.unique(np.r_[u, v]))}")
tick("done")
