"""Injected-campaign stress test on REAL YelpZip background (account-level evaluation).
Background = accounts whose every review was kept by Yelp's filter ("genuine" by platform label). Campaigns are injected with seeded,
documented attack strategies; campaign scripts are real review texts taken from Yelp-filtered reviews (never from the background).
Usage: python yelp_inject.py <dir> <scenario> <seed>"""
import sys, time, json, pickle, numpy as np, pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
import coordgraph as cg, pipeline as pl

SCEN = {  # attack strategy parameters (see paper): window days, text drop/replace prob, cell size, n targets, camouflage hub
    "I0_baseline": dict(win=7.0, drop=0.10, repl=0.05, cell=(6, 18), tg=(3, 6), hub=False, pr=0.5),
    "I1_paraphrase": dict(win=7.0, drop=0.35, repl=0.25, cell=(6, 18), tg=(3, 6), hub=False, pr=0.5),
    "I2_jitter": dict(win=60.0, drop=0.10, repl=0.05, cell=(6, 18), tg=(3, 6), hub=False, pr=0.5),
    "I3_fragmented": dict(win=7.0, drop=0.10, repl=0.05, cell=(3, 6), tg=(3, 6), hub=False, pr=0.5),
    "I4_camouflage": dict(win=7.0, drop=0.10, repl=0.05, cell=(6, 18), tg=(3, 6), hub=True, pr=0.5),
    "I5_adaptive": dict(win=60.0, drop=0.35, repl=0.25, cell=(3, 6), tg=(2, 4), hub=True, pr=0.5)}
D, scen, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); cfg = SCEN[scen]; rng = np.random.default_rng(seed * 7919 + 13)
t0 = time.time(); tick = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
P = pickle.load(open(f"{D}/graph_layers.pkl", "rb")); R0, A0, y_rev = P["R"], P["A"], P["y_rev"]
txt = pd.read_csv(f"{D}/reviews.csv", usecols=["text"], keep_default_na=False)["text"].values
acc_fake = A0["fake"].values
bg_mask = ~acc_fake[R0["user"].values]                                   # reviews of accounts with no filtered review
Rb = R0[bg_mask].reset_index(drop=True); tb = txt[bg_mask]
old_users = np.unique(Rb["user"].values); remap = np.full(len(A0), -1); remap[old_users] = np.arange(len(old_users)); Rb["user"] = remap[Rb["user"].values]
nb = len(old_users); tick(f"background: {nb} accounts, {len(Rb)} reviews")
T = float(R0["t"].max() + 1)
# empirical ingredients
pop = Rb.groupby("product").size(); big = pop.index[pop.values >= 25].values; hubs = pop.sort_values(ascending=False).index[:40].values
nrev_emp = Rb.groupby("user").size().values; rate_emp = Rb["rating"].values; s_gen = Rb["s_text"].values
flag_idx = np.where(y_rev == 1)[0]; flag_idx = flag_idx[np.array([len(txt[i]) >= 150 for i in flag_idx])]; s_flag = R0["s_text"].values[flag_idx]
words = pd.Series(" ".join(txt[flag_idx[:20000]]).lower().split()).value_counts().index[:5000].values
times_by_prod = Rb.groupby("product")["t"].apply(np.asarray).to_dict(); prod_p = (pop / pop.sum())
# ---- inject campaigns
FR = float(__import__('os').environ.get('INJ_FRAC', '0.04')); n_inj = int(FR * nb); rows, texts, cells, cell_of = [], [], [], []; uid = nb; c = 0; lo, hi = cfg["cell"]
def perturb(s):
    w = s.split(); out = []
    for x in w:
        r = rng.random()
        if r < cfg["drop"]: continue
        out.append(words[rng.integers(len(words))] if r < cfg["drop"] + cfg["repl"] else x)
    return " ".join(out)
o = np.argsort(Rb["user"].values, kind="stable"); su, st, sr = Rb["user"].values[o], Rb["t"].values[o], Rb["rating"].values[o]
bounds = np.r_[0, np.flatnonzero(np.diff(su)) + 1, len(su)]; donors = [(st[a:b] - st[a:b].min(), sr[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
members_all = []
while uid - nb < n_inj:
    m = min(int(rng.integers(lo, hi + 1)), nb + n_inj - uid); tg = rng.choice(big, size=int(rng.integers(cfg["tg"][0], cfg["tg"][1] + 1)), replace=False)
    t_start = rng.uniform(0, T - cfg["win"] - 1); script = txt[flag_idx[rng.integers(len(flag_idx))]]
    for j in range(m):
        u = uid + j; k = 0
        for p in tg:
            if rng.random() < cfg["pr"] or (k == 0 and p == tg[-1]):
                rows.append((u, p, t_start + rng.uniform(0, cfg["win"]), 5.0 if rng.random() < 0.85 else 4.0, float(rng.choice(s_flag)), 1)); texts.append(perturb(script)); k += 1
        off, dr = donors[int(rng.integers(len(donors)))]                    # clone a real honest account's activity pattern (timing offsets, ratings)
        pad = max(0, len(off) - k); start = max(0.0, t_start - rng.uniform(0, 1000.0))
        for q_ in range(pad):                                                # honest-looking padding that matches an actual honest account
            if cfg["hub"]:
                p = hubs[rng.integers(len(hubs))]; tt = float(rng.choice(times_by_prod[p]))
            else:
                p = rng.choice(prod_p.index.values, p=prod_p.values); tt = float(np.clip(start + off[q_], 0, T - 1))
            rows.append((u, p, tt, float(dr[q_]), float(rng.choice(s_gen)), 0)); texts.append("")
        cells.append(f"cell{c}"); members_all.append(u)
    uid += m; c += 1
Ri = pd.DataFrame(rows, columns=["user", "product", "t", "rating", "s_text", "is_camp"])
n = uid; tick(f"injected {n - nb} accounts in {c} cells, {len(Ri)} reviews")
R = pd.concat([Rb[["user", "product", "t", "rating", "s_text"]], Ri[["user", "product", "t", "rating", "s_text"]]], ignore_index=True)
R["s_img"] = 0.0; R["s_gap"] = 0.0; R["img"] = -1; R["has_img"] = False; R["rid"] = np.arange(len(R)); R["review_id"] = R["rid"]
alltxt = np.concatenate([tb, np.array(texts, dtype=object)])
first = R.groupby("user")["t"].min().reindex(np.arange(n)).values
A = pd.DataFrame({"user": np.arange(n), "fake": np.arange(n) >= nb, "age": np.clip(T - first, 0, None)})
A["cell"] = ""; A.loc[nb:, "cell"] = cells; A["group"] = np.where(A["fake"], A["cell"], "u" + A["user"].astype(str))
A["group"] = pd.factorize(A["group"])[0]          # integer group ids (string isin() inside base_oof is very slow at this scale)
for col in ("terse", "family", "flash", "ai_assist"): A[col] = False
# ---- content layer on real text: exact shingle cosine, degree-corrected Poisson null (same procedure as the main experiment)
hv = HashingVectorizer(ngram_range=(4, 4), n_features=2 ** 24, alternate_sign=False, norm=None, binary=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9']*\b")
H = hv.transform(alltxt).tocsc(); dfc = np.diff(H.indptr); H = H[:, np.where((dfc >= 2) & (dfc <= 200))[0]].tocsr(); Xs = normalize(H)
elig = np.where(np.diff(Xs.indptr) >= 5)[0]; Xe = Xs[elig]; XT = Xe.T.tocsr(); pi, pj = [], []
for i0 in range(0, len(elig), 2000):
    S = (Xe[i0:i0 + 2000] @ XT).tocoo(); m_ = (S.data >= 0.6) & (S.col > S.row + i0); pi.append(elig[S.row[m_] + i0]); pj.append(elig[S.col[m_]])
pi, pj = np.concatenate(pi), np.concatenate(pj); user = R["user"].values; cr = user[pi] != user[pj]; pi, pj = pi[cr], pj[cr]; N = len(R)
tick(f"near-duplicate review pairs: {len(pi)}")
def content_patch(R_, X_, theta=0.8, q=0.01, smax=10.0, discount=True, chunk=None, min_k=2):
    a_, b_ = np.minimum(user[pi], user[pj]), np.maximum(user[pi], user[pj])
    df = pd.DataFrame({"u": a_, "v": b_}).groupby(["u", "v"], as_index=False).size().rename(columns={"size": "k"}); n_tests = len(df)
    if not discount: return df.u.values, df.v.values, np.minimum(1.0, df.k.values / 4.0)
    deg = np.bincount(pi, minlength=N) + np.bincount(pj, minlength=N); Dg = np.bincount(user, weights=deg, minlength=n)
    mu = Dg[df.u.values] * Dg[df.v.values] / max(deg.sum(), 1); keep, a = cg.poisson_validate(df.k.values, mu, n_tests, q, smax)
    return df.u.values[keep], df.v.values[keep], a
cg.content_layer = content_patch
res = pl.evaluate_market(R, np.zeros((N, 1), np.float32), A, T, seed, "main", t0); res["_scenario"] = scen; res["_seed"] = seed; res["_frac"] = FR
res["_inj"] = dict(accounts=int(n - nb), cells=int(c), reviews=int(len(Ri)), bg_accounts=int(nb))
with open(f"{D}/yelp_inject.jsonl" if FR == 0.04 else f"{D}/yelp_inject_frac.jsonl", "a") as f: f.write(json.dumps(res, default=float) + "\n")
for k, v in res.items():
    if isinstance(v, dict) and "ap" in v: print(f"{k:20s} AP {v['ap']:.4f} AUROC {v['auroc']:.4f} F1 {v.get("f1", float("nan")):.4f} Rec@1%FPR {v.get("rec_fpr1", float("nan")):.4f}")
tick("done")
