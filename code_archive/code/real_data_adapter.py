"""Run MultiSentinel-G on REAL review data (or on semi-synthetic data with injected campaigns).

Input files (CSV, UTF-8):
  reviews.csv  : review_id, user_id, product_id, timestamp, rating, text_ai_score, image_mismatch_score, sentiment_gap,
                 image_hash (empty if no image)            [+ optional column `text` if no embeddings are supplied]
  accounts.csv : user_id, account_created (same time unit as timestamp) [+ optional: label (1=campaign, 0=honest), group]
  embeddings.npy (optional): float array [n_reviews, d] aligned with the row order of reviews.csv (e.g. a multilingual
                 sentence encoder). If absent, character n-gram TF-IDF + SVD is used (language-agnostic but weaker).
Timestamps are interpreted as days (float) or ISO-8601 strings.

Usage:
  python real_data_adapter.py reviews.csv accounts.csv [embeddings.npy] --out scores.csv
Without labels it writes per-account graph features, the community id and a coordination score; with labels it also
runs the same campaign-grouped 5-fold evaluation used in the paper (use `group` to keep households / cells together).
"""
import sys, argparse, numpy as np, pandas as pd
import coordgraph as cg
from pipeline import account_features, build_graph, evaluate_market


def load(reviews_csv, accounts_csv, emb_npy=None, dim=64):
    rv = pd.read_csv(reviews_csv); ac = pd.read_csv(accounts_csv)
    t = pd.to_numeric(rv["timestamp"], errors="coerce")
    if t.isna().any(): t = (pd.to_datetime(rv["timestamp"]) - pd.to_datetime(rv["timestamp"]).min()).dt.total_seconds() / 86400.0
    uid = {u: i for i, u in enumerate(pd.unique(pd.concat([ac["user_id"], rv["user_id"]])))}
    pid = {p: i for i, p in enumerate(pd.unique(rv["product_id"]))}
    R = pd.DataFrame({"user": rv["user_id"].map(uid), "product": rv["product_id"].map(pid), "t": t - t.min(), "rating": rv["rating"],
                      "s_text": rv["text_ai_score"], "s_img": rv["image_mismatch_score"].fillna(0.0), "s_gap": rv["sentiment_gap"].abs()})
    h = rv.get("image_hash", pd.Series([None] * len(rv)))
    codes = pd.factorize(h)[0]; R["img"] = np.where(h.isna(), -1, codes + 1_000_000); R["has_img"] = R["img"] >= 0
    R["rid"] = np.arange(len(R)); R["review_id"] = R["rid"]
    if emb_npy:
        X = np.load(emb_npy).astype(np.float32)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        M = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit_transform(rv["text"].fillna(""))
        X = TruncatedSVD(dim, random_state=0).fit_transform(M).astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    n = len(uid); A = pd.DataFrame({"user": np.arange(n)})
    ac = ac.assign(user=ac["user_id"].map(uid)).drop_duplicates("user").set_index("user")
    created = pd.to_numeric(ac.get("account_created"), errors="coerce")
    A["age"] = (t.max() - created.reindex(A.user).fillna(t.min())).clip(lower=0).values if created is not None else 0.0
    A["fake"] = ac["label"].reindex(A.user).fillna(0).astype(int).astype(bool).values if "label" in ac else False
    A["group"] = ac["group"].reindex(A.user).fillna(pd.Series(A.user.astype(str).radd("solo"), index=A.user)).values if "group" in ac else ("solo" + A.user.astype(str)).values
    A["cell"] = ""; A["terse"] = A["family"] = A["flash"] = A["ai_assist"] = False
    return R, X, A, uid, float(R["t"].max() + 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("reviews"); ap.add_argument("accounts"); ap.add_argument("emb", nargs="?")
    ap.add_argument("--out", default="scores.csv"); a = ap.parse_args()
    R, X, A, uid, T = load(a.reviews, a.accounts, a.emb)
    W, mats, gf, cid, L = build_graph(R, X, len(A), T)
    F = account_features(R, A, T)
    out = pd.concat([gf, F], axis=1); out.insert(0, "user_id", list(uid.keys())[:len(A)]); out["coordination_score"] = cg.propagate(W, np.minimum(1.0, out["g_str_multi"].values))
    out.to_csv(a.out, index=False); print("wrote", a.out, "| layer edges:", {k: len(v[0]) for k, v in L.items()})
    if A["fake"].any():
        print(evaluate_market(R, X, A, T, seed=0, which="lite"))
