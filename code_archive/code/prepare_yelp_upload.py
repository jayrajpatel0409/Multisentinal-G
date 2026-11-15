#!/usr/bin/env python3
"""Prepare YelpZip / YelpNYC / YelpChi for MultiSentinel-G (run in Google Colab or on your computer).

What it does
  1. Reads the two raw files (metadata + reviewContent), joins them, parses dates.
  2. Re-runs a TF-IDF + logistic-regression text baseline with reviewer-grouped 5-fold cross-validation, so every review gets an
     out-of-fold text score (no review is scored by a model that trained on the same reviewer). This score is the `text_ai_score` column.
  3. Writes reviews.csv + accounts.csv (text shortened to 300 characters, enough for near-duplicate detection) and zips them.

Usage (Colab cell):
  !python prepare_yelp_upload.py --meta metadata --content reviewContent --out yelpzip_prepared.zip
Options:
  --spam_value -1      value in the metadata label column that means "filtered/fake" (original Yelp files use -1)
  --max_reviews 0      if the zip is too big to upload, e.g. 300000: keeps WHOLE restaurants until this many reviews are reached
  --account_label any  account is fake if ANY of its reviews is filtered ("any") or the MAJORITY are ("majority")
Files may be tab- or comma-separated, with or without a header row. If anything looks wrong the script prints what it found.
"""
import argparse, csv, os, re, sys, zipfile, numpy as np, pandas as pd

def read_table(path, names):
    with open(path, encoding="utf-8", errors="replace") as f: first = f.readline()
    sep = "\t" if first.count("\t") >= first.count(",") else ","
    fields = [x.strip().strip('"') for x in first.rstrip("\n").split(sep)]
    has_header = any(re.match(r"^(user|prod|rest|rating|label|date|review|flag)", x, re.I) for x in fields)
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None, quoting=csv.QUOTE_NONE, on_bad_lines="skip",
                     encoding="utf-8", encoding_errors="replace", dtype=str)
    if has_header:
        low = {c: c.lower().replace("_", "").replace("id", "").strip() for c in df.columns}
        print(f"[{os.path.basename(path)}] header found: {list(df.columns)}")
    else:
        if df.shape[1] < len(names): sys.exit(f"{path}: expected at least {len(names)} columns, found {df.shape[1]}")
        df = df.iloc[:, :len(names)]; df.columns = names
        print(f"[{os.path.basename(path)}] no header; assumed columns {names}")
    return df

def pick(df, keys, what):
    for c in df.columns:
        if any(k in c.lower() for k in keys): return c
    sys.exit(f"cannot find the {what} column among {list(df.columns)}")

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True); ap.add_argument("--content", required=True); ap.add_argument("--out", default="prepared.zip")
ap.add_argument("--spam_value", default="-1"); ap.add_argument("--max_reviews", type=int, default=0)
ap.add_argument("--account_label", choices=["any", "majority"], default="any"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

M = read_table(a.meta, ["user_id", "prod_id", "rating", "label", "date"])
C = read_table(a.content, ["user_id", "prod_id", "date", "text"])
mu, mp, mr, ml, md = (pick(M, k, w) for k, w in [(["user"], "user"), (["prod", "rest", "item"], "product"), (["rating", "star"], "rating"), (["label", "flag"], "label"), (["date"], "date")])
cu, cp, cd, ct = (pick(C, k, w) for k, w in [(["user"], "user"), (["prod", "rest", "item"], "product"), (["date"], "date"), (["text", "review", "content"], "text")])
M = M.rename(columns={mu: "user_id", mp: "product_id", mr: "rating", ml: "label_raw", md: "date"})[["user_id", "product_id", "rating", "label_raw", "date"]]
C = C.rename(columns={cu: "user_id", cp: "product_id", cd: "date", ct: "text"})[["user_id", "product_id", "date", "text"]]
print("metadata rows:", len(M), "| content rows:", len(C))
if len(M) == len(C) and (M[["user_id", "product_id", "date"]].values == C[["user_id", "product_id", "date"]].values).all():
    D = M.copy(); D["text"] = C["text"].values; print("files align row by row: joined directly")
else:
    n0 = len(M); D = M.drop_duplicates(["user_id", "product_id", "date"]).merge(C.drop_duplicates(["user_id", "product_id", "date"]), on=["user_id", "product_id", "date"], how="inner")
    print(f"joined on (user, product, date): kept {len(D)} of {n0} metadata rows")
D["t"] = pd.to_datetime(D["date"], errors="coerce"); D = D.dropna(subset=["t"]).copy()
D["rating"] = pd.to_numeric(D["rating"], errors="coerce").fillna(3.0)
D["label"] = (D["label_raw"].astype(str).str.strip() == str(a.spam_value)).astype(int)
D["text"] = D["text"].fillna("").astype(str).str.replace(r"[\t\r\n]+", " ", regex=True)
print(f"reviews {len(D)} | filtered share {D.label.mean():.4f} | reviewers {D.user_id.nunique()} | products {D.product_id.nunique()} | {D.t.min().date()} .. {D.t.max().date()}")
if not 0.02 < D.label.mean() < 0.6: print("WARNING: filtered share looks odd. Check --spam_value (values seen:", D.label_raw.value_counts().head(5).to_dict(), ")")
rng = np.random.RandomState(a.seed)
if a.max_reviews and len(D) > a.max_reviews:
    sizes = D.groupby("product_id").size(); prods = rng.permutation(sizes.index.values); keep = prods[np.cumsum(sizes[prods].values) <= a.max_reviews]
    D = D[D.product_id.isin(keep)].copy(); print(f"subsampled to {len(D)} reviews from {len(keep)} whole restaurants")
D = D.reset_index(drop=True); D["review_id"] = np.arange(len(D)); D["days"] = (D.t - D.t.min()).dt.total_seconds() / 86400.0

# ---- out-of-fold text score (reviewer-grouped, stratified): text-only TF-IDF + logistic regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
print("computing out-of-fold text scores ...", flush=True)
X = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True, dtype=np.float32).fit_transform(D.text)
oof = np.zeros(len(D), dtype=np.float32)
for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(X, D.label, D.user_id)):
    m = LogisticRegression(C=1.0, max_iter=200).fit(X[tr], D.label.values[tr]); oof[te] = m.predict_proba(X[te])[:, 1]; print(f"  fold {k + 1}/5", flush=True)
print(f"out-of-fold text score: review-level AP {average_precision_score(D.label, oof):.4f}, AUROC {roc_auc_score(D.label, oof):.4f}")

rv = pd.DataFrame({"review_id": D.review_id, "user_id": D.user_id, "product_id": D.product_id, "timestamp": D.days.round(4), "rating": D.rating,
                   "text": D.text.str.slice(0, 300), "text_ai_score": oof.round(5), "image_mismatch_score": 0.0, "sentiment_gap": 0.0, "image_hash": "", "label": D.label})
g = D.groupby("user_id"); lab = g.label.max() if a.account_label == "any" else (g.label.mean() > 0.5).astype(int)
ac = pd.DataFrame({"user_id": lab.index, "account_created": g.days.min().reindex(lab.index).values, "label": lab.values})
print(f"accounts {len(ac)} | fake accounts {ac.label.sum()} ({ac.label.mean():.3f}) | accounts with >=2 reviews {(g.size() >= 2).sum()}")
rv.to_csv("reviews.csv", index=False); ac.to_csv("accounts.csv", index=False)
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z: z.write("reviews.csv"); z.write("accounts.csv")
print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB.  Upload this zip here.")
if os.path.getsize(a.out) > 90e6: print("The zip is large. If upload fails, re-run with --max_reviews 300000.")
