#!/usr/bin/env python3
"""Prepare the Kaggle YelpNYC files for MultiSentinel-G (Google Colab).
Inputs: "Yelp NYC Metadata.csv" (has user id, label, rating) and "yelp.csv" (Review_id, Product_id, Date, Review).
The review text file has no user id, so the two files are joined row by row; the script CHECKS that product and date agree on every row
and stops with a clear message if they do not. Output: yelpnyc_part_aa, yelpnyc_part_ab, ... (each about 20 MB) to upload here.
"""
import os, re, sys, glob, zipfile, numpy as np, pandas as pd
META, CONT = sys.argv[1], sys.argv[2]; SEED = 42
M = pd.read_csv(META, dtype=str, keep_default_na=False); C = pd.read_csv(CONT, dtype=str, keep_default_na=False)
print("METADATA columns:", list(M.columns), "rows", len(M)); print(M.head(3).to_string()); print("CONTENT columns:", list(C.columns), "rows", len(C)); print(C.drop(columns=[c for c in C.columns if 'review' == c.lower()]).head(3).to_string())
def col(df, keys, what):
    for c in df.columns:
        if any(k in c.lower().replace(" ", "") for k in keys): return c
    sys.exit(f"cannot find {what} column in {list(df.columns)}; paste the two printed headers to Claude")
# Kaggle YelpNYC layout (verified): metadata columns are [reviewer id (mislabelled Product_id), restaurant id (Product_id2), Rating, Label, Date];
# in yelp.csv the column named Review_id is the reviewer id and Product_id is the restaurant.
mu, mp, mr, ml, md = M.columns[0], M.columns[1], "Rating", "Label", "Date"
cp, cd = "Product_id", "Date"; ct = "Review"
if len(M) != len(C): sys.exit(f"row counts differ ({len(M)} vs {len(C)}); cannot join safely. Paste the printed output to Claude.")
dm, dc = pd.to_datetime(M[md], errors="coerce"), pd.to_datetime(C[cd], errors="coerce")
agree_p = (M[mp].values == C[cp].values).mean(); agree_d = (dm.values == dc.values).mean()
print(f"row-by-row agreement: product {agree_p:.4f}, date {agree_d:.4f}")
if agree_p < 0.99 or agree_d < 0.99: sys.exit("The files do NOT align row by row. Paste the printed output to Claude; do not upload anything yet.")
D = pd.DataFrame({"user_id": M[mu].values, "product_id": M[mp].values, "rating": pd.to_numeric(M[mr], errors="coerce").fillna(3.0).values, "label_raw": M[ml].values, "t": dm.values, "text": C[ct].fillna("").astype(str).str.replace(r"[\t\r\n]+", " ", regex=True).values})
D = D.dropna(subset=["t"]).reset_index(drop=True)
vals = D.label_raw.value_counts().to_dict(); print("label values:", vals)
spam = "-1" if "-1" in vals else sorted(vals, key=lambda v: vals[v])[0]   # original Yelp files use -1 for filtered; fallback: rarer value
D["label"] = (D.label_raw == spam).astype(int); print(f"treating '{spam}' as filtered: share {D.label.mean():.4f}")
if not 0.03 < D.label.mean() < 0.5: sys.exit("filtered share looks wrong; paste the printed output to Claude")
D["days"] = (D.t - D.t.min()).dt.total_seconds() / 86400.0; D["review_id"] = np.arange(len(D))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as aps
X = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True, dtype=np.float32).fit_transform(D.text); oof = np.zeros(len(D), np.float32)
for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, D.label, D.user_id)):
    oof[te] = LogisticRegression(C=1.0, max_iter=200).fit(X[tr], D.label.values[tr]).predict_proba(X[te])[:, 1]; print("  text fold", k + 1, "/5", flush=True)
print("out-of-fold text AP", round(aps(D.label, oof), 4))
rv = pd.DataFrame({"review_id": D.review_id, "user_id": D.user_id, "product_id": D.product_id, "timestamp": D.days.round(4), "rating": D.rating, "text": D.text.str.slice(0, 300), "text_ai_score": oof.round(5), "image_mismatch_score": 0.0, "sentiment_gap": 0.0, "image_hash": "", "label": D.label})
g = D.groupby("user_id"); ac = pd.DataFrame({"user_id": g.label.max().index, "account_created": g.days.min().values, "label": g.label.max().values})
print(f"reviews {len(rv)} | reviewers {len(ac)} | products {D.product_id.nunique()} | filtered {D.label.mean():.3f}")
rv.to_csv("reviews.csv", index=False); ac.to_csv("accounts.csv", index=False)
with zipfile.ZipFile("yelpnyc_prepared.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z: z.write("reviews.csv"); z.write("accounts.csv")
data = open("yelpnyc_prepared.zip", "rb").read(); P = 20 * 1024 * 1024
for i in range(0, len(data), P): open(f"yelpnyc_part_{i // P:02d}", "wb").write(data[i:i + P])
print("DONE. Upload these files here:", sorted(glob.glob("yelpnyc_part_*")), "| total MB", round(len(data) / 1e6, 1))
