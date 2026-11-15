#!/usr/bin/env python3
"""Shrink the Yelp Open Dataset (JSON lines) for MultiSentinel-G. Run in Colab.
Usage:  !python prepare_yelp_open.py <folder with the json files> [STATE] [MAX_REVIEWS]
   e.g. !python prepare_yelp_open.py /content PA 400000
Needs: yelp_academic_dataset_business.json and yelp_academic_dataset_review.json (required),
       yelp_academic_dataset_user.json (optional: gives the friend links used for the honest-group test).
Output: yelpopen_part_00, yelpopen_part_01, ... (about 20 MB each). Upload them to Claude."""
import sys, os, glob, json, zipfile, numpy as np, pandas as pd
D = sys.argv[1]; STATE = sys.argv[2] if len(sys.argv) > 2 else "PA"; MAXR = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
B = pd.read_json(f"{D}/yelp_academic_dataset_business.json", lines=True)
B = B[(B.state == STATE) & B.categories.fillna("").str.contains("Restaurants")]
print(f"restaurants in {STATE}: {len(B)}")
if len(B) < 300: sys.exit("too few restaurants; try another state code (PA, FL, TN, LA, MO, NV, AZ, IN)")
keep_b = set(B.business_id); parts = []; n = 0
for ch in pd.read_json(f"{D}/yelp_academic_dataset_review.json", lines=True, chunksize=200000, dtype=False):
    c = ch[ch.business_id.isin(keep_b)][["review_id", "user_id", "business_id", "stars", "date", "text"]]; parts.append(c); n += len(ch); print("read", n, "kept", sum(len(p) for p in parts), flush=True)
R = pd.concat(parts, ignore_index=True)
if len(R) > MAXR:
    rng = np.random.RandomState(42); sz = R.groupby("business_id").size(); order = rng.permutation(sz.index.values); keep = order[np.cumsum(sz[order].values) <= MAXR]
    R = R[R.business_id.isin(keep)].reset_index(drop=True); print("subsampled to", len(R), "reviews from", len(keep), "restaurants")
R["t"] = pd.to_datetime(R["date"]); R["days"] = (R.t - R.t.min()).dt.total_seconds() / 86400.0
R["text"] = R.text.fillna("").astype(str).str.replace(r"[\t\r\n]+", " ", regex=True)
users = set(R.user_id)
rv = pd.DataFrame({"review_id": np.arange(len(R)), "user_id": R.user_id, "product_id": R.business_id, "timestamp": R.days.round(4), "rating": R.stars, "text": R.text.str.slice(0, 300),
                   "text_ai_score": 0.0, "image_mismatch_score": 0.0, "sentiment_gap": 0.0, "image_hash": "", "label": 0})
g = R.groupby("user_id"); ac = pd.DataFrame({"user_id": g.size().index, "account_created": g.days.min().values, "label": 0})
print(f"reviews {len(rv)} | reviewers {len(ac)} | restaurants {R.business_id.nunique()}")
rv.to_csv("reviews.csv", index=False); ac.to_csv("accounts.csv", index=False); files = ["reviews.csv", "accounts.csv"]
up = f"{D}/yelp_academic_dataset_user.json"
if os.path.exists(up):
    ed = []; meta = []
    for ch in pd.read_json(up, lines=True, chunksize=100000, dtype=False):
        ch = ch[ch.user_id.isin(users)]
        for u, f in zip(ch.user_id, ch.friends):
            for v in ([] if f in (None, "None", "") else str(f).split(", ")):
                if v in users and u < v: ed.append((u, v))
        meta.append(ch[["user_id", "review_count", "fans", "yelping_since"]])
    pd.DataFrame(ed, columns=["u", "v"]).drop_duplicates().to_csv("friends.csv", index=False); pd.concat(meta).to_csv("users_meta.csv", index=False)
    files += ["friends.csv", "users_meta.csv"]; print("friend pairs inside the sample:", len(ed))
else: print("user file not found: skipping friend links")
with zipfile.ZipFile("yelpopen_prepared.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in files: z.write(f)
data = open("yelpopen_prepared.zip", "rb").read(); P = 20 * 1024 * 1024
for i in range(0, len(data), P): open(f"yelpopen_part_{i // P:02d}", "wb").write(data[i:i + P])
print("DONE. Upload:", sorted(glob.glob("yelpopen_part_*")), "| total MB", round(len(data) / 1e6, 1))
