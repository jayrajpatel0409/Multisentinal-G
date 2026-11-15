"""(1) Null calibration of the validated projection; (2) scalability. Run after the main suite (timings need an idle machine)."""
import json, time, sys
import numpy as np, pandas as pd
from dataclasses import replace
from campaignsim import SimConfig, simulate
import coordgraph as cg
from pipeline import build_graph

out = {}
# ---- (1) null calibration: no campaigns, no families, no flash crowds, no terse/generic reviewers, no AI-assist
null = replace(SimConfig(), prevalence=0.0, family_frac=0.0, flash_frac=0.0, terse_frac=0.0, catalog_share=0.0)
res = []
for seed in range(30):
    R, X, A, info = simulate(null, 500 + seed); n = len(A)
    row = dict(seed=seed, users=n, reviews=len(R))
    for q in (0.01, 0.05):
        u, v, a = cg.sync_layer(R, n, q=q); row[f"sync_{q}"] = len(u)
        u, v, a = cg.content_layer(R, X, q=q); row[f"content_{q}"] = len(u)
        u, v, a = cg.media_layer(R, q=q); row[f"media_{q}"] = len(u)
    # raw (unvalidated) edge counts for contrast
    u, v, a = cg.sync_layer(R, n, validated=False); row["sync_raw"] = len(u)
    res.append(row)
out["null"] = res
# ---- (1b) generic-phrase stress: honest terse reviewers present (content collisions are real but non-coordinated)
terse = replace(null, terse_frac=0.30)
res = []
for seed in range(10):
    R, X, A, info = simulate(terse, 700 + seed); n = len(A)
    row = dict(seed=seed)
    for disc in (True, False):
        u, v, a = cg.content_layer(R, X, discount=disc); row[f"content_disc={disc}"] = len(u)
    res.append(row)
out["terse"] = res
# ---- (2) scalability
sc = []
for ng in (1000, 2000, 4000, 8000, 16000):
    cfg = replace(SimConfig(), n_genuine=ng, n_products=int(250 * ng / 4000) if ng >= 4000 else 250)
    R, X, A, info = simulate(cfg, 1)
    n = len(A); T = info["T"]; row = dict(users=n, reviews=len(R))
    t = time.time(); u, v, a = cg.sync_layer(R, n); row["t_sync"] = time.time() - t
    t = time.time(); u2, v2, a2 = cg.content_layer(R, X); row["t_content"] = time.time() - t
    t = time.time(); u3, v3, a3 = cg.media_layer(R); row["t_media"] = time.time() - t
    t = time.time(); W, mats, gf, cid, L = build_graph(R, X, n, T); row["t_total_graph"] = time.time() - t
    row["edges"] = int(W.nnz // 2); sc.append(row); print(row, flush=True)
out["scale"] = sc
json.dump(out, open("../results/extra.json", "w"), default=float, indent=1)
