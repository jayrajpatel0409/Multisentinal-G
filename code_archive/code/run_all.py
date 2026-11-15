import json, os, sys, time, traceback
import multiprocessing as mp
from dataclasses import replace, asdict
import numpy as np
from campaignsim import SimConfig
from pipeline import run_once

BASE = SimConfig()
SCEN = {
 "S0_baseline": {},
 "S1_paraphrase": dict(sigma_text=1.0, text_drop=0.5),
 "S2_temporal_jitter": dict(win_scale=30.0, camp_sale=0.0),
 "S3_fragmentation": dict(cell_size=(3, 6), n_targets=(2, 4)),
 "S4_camouflage": dict(camo_mean=2.0, camo_hub=True),
 "S5_image_diversify": dict(img_reuse=0.0, auc_img=0.7),
 "S6_sale_blending": dict(camp_sale=1.0),
 "S7_adaptive_combined": dict(sigma_text=1.0, text_drop=0.5, win_scale=20.0, camo_mean=2.0, camo_hub=True, img_reuse=0.1, cell_size=(4, 10), camp_sale=0.6),
 "S8_dense_honest": dict(sale_share=0.6, flash_frac=0.15, family_frac=0.08),
}
def tasks():
    T = []
    for s in range(15): T.append(("S0_baseline", s, "main", {}))
    for name in ["S1_paraphrase", "S7_adaptive_combined", "S8_dense_honest"]:
        for s in range(10): T.append((name, s, "main", {}))
    for name in SCEN:
        if name in ("S0_baseline", "S1_paraphrase", "S7_adaptive_combined", "S8_dense_honest"): continue
        for s in range(10): T.append((name, s, "lite", {}))
    for a in [0.55, 0.65, 0.75, 0.85, 0.95]:
        for s in range(4): T.append((f"sweep_auc={a}", s, "lite", dict(auc_text=(a,) * 6, auc_img=a)))
    for pv in [0.02, 0.04, 0.12, 0.16]:
        for s in range(4): T.append((f"sweep_prev={pv}", s, "lite", dict(prevalence=pv)))
    for of in [0.1, 0.2, 0.4, 0.6, 0.8]:
        for s in range(4): T.append((f"sweep_obs={of}", s, "lite", dict(obs_frac=of)))
    return T

def work(t):
    name, seed, which, over = t
    cfg = replace(BASE, **{**SCEN.get(name.split("sweep")[0] if name.startswith("S") else "S0_baseline", {}), **over}) if name.startswith("sweep") else replace(BASE, **SCEN[name])
    try:
        out = run_once(cfg, seed, which)
        return dict(scenario=name, seed=seed, which=which, ok=True, res=out)
    except Exception as e:
        return dict(scenario=name, seed=seed, which=which, ok=False, err=traceback.format_exc())

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/results.jsonl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r.get("ok"): done.add((r["scenario"], r["seed"]))
    todo = [t for t in tasks() if (t[0], t[1]) not in done]
    print(len(todo), "tasks to run", flush=True)
    with mp.Pool(2) as pool, open(path, "a") as f:
        for i, r in enumerate(pool.imap_unordered(work, todo)):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
            print(i + 1, len(todo), r["scenario"], r["seed"], r["ok"], flush=True)
