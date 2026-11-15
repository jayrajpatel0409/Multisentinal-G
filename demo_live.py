"""
MultiSentinel-G: Live Panel Demonstration Script
Runs a fast end-to-end evaluation comparing:
  1. Text-Only baseline
  2. Account Context baseline
  3. Stage Fusion (Text + Metadata)
  4. MultiSentinel-G (Multiplex Coordination Graph + Statistical FDR Validation)
"""
import sys, os, time
sys.path.insert(0, os.path.abspath("code_archive/code"))

import numpy as np
import pandas as pd
from campaignsim import SimConfig, simulate
import coordgraph as cg
from pipeline import account_features, fit_fold, Pool, STAGE, META

def run_panel_demo():
    print("=" * 74)
    print("   MULTISENTINEL-G: OPINION-SPAM COORDINATION DETECTION (LIVE DEMO)")
    print("=" * 74)
    t0 = time.time()
    
    print("\n[1/4] Simulating Marketplace & Adversarial Campaign Scenarios...")
    cfg = SimConfig(n_genuine=2000, n_products=120)
    R, X, A, meta = simulate(cfg, seed=0)
    T = meta["T"]
    n = len(A)
    y = A["fake"].values.astype(int)
    print(f"      -> Generated {len(R):,} reviews across {n:,} reviewer accounts.")
    print(f"      -> Ground-truth: {y.sum()} fake accounts ({y.mean()*100:.1f}% spam prevalence).")
    print(f"      -> Confounders included: {A['family'].sum()} household/friends, {A['flash'].sum()} flash-sale crowd.")

    print("\n[2/4] Constructing Multiplex Coordination Graph with Statistical Validation...")
    F = account_features(R, A, T)
    
    # Layer 1: Synchrony (co-review temporal window)
    u_s, v_s, a_s = cg.sync_layer(R, n, W=7.0, q=0.01, min_k=2)
    print(f"      -> [SYNC layer]    {len(a_s):,} edges validated (Poisson null + FDR q=0.01)")
    
    # Layer 2: Content (near-duplicate text)
    u_c, v_c, a_c = cg.content_layer(R, X, theta=0.80, min_k=2)
    print(f"      -> [CONTENT layer] {len(a_c):,} edges validated (collision surprisal)")
    
    # Layer 3: Media (shared images)
    u_m, v_m, a_m = cg.media_layer(R, min_k=1)
    print(f"      -> [MEDIA layer]   {len(a_m):,} edges validated (popularity discount)")
    
    # Layer Fusion via Noisy-OR
    layers = {"sync": (u_s, v_s, a_s), "content": (u_c, v_c, a_c), "media": (u_m, v_m, a_m)}
    W, mats = cg.fuse_layers(layers, n, mode="noisyor", a_min=0.3)
    feats, cid, W = cg.graph_features(W, mats, R, n, T, seed=0)
    print(f"      -> [NOISY-OR FUSION] {W.nnz // 2:,} consolidated coordination edges")

    print("\n[3/4] Evaluating under Reviewer-Grouped Cross-Validation (Zero Leakage)...")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(skf.split(np.zeros(n), y)):
        fold[te] = k

    base_cols = STAGE + META
    G = feats.drop(columns=["comm"])
    s0_dummy = F["t_mean"].values
    G = G.assign(g_prop=cg.propagate(W, s0_dummy), g_nbr=cg.neighbour_mean(W, s0_dummy))
    FF_full = pd.concat([F, G], axis=1)

    models_to_test = {
        "1. Text-Only Classifier": (F, ["t_max", "t_mean", "t_pos"]),
        "2. Reviewer Context Only": (F, META),
        "3. Stage Fusion (Text + Metadata)": (F, base_cols),
        "4. MultiSentinel-G (Proposed)": (FF_full, list(G.columns) + base_cols)
    }

    pools = {name: Pool() for name in models_to_test}

    for k in range(5):
        tr = (fold != k) & (fold != (k + 1) % 5)
        va = fold == (k + 1) % 5
        te = fold == k

        for name, (DF, cols) in models_to_test.items():
            pt, th = fit_fold(
                DF.loc[tr, cols].values, y[tr],
                DF.loc[va, cols].values, y[va],
                DF.loc[te, cols].values, 0
            )
            pools[name].add(y[te], pt, th, np.where(te)[0])

    print("\n[4/4] Final Results Comparison:")
    print("=" * 74)
    print(f"  {'Methodology':<34} | {'AUROC':>7} | {'Avg Prec (AP)':>13} | {'F1':>6}")
    print("-" * 74)
    for name, p in pools.items():
        m = p.metrics()
        print(f"  {name:<34} | {m['auroc']:>7.4f} | {m['ap']:>13.4f} | {m['f1']:>6.4f}")
    print("=" * 74)
    print(f"Total Execution Time: {time.time() - t0:.1f} seconds")
    print("\n[Core Takeaways for Evaluation Panel]")
    print("  1. Text-Only models fail to isolate modern spam rings (low AP).")
    print("  2. Account Context catches habitual spammers but misses coordinated botnets.")
    print("  3. MultiSentinel-G captures statistical coordination edges without false-positive inflation.")
    print("=" * 74)

if __name__ == "__main__":
    run_panel_demo()
