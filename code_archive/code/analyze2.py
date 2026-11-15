"""Tables for subgroup false-positive rates, null calibration and scalability -> ../paper/tab_*.tex and numbers2.tex"""
import json, sys, collections, numpy as np, pandas as pd
RES = "../results/results.jsonl"; EX = "../results/extra.json"; OUT = "../paper"
rows = [json.loads(l) for l in open(RES)]; rows = [r for r in rows if r["ok"]]
nums = {}
# ---------------- subgroup FPR
G = ["regular", "terse", "family", "flash", "ai_assist"]
M = [("stage_fusion", "Stage-fusion (no graph)"), ("sign_rawgraph", "Raw co-review graph + aggregation"), ("full", "MultiSentinel-G")]
L = []
for scn, nm in [("S0_baseline", "Baseline (S0)"), ("S8_dense_honest", "Dense honest crowds (S8)"), ("S7_adaptive_combined", "Adaptive (S7)")]:
    acc = collections.defaultdict(list)
    for r in rows:
        if r["scenario"] == scn and "_fpr_groups" in r["res"]:
            for m, d in r["res"]["_fpr_groups"].items():
                for g in G:
                    if d.get(g) is not None: acc[(m, g)].append(d[g])
    L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{nm}}}}} \\\\")
    for m, mn in M:
        vals = [100 * np.mean(acc[(m, g)]) if acc[(m, g)] else np.nan for g in G]
        cells = [f"{v:.2f}" for v in vals]
        if m == "full": cells = [f"\\textbf{{{c}}}" for c in cells]
        L.append(" & ".join([mn] + cells) + " \\\\")
        for g, v in zip(G, vals): nums[f"fpr_{scn[:2]}_{m}_{g}"] = f"{v:.2f}"
open(f"{OUT}/tab_fpr.tex", "w").write("\n".join(L))
# ---------------- null calibration
e = json.load(open(EX)); n = pd.DataFrame(e["null"])
L = []
for lay, nm in [("sync", "Sync (co-review)"), ("content", "Content (near-duplicate)"), ("media", "Media (shared image)")]:
    cells = []
    for q in (0.01, 0.05):
        c = f"{lay}_{q}"; cells += [f"{n[c].mean():.2f}", f"{100 * (n[c] > 0).mean():.0f}\\%"]
        nums[f"null_{lay}_{q}_edges"] = f"{n[c].mean():.2f}"; nums[f"null_{lay}_{q}_share"] = f"{100 * (n[c] > 0).mean():.0f}"
    L.append(" & ".join([nm] + cells) + " \\\\")
open(f"{OUT}/tab_null.tex", "w").write("\n".join(L))
nums["null_sync_raw"] = f"{n['sync_raw'].mean():,.0f}"; nums["null_nrep"] = len(n)
t = pd.DataFrame(e["terse"]); nums["terse_disc"] = f"{t['content_disc=True'].mean():,.0f}"; nums["terse_nodisc"] = f"{t['content_disc=False'].mean():,.0f}"
nums["terse_red"] = f"{100 * (1 - t['content_disc=True'].mean() / t['content_disc=False'].mean()):.0f}"
# ---------------- scalability
s = pd.DataFrame(e["scale"]); L = []
for _, r in s.iterrows():
    L.append(f"{int(r.users):,} & {int(r.reviews):,} & {r.t_sync:.2f} & {r.t_content:.2f} & {r.t_media:.2f} & {r.t_total_graph:.1f} & {int(r.edges):,} \\\\")
open(f"{OUT}/tab_scale.tex", "w").write("\n".join(L))
sl = np.polyfit(np.log(s.reviews), np.log(s.t_content), 1)[0]; sl2 = np.polyfit(np.log(s.reviews), np.log(s.t_total_graph), 1)[0]
nums["slope_content"] = f"{sl:.2f}"; nums["slope_total"] = f"{sl2:.2f}"; nums["t_big"] = f"{s.t_total_graph.iloc[-1]:.0f}"; nums["n_big"] = f"{int(s.users.iloc[-1]):,}"
with open(f"{OUT}/numbers2.tex", "w") as f:
    f.write("\\makeatletter\n")
    for k, v in nums.items(): f.write(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n")
    f.write("\\makeatother\n")
print("ok", len(nums))
