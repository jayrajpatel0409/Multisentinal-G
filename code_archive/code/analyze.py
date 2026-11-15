"""Aggregate results.jsonl -> LaTeX tables, figures (PDF) and numbers.tex macros. No numbers are typed by hand."""
import json, sys, os, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "../results/results.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../paper"
os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/figs", exist_ok=True)
rows = [json.loads(l) for l in open(RES)]
rows = [r for r in rows if r["ok"] and r["scenario"] != "sweep_prev=0.01"]

NICE = {"text_only": "Text-AI detector only", "image_only": "Image check only", "meta_only": "Reviewer metadata only",
        "stage_fusion": "Stage-fusion (original MultiSentinel)", "dense_block": "Dense-block peeling (unsup.)",
        "sign_rawgraph": "Raw co-review graph + aggregation", "graph_only": "Graph features only",
        "full": "MultiSentinel-G (proposed)"}
SC = {"S0_baseline": "Baseline campaign", "S1_paraphrase": "Paraphrase evasion", "S2_temporal_jitter": "Temporal jitter",
      "S3_fragmentation": "Cell fragmentation", "S4_camouflage": "Heavy camouflage", "S5_image_diversify": "Image diversification",
      "S6_sale_blending": "Sale-window blending", "S7_adaptive_combined": "Adaptive (combined)", "S8_dense_honest": "Dense honest crowds"}
MET = ["auroc", "ap", "f1", "ece", "rec_fpr1"]

def table(scn):
    recs = []
    for r in rows:
        if r["scenario"] != scn: continue
        for m, v in r["res"].items():
            if m.startswith("_") or not isinstance(v, dict): continue
            recs.append(dict(seed=r["seed"], method=m, **{k: v.get(k, np.nan) for k in MET}))
    return pd.DataFrame(recs)

def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.mean(x) if len(x) else np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), h

def fmt(m, h, d=3): return f"{m:.{d}f}" + (f"$\\pm${h:.{d}f}" if not np.isnan(h) else "")

def paired(df, a, b, metric="ap"):
    pa = df[df.method == a].set_index("seed")[metric]; pb = df[df.method == b].set_index("seed")[metric]
    s = pa.index.intersection(pb.index); d = (pa[s] - pb[s]).values
    if len(d) < 3 or np.allclose(d, 0): return np.mean(d) if len(d) else np.nan, np.nan, np.nan, len(d)
    p = stats.wilcoxon(d, zero_method="wilcox").pvalue
    rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, len(d)

nums = {}
def num(k, v): nums[k] = v

# ------------------------------------------------------------ main table (S0)
d0 = table("S0_baseline")
order = ["text_only", "image_only", "meta_only", "stage_fusion", "dense_block", "sign_rawgraph", "graph_only", "full"]
L = []
for m in order:
    sub = d0[d0.method == m]
    cells = [fmt(*ci(sub[k])) if k in sub and not sub[k].isna().all() else "--" for k in ["auroc", "ap", "f1", "rec_fpr1", "ece"]]
    nm = NICE[m];
    if m == "full": nm = "\\textbf{" + nm + "}"; cells = ["\\textbf{" + c + "}" for c in cells]
    L.append(" & ".join([nm] + cells) + " \\\\")
open(f"{OUT}/tab_main.tex", "w").write("\n".join(L))
n0 = d0.seed.nunique(); num("nseedsmain", n0)
for m in ["stage_fusion", "full", "sign_rawgraph", "meta_only", "text_only", "graph_only", "dense_block"]:
    mm, hh = ci(d0[d0.method == m]["ap"]); num(f"ap_{m}", f"{mm:.3f}"); num(f"apci_{m}", f"{hh:.3f}")
    mm, hh = ci(d0[d0.method == m]["auroc"]); num(f"auc_{m}", f"{mm:.3f}")
    mm, hh = ci(d0[d0.method == m]["f1"]); num(f"f1_{m}", f"{mm:.3f}")
    mm, hh = ci(d0[d0.method == m]["rec_fpr1"]); num(f"rec_{m}", f"{mm:.3f}")
    mm, hh = ci(d0[d0.method == m]["ece"]); num(f"ece_{m}", f"{mm:.4f}")

# ------------------------------------------------------------ scenario table
L = []; sc_stats = {}
for scn, nm in SC.items():
    d = table(scn)
    if d.empty: continue
    row = [nm]
    for m in ["stage_fusion", "sign_rawgraph", "full"]:
        row.append(fmt(*ci(d[d.method == m]["ap"]), d=3))
    dm, lo, hi, p, n = paired(d, "full", "stage_fusion")
    row.append(f"{dm:+.3f} [{lo:+.3f}, {hi:+.3f}]"); sc_stats[scn] = (dm, p)
    cr = [r["res"].get("_cell_recovery", np.nan) for r in rows if r["scenario"] == scn]
    row.append(f"{np.nanmean(cr):.2f}")
    L.append(" & ".join(row) + " \\\\")
open(f"{OUT}/tab_scen.tex", "w").write("\n".join(L))
# Holm correction over scenario p-values
ps = {k: v[1] for k, v in sc_stats.items() if not np.isnan(v[1])}
srt = sorted(ps.items(), key=lambda kv: kv[1]); holm = {}; prev = 0
for i, (k, p) in enumerate(srt):
    adj = min(1.0, (len(srt) - i) * p); prev = max(prev, adj); holm[k] = prev
num("holmmax", f"{max(holm.values()):.3f}" if holm else "n/a")
num("nsigscen", sum(v < 0.05 for v in holm.values())); num("nscen", len(holm))

# ------------------------------------------------------------ ablations (S0, S1, S7)
abl = ["full", "abl_no_sync", "abl_no_content", "abl_no_media", "abl_no_validation", "abl_no_discount", "abl_sum_fusion", "abl_no_propagation", "abl_no_community", "graph_only", "stage_fusion"]
ANAME = {"full": "Full model", "abl_no_sync": "$-$ sync layer", "abl_no_content": "$-$ content layer", "abl_no_media": "$-$ media layer",
         "abl_no_validation": "$-$ statistical validation (raw counts)", "abl_no_discount": "$-$ genericness discount",
         "abl_sum_fusion": "noisy-OR $\\to$ clipped sum", "abl_no_propagation": "$-$ score propagation",
         "abl_no_community": "$-$ community features", "graph_only": "graph features only (no Stage 1--3)", "stage_fusion": "no graph (stage-fusion)"}
L = []; ab_vals = {}
dd = {s: table(s) for s in ["S0_baseline", "S1_paraphrase", "S7_adaptive_combined", "S8_dense_honest"]}
for m in abl:
    row = [ANAME[m]]
    for s in dd:
        sub = dd[s][dd[s].method == m]
        row.append(f"{ci(sub['ap'])[0]:.3f}" if len(sub) else "--"); ab_vals[(m, s)] = ci(sub["ap"])[0] if len(sub) else np.nan
        if m not in ("full",) and len(sub):
            dm, lo, hi, p, n = paired(dd[s], m, "full") if True else (0, 0, 0, 1, 0)
            row[-1] += f" ({dm:+.3f}{'*' if (not np.isnan(p) and p < 0.05) else ''})"
    L.append(" & ".join(row) + " \\\\")
open(f"{OUT}/tab_abl.tex", "w").write("\n".join(L))

# ------------------------------------------------------------ layer agreement / edge quality
agree = {1: [], 2: [], 3: []}; nedges = {1: [], 2: [], 3: []}
for r in rows:
    if r["scenario"] != "S0_baseline": continue
    e = r["res"].get("_edge_by_layers")
    if not e: continue
    for c in (1, 2, 3):
        v = e[str(c)]
        if v["same_cell"] is not None: agree[c].append(v["same_cell"]); nedges[c].append(v["n"])
L = []
for c in (1, 2, 3):
    if agree[c]:
        m, h = ci(agree[c]); L.append(f"{c} & {np.mean(nedges[c]):,.0f} & {fmt(m, h)} \\\\"); num(f"edgeprec{c}", f"{100*m:.1f}")
open(f"{OUT}/tab_agree.tex", "w").write("\n".join(L))

# ------------------------------------------------------------ cell recovery
for scn in ["S0_baseline", "S7_adaptive_combined"]:
    cr = [r["res"].get("_cell_recovery") for r in rows if r["scenario"] == scn and "_cell_recovery" in r["res"]]
    if cr: num(f"cellrec_{scn[:2]}", f"{np.mean(cr):.2f}")

# ------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150})
C = {"stage_fusion": "#7a7a7a", "sign_rawgraph": "#d98c1f", "full": "#1f5fa8", "meta_only": "#b04a4a", "dense_block": "#8a6bb1"}
# Fig: AP per scenario
fig, ax = plt.subplots(figsize=(7.2, 3.1)); scn_list = [s for s in SC if not table(s).empty]; w = 0.26
for i, m in enumerate(["stage_fusion", "sign_rawgraph", "full"]):
    mu, hw = [], []
    for s in scn_list:
        d = table(s); a = ci(d[d.method == m]["ap"]); mu.append(a[0]); hw.append(0 if np.isnan(a[1]) else a[1])
    ax.bar(np.arange(len(scn_list)) + (i - 1) * w, mu, w, yerr=hw, color=C[m], label=NICE[m], capsize=1.5, error_kw=dict(lw=.7))
ax.set_xticks(range(len(scn_list))); ax.set_xticklabels([SC[s].replace(" ", "\n", 1) for s in scn_list], fontsize=6.6)
ax.set_ylabel("Average precision (test accounts)"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/figs/fig_scenarios.pdf"); plt.close(fig)

# Fig: sweeps
def sweep(prefix, base_name="S0_baseline", base_val=None):
    pts = {}
    for r in rows:
        if r["scenario"].startswith(prefix):
            pts.setdefault(float(r["scenario"].split("=")[1]), []).append(r)
    return pts
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for ax, (prefix, xl, bv) in zip(axs, [("sweep_auc", "Upstream detector AUC (text and image)", .85), ("sweep_prev", "Fake-account prevalence", 0.08), ("sweep_obs", "Fraction of reviews observed", 1.0)]):
    pts = sweep(prefix)
    for m in ["meta_only", "stage_fusion", "sign_rawgraph", "full"]:
        xs, mu, hw = [], [], []
        for x, rr in sorted(pts.items()):
            v = [rw["res"][m]["ap"] for rw in rr if m in rw["res"]]
            if v: xs.append(x); a = ci(v); mu.append(a[0]); hw.append(0 if np.isnan(a[1]) else a[1])
        # add the baseline value
        if prefix == "sweep_prev": bx = 0.08
        elif prefix == "sweep_obs": bx = 1.0
        else: bx = None
        if bx is not None:
            v = d0[d0.method == m]["ap"].values
            xs.append(bx); a = ci(v); mu.append(a[0]); hw.append(0 if np.isnan(a[1]) else a[1])
        o = np.argsort(xs); xs, mu, hw = np.array(xs)[o], np.array(mu)[o], np.array(hw)[o]
        ax.errorbar(xs, mu, yerr=hw, marker="o", ms=3, lw=1.3, capsize=1.5, color=C[m], label=NICE[m] if ax is axs[0] else None)
    ax.set_xlabel(xl, fontsize=8)
axs[0].set_ylabel("Average precision"); axs[0].set_ylim(0, 1.02)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=6.3, frameon=False)
fig.tight_layout(rect=[0, 0.09, 1, 1]); fig.savefig(f"{OUT}/figs/fig_sweeps.pdf"); plt.close(fig)

# numbers for sweeps
def sweep_ap(prefix, x, m):
    rr = sweep(prefix).get(x, []); v = [rw["res"][m]["ap"] for rw in rr if m in rw["res"]]; return np.mean(v) if v else np.nan
for x in [0.1, 0.2, 0.4]:
    num(f"obs{int(x*100)}_full", f"{sweep_ap('sweep_obs', x, 'full'):.3f}"); num(f"obs{int(x*100)}_stage", f"{sweep_ap('sweep_obs', x, 'stage_fusion'):.3f}")
for x in [0.55, 0.65, 0.95]:
    num(f"auc{int(x*100)}_full", f"{sweep_ap('sweep_auc', x, 'full'):.3f}"); num(f"auc{int(x*100)}_stage", f"{sweep_ap('sweep_auc', x, 'stage_fusion'):.3f}")
for x in [0.02]:
    num(f"prev{int(x*100)}_full", f"{sweep_ap('sweep_prev', x, 'full'):.3f}"); num(f"prev{int(x*100)}_stage", f"{sweep_ap('sweep_prev', x, 'stage_fusion'):.3f}")

# per-scenario headline numbers
for scn in SC:
    d = table(scn)
    if d.empty: continue
    k = scn[:2]
    for m in ["stage_fusion", "full", "sign_rawgraph", "meta_only"]:
        num(f"{k}_ap_{m}", f"{ci(d[d.method == m]['ap'])[0]:.3f}")
    dm, lo, hi, p, n = paired(d, "full", "stage_fusion"); num(f"{k}_diff", f"{dm:+.3f}"); num(f"{k}_p", f"{p:.4f}" if not np.isnan(p) else "n/a")
    num(f"{k}_n", n)

# ablation macros
for m in ["abl_no_sync", "abl_no_content", "abl_no_media", "abl_no_validation", "abl_no_discount", "abl_sum_fusion", "abl_no_propagation", "abl_no_community"]:
    for s in dd:
        v = ab_vals.get((m, s), np.nan); f = ab_vals.get(("full", s), np.nan)
        num(f"{m[4:]}_{s[:2]}", f"{v - f:+.3f}" if not np.isnan(v) and not np.isnan(f) else "n/a")
for s in dd:
    num(f"graphonly_{s[:2]}", f"{ab_vals.get(('graph_only', s), np.nan):.3f}")
    num(f"nogr_{s[:2]}", f"{ab_vals.get(('stage_fusion', s), np.nan):.3f}")

# runtime
secs = [r["res"]["_secs"] for r in rows if "_secs" in r["res"]]
num("runsecs", f"{np.mean(secs):.0f}"); num("nruns", len(rows))
tot_users = np.mean([r["res"]["_n"]["users"] for r in rows]); num("meanusers", f"{tot_users:,.0f}")
num("meanrev", f"{np.mean([r['res']['_n']['reviews'] for r in rows]):,.0f}")
num("meanfake", f"{np.mean([r['res']['_n']['fake'] for r in rows]):.0f}")

with open(f"{OUT}/numbers.tex", "w") as f:
    f.write("\\makeatletter\n\\newcommand{\\num}[1]{\\csname num@#1\\endcsname}\n")
    for k, v in nums.items():
        f.write(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n")
    f.write("\\makeatother\n")
json.dump({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in nums.items()}, open(f"{OUT}/numbers.json", "w"), indent=1)
print("done;", len(rows), "runs;", len(nums), "numbers")
