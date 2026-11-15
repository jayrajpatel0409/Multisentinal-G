"""Illustrative figure: fused graph vs layer-agreement subgraph on one baseline market (seed 3)."""
import numpy as np, networkx as nx, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, scipy.sparse as sp
from campaignsim import SimConfig, simulate
from pipeline import build_graph
R, X, A, info = simulate(SimConfig(), 3); n = len(A)
W, mats, gf, cid, L = build_graph(R, X, n, info["T"])
supp = sum(((M + M.T) > 0).astype(float) for M in mats.values()).tocsr()
Wm = W.multiply(supp >= 2).tocsr()
rng = np.random.default_rng(0)
fake = A.fake.values; deg = np.asarray((W > 0).sum(1)).ravel()
honest = np.where(~fake & (deg > 0))[0]; honest = rng.choice(honest, min(500, len(honest)), replace=False)
nodes = np.r_[np.where(fake)[0], honest]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
for ax, M, ttl in [(axs[0], W, "(a) Fused graph (all validated edges)"), (axs[1], Wm, "(b) Edges backed by $\\geq 2$ layers")]:
    G = nx.from_scipy_sparse_array(M[nodes][:, nodes]); keep = [v for v in G if G.degree(v) > 0]
    H = G.subgraph(keep); pos = nx.spring_layout(H, seed=1, k=0.12, iterations=60)
    col = ["#c0392b" if fake[nodes[v]] else "#3b6ea5" for v in H]
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.12, width=0.4, edge_color="#555555")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=5, node_color=col, linewidths=0)
    ax.set_title(ttl, fontsize=8); ax.axis("off")
axs[0].plot([], [], "o", color="#c0392b", ms=4, label="campaign account"); axs[0].plot([], [], "o", color="#3b6ea5", ms=4, label="honest account (sample of 500)")
axs[0].legend(fontsize=6.5, loc="lower left", frameon=False)
fig.tight_layout(); fig.savefig("../paper/figs/fig_network.pdf"); print("ok", int(Wm.nnz // 2), int(W.nnz // 2))
