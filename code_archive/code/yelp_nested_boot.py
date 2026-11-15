import sys, json, pickle, numpy as np
from sklearn.metrics import average_precision_score as aps, roc_auc_score as auc
D=sys.argv[1]; B=int(sys.argv[2]); sd=[int(x) for x in sys.argv[3].split(",")]
P=pickle.load(open(f"{D}/graph_layers.pkl","rb")); user=P["R"]["user"].values; n=P["n"]
Z=np.load(f"{D}/yelp_nested_gnn_preds_seed0.npz"); idx=Z["full_k2__idx"]; y=Z["full_k2__y"]
p={m:Z[f"{m}__p"] for m in ("text_only","stage_fusion","acct_context","full_k2","gnn","mlp_nograph")}
u=user[idx]; rng=np.random.default_rng(5); ab={m:[] for m in p}
for b in range(B):
    w=np.bincount(rng.integers(0,n,n),minlength=n)[u].astype(float)
    for m in p: ab[m].append(aps(y,p[m],sample_weight=w))
out={}
for a,c in [("full_k2","stage_fusion"),("full_k2","acct_context"),("gnn","full_k2"),("mlp_nograph","full_k2"),("gnn","mlp_nograph"),("full_k2","text_only")]:
    d=np.array(ab[a])-np.array(ab[c]); out[f"{a}-{c}"]=dict(d_ap=float(aps(y,p[a])-aps(y,p[c])),ap_ci=[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))])
# 3-seed means of non-GNN methods
R=[json.load(open(f"{D}/yelp_nested_seed{s}.json")) for s in sd]
out["means"]={m:{k:[float(np.mean([r[m][k] for r in R])),float(np.std([r[m][k] for r in R],ddof=1))] for k in("ap","auroc","f1","rec_fpr1")} for m in R[0]}
G=json.load(open(f"{D}/yelp_nested_gnn_seed0.json")); out["gnn_seed0"]={m:G[m] for m in ("gnn","mlp_nograph","full_k2","acct_context")}
json.dump(out,open(f"{D}/analysis_nested.json","w"),indent=1); print(json.dumps({k:v for k,v in out.items() if k not in("means",)},indent=1)[:2500])
