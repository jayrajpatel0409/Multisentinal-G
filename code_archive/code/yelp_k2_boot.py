import sys, json, pickle, numpy as np
from sklearn.metrics import average_precision_score as aps, roc_auc_score as auc
D=sys.argv[1]; B=int(sys.argv[2])
P=pickle.load(open(f"{D}/graph_layers.pkl","rb")); user=P["R"]["user"].values; n=P["n"]
Z=np.load(f"{D}/yelp_preds_seed0.npz"); G=np.load(f"{D}/yelp_gnn_preds_seed0.npz")
idx=Z["full_k2__idx"]; y=Z["full_k2__y"]; p={m:Z[f"{m}__p"] for m in ("full_k2","full","acct_context","stage_fusion","sign_rawgraph")}
for m in ("gnn","mlp_nograph"): assert (G[f"{m}__idx"]==idx).all(); p[m]=G[f"{m}__p"]
u=user[idx]; rng=np.random.default_rng(3); ab={m:[] for m in p}; ub={m:[] for m in p}
for b in range(B):
    w=np.bincount(rng.integers(0,n,n),minlength=n)[u].astype(float)
    for m in p: ab[m].append(aps(y,p[m],sample_weight=w)); ub[m].append(auc(y,p[m],sample_weight=w))
out={}
for c in ("stage_fusion","acct_context","sign_rawgraph","mlp_nograph","gnn","full"):
    d1=np.array(ab["full_k2"])-np.array(ab[c]); d2=np.array(ub["full_k2"])-np.array(ub[c])
    out[f"full_k2-{c}"]=dict(d_ap=float(aps(y,p["full_k2"])-aps(y,p[c])),ap_ci=[float(np.percentile(d1,2.5)),float(np.percentile(d1,97.5))],d_auroc=float(auc(y,p["full_k2"])-auc(y,p[c])),auroc_ci=[float(np.percentile(d2,2.5)),float(np.percentile(d2,97.5))])
json.dump(out,open(f"{D}/analysis_k2.json","w"),indent=1); print(json.dumps({k:(round(v['d_ap'],4),[round(x,4) for x in v['ap_ci']]) for k,v in out.items()}))
