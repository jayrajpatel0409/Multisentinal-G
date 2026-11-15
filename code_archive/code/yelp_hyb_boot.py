import sys, json, pickle, numpy as np
from sklearn.metrics import average_precision_score as aps
D=sys.argv[1]; B=int(sys.argv[2])
P=pickle.load(open(f"{D}/graph_layers.pkl","rb")); user=P["R"]["user"].values; n=P["n"]
H=np.load(f"{D}/yelp_nested_hyb_preds_seed0.npz"); G=np.load(f"{D}/yelp_nested_gnn_preds_seed0.npz")
idx=H["full_k2__idx"]; y=H["full_k2__y"]
for m in ("gnn","mlp_nograph","full_k2"): assert (G[f"{m}__idx"]==idx).all()
p={"gnn_val":H["gnn_val__p"],"gnn_mix":H["gnn_mix__p"],"gnn":G["gnn__p"],"mlp_nograph":G["mlp_nograph__p"],"full_k2":G["full_k2__p"]}
u=user[idx]; rng=np.random.default_rng(9); ab={m:[] for m in p}
for b in range(B):
    w=np.bincount(rng.integers(0,n,n),minlength=n)[u].astype(float)
    for m in p: ab[m].append(aps(y,p[m],sample_weight=w))
out={"ap":{m:float(aps(y,p[m])) for m in p}}
for a,c in [("gnn_val","full_k2"),("gnn_val","mlp_nograph"),("gnn_mix","gnn"),("gnn_val","gnn"),("gnn","mlp_nograph")]:
    d=np.array(ab[a])-np.array(ab[c]); out[f"{a}-{c}"]=dict(d_ap=float(aps(y,p[a])-aps(y,p[c])),ap_ci=[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))])
json.dump(out,open(f"{D}/analysis_hyb.json","w"),indent=1); print(json.dumps(out,indent=1))
