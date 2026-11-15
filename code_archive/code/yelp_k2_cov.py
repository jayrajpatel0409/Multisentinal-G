import sys, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as aps
import coordgraph as cg
D=sys.argv[1]; P=pickle.load(open(f"{D}/graph_layers.pkl","rb")); R,n,(pi,pj),L=P["R"],P["n"],P["pairs"],P["L"]; user=R["user"].values; N=len(R)
a_,b_=np.minimum(user[pi],user[pj]),np.maximum(user[pi],user[pj])
df=pd.DataFrame({"u":a_,"v":b_}).groupby(["u","v"],as_index=False).size().rename(columns={"size":"k"}); nt=len(df)
deg=np.bincount(pi,minlength=N)+np.bincount(pj,minlength=N); Dg=np.bincount(user,weights=deg,minlength=n)
dd=df[df.k>=2]; mu=Dg[dd.u.values]*Dg[dd.v.values]/max(deg.sum(),1); keep,a=cg.poisson_validate(dd.k.values,mu,nt,0.01,10.0)
cu=(dd.u.values[keep],dd.v.values[keep],a)
W,mats=cg.fuse_layers({"sync":L["sync"],"content":cu},n); cov=np.asarray((W>0).sum(1)).ravel()>0
Z=np.load(f"{D}/yelp_preds_seed0.npz"); out={"covered_accounts":int(cov.sum()),"covered_share_reviews":float(cov[user[Z['full_k2__idx']]].mean())}
c=cov[user[Z["full_k2__idx"]]]; y=Z["full_k2__y"]
for m in ("stage_fusion","acct_context","full_k2"): out[f"ap_cov_{m}"]=float(aps(y[c],Z[f"{m}__p"][c])); out[f"ap_unc_{m}"]=float(aps(y[~c],Z[f"{m}__p"][~c]))
json.dump(out,open(f"{D}/k2_cov.json","w"),indent=1); print(out)
