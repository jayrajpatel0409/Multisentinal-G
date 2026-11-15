import json, sys, numpy as np
D=sys.argv[1]; sd=[int(x) for x in sys.argv[2].split(",")]
R=[json.load(open(f"{D}/yelp_nested_ctrl_seed{s}.json")) for s in sd]
ap=lambda m:np.array([r[m]["ap"] for r in R])
out={}
for meth,key in (("Propagation of base score (label-propagation style)","lp"),("Neighbour aggregation + learned model (SIGN style)","sign")):
    for g in ("sync","union"):
        a=ap(f"{key}_noctx_{g}")-ap("stage_fusion"); b=ap(f"{key}_ctx_{g}")-ap("acct_context")
        out[f"{key}_{g}"]=dict(label=meth,graph=g,no_control=[float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else 0.0],with_control=[float(b.mean()),float(b.std(ddof=1)) if len(b)>1 else 0.0])
out["ctx_gain"]=[float((ap("acct_context")-ap("stage_fusion")).mean())]
json.dump(out,open(f"{D}/ctrl_table.json","w"),indent=1); print(json.dumps(out,indent=1))
