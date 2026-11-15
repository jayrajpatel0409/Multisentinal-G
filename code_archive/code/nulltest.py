import numpy as np, pandas as pd
from dataclasses import replace
from campaignsim import SimConfig, simulate
import coordgraph as cg
null = replace(SimConfig(), prevalence=0.0, family_frac=0.0, flash_frac=0.0, terse_frac=0.0, catalog_share=0.0)
rows=[]
for seed in range(20):
    R,X,A,info=simulate(null,500+seed); n=len(A); r={}
    for q in (0.01,0.05):
        r[f"sync{q}"]=len(cg.sync_layer(R,n,q=q)[0]); r[f"content{q}"]=len(cg.content_layer(R,X,q=q)[0]); r[f"media{q}"]=len(cg.media_layer(R,q=q)[0])
    rows.append(r)
d=pd.DataFrame(rows); print(d.mean().round(2)); print((d>0).mean())
