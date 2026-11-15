import json,numpy as np,collections
def load(p):
    d=collections.defaultdict(list)
    for l in open(p): r=json.loads(l); d[r['_scenario']].append(r)
    return d
import os,sys
OP=sys.argv[1] if len(sys.argv)>1 else '../data/open'
B=load(OP+'/open_inject.jsonl'); S=load(OP+'/open_inject_frsup.jsonl')
out={}
for sc in sorted(B):
    if sc not in S: continue
    def m(D,k,f='ap'): return float(np.mean([r[k][f] for r in D[sc]]))
    fs=lambda D: float(np.mean([r['_friend_stats']['sync_validated']['friend_edges']/r['_friend_stats']['sync_validated']['edges'] for r in D[sc]]))
    out[sc]=dict(n_base=len(B[sc]),n_sup=len(S[sc]),stage=m(B,'stage_fusion'),full_base=m(B,'full'),full_sup=m(S,'full'),gain_base=m(B,'full')-m(B,'stage_fusion'),gain_sup=m(S,'full')-m(S,'stage_fusion'),rec_base=m(B,'full','rec_fpr1'),rec_sup=m(S,'full','rec_fpr1'),friend_share=fs(B))
    print(sc,{k:round(v,4) for k,v in out[sc].items()})
json.dump(out,open(OP+'/open_inj_table.json','w'),indent=1)
