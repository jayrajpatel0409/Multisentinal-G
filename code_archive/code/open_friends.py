import pickle, numpy as np, pandas as pd, json
d=pickle.load(open('graph_layers.pkl','rb'))
A=d['A']; L=d['L']
rv=pd.read_csv('reviews.csv',usecols=['review_id','user_id'])
m0=d['R'][['review_id','user']].merge(rv,on='review_id').drop_duplicates('user').set_index('user')['user_id']
uid=np.array([m0.get(i,'') for i in range(len(A))],dtype=object)
fr=pd.read_csv('friends.csv')
idx={u:i for i,u in enumerate(uid)}
fu=fr['u'].map(idx); fv=fr['v'].map(idx)
m=fu.notna()&fv.notna()
fu=fu[m].astype(int).values; fv=fv[m].astype(int).values
n=len(uid)
F=set(zip(np.minimum(fu,fv),np.maximum(fu,fv)))
print('friend pairs in index',len(F))
deg=np.bincount(np.r_[fu,fv],minlength=n)
rng=np.random.default_rng(0)
out={}
for name in ['sync','sync_raw','content','content_k1']:
    u,v=L[name][0],L[name][1]
    u=np.asarray(u);v=np.asarray(v)
    a=np.minimum(u,v);b=np.maximum(u,v)
    hit=sum((x,y) in F for x,y in zip(a,b))
    # chance: same nodes, random partner among nodes with friend-degree-matched? use random pairs of the participating nodes
    nodes=np.unique(np.r_[u,v])
    ch=[]
    for _ in range(200):
        p=rng.choice(nodes,len(u));q=rng.choice(nodes,len(u))
        ok=p!=q
        aa=np.minimum(p,q)[ok];bb=np.maximum(p,q)[ok]
        ch.append(sum((x,y) in F for x,y in zip(aa,bb))/ok.sum())
    base=np.mean(ch)
    out[name]=dict(edges=len(u),friend_edges=hit,frac=hit/len(u),chance=base,lift=(hit/len(u))/base if base>0 else None,
                   nodes_with_friends=float((deg[nodes]>0).mean()))
    print(name,out[name])
json.dump(out,open('friends_test.json','w'),indent=1)
