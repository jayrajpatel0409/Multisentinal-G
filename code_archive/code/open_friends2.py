import pickle, numpy as np, pandas as pd, json
d=pickle.load(open('graph_layers.pkl','rb')); R=d['R']; A=d['A']
rv=pd.read_csv('reviews.csv',usecols=['review_id','user_id'])
m0=R[['review_id','user']].merge(rv,on='review_id').drop_duplicates('user').set_index('user')['user_id']
idx={u:i for i,u in m0.items()}  # user int -> id ; invert
inv={v:k for k,v in m0.items()}
fr=pd.read_csv('friends.csv'); fu=fr.u.map(inv); fv=fr.v.map(inv); m=fu.notna()&fv.notna()
F=set(zip(np.minimum(fu[m],fv[m]).astype(int),np.maximum(fu[m],fv[m]).astype(int)))
rng=np.random.default_rng(1)
# pairs co-reviewing same product: same week vs. different weeks (>8 weeks apart)
R=R[['user','product','t']].copy(); R['w']=(R.t//7).astype(int)
g=R.groupby('product')
same=[];diff=[]
for p,df in g:
    if len(df)<2 or len(df)>400: continue
    u=df.user.values;w=df.w.values
    k=min(len(u),60)
    ii=rng.choice(len(u),k,replace=False)
    for a in range(k):
        for b in range(a+1,k):
            x,y=u[ii[a]],u[ii[b]]
            if x==y: continue
            dw=abs(w[ii[a]]-w[ii[b]])
            key=(min(x,y),max(x,y))
            if dw==0: same.append(key in F)
            elif dw>=8: diff.append(key in F)
    if len(same)>3e5 and len(diff)>3e6: break
res=dict(same_week_pairs=len(same),same_week_friend_rate=float(np.mean(same)),other_week_pairs=len(diff),other_week_friend_rate=float(np.mean(diff)))
res['ratio']=res['same_week_friend_rate']/res['other_week_friend_rate']
print(res); json.dump(res,open('friends_test2.json','w'),indent=1)
