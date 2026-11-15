import json, csv, os, glob
import sys; W=sys.argv[1] if len(sys.argv)>1 else '.'; O=W+'/data_release'
def wr(name,rows,cols):
    with open(f'{O}/{name}','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    print(name,len(rows))
M=['auroc','ap','f1','ece','brier','rec_fpr1']
# 1/2 nested real-data results per seed (corrected protocol)
for ds,d,out in (('YelpZip',W+'/yelp','d01_yelpzip_main_results_per_seed.csv'),('YelpNYC',W+'/nyc','d02_yelpnyc_main_results_per_seed.csv')):
    rows=[]
    for s in range(3):
        for tag,fn in (('main',f'yelp_abl_nested_seed{s}.json' if ds=='YelpZip' else f'yelp_nested_seed{s}.json'),('gnn_control',f'yelp_nested_gctrl_seed{s}.json')):
            p=f'{d}/{fn}'
            if not os.path.exists(p): continue
            for k,v in json.load(open(p)).items():
                if isinstance(v,dict) and 'ap' in v and (tag=='main' or k.startswith(('gnn','mlp'))): rows.append(dict(dataset=ds,seed=s,method=k,**{m:v.get(m) for m in M}))
    wr(out,rows,['dataset','seed','method']+M)
# 3 account-context control
c=json.load(open(W+'/yelp/ctrl_table.json')); rows=[]
for k in ('lp_sync','lp_union','sign_sync','sign_union'):
    v=c[k]; rows.append(dict(addition=v['label'],graph=v['graph'],gain_without_control_ap=v['no_control'][0],sd_without=v['no_control'][1],gain_with_control_ap=v['with_control'][0],sd_with=v['with_control'][1]))
wr('d03_account_context_control_yelpzip.csv',rows,list(rows[0]))
# 4 permutation diagnostic
rows=[]
for ds,p in (('YelpZip',W+'/yelp/kmin_rule.json'),('YelpNYC',W+'/nyc/kmin_rule.json')):
    d=json.load(open(p))
    for k,v in d['real'].items(): rows.append(dict(dataset=ds,condition='real_authorship',k_min=k,validated_edges=v[0],candidate_pairs=v[1]))
    for cond in ('perm_global','perm_product'):
        for k,v in d[cond].items():
            for i,x in enumerate(v): rows.append(dict(dataset=ds,condition=cond,k_min=k,permutation=i,validated_edges=x,candidate_pairs=''))
wr('d04_kmin_permutation_diagnostic.csv',rows,['dataset','condition','k_min','permutation','validated_edges','candidate_pairs'])
# 5 sensitivity
rows=[]
for s in range(3):
    for k,v in json.load(open(f'{W}/yelp/yelp_sens_nested_seed{s}.json')).items(): rows.append(dict(seed=s,setting=k,**{m:v.get(m) for m in M}))
wr('d05_parameter_sensitivity_yelpzip.csv',rows,['seed','setting']+M)
# 6/7 injection
MET=['text_only','image_only','meta_only','stage_fusion','sign_rawgraph','full','graph_only','abl_no_sync','abl_no_content','abl_no_media','abl_no_validation','abl_no_discount','abl_sum_fusion','abl_no_propagation','abl_no_community']
def inj(files,name,extra):
    rows=[]
    for fn,tag in files:
        for l in open(fn):
            r=json.loads(l)
            for m in MET:
                if m in r: rows.append(dict(background=extra,variant=tag if tag else r.get('_frsup',''),scenario=r['_scenario'],seed=r['_seed'],injected_fraction=r.get('_frac',0.04),method=m,ap=r[m]['ap'],auroc=r[m]['auroc'],f1=r[m].get('f1'),rec_fpr1=r[m].get('rec_fpr1')))
    wr(name,rows,['background','variant','scenario','seed','injected_fraction','method','ap','auroc','f1','rec_fpr1'])
inj([(W+'/yelp/yelp_inject.jsonl','main'),(W+'/yelp/yelp_inject_frac.jsonl','prevalence_sweep')],'d06_injection_yelpzip_background.csv','YelpZip genuine accounts')
inj([(W+'/open/open_inject.jsonl','friends_kept'),(W+'/open/open_inject_frsup.jsonl','friends_removed')],'d07_injection_yelp_open_background.csv','Yelp Open')
# 8 friend-group test
f1=json.load(open(W+'/open/friends_test.json')); rows=[dict(layer=k,validated_edges=v['edges'],friend_edges=v['friend_edges'],friend_share=v['frac'],chance_share=v['chance'],lift_over_chance=v['lift']) for k,v in f1.items()]
wr('d08_friend_group_test_yelp_open.csv',rows,list(rows[0]))
f2=json.load(open(W+'/open/friends_test2.json')); wr('d09_cooccurrence_friend_baseline_yelp_open.csv',[f2],list(f2))
# 10 simulation (CampaignSim, fully synthetic)
rows=[]
for l in open(W+'/results/results.jsonl'):
    r=json.loads(l)
    if not r.get('ok'): continue
    for m,v in r['res'].items():
        if isinstance(v,dict) and 'ap' in v: rows.append(dict(scenario=r['scenario'],seed=r['seed'],setting=r.get('which'),method=m,ap=v['ap'],auroc=v.get('auroc'),f1=v.get('f1'),rec_fpr1=v.get('rec_fpr1'),ece=v.get('ece')))
wr('d10_campaignsim_simulation_results.csv',rows,['scenario','seed','setting','method','ap','auroc','f1','rec_fpr1','ece'])
