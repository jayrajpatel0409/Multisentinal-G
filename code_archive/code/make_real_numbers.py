"""Generate numbers3.tex and real-data tables from yelp analysis.json and inject_summary / yelp_inject.jsonl."""
import json,collections,numpy as np,sys
import os
Y=os.environ.get('YELP_DIR','../data/yelp').rstrip('/')+'/'
A0=json.load(open(Y+'analysis.json')); A=json.load(open(Y+'analysis_corrected.json')); A['meta']=A0['meta']; rows=[json.loads(l) for l in open(Y+'yelp_inject.jsonl')]
N={}
f=lambda x,k=3:f"{x:.{k}f}"
m=A['meta']; N['yz_nrev']=f"{m['n_reviews']:,}".replace(',','{,}'); N['yz_nacc']=f"{m['n_accounts']:,}".replace(',','{,}')
N['yz_prev']=f(100*m['prevalence'],1)
ap=lambda k:A[k]['ap'][0]
N['yz_ap_stage']=f(ap('stage_fusion'));N['yz_ap_full']=f(ap('full_k2'));N['yz_ap_k1']=f(ap('full'));N['yz_auc_k1']=f(A['full']['auroc'][0]);N['yz_ap_ctx']=f(ap('acct_context'));N['yz_ap_raw']=f(ap('sign_rawgraph'))
N['yz_auc_full']=f(A['full']['auroc'][0]);N['yz_auc_dense']=f(A['dense_block']['auroc'][0]);N['yz_ap_dense']=f(ap('dense_block'))
CB=A['bootstrap']; B={'full-stage_fusion':CB['full_k2-stage_fusion'],'full-acct_context':CB['full_k2-acct_context'],'full-sign_rawgraph':CB['full_k2-sign_rawgraph'],'acct_context-stage_fusion':CB['acct_context-stage_fusion'],'full-text_only':CB['full_k2-text_only'],'k1-k2':CB['full-full_k2']}
for key,tag in [('full-stage_fusion','stage'),('full-acct_context','ctx'),('full-sign_rawgraph','raw'),('acct_context-stage_fusion','ctxstage'),('full-text_only','text'),('k1-k2','k1k2')]:
    b=B[key];N[f'yz_d_{tag}']=f(b['d_ap']);N[f'yz_d_{tag}_lo']=f(b['ap_ci'][0]);N[f'yz_d_{tag}_hi']=f(b['ap_ci'][1])
    N[f'yz_dauc_{tag}']=f(b['d_auroc']);N[f'yz_dauc_{tag}_lo']=f(b['auroc_ci'][0]);N[f'yz_dauc_{tag}_hi']=f(b['auroc_ci'][1])
C=A['cov_k2'];C={**C,'ap_cov_full_k2':C['ap_cov_full'],'ap_unc_full_k2':C['ap_unc_full']};N['yz_covacc']=f(100*C['covered_accounts']/m['n_accounts'],1);N['yz_covrev']=f(100*C['covered_share_reviews'],1);N['yz_covacc_n']=f"{C['covered_accounts']:,}".replace(',','{,}')
N['yz_apcov_stage']=f(C['ap_cov_stage_fusion']);N['yz_apcov_ctx']=f(C['ap_cov_acct_context']);N['yz_apcov_full']=f(C['ap_cov_full_k2'])
N['yz_apunc_full']=f(C['ap_unc_full_k2']);N['yz_apunc_stage']=f(C['ap_unc_stage_fusion']);N['yz_apunc_ctx']=f(C['ap_unc_acct_context'])
K1=A['cov_k1'];N['yz_covacc_k1']=f(100*K1['covered_accounts']/m['n_accounts'],1);N['yz_covrev_k1']=f(100*K1['covered_share_reviews'],1);N['yz_apcov_k1']=f(K1['ap_cov_full']);N['yz_apcov_ctx_k1']=f(K1['ap_cov_acct_context'])
N['yz_e_sync']=f"{m['layer_edges']['sync']:,}".replace(',','{,}');N['yz_e_c1']=str(m['layer_edges']['content_k1']);N['yz_e_c2']=str(m['layer_edges']['content_k2'])
N['yz_d_ctx_mean']=f(A['full_k2']['ap'][0]-A['acct_context']['ap'][0],4)
HY=json.load(open(Y+'analysis_hyb.json'))
N['hyb_ap_val']=f(HY['ap']['gnn_val']);N['hyb_ap_mix']=f(HY['ap']['gnn_mix'])
for key,tag in [('gnn_val-full_k2','valk2'),('gnn_val-mlp_nograph','valmlp'),('gnn_mix-gnn','mixgnn'),('gnn_val-gnn','valgnn')]:
    b=HY[key];N[f'hyb_d_{tag}']=f(b['d_ap']);N[f'hyb_d_{tag}_lo']=f(b['ap_ci'][0]);N[f'hyb_d_{tag}_hi']=f(b['ap_ci'][1])
CT=json.load(open(Y+'ctrl_table.json'));tc=[]
for k in ('lp_sync','lp_union','sign_sync','sign_union'):
    c=CT[k];nm_,wc=c['no_control'],c['with_control'];share=100*(1-wc[0]/nm_[0]) if nm_[0]>0 else float('nan')
    tc.append(f"{'Propagation (LP style)' if k.startswith('lp') else 'Aggregation (SIGN style)'} & {c['graph']} & {nm_[0]:+.4f} & {wc[0]:+.4f} & {share:.0f}\\% \\\\"); N['ctrl_'+k+'_no']=f(nm_[0],4);N['ctrl_'+k+'_with']=f(wc[0],4);N['ctrl_'+k+'_share']=f"{share:.0f}"
N['ctrl_ctxgain']=f(CT['ctx_gain'][0],4)
open('tab_ctrl.tex','w').write('\n'.join(tc)+'\n\n')
# injection
S=collections.defaultdict(list)
for r in rows:S[r['_scenario']].append(r)
names={'I0_baseline':'I0 baseline','I1_paraphrase':'I1 paraphrase','I2_jitter':'I2 temporal jitter','I3_fragmented':'I3 fragmentation','I4_camouflage':'I4 camouflage','I5_adaptive':'I5 adaptive'}
ms=lambda v:(np.mean(v),np.std(v,ddof=1))
tab=[]; ds={}; dr={}; rec={}
for s,rs in sorted(S.items()):
    g=lambda m:np.array([r[m]['ap'] for r in rs])
    d1=g('full')-g('stage_fusion'); d2=g('full')-g('sign_rawgraph')
    ds[s]=d1.mean();dr[s]=d2.mean();rec[s]=np.mean([r['_cell_recovery'] for r in rs])
    eq=np.mean([r['_edge_quality']['same_cell'] for r in rs])
    tab.append(f"{names[s]} & {f(g('meta_only').mean(),2)} & {f(g('stage_fusion').mean(),2)} & {f(g('sign_rawgraph').mean(),2)} & {f(g('full').mean(),2)} $\\pm$ {f(g('full').std(ddof=1),3)} & {f(d1.mean(),3)} & {f(d2.mean(),3)} & {f(rec[s],2)} \\\\")
    k=s.split('_')[0]; N[f'inj_{k}_full']=f(g('full').mean(),2);N[f'inj_{k}_stage']=f(g('stage_fusion').mean(),2);N[f'inj_{k}_raw']=f(g('sign_rawgraph').mean(),2)
    N[f'inj_{k}_dstage']=f(d1.mean(),3);N[f'inj_{k}_draw']=f(d2.mean(),3);N[f'inj_{k}_rec']=f(rec[s],2);N[f'inj_{k}_edgeq']=f(eq,2)
    N[f'inj_{k}_nosync']=f(g('abl_no_sync').mean()-g('full').mean(),3);N[f'inj_{k}_nocont']=f(g('abl_no_content').mean()-g('full').mean(),3)
    N[f'inj_{k}_graph']=f(g('graph_only').mean(),2)
nonad=[ds[s] for s in ds if s!='I5_adaptive']
N['inj_dmin']=f(min(nonad),2);N['inj_dmax']=f(max(nonad),2);N['inj_dmaxraw']=f(max(dr.values()),2);N['inj_dadapt']=f(ds['I5_adaptive'],3);N['inj_recadapt']=f(rec['I5_adaptive'],2)
N['inj_bg']=f"{rows[0]['_inj']['bg_accounts']:,}".replace(',','{,}');N['inj_prev']=f(100*rows[0]['_n']['prevalence'],1)
N['inj_fpr_stage']=f(100*np.mean([r['_fpr_groups']['stage_fusion']['regular'] for r in S['I0_baseline']]),2)
N['inj_fpr_full']=f(100*np.mean([r['_fpr_groups']['full']['regular'] for r in S['I0_baseline']]),2)
N['inj_fpr_raw']=f(100*np.mean([r['_fpr_groups']['sign_rawgraph']['regular'] for r in S['I0_baseline']]),2)
N['inj_fpr5_stage']=f(100*np.mean([r['_fpr_groups']['stage_fusion']['regular'] for r in S['I5_adaptive']]),2)
N['inj_fpr5_full']=f(100*np.mean([r['_fpr_groups']['full']['regular'] for r in S['I5_adaptive']]),2)
NS=json.load(open(Y+'analysis_nested.json'));nm=NS['means']
# earlier-protocol values kept under yz_e_*
N['yz_d_stage_mean']=f(A['full_k2']['ap'][0]-A['stage_fusion']['ap'][0])
for key,tag in [('gnn-full_k2','gnnfull'),('mlp_nograph-full_k2','mlpfull'),('gnn-mlp_nograph','gnnmlp')]:
    b=NS[key];N[f'yz_d_{tag}']=f(b['d_ap']);N[f'yz_d_{tag}_lo']=f(b['ap_ci'][0]);N[f'yz_d_{tag}_hi']=f(b['ap_ci'][1])
GC=[json.load(open(Y+f'yelp_nested_gctrl_seed{k}.json')) for k in (0,1,2)]
def gm(name,m): return np.array([g[name][m] for g in GC])
N['yz0_ap_gnn']=f(NS['gnn_seed0']['gnn']['ap']);N['yz0_ap_mlp']=f(NS['gnn_seed0']['mlp_nograph']['ap'])
N['yz_ap_gnn']=f(gm('gnn_ctx','ap').mean());N['yz_ap_mlp']=f(gm('mlp_nograph_ctx','ap').mean());N['yz_auc_gnn']=f(gm('gnn_ctx','auroc').mean())
t=[]
for lab_,k in [('Text only (TF-IDF LR)','text_only'),('Stage-fusion (text + behaviour)','stage_fusion'),('Stage-fusion + account context (no graph)','acct_context'),('MultiSentinel-G (proposed, $k_{\\min}=2$)','full_k2')]:
    v=A[k];t.append(f"{lab_} & {v['auroc'][0]:.3f} & {v['ap'][0]:.3f} $\\pm$ {v['ap'][1]:.3f} & {v['f1'][0]:.3f} & {v['rec_fpr1'][0]:.3f} \\\\")
for lab_,k in [('Neural account model, no graph (MLP)','mlp_nograph'),('GraphSAGE (trained GNN)','gnn')]:
    nmk={'mlp_nograph':'mlp_nograph_ctx','gnn':'gnn_ctx'}[k];t.append(f"{lab_} & {gm(nmk,'auroc').mean():.3f} & {gm(nmk,'ap').mean():.3f} $\\pm$ {gm(nmk,'ap').std(ddof=1):.3f} & {gm(nmk,'f1').mean():.3f} & {gm(nmk,'rec_fpr1').mean():.3f} \\\\")
open('tab_yelp_nested.tex','w').write('\n'.join(t)+'\n\n')
import os
SR=[json.load(open(Y+f'yelp_sens_nested_seed{k}.json')) for k in (0,1,2)]
lab={'base':'Proposed (7 days, $q=0.01$, $\\alpha=0.8$)','W3':'Window 3 days','W14':'Window 14 days','W30':'Window 30 days','q0.001':'$q=0.001$','q0.05':'$q=0.05$','a0.5':'$\\alpha=0.5$','a0.95':'$\\alpha=0.95$','acct_context':'No graph (account context)'}
t=[]
for k in ['acct_context','base','W3','W14','W30','q0.001','q0.05','a0.5','a0.95']:
    a=np.array([x[k]['ap'] for x in SR]);u=np.array([x[k]['auroc'] for x in SR]);t.append(f"{lab[k]} & {a.mean():.4f} $\\pm$ {a.std(ddof=1):.4f} & {u.mean():.4f} \\\\")
    if k!='acct_context': N['sens_'+k.replace('.','')]=f(a.mean(),4)
open('tab_sens.tex','w').write('\n'.join(t)+'\n\n')
allsens=[np.mean([x[k]['ap'] for x in SR]) for k in lab if k!='acct_context'];N['sens_min']=f(min(allsens),4);N['sens_max']=f(max(allsens),4)
if os.path.exists(Y+'yelp_inject_frac.jsonl'):
    FR=[json.loads(l) for l in open(Y+'yelp_inject_frac.jsonl')]+[r|{'_frac':0.04} for r in rows if r['_seed']==0 and r['_scenario'] in('I0_baseline','I2_jitter')]
    tf=[]
    for sc,nm in (('I0_baseline','I0'),('I2_jitter','I2')):
        for r in sorted([r for r in FR if r['_scenario']==sc],key=lambda r:r['_frac']):
            tf.append(f"{nm} & {100*r['_frac']:.0f}\\% & {r['stage_fusion']['ap']:.3f} & {r['sign_rawgraph']['ap']:.3f} & {r['full']['ap']:.3f} & {r['full']['ap']-r['stage_fusion']['ap']:.3f} \\\\")
    open('tab_injfrac.tex','w').write('\n'.join(tf)+'\n\n')
KR=json.load(open(Y+'kmin_rule.json'));pv=KR['perm_global']['1']+KR['perm_product']['1']
N['yz_perm1']=f"{int(round(np.mean(pv))):,}".replace(',','{,}');N['yz_ntest']=f"{KR['real']['1'][1]:,}".replace(',','{,}')
N['yz_perm2']=str(max(KR['perm_global']['2']+KR['perm_product']['2']))
open('numbers3.tex','w').write(''.join(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}\n" for k,v in N.items()))
open('tab_inject.tex','w').write('\n'.join(tab)+'\n\n')
# yelp main table
rowsY=[('Text only (TF-IDF LR)','text_only'),('Behavioural only','behav_only'),('Stage-fusion (text + behaviour)','stage_fusion'),('Stage-fusion + account context (no graph)','acct_context'),('Raw co-review graph + aggregation','sign_rawgraph'),('Dense-block peeling','dense_block'),('Graph features only','graph_only'),('Neural account model, no graph (MLP)','mlp_nograph'),('GraphSAGE (trained GNN)','gnn'),('MultiSentinel-G (proposed, rule-selected $k_{\\min}=2$)','full_k2'),('Exploratory variant: content layer with $k_{\\min}=1$','full')]
t=[]
for lab,k in rowsY:
    v=A[k];g=lambda x,d=3:f"{v[x][0]:.{d}f}"
    t.append(f"{lab} & {g('auroc')} & {g('ap')} & {g('f1')} & {g('rec_fpr1')} \\\\")
open('tab_yelp.tex','w').write('\n'.join(t)+'\n\n')
ab=[('Exploratory $k_{\\min}=1$ configuration (base of the ablations below)','full'),('Proposed, rule-selected $k_{\\min}=2$','full_k2'),('Without sync layer','abl_no_sync'),('Without content layer','abl_no_content'),('Without statistical validation','abl_no_validation'),('Without propagation','abl_no_propagation'),('Without community features','abl_no_community')]
t=[]
for lab,k in ab:
    v=A[k];t.append(f"{lab} & {v['ap'][0]:.4f} & {v['auroc'][0]:.4f} & {v['ap'][0]-A['full']['ap'][0]:+.4f} \\\\")
open('tab_yelp_abl.tex','w').write('\n'.join(t)+'\n\n')
print(N)
