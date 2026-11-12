"""
Revision figures (Figures 1, 3, 4, 5 of the revised manuscript), drawn only from results/tables.
Figure 2 (IBIP illustration) is a static diagram. Error bars: 95% CI (t-based) across seeds.
Usage: python experiments/make_revision_figures.py
"""
import os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=os.path.join(ROOT,'results','figures','revision')+os.sep
os.makedirs(OUT,exist_ok=True)
T=lambda f: os.path.join(ROOT,'results','tables',f)
C={'Sequential':'#7f8fa6','Random':'#c8d0db','Importance-Blocked':'#e0a040','IBIP':'#2a5caa'}
def ci(v): v=np.asarray(v); return 2.776*v.std(ddof=1)/np.sqrt(len(v)) if len(v)==5 else 1.96*v.std(ddof=1)/np.sqrt(len(v))
# Fig 1 workflow
fig,ax=plt.subplots(figsize=(3.6,5.6)); ax.axis('off'); ax.set_xlim(-0.5,10.5); ax.set_ylim(0,20)
steps=["WDBC dataset (569 x 30)","Stratified split (80:20 or 5-fold CV);\nz-score fitted on training data","Feature ranking: XGBoost\npermutation importance (train)","Feature allocation\nSequential | Random |\nImportance-Blocked | IBIP","Classical projection phi_q = pi tanh(W_q x_q / sqrt(d_q));\nencoding Ry(phi_q) Rz(e_q)","Interaction topology\n(2-qubit gates / 3-qubit graphs)","Variational: Ry(v_q) Rz(v_q)","Pauli-Z expectations -> linear readout\n-> sigmoid; BCE loss, SuperGD","Benign / Malignant"]
ys=np.linspace(19.0,1.0,len(steps))
for i,(s,y) in enumerate(zip(steps,ys)):
    h=1.2 if s.count('\n')<1 else (1.55 if s.count('\n')<2 else 2.0)
    ax.add_patch(FancyBboxPatch((-0.2,y-h/2),10.4,h,boxstyle="round,pad=0.02,rounding_size=0.15",fc='#eef2f8' if i not in (3,5) else '#d6e2f3',ec='#33475b',lw=0.8))
    ax.text(5,y,s.replace('phi_q','φ_q').replace('pi ','π ').replace('sqrt','√'),ha='center',va='center',fontsize=7.2)
    if i<len(steps)-1: ax.annotate('',xy=(5,ys[i+1]+(1.2 if steps[i+1].count('\n')<1 else (1.55 if steps[i+1].count('\n')<2 else 2.0))/2),xytext=(5,y-(h/2)),arrowprops=dict(arrowstyle='->',lw=0.8,color='#33475b'))
plt.savefig(OUT+'fig1_workflow.png',dpi=300,bbox_inches='tight'); plt.close()
# Fig 3 four-way ablation WDBC
a=pd.read_csv(T('ablation_4way_per_seed.csv'))
m={'sequential':'Sequential','random':'Random','importance_blocked':'Importance-Blocked','ibip':'IBIP'}
fig,ax=plt.subplots(figsize=(6.2,3.2))
for i,(k,lab) in enumerate(m.items()):
    v=a[a.Strategy==k].Accuracy.values
    ax.bar(i,v.mean(),yerr=ci(v),color=C[lab],edgecolor='#333',lw=0.6,capsize=4,width=0.62)
    ax.scatter(np.full(5,i)+np.linspace(-0.15,0.15,5),v,s=9,color='k',zorder=3)
    ax.text(i,v.mean()+ci(v)+0.25,f"{v.mean():.2f}",ha='center',fontsize=8.5)
ax.set_xticks(range(4)); ax.set_xticklabels(['Sequential','Random','Importance-\nBlocked','IBIP']); ax.set_ylim(86,101.5); ax.set_ylabel('Test accuracy (%)')
plt.tight_layout(); plt.savefig(OUT+'fig3_allocation_ablation.png',dpi=300); plt.close()
# Fig 4 three-qubit suite
t=pd.read_csv(T('three_qubit_suite_per_seed.csv'))
tops=['cnot_star','cnot_chain','full_mesh','swap_star','swap_chain','none','cnot_ring']
labs=['CNOT-Star','CNOT-Chain','Full-Mesh','SWAP-Star','SWAP-Chain','Independent','CNOT-Ring']
fig,ax=plt.subplots(figsize=(6.6,3.2)); x=np.arange(len(tops)); w=0.38
for j,al in enumerate(['IBIP','Sequential']):
    mu=[t[(t.Topology==tp)&(t.Allocation==al)].Accuracy.mean() for tp in tops]
    e=[ci(t[(t.Topology==tp)&(t.Allocation==al)].Accuracy.values) for tp in tops]
    ax.bar(x+(j-0.5)*w,mu,w,yerr=e,capsize=3,color=C[al],edgecolor='#333',lw=0.6,label=al)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8.5); ax.set_ylim(70,100.5); ax.set_ylabel('Test accuracy (%)'); ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.13),fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT+'fig4_three_qubit_topologies.png',dpi=300); plt.close()
# Fig 5 external datasets
d=pd.read_csv(T('multi_dataset_per_seed.csv'))
fig,axs=plt.subplots(1,2,figsize=(6.6,2.9),sharey=False)
for ax,(ds,title) in zip(axs,[("Parkinson's Disease","Parkinson's disease (195 x 22)"),("Heart Disease","Cleveland heart disease (303 x 13)")]):
    for i,(k,lab) in enumerate(m.items()):
        v=d[(d.Dataset==ds)&(d.Strategy==k)].Accuracy.values
        ax.bar(i,v.mean(),yerr=ci(v),color=C[lab],edgecolor='#333',lw=0.6,capsize=3,width=0.62)
        ax.text(i,61.2,f"{v.mean():.1f}",ha='center',fontsize=8,color='white' if lab in ('IBIP','Sequential') else 'black')
    sv=d[(d.Dataset==ds)&(d.Strategy=='SVM_RBF')].Accuracy.mean()
    ax.axhline(sv,ls='--',lw=0.8,color='#555'); ax.text(-0.45,sv+0.6,f'RBF-SVM {sv:.1f}',ha='left',fontsize=7.5,color='#444')
    ax.set_xticks(range(4)); ax.set_xticklabels(['Seq.','Random','Imp.-\nBlocked','IBIP'],fontsize=8); ax.set_ylim(60,95); ax.set_title(title,fontsize=9)
axs[0].set_ylabel('Test accuracy (%)')
plt.tight_layout(); plt.savefig(OUT+'fig5_external_datasets.png',dpi=300); plt.close()
print('done')
