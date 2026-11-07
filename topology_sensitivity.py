#!/usr/bin/env python3
"""Post-hoc specificity check, added after seeing the prespecified test results.
No new simulations or parameter changes. Tests whether core features beat a
simpler predictor given topology-by-product labels, in the same seed folds.
"""
import csv,json
import numpy as np
from pathlib import Path
from topology_experiment import ROOT,OUT,fit_predict,metrics,ci
rows=list(csv.DictReader((OUT/'constructions.csv').open()))
for r in rows:
    for k in r:
        if k not in ('topology','core','core_viable'):r[k]=float(r[k])
y=np.array([r['success'] for r in rows]);seeds=np.array([r['seed'] for r in rows],int)
L=np.log1p([[r[k] for k in ['cumulative_reg','cumulative_ip','cumulative_stock']] for r in rows])
labels=np.array([int(r['construction'])+4*(r['topology']=='B') for r in rows])
X=np.c_[np.eye(8)[labels][:,1:],[[r[k] for k in ['initial_stock','initial_reg','initial_ip']] for r in rows],L,L**2,L[:,0]*L[:,1],L[:,0]*L[:,2],L[:,1]*L[:,2]]
XA=np.c_[X,[[r['core_fraction'],r['all_in_core']] for r in rows]]
preds=[]
for A in [X,XA]:
    p=np.zeros(len(rows))
    for fold in range(5):
        test=seeds%5==fold;p[test]=fit_predict(A,y,~test,test)
    preds.append(p)
losses=[-y*np.log(np.clip(p,1e-12,1))-(1-y)*np.log(np.clip(1-p,1e-12,1)) for p in preds]
d=np.array([(losses[0]-losses[1])[seeds==s].mean() for s in range(300)])
summary=dict(analysis_status='post-hoc, no parameter changes',label_baseline=metrics(y,preds[0]),label_plus_core=metrics(y,preds[1]),
 loss_improvement_seed_bootstrap=ci(d,np.random.default_rng(8813)),
 cumulative_reg_ranges={name:[min(r['cumulative_reg'] for r in rows if r['topology']==name),max(r['cumulative_reg'] for r in rows if r['topology']==name)] for name in ['A','B']},
 joint_success={name:[int(sum(r['joint'] for r in rows if r['topology']==name and r['construction']==j)) for j in range(4)] for name in ['A','B']})
(OUT/'sensitivity.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
