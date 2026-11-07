#!/usr/bin/env python3
"""Independent numeric diagnostics for both chemistries; no network access."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import run_all as first
import topology_experiment as second
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
rows=list(csv.DictReader((OUT/'replicates.csv').open()))
rs=[{k:(v=='True' if v in ('True','False') else v if k=='condition' else float(v)) for k,v in r.items()} for r in rows]
low=sorted([r for r in rs if r['condition']=='subthreshold'],key=lambda r:r['seed'])
high=sorted([r for r in rs if r['condition']=='superthreshold'],key=lambda r:r['seed'])
diff=np.array([h['early_product']-l['early_product'] for h,l in zip(high,low)])
rng=np.random.default_rng(7717);boots=rng.choice(diff,(20000,200)).mean(1)
rng=np.random.default_rng(first.CONFIG['seed']);multipliers=np.exp(rng.normal(0,np.sqrt(np.log1p(first.CONFIG['cv']**2)),(200,3)))
supercritical=[]
for r in rs:
 k=r['k']*multipliers[int(r['seed'])];rho=np.cbrt(np.prod(k))
 if rho<=.2:continue
 v=np.array([1,k[1]/rho,k[1]*k[2]/rho**2]);F=.2/rho;c=(1-F)/(.2*v.sum()+.1*F)*v
 supercritical.append(dict(seed=int(r['seed']),condition=r['condition'],rho=float(rho),min_equilibrium=float(c.min()),durable=r['durable_recovery']))
miss=[r for r in supercritical if not r['durable']];exceptions=[r for r in miss if r['min_equilibrium']>=.05]
assert len(supercritical)==199 and len(miss)==21 and len(exceptions)==1
ex=exceptions[0];k=(.22 if ex['condition']=='superthreshold' else .18)*multipliers[ex['seed']]
long=first.solve(first.initial(),0,1000,k,times=np.arange(600,1001))
ex['first_integer_time_above_cut']=float(long.t[long.y[1:4].min(0)>=.05][0])
# Independent integration of full S/P branches, without integrated-hazard states.
network=list(csv.DictReader((OUT/'topology/networks.csv').open()))
constructions=list(csv.DictReader((OUT/'topology/constructions.csv').open()))
errors=[]
for seed in [0,121,299]:
 for name in ['A','B']:
  nr=next(r for r in network if int(r['seed'])==seed and r['topology']==name)
  k=np.array([float(nr[f'k{j}']) for j in range(6)]);W=second.matrix(k,second.TOPO[name])
  def rhs(t,y,assay):
   F=y[0];c=y[1:7];I=y[7:10];reg=F*(W@c);ip=.04*F*c[[0,2,3]]
   hazards=second.Q*np.array([np.prod(I[list(r)]) for r in second.REQ])
   flux=hazards*y[10:14] if assay else np.zeros(4)
   return np.r_[1-F-reg.sum()-ip.sum(),reg-.2*c,ip-.1*I,-flux,flux]
  y=np.r_[1.,np.full(6,.015),np.zeros(11)]
  pre=solve_ivp(lambda t,y:rhs(t,y,False),(0,150),y,method='DOP853',rtol=1e-11,atol=1e-13)
  y=pre.y[:,-1];y[10:14]=1
  ass=solve_ivp(lambda t,y:rhs(t,y,True),(150,170),y,method='DOP853',rtol=1e-11,atol=1e-13)
  expected=np.array([float(next(r['product'] for r in constructions if int(r['seed'])==seed and r['topology']==name and int(r['construction'])==j)) for j in range(4)])
  errors.append(float(abs(ass.y[14:18,-1]-expected).max()))
assert max(errors)<1e-6
result=dict(first_experiment=dict(n_supercritical=len(supercritical),n_misses=len(miss),n_miss_equilibrium_below_cut=sum(r['min_equilibrium']<.05 for r in miss),exceptions=exceptions,
 early_difference=float(diff.mean()),early_paired_ci=np.quantile(boots,[.025,.975]).tolist(),rescue_success=sum(r['rescue_success'] for r in low),rescue_viable=sum(r['rescue_viable'] for r in low),
 rescue_product_lower_bound=float(1-np.exp(-2*(1-np.exp(-2))))),
 topology_independent_full_product_integration=dict(networks=6,assays=24,maximum_yield_error=max(errors),relative_tolerance=1e-11,absolute_tolerance=1e-13))
(OUT/'round3_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
