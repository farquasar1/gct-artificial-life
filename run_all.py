#!/usr/bin/env python3
"""GCT catalytic-loop experiment. No fitted parameters; all decisions in CONFIG.
Run: python run_all.py. Writes results/ and figures/."""
from pathlib import Path
import csv, json, platform, sys
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
CONFIG=dict(seed=20261107,n_main=200,n_sweep=100, k_low=.18,k_high=.22,
    delta=.2,D=1.,F0=1.,beta=.1,mu=.1,q=1.,cv=.08,initial_c=.15,
    initial_I=.15,washout_fraction=.9,assay_time=150.,early_time=30.,
    assay_duration=20.,long_time=600.,recovery_cut=.05,viability_cut=.001,
    product_cut=.5,rescue_dose=.2,substrate_dose=1.,budget=172.2,
    rtol=1e-8,atol=1e-10,base_probe=1e-7,probe_step=1e-8,probe_h=1e-4,
    sweep_k=np.linspace(.16,.24,21).tolist())
# State: F,A,B,C,I,S,P, cumulative food supplied (initial F excluded).
# Reactions: F+Cj -> Cj+Ci, cyclic i=(j+1)%3; Ci -> waste;
# F+A -> A+I; I -> waste; I+S -> I+P; source -> F (D F0), F -> waste (D F).
# The inflow/outflow form for F is equivalent to D(F0-F).
def rhs(t,y,k,cut=False):
    f=y[0]; c=y[1:4]; prev=np.roll(c,1)
    prod=k*f*prev
    if cut: prod[0]=0. # delete C -> A maintenance, no replacement support
    i_flux=CONFIG['beta']*f*c[0]
    pflux=CONFIG['q']*y[4]*y[5]
    return np.r_[CONFIG['D']*(1-f)-sum(prod)-i_flux,
        prod-CONFIG['delta']*c,i_flux-CONFIG['mu']*y[4],-pflux,pflux,CONFIG['D']]
def solve(y,t0,t1,k,cut=False,times=None,tight=False):
    ans=solve_ivp(rhs,(t0,t1),y,args=(k,cut),method='DOP853',t_eval=times,
        rtol=1e-10 if tight else CONFIG['rtol'],atol=1e-12 if tight else CONFIG['atol'])
    if not ans.success: raise RuntimeError(ans.message)
    if np.min(ans.y[:7]) < -1e-7: raise RuntimeError('Negative concentration')
    return ans

def initial():
    return np.r_[1., np.repeat(CONFIG['initial_c']*.1,3),CONFIG['initial_I']*.1,0.,0.,0.]
def estimate(k):
    # Independent, brief in-silico interventions; never use recovery data.
    y=np.r_[1.,np.repeat(CONFIG['base_probe'],3),0.,0.,0.,0.]
    h=CONFIG['probe_h']; eps=CONFIG['probe_step']
    base=solve(y,0,h,k,tight=True).y[:3+1,-1][1:]
    jac=np.zeros((3,3))
    for j in range(3):
        pert=y.copy(); pert[j+1]+=eps
        jac[:,j]=(solve(pert,0,h,k,tight=True).y[1:4,-1]-base)/(eps*h)
    # Off-diagonal local maintenance rates; diagonal loss independently known.
    np.fill_diagonal(jac,0.)
    return jac,float(max(abs(np.linalg.eigvals(jac))))
def assay(state,t,k,rescue=False):
    y=state.copy(); y[5]=CONFIG['substrate_dose']
    if rescue: y[4]+=CONFIG['rescue_dose']
    sol=solve(y,t,t+20,k,times=np.linspace(t,t+20,81))
    p=float(sol.y[6,-1]); viable=bool(np.min(sol.y[1:4])>=CONFIG['viability_cut'])
    # All treatments have the same allowance; unused I allowance is not recycled.
    cost=1.+float(sol.y[7,-1])+1.+(CONFIG['rescue_dose'] if rescue else 0.)
    return dict(product=p,viable=viable,success=bool(p>=CONFIG['product_cut']),
        joint=bool(viable and p>=CONFIG['product_cut'] and cost<=CONFIG['budget']+1e-8),
        resource_used=cost)
def experiment(k0,mult,seed,condition,full=True):
    k=k0*mult; grid=np.linspace(0,150,301)
    sol=solve(initial(),0,150,k,times=grid)
    late=assay(sol.y[:,-1],150,k)
    early=assay(sol.y[:,60],30,k) if full else {}
    m,rho=estimate(k)
    row=dict(seed=seed,condition=condition,k=k0,rho_analytic=float(np.prod(k)**(1/3)),
        rho_estimated=rho,recovery=bool(min(sol.y[1:4,-1])>=.05),
        min_core_150=float(min(sol.y[1:4,-1])),intermediate_150=float(sol.y[4,-1]),
        trajectory_viable=bool(np.min(sol.y[1:4])>=.001),**{'late_'+a:b for a,b in late.items()},
        **{'early_'+a:b for a,b in early.items()})
    if full:
        rescued=assay(sol.y[:,-1],150,k,True)
        row.update({'rescue_'+a:b for a,b in rescued.items()})
        long=solve(sol.y[:,-1],150,600,k,times=np.linspace(150,600,451))
        row['durable_recovery']=bool(np.min(long.y[1:4,-51:])>=.05)
        # Remove all non-core species at disruption. They provide no catalytic input.
        boundary=initial(); boundary[4:7]=0.
        b=solve(boundary,0,150,k,times=grid)
        row['boundary_max_core_error']=float(np.max(np.abs(b.y[1:4]-sol.y[1:4])))
        broken=solve(initial(),0,600,k,cut=True,times=np.linspace(550,600,51))
        row['cut_durable_recovery']=bool(np.min(broken.y[1:4])>=.05)
        row['matrix_max_error']=float(np.max(np.abs(m-np.array([[0,0,k[0]],[k[1],0,0],[0,k[2],0]]))))
    return row,sol

def write_csv(name,rows):
    with open(OUT/name,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def wilson(v):
    n=len(v); p=float(np.mean(v)); z=1.95996398454
    c=(p+z*z/(2*n))/(1+z*z/n); d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [p,max(0.,c-d),min(1.,c+d)]
def mean_ci(v):
    v=np.asarray(v,dtype=float)
    if not len(v): return [None,None,None]
    r=np.random.default_rng(4831)
    boots=np.mean(r.choice(v,(4000,len(v)),replace=True),axis=1)
    return [float(np.mean(v)),*map(float,np.quantile(boots,[.025,.975]))]
def summary(rows):
    d={'n':len(rows)}
    for key in rows[0]:
        vals=[r[key] for r in rows]
        if isinstance(vals[0],bool): d[key]=wilson(vals)
        elif isinstance(vals[0],(float,np.floating)): d[key]=mean_ci(vals)
    survivors=[r for r in rows if r['late_viable'] and r['trajectory_viable']]
    early=[r for r in rows if r.get('early_viable',False)]
    d['late_survivors_n']=len(survivors)
    d['late_product_given_viable']=mean_ci([r['late_product'] for r in survivors])
    d['early_survivors_n']=len(early)
    d['early_product_given_viable']=mean_ci([r['early_product'] for r in early])
    return d

def savefig(name):
    plt.tight_layout()
    for ext in ('pdf','png'): plt.savefig(FIG/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close()
def main():
    if '--reuse-trajectories' in sys.argv:
        # Recompute summaries/plots from existing raw trajectories and outcomes.
        # This changes no dynamics and is used only for deterministic QA/reanalysis.
        def read_csv(name):
            rows=list(csv.DictReader((OUT/name).open()))
            for r in rows:
                for key,value in r.items():
                    if value in ('True','False'): r[key]=value=='True'
                    elif key=='seed': r[key]=int(float(value))
                    elif key!='condition': r[key]=float(value)
            return rows
        old=json.loads((OUT/'summary.json').read_text())
        rows=read_csv('replicates.csv'); sweep=read_csv('sweep.csv')
        trajectories=read_csv('trajectories.csv'); nominal=old['nominal']
        def fix(r):
            for prefix in ('late','early','rescue'):
                if prefix+'_product' in r:
                    r[prefix+'_joint']=bool(r[prefix+'_viable'] and r[prefix+'_success'] and r[prefix+'_resource_used']<=CONFIG['budget']+1e-8)
        for r in rows+sweep+list(nominal.values()): fix(r)
        write_csv('replicates.csv',rows); write_csv('sweep.csv',sweep)
        sweep_summary=[]
        for k in CONFIG['sweep_k']:
            ss=summary([r for r in sweep if abs(r['k']-k)<1e-10]);ss['k']=k;sweep_summary.append(ss)
    else:
        rng=np.random.default_rng(CONFIG['seed']); sig=np.sqrt(np.log(1+CONFIG['cv']**2))
        # Geometric-mean-one lognormal uncertainty. Same three draws paired by seed.
        multipliers=np.exp(rng.normal(0,sig,(CONFIG['n_main'],3)))
        rows=[]; nominal={}; trajectories=[]
        for name,k in [('subthreshold',.18),('superthreshold',.22)]:
            r,s=experiment(k,np.ones(3),-1,name)
            nominal[name]=r
            for j,t in enumerate(s.t):
                trajectories.append(dict(condition=name,t=t,**dict(zip(['F','A','B','C','I','S','P','food_in'],s.y[:,j]))))
            for seed,m in enumerate(multipliers):
                rr,_=experiment(k,m,seed,name); rows.append(rr)
            print('Finished',name,flush=True)
        write_csv('replicates.csv',rows); write_csv('trajectories.csv',trajectories)
        sweep=[]; sweep_summary=[]
        for k in CONFIG['sweep_k']:
            rr=[experiment(k,m,int(i),'sweep',False)[0] for i,m in enumerate(multipliers[:CONFIG['n_sweep']])]
            sweep.extend(rr); ss=summary(rr); ss['k']=k; sweep_summary.append(ss)
        write_csv('sweep.csv',sweep)
    summaries={c:summary([r for r in rows if r['condition']==c]) for c in nominal}
    error=max(r['matrix_max_error'] for r in rows)
    prediction=[(r['rho_estimated']>.2)==r['recovery'] for r in rows]
    durable=[(r['rho_estimated']>.2)==r['durable_recovery'] for r in rows]
    # Numerical accuracy check on nominal complete assay trajectories.
    convergence=[]
    for name,k in [('subthreshold',.18),('superthreshold',.22)]:
        sol=solve(initial(),0,150,np.repeat(k,3),tight=True)
        convergence.append(abs(assay(sol.y[:,-1],150,np.repeat(k,3))['product']-nominal[name]['late_product']))
    report=dict(config=CONFIG,nominal=nominal,groups=summaries,sweep=sweep_summary,
        max_matrix_error=error,threshold_recovery_agreement=float(np.mean(prediction)),
        threshold_durable_agreement=float(np.mean(durable)),numerical_yield_error=max(convergence),
        software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'parameters.json').write_text(json.dumps(CONFIG,indent=2))
    # Figure 1: equal initial conditions, postwashout recovery and I retention.
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    for c,color in [('subthreshold','tab:orange'),('superthreshold','tab:blue')]:
        ts=[r for r in trajectories if r['condition']==c]
        ax[0].plot([r['t'] for r in ts],[r['A'] for r in ts],label=c,color=color)
        ax[1].plot([r['t'] for r in ts],[r['I'] for r in ts],label=c,color=color)
    ax[0].axhline(.05,color='gray',ls=':',label='Recovery cut-off')
    for a in ax: a.set_xlabel('Time after 90% washout'); a.legend(fontsize=8)
    ax[0].set_ylabel('Catalyst concentration (each)'); ax[1].set_ylabel('Reusable intermediate I')
    savefig('Vieira.fig01')
    # Figure 2: stochastic parameter robustness, Wilson intervals.
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    x=[s['rho_estimated'][0] for s in sweep_summary]
    for key,lab in [('recovery','Recovery'),('late_viable','Viable during assay'),('late_joint','Viable product')]:
        y=np.array([s[key] for s in sweep_summary]); ax[0].plot(x,y[:,0],label=lab); ax[0].fill_between(x,y[:,1],y[:,2],alpha=.15)
    y=np.array([s['late_product'] for s in sweep_summary]); ax[1].plot(x,y[:,0]);ax[1].fill_between(x,y[:,1],y[:,2],alpha=.2)
    for a in ax: a.axvline(.2,color='k',ls='--');a.set_xlabel('Mean interventionally estimated rho(B)')
    ax[0].set_ylabel('Fraction (95% Wilson intervals)');ax[0].legend(fontsize=8)
    ax[1].set_ylabel('Product yield (bootstrap 95% CI)'); savefig('Vieira.fig02')
    # Figure 3: rescue restores construction, not catalyst viability.
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    low=summaries['subthreshold']; high=summaries['superthreshold']
    labels=['Below','Above','Below + I']
    for a,items in zip(ax,[[low['late_product'],high['late_product'],low['rescue_product']],
                           [low['late_viable'],high['late_viable'],low['rescue_viable']]]):
        val=np.array(items);a.bar(labels,val[:,0],color=['tab:orange','tab:blue','tab:green']);a.errorbar(range(3),val[:,0],yerr=[val[:,0]-val[:,1],val[:,2]-val[:,0]],fmt='none',color='k',capsize=4)
    ax[0].set_ylabel('Product yield (bootstrap 95% CI)');ax[1].set_ylabel('Core viability fraction (Wilson 95% CI)')
    savefig('Vieira.fig03')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sweep','config')},indent=2))
if __name__=='__main__': main()
