#!/usr/bin/env python3
"""Deterministic organization test. See topology_protocol.md written before execution."""
from pathlib import Path
import csv,json,platform,hashlib
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'topology'; OUT.mkdir(exist_ok=True,parents=True)
FIG=ROOT/'figures'
CFG=dict(master_seed=20261108,n=300,cv=.12,rates=[.28]*3+[.15]*3,d=.2,D=1.,beta=.04,mu=.1,
 initial_c=.015,assay_start=150.,end=170.,cut=.5,viability=.001,rtol=1e-8,atol=1e-10,
 bootstrap_seed=8813,bootstrap_n=10000,folds=5,ridge=1.)
TOPO={'A':[(0,1,2),(3,4,5)],'B':[(0,1,3),(2,4,5)]}
REQ=[(0,1),(0,2),(1,2),(0,1,2)]; PRODUCERS=np.array([0,2,3]); Q=np.array([100.,100.,100.,5000.])
def matrix(k,cycles):
    W=np.zeros((6,6))
    for cyc in cycles:
        for j,i in zip(cyc,cyc[1:]+cyc[:1]): W[i,j]=k[j]
    return W
# state F,c[6],I[3], integrated regeneration, I-production, total c,
# four integrated assay hazards. S,P recovered exactly from hazard integrals.
def rhs(t,y,W,assay=False):
    F=y[0];c=y[1:7];I=y[7:10]
    reg=F*(W@c);ip=CFG['beta']*F*c[PRODUCERS]
    hazards=Q*np.array([np.prod(I[list(r)]) for r in REQ]) if assay else np.zeros(4)
    return np.r_[1-F-reg.sum()-ip.sum(),reg-CFG['d']*c,ip-CFG['mu']*I,reg.sum(),ip.sum(),c.sum(),hazards]
def estimate(W):
    y=np.r_[1.,np.full(6,1e-7),np.zeros(10)]
    base=rhs(0,y,W)[1:7];M=np.zeros((6,6)); eps=1e-8
    for j in range(6):
        yp=y.copy();yp[1+j]+=eps
        M[:,j]=(rhs(0,yp,W)[1:7]-base)/eps
    np.fill_diagonal(M,0.)
    n,labels=connected_components((M>1e-6).astype(int),directed=True,connection='strong')
    core=np.zeros(6,dtype=bool);rhos=[]
    for lab in range(n):
        ix=np.where(labels==lab)[0];r=float(max(abs(np.linalg.eigvals(M[np.ix_(ix,ix)]))))
        rhos.append(r)
        if r>.2: core[ix]=True
    return M,core,rhos

def fit_predict(X,y,train,test):
    m=X[train].mean(0);s=X[train].std(0);s[s<1e-12]=1
    Z=np.c_[np.ones(len(X)),(X-m)/s];z=Z[train];v=y[train]
    def objective(w):
        a=z@w
        return np.logaddexp(0,a).sum()-v@a+.5*CFG['ridge']*np.dot(w[1:],w[1:])
    def grad(w):
        g=z.T@(expit(z@w)-v);g[1:]+=CFG['ridge']*w[1:];return g
    opt=minimize(objective,np.zeros(Z.shape[1]),jac=grad,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7})
    if not opt.success: raise RuntimeError(opt.message)
    return expit(Z[test]@opt.x)
def metrics(y,p):
    p=np.clip(p,1e-12,1-1e-12);n1=y.sum();n0=len(y)-n1
    auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n0*n1) if n0*n1 else None
    return dict(log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),brier=float(np.mean((p-y)**2)),auc=float(auc) if auc is not None else None)
def ci(v,rng):
    v=np.asarray(v);b=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
    return [float(v.mean()),*map(float,np.quantile(b,[.025,.975]))]
def main():
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((ROOT/'topology_protocol.md').read_bytes()).hexdigest()+'\n')
    rng=np.random.default_rng(CFG['master_seed']);sig=np.sqrt(np.log(1+CFG['cv']**2))
    mult=np.exp(rng.normal(0,sig,(CFG['n'],6)));rows=[];nets=[];maxerr=0
    for seed,m in enumerate(mult):
        k=np.array(CFG['rates'])*m
        for name,cycles in TOPO.items():
            W=matrix(k,cycles);M,core,rhos=estimate(W);maxerr=max(maxerr,float(abs(W-M).max()))
            y0=np.r_[1.,np.full(6,.015),np.zeros(10)]
            pre=solve_ivp(rhs,(0,150),y0,args=(W,False),method='DOP853',rtol=CFG['rtol'],atol=CFG['atol'])
            ass=solve_ivp(rhs,(150,170),pre.y[:,-1],args=(W,True),t_eval=np.linspace(150,170,81),method='DOP853',rtol=CFG['rtol'],atol=CFG['atol'])
            if not pre.success or not ass.success: raise RuntimeError('integration failed')
            if min(pre.y[:10].min(),ass.y[:10].min()) < -1e-7: raise RuntimeError('negative concentrations')
            end=ass.y[:,-1];p=-np.expm1(-end[13:17]);viable=bool(core.any() and ass.y[1:7][core].min()>=.001)
            record=dict(seed=seed,topology=name,initial_stock=.09,initial_reg=float((W@y0[1:7]).sum()),initial_ip=float(CFG['beta']*y0[1:7][PRODUCERS].sum()),
                cumulative_reg=float(end[10]),cumulative_ip=float(end[11]),cumulative_stock=float(end[12]),core=''.join(map(str,core.astype(int))),
                core_viable=viable,min_rho=min(rhos),max_rho=max(rhos),**{f'k{j}':float(v) for j,v in enumerate(k)})
            nets.append(record)
            for j,req in enumerate(REQ):
                inside=core[PRODUCERS[list(req)]]
                rows.append(dict(**record,construction=j,product=float(p[j]),hazard=float(end[13+j]),success=int(p[j]>=.5),joint=int(p[j]>=.5 and viable),core_fraction=float(inside.mean()),all_in_core=int(inside.all())))
        if seed%50==0: print('Completed paired seed',seed,flush=True)
    def write(name,data):
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write('networks.csv',nets)
    # Explicit pairing and exact initial matching checks.
    for a,b in zip(nets[::2],nets[1::2]):
        for key in ['initial_stock','initial_reg','initial_ip']:
            assert abs(a[key]-b[key])<1e-12,(key,a[key],b[key])
    throughput=np.array([[r[k] for k in ['cumulative_reg','cumulative_ip','cumulative_stock']] for r in rows])
    L=np.log1p(throughput)
    X=np.c_[np.array([[r[k] for k in ['initial_stock','initial_reg','initial_ip']] for r in rows]),
        np.eye(4)[[r['construction'] for r in rows]][:,1:],L,L**2,L[:,0]*L[:,1],L[:,0]*L[:,2],L[:,1]*L[:,2]]
    XA=np.c_[X,[[r['core_fraction'],r['all_in_core']] for r in rows]]
    y=np.array([r['success'] for r in rows]);seeds=np.array([r['seed'] for r in rows]);pred=[]
    for A in [X,XA]:
        p=np.zeros(len(rows))
        for fold in range(5):
            test=seeds%5==fold; p[test]=fit_predict(A,y,~test,test)
        pred.append(p)
    for r,p0,p1 in zip(rows,*pred): r.update(baseline_prediction=float(p0),core_prediction=float(p1))
    write('constructions.csv',rows)
    losses=[-y*np.log(np.clip(p,1e-12,1))-(1-y)*np.log(np.clip(1-p,1e-12,1)) for p in pred]
    perseed=np.array([(losses[0]-losses[1])[seeds==s].mean() for s in range(CFG['n'])])
    bootrng=np.random.default_rng(8813)
    summary=dict(config=CFG,software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
      matrix_max_abs_error=maxerr,initial_matching_max_error=max(abs(a['initial_reg']-b['initial_reg']) for a,b in zip(nets[::2],nets[1::2])),
      baseline=metrics(y,pred[0]),augmented=metrics(y,pred[1]),loss_improvement_seed_bootstrap=ci(perseed,bootrng),
      outcomes={},paired_yield_B_minus_A={},throughput_B_minus_A={},n_networks=len(nets),n_construction_assays=len(rows),
      exact_exposure_oracle_mismatches=int(sum(int((-np.expm1(-r['hazard']))>=.5)!=r['success'] for r in rows)))
    for name in TOPO:
        rr=[r for r in rows if r['topology']==name]
        summary['outcomes'][name]=dict(core_viable=sum(r['core_viable'] for r in nets if r['topology']==name),
            construction_success=[sum(r['success'] for r in rr if r['construction']==j) for j in range(4)],
            mean_product=[float(np.mean([r['product'] for r in rr if r['construction']==j])) for j in range(4)],
            mean_cumulative_reg=float(np.mean([r['cumulative_reg'] for r in nets if r['topology']==name])))
    for j in range(4):
        a=np.array([r['product'] for r in rows if r['topology']=='A' and r['construction']==j]);b=np.array([r['product'] for r in rows if r['topology']=='B' and r['construction']==j])
        summary['paired_yield_B_minus_A'][str(j)]=ci(b-a,bootrng)
    for key in ['cumulative_reg','cumulative_ip','cumulative_stock']:
        summary['throughput_B_minus_A'][key]=ci(np.array([b[key]-a[key] for a,b in zip(nets[::2],nets[1::2])]),bootrng)
    # Conditional finite-ensemble diagnostic. Counts do not imply causal identification.
    summary['by_core_fraction']={str(f):dict(n=sum(r['core_fraction']==f for r in rows),successes=sum(r['success'] for r in rows if r['core_fraction']==f)) for f in sorted(set(r['core_fraction'] for r in rows))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'parameters.json').write_text(json.dumps(CFG,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(10,3.8))
    for name,color,offset in [('A','tab:blue',-.18),('B','tab:orange',.18)]:
        vals=summary['outcomes'][name]['construction_success'];ax[0].bar(np.arange(4)+offset,np.array(vals)/300,.36,label=name,color=color)
    ax[0].set_xticks(range(4),['I1 I2','I1 I3','I2 I3','I1 I2 I3']);ax[0].set_ylabel('Construction success fraction');ax[0].legend(title='Topology')
    for j,key in enumerate(['baseline','augmented']): ax[1].bar(j,summary[key]['log_loss'],color=['gray','tab:green'][j])
    ax[1].set_xticks([0,1],['Stocks + throughput','Add core membership']);ax[1].set_ylabel('Grouped held-out log loss (lower is better)')
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(FIG/f'Vieira.fig04.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
