"""Checks saved outcomes against rerun nominal ODEs and accounting identities."""
import csv,json
import numpy as np
import run_all as m
s=json.loads((m.OUT/'summary.json').read_text())
r=list(csv.DictReader((m.OUT/'replicates.csv').open()))
w=list(csv.DictReader((m.OUT/'sweep.csv').open()))
assert len(r)==400 and len(w)==2100
for row in r+w:
    for p in ('late','early','rescue'):
        if p+'_product' not in row: continue
        expected=(row[p+'_viable']=='True' and float(row[p+'_product'])>=m.CONFIG['product_cut'] and float(row[p+'_resource_used'])<=m.CONFIG['budget']+1e-8)
        assert (row[p+'_joint']=='True')==expected
errs={}
for condition,k in [('subthreshold',.18),('superthreshold',.22)]:
    row,_=m.experiment(k,np.ones(3),-1,condition)
    errs[condition]=max(abs(row[x]-s['nominal'][condition][x]) for x in ['late_product','rho_estimated','min_core_150','rescue_product'])
    assert errs[condition]<1e-9
report={'checks':'passed','main_replicates':len(r),'sweep_replicates':len(w),'nominal_rerun_max_differences':errs,'budget':m.CONFIG['budget']}
(m.OUT/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
