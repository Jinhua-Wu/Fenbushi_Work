from model import Config,simulate
from dataclasses import replace
from pathlib import Path
import json,csv,time
import numpy as np

SCENARIOS={
 'baseline':{}, 'mask':{'mask':.8}, 'isolation':{'isolation':.8},
 'combined':{'mask':.8,'isolation':.8,'triage':1},
 'triage':{'triage':1}, 'no_hospital':{'hospital':False},
 'random10':{'target':'random'}, 'degree10':{'target':'degree'},
 'homogeneous':{'dispersion':0}, 'late_combined':{'mask':.8,'isolation':.8,'triage':1,'policy_day':25},
 'game_high_cost':{'game':True,'mask':.8,'isolation':.8,'triage':1,'income_cost':2.5},
 'game_subsidy':{'game':True,'mask':.8,'isolation':.8,'triage':1,'income_cost':2.5,'subsidy':1.8},
 'game_low_copay':{'game':True,'mask':.8,'isolation':.8,'triage':1,'income_cost':2.5,'copay':.1},
}

def main():
    out=Path(__file__).parent/'results'; out.mkdir(exist_ok=True)
    runs=[]; summary={}; demos={}; start=time.time()
    for name,kw in SCENARIOS.items():
        group=[]
        for seed in range(100,130):
            r=simulate(Config(seed=seed,**kw)); group.append(r['metrics']); runs.append(dict(scenario=name,seed=seed,**r['metrics']))
        s={}
        for key in group[0]:
            a=np.array([g[key] for g in group],float)
            s[key]={'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'ci95':float(2.045*a.std(ddof=1)/np.sqrt(len(a)))}
        summary[name]=s
        demos[name]=simulate(Config(seed=100,**kw),True)
        print(name,round(s['attack']['mean'],3),flush=True)
    sensitivity=[]
    for beta in [.035,.055,.075]:
        for day in [5,10,25]:
            vals=[simulate(Config(seed=s,beta=beta,policy_day=day,mask=.8,isolation=.8,triage=1))['metrics']['attack'] for s in range(200,215)]
            sensitivity.append(dict(beta=beta,day=day,attack_mean=float(np.mean(vals)),attack_sd=float(np.std(vals,ddof=1))))
    with (out/'runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(runs[0])); w.writeheader(); w.writerows(runs)
    for name,data in [('summary',summary),('demos',demos),('sensitivity',sensitivity)]:
        (out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    # Paired CI reflects correlation across shared network and natural-history seeds.
    paired={}
    baseline=[r['attack'] for r in runs if r['scenario']=='baseline']
    for name in SCENARIOS:
        diff=np.array([r['attack'] for r in runs if r['scenario']==name])-baseline
        paired[name]={'difference':float(diff.mean()),'ci95':float(2.045*diff.std(ddof=1)/np.sqrt(30))}
    (out/'paired.json').write_text(json.dumps(paired),encoding='utf-8')
    print('seconds',round(time.time()-start,1))

if __name__=='__main__': main()
