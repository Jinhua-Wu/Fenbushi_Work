"""Synthetic multilayer epidemic simulation. Requires Python 3.10+ and numpy."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class Config:
    n: int = 600
    days: int = 100
    beta: float = .055
    seed: int = 42
    policy_day: int = 10
    mask: float = 0.
    isolation: float = 0.
    triage: float = 0.
    target: str = 'none'
    dispersion: float = .35
    hospital: bool = True
    game: bool = False
    income_cost: float = 1.2
    subsidy: float = 0.
    copay: float = .5

    def __post_init__(self):
        if not isinstance(self.seed,int) or self.seed<0: raise ValueError('seed must be a nonnegative integer')
        if not isinstance(self.n,int) or not isinstance(self.days,int): raise ValueError('n and days must be integers')
        for key in ['mask','isolation','triage','copay']:
            if not 0<=getattr(self,key)<=1: raise ValueError(key+' must be 0..1')
        if not 0<=self.beta<=1 or self.dispersion<0: raise ValueError('invalid transmission parameter')
        if not 0<=self.policy_day<=250: raise ValueError('invalid policy_day')
        if self.income_cost<0 or self.subsidy<0: raise ValueError('cost must be nonnegative')
        if self.target not in ['none','random','degree']: raise ValueError('invalid target')

def simulate(c=Config(), snapshots=False):
    if not 100 <= c.n <= 3000 or not 20 <= c.days <= 250: raise ValueError('n 100..3000; days 20..250')
    rng = np.random.default_rng(c.seed)
    n = c.n
    # Independent streams prevent an intervention's random draws changing natural history.
    natural = np.random.default_rng(c.seed + 100000)
    pairs = []
    for start in range(0,n,4):
        for i in range(start,min(start+4,n)):
            for j in range(i+1,min(start+4,n)): pairs.append((i,j,0))
    for start in range(0,n,20):
        for i in range(start,min(start+20,n)):
            for j in range(i+1,min(start+20,n)):
                if rng.random()<.20: pairs.append((i,j,1))
    # Preferential attachment community graph, 3 edges per arriving node.
    urn=list(range(4))*3
    for i in range(4,n):
        js=set()
        while len(js)<3: js.add(int(rng.choice(urn)))
        for j in js: pairs.append((i,j,2)); urn.extend([i,j])
    edges=np.array(pairs,dtype=int)
    degree=np.bincount(edges[:,:2].ravel(),minlength=n)
    z=natural.gamma(c.dispersion,1/c.dispersion,n) if c.dispersion>0 else np.ones(n)
    latent=natural.integers(2,5,n); duration=natural.integers(6,10,n)
    symptomatic=natural.random(n)<.7; severe=natural.random(n)<.08
    mask_rank=natural.random(n); iso_rank=natural.random(n)
    selected=np.zeros(n,bool); budget=max(1,int(n*.1))
    if c.target=='degree': selected[np.argsort(-degree,kind='stable')[:budget]]=True
    if c.target=='random': selected[rng.choice(n,budget,replace=False)]=True
    state=np.zeros(n,dtype=int); infected=np.full(n,-1); infectious=np.full(n,-1)
    initial=natural.choice(n,5,replace=False); state[initial]=2; infectious[initial]=0; infected[initial]=-3
    parent=np.full(n,-1); layer=np.full(n,-1); secondary=np.zeros(n,int)
    history=[]; frames=[]; x=.25; burden=0.; over=0
    for t in range(c.days):
        e=(state==1)&(t-infected>=latent); state[e]=2; infectious[e]=t
        r=(state==2)&(t-infectious>=duration); state[r]=3
        active=t>=c.policy_day
        if c.game and active:
            risk=1-np.exp(-10*np.mean(state==2))
            delta=.65*risk*(8+12*c.copay) - max(0,c.income_cost-c.subsidy)
            desired=1/(1+np.exp(-np.clip(delta/.35,-30,30)))
            x=.8*x+.2*desired
        compliance=x if c.game else 1.
        masked=mask_rank<(c.mask*compliance if active else 0)
        isolated=(state==2)&symptomatic&(t-infectious>=2)&(iso_rank<c.isolation*compliance)&active
        a=edges[:,0]; b=edges[:,1]; l=edges[:,2]
        if c.hospital:
            visitors=np.flatnonzero((state==2)&symptomatic&(t-infectious>=2)&~isolated)
            visitors=visitors[rng.random(len(visitors))<.65]
            healthy=rng.choice(n,max(6,n//60),replace=False)
            people=np.unique(np.r_[visitors,healthy,np.arange(max(6,n//50))])
            hp=[]
            for i in people:
                others=people[people!=i]
                for j in rng.choice(others,min(10,len(others)),replace=False):
                    if i<j: hp.append((i,j))
            if hp:
                hp=np.array(hp); a=np.r_[a,hp[:,0]]; b=np.r_[b,hp[:,1]]; l=np.r_[l,np.full(len(hp),3)]
        src=np.r_[a,b]; dst=np.r_[b,a]; ll=np.r_[l,l]
        valid=(state[src]==2)&(state[dst]==0)
        src=src[valid]; dst=dst[valid]; ll=ll[valid]
        w=np.array([1.,.65,.55,1.6])[ll]
        w*=np.where((ll!=0)&(isolated[src]|isolated[dst]),.12,1.)
        if active:
            w*=np.where((ll!=0)&(selected[src]|selected[dst]),.2,1.)
            w*=np.where(ll==3,1-.85*c.triage,1.)
        hazard=c.beta*w*z[src]*(1-.5*masked[src])*(1-.3*masked[dst])
        times=rng.exponential(1/np.maximum(hazard,1e-15))
        hits=np.flatnonzero(times<1)
        hits=hits[np.argsort(times[hits])]
        _, first=np.unique(dst[hits],return_index=True); hits=hits[first]
        new=dst[hits]; state[new]=1; infected[new]=t; parent[new]=src[hits]; layer[new]=ll[hits]
        np.add.at(secondary,src[hits],1)
        demand=int(np.sum((state==2)&severe&(t-infectious>=3)))
        capacity=max(1,round(n*.008)); over+=max(0,demand-capacity)
        burden+=np.sum(isolated)+.2*np.sum(selected)*active
        history.append(dict(day=t,S=int(np.sum(state==0)),E=int(np.sum(state==1)),I=int(np.sum(state==2)),R=int(np.sum(state==3)),new=len(new),demand=demand,capacity=capacity,compliance=float(compliance)))
        if snapshots: frames.append(state.tolist())
        assert np.bincount(state,minlength=4).sum()==n
    cases=int(np.sum(state!=0)); peak=max(h['I'] for h in history)
    transmitters=secondary[state!=0]
    top=max(1,int(np.ceil(.2*len(transmitters))))
    metrics=dict(cases=cases,attack=cases/n,peak=peak,peak_day=max(history,key=lambda h:h['I'])['day'],hospital=int(np.sum(layer==3)),overflow=over,restriction_days=float(burden),top20_share=float(np.sort(transmitters)[-top:].sum()/max(1,secondary.sum())),extinct=int(not np.any((state==1)|(state==2))))
    result=dict(config=asdict(c),metrics=metrics,history=history)
    if snapshots: result.update(frames=frames,edges=edges.tolist(),degree=degree.tolist(),parents=parent.tolist(),layers=layer.tolist(),secondary=secondary.tolist())
    return result
