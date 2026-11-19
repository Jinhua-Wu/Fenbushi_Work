from model import Config,simulate
import json
from pathlib import Path

def main():
    a=simulate(Config(seed=100),True); b=simulate(Config(seed=100),True)
    assert a==b,'reproducibility'
    z=simulate(Config(beta=0),True)
    assert z['metrics']['cases']==5,'zero transmission'
    for r in [a,z,simulate(Config(mask=1,isolation=1,triage=1),True)]:
        n=r['config']['n']; hist=r['history']
        assert all(h['S']+h['E']+h['I']+h['R']==n for h in hist)
        assert all(hist[t]['S']>=hist[t+1]['S'] for t in range(len(hist)-1))
        assert sum(p>=0 for p in r['parents'])==r['metrics']['cases']-5
        assert sum(r['secondary'])==r['metrics']['cases']-5
        assert all(r['parents'][i]!=i for i in range(n))
        for t in range(1,len(r['frames'])):
            assert all((old==new or (old,new) in [(0,1),(1,2),(2,3)]) for old,new in zip(r['frames'][t-1],r['frames'][t]))
    saved=json.loads((Path(__file__).parent/'results/demos.json').read_text())
    assert saved['baseline']==a,'demo parity'
    print('PASS deterministic replay; zero transmission; conservation; monotone susceptibility; parent accounting; state transition; demo parity')

if __name__=='__main__':main()
