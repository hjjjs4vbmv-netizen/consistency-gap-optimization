"""Recompute all published statistics without GPU work or network access."""
import argparse,json,math
from pathlib import Path
from ..analyze import summarize
ROOT=Path(__file__).parent

def compare(a,b,path='$'):
    if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):
        if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):raise AssertionError((path,a,b))
    elif isinstance(a,dict) and isinstance(b,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
    else:assert a==b,(path,a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
    actual=json.loads(json.dumps(summarize(json.loads((ROOT/'quality_slots_320.json').read_text()))))
    expected=json.loads((ROOT/'statistics.json').read_text());compare(actual,expected)
    assert actual['status']=='COMPLETE' and actual['n']==14
    if args.output:args.output.write_text(json.dumps(actual,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','n':actual['n'],'exact_json_match':actual==expected,'numeric_tolerance':1e-12}))
if __name__=='__main__':main()
