"""Pure integrity and scalar-analysis contracts for the source backfill."""
import csv
import hashlib
import json
import math
from pathlib import Path

EXPERIMENT_ID = 'q256_terminal_history_512_source_backfill_v1'
PR101_COMMIT = '890a85a8ef4d9effb48f653111a70b5f15b249de'
PROTOCOL_SHA = '87b51a7383c67772cdbc1f96ef1bda3766af233995c41f2b36ce57ba1abcad72'
TERMINAL = {'PASS', 'FAILED', 'NOT_RUN'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def require(condition, message):
    if not condition: raise RuntimeError(message)

def load(path):
    return json.loads(Path(path).read_text())

def write_json(path, value, exclusive=True):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def check_hash(path, expected):
    actual=sha256(path)
    require(actual==expected, 'SHA mismatch: '+str(path))
    return actual

def frozen_inputs(repo, work):
    p=repo/'analysis'/EXPERIMENT_ID/'protocol.json'
    check_hash(p,PROTOCOL_SHA)
    protocol=load(p)
    invp=work/'source_inventory/source_inventory.json'
    check_hash(invp,protocol['cohort']['source_inventory_json_sha256'])
    check_hash(work/'source_inventory/source_inventory.csv',protocol['cohort']['source_inventory_csv_sha256'])
    inv=load(invp)
    rows=[r for r in inv['rows'] if r['included_in_paired_cohort']]
    require(len(rows)==protocol['cohort']['formal_jobs'],'formal job count mismatch')
    keys=[(r['seed'],r['arm']) for r in rows]
    expected=[(s,a) for s in protocol['cohort']['source_valid_pairs'] for a in 'AB']
    require(keys==expected and len(set(keys))==len(keys),'source cohort/order mismatch')
    require(all(r['source_valid'] and r['chain_status']=='PASS' for r in rows),'source inventory invalid')
    return protocol,inv,rows

def source_path(work, row):
    return work/'source_checkpoints'/'training'/('seed%d'%row['seed'])/('prefix_'+row['arm'])/'training-state-kimg000512.pt'

def check_positive_fid(value):
    value=float(value)
    require(math.isfinite(value) and value>0,'FID must be finite and positive')
    return value

def source_pairs(decoded):
    by_seed={}
    for r in decoded:
        seed=int(r['seed']); arm=r['arm']
        require(arm in 'AB' and len(arm)==1,'unknown source arm')
        require(arm not in by_seed.setdefault(seed,{}),'duplicate source arm')
        by_seed[seed][arm]=r
    pairs=[]
    for seed,arms in sorted(by_seed.items()):
        if set(arms)!=set('AB') or any(r['evaluation_status']!='PASS' for r in arms.values()): continue
        a=check_positive_fid(arms['A']['fid50k']); b=check_positive_fid(arms['B']['fid50k'])
        ka=float(arms['A']['kid50k']); kb=float(arms['B']['kid50k'])
        require(math.isfinite(ka) and math.isfinite(kb),'nonfinite KID in PASS row')
        q=math.log(b)-math.log(a)
        pairs.append(dict(seed=seed,fid_A_512=a,fid_B_512=b,kid_A_512=ka,kid_B_512=kb,
                          Q_logfid_B_minus_A=q,B_worse_at_512=q>0))
    return pairs

def join_future(pairs,endpoints):
    by_seed={}
    for r in endpoints:
        seed=int(r['seed']); require(seed not in by_seed,'duplicate endpoint seed')
        aa=check_positive_fid(r['aa_fid50k']); ba=check_positive_fid(r['ba_fid50k'])
        h=math.log(ba)-math.log(aa)
        require(math.isclose(h,float(r['log_fid_contrast_ba_minus_aa']),rel_tol=1e-12,abs_tol=1e-12),
                'frozen H_A cannot be reproduced from endpoint scalars')
        by_seed[seed]=(aa,ba,h)
    out=[]
    for r in pairs:
        seed=int(r['seed'])
        if seed not in by_seed: continue
        aa,ba,h=by_seed[seed]; q=float(r['Q_logfid_B_minus_A'])
        out.append(dict(seed=seed,fid_A_512=r['fid_A_512'],fid_B_512=r['fid_B_512'],Q=q,
                        fid_AA_1024=aa,fid_BA_1024=ba,H_A=h,source_B_worse=q>0,
                        future_B_history_better=h<0,delayed_reversal=q>0 and h<0))
    return out

def quadrants(rows):
    counts=dict(reversal=0,bad_to_bad=0,good_to_good=0,reverse_loss=0,on_axis=0)
    for r in rows:
        q=float(r['Q']); h=float(r['H_A'])
        require(math.isfinite(q) and math.isfinite(h),'nonfinite quadrant value')
        if q==0 or h==0: key='on_axis'
        elif q>0: key='reversal' if h<0 else 'bad_to_bad'
        else: key='good_to_good' if h<0 else 'reverse_loss'
        counts[key]+=1
    return counts
