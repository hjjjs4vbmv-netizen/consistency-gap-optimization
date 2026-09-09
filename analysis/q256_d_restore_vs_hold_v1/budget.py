"""Disjoint node allocations with explicit process leases; metrics never enter this ledger."""
from __future__ import annotations
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from scripts.run_m1_training_slot import write_json as write


@contextlib.contextmanager
def locked(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX); yield


def initialize(path,allocation,node):
    nodes=allocation['nodes']; cap=min(80.0,allocation['remaining_quota_gpuh'])
    if sum(n['cap_gpuh'] for n in nodes)+allocation.get('engineering_cap_gpuh',0)>cap: raise RuntimeError('allocations exceed global cap')
    seeds=[s for n in nodes for s in n['seeds']]
    if sorted(seeds)!=list(range(50,66)): raise RuntimeError('allocation must cover exactly 16 unique seeds')
    row=next(n for n in nodes if n['node']==node)
    value={'protocol':'q256_d_restore_vs_hold_v1','cap_gpuh':row['cap_gpuh'],'scoped_seeds':row['seeds'],
           'allocation_id':allocation['id'],'processes':{},'status':'READY'}
    if Path(path).exists():
        old=json.loads(Path(path).read_text())
        if any(old[k]!=value[k] for k in ('protocol','cap_gpuh','scoped_seeds','allocation_id')):
            raise RuntimeError('existing allocation changed')
    else: write(Path(path),value)


def reserve(path,key,max_gpuh,seed=None):
    with locked(str(path)+'.lock'):
        data=json.loads(Path(path).read_text())
        if seed is not None and seed not in data['scoped_seeds']: raise RuntimeError('seed outside node allocation')
        old=data['processes'].get(key,{})
        if old.get('status')=='RUNNING': raise RuntimeError('active lease must be reconciled before retry')
        if old.get('status') in {'PASS','SCIENTIFIC_FAILURE','NO_ENDPOINT'}: return None
        spent=sum(a['process_gpuh'] for r in data['processes'].values() for a in r.get('attempts',[]))
        held=sum(r['max_gpuh'] for r in data['processes'].values() if r.get('status')=='RUNNING')
        if spent+held+max_gpuh>data['cap_gpuh']+1e-12:
            data['status']='INCOMPLETE_BUDGET'; write(Path(path),data); return None
        data['processes'][key]={**old,'status':'RUNNING','max_gpuh':max_gpuh,'started_unix':time.time(),'pid':os.getpid()}
        write(Path(path),data); return max_gpuh


def finish(path,key,record):
    with locked(str(path)+'.lock'):
        data=json.loads(Path(path).read_text()); row=data['processes'][key]
        row.setdefault('attempts',[]).append({k:record[k] for k in ('status','process_gpuh','exit_code')})
        row['status']=record['status']; write(Path(path),data)


def run(cmd,*,cwd,env,log,ledger,key,max_gpuh,seed=None):
    """Hard timeout applies to entire child process group, leaving durable checkpoints."""
    amount=reserve(ledger,key,max_gpuh,seed)
    if amount is None: return {'status':'BUDGET_PAUSED','process_gpuh':0.0,'exit_code':None}
    started=time.monotonic(); timed_out=False
    with Path(log).open('xb') as output:
        process=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        # Reserve shutdown margin inside the lease, never after exhausting the global budget.
        deadline=started+amount*3600-20
        while process.poll() is None:
            if time.monotonic()>=deadline:
                timed_out=True; os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                break
            time.sleep(min(2,max(.05,deadline-time.monotonic())))
    return {'status':'BUDGET_PAUSED' if timed_out else 'EXITED','exit_code':process.returncode,
            'process_gpuh':(time.monotonic()-started)/3600,'pid':process.pid}
