import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/ECT002/q256-pr101-512-source-fid-backfill-v1')
SRC=str(ROOT)+'/'
DST='root@px-cloud1.matpool.com:/root/q256-pr101-512-source-fid-backfill-v1/'

def run(lane, files, label, destination):
    f=ROOT/(label+'.files'); f.write_text('\n'.join(files)+'\n')
    sock=ROOT/('a100-control.sock' if lane==0 else 'a100-lane%d.sock'%lane)
    command=['rsync','-a','--partial','--append-verify','--no-owner','--no-group',
             '--stats','--files-from='+str(f),'-e','ssh -o BatchMode=yes -S %s -p 27815'%sock,SRC,DST+destination]
    receipt={'label':label,'lane':lane,'command':command,'started_epoch':time.time(),
             'file_list_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'file_count':len(files)}
    with (ROOT/'logs'/(label+'.log')).open('xb') as log:
        rc=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    receipt.update(ended_epoch=time.time(),exit_code=rc,status='PASS' if rc==0 else 'FAILED')
    (ROOT/'receipts'/(label+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(label,receipt['status'],flush=True)
    return rc

def worker(lane, rows):
    codes=[]
    for i,r in rows:
        rel='source_snapshots/job%03d'%i
        while not (ROOT/rel/'export_receipt.json').exists(): time.sleep(2)
        codes.append(run(lane,[rel+'/network-snapshot.pkl',rel+'/export_receipt.json'],'ema_job%03d'%i,''))
    return int(any(codes))

def main():
    inv=json.loads((ROOT/'source_inventory/source_inventory.json').read_text())
    rows=[r for r in inv['rows'] if r['included_in_paired_cohort']]
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        fs=[pool.submit(worker,lane,list(enumerate(rows))[lane::32]) for lane in range(32)]
        codes=[f.result() for f in fs]
    out={'status':'PASS' if not any(codes) else 'FAILED','lane_exit_codes':codes,
         'started_epoch':start,'ended_epoch':time.time(),'n_checkpoints':len(rows),
         'note':'Transport completion only; independent destination SHA verification remains mandatory.'}
    (ROOT/'receipts/ema_transfer_transport.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)
    return int(any(codes))

if __name__=='__main__': raise SystemExit(main())
