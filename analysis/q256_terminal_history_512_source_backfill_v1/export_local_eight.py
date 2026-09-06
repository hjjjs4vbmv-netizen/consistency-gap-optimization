"""CPU-only retained-state EMA export; never mutates or rehashes source states."""
import concurrent.futures, copy, hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
sys.modules['numpy._core']=np.core
sys.modules['numpy._core.multiarray']=np.core.multiarray
from training import reproducibility as rep
ROOT=Path('/root/q256-pr101-512-source-fid-backfill-v1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
def export(item):
 i,r=item; torch.set_num_threads(2)
 out=ROOT/'source_snapshots_local'/('job%03d'%i); out.mkdir(parents=True,exist_ok=False)
 pstate=ROOT/'source_checkpoints/training'/('seed%d'%r['seed'])/('prefix_'+r['arm'])/'training-state-kimg000512.pt'
 assert sha(pstate)==r['checkpoint_sha256']
 s=torch.load(pstate,map_location='cpu',weights_only=False)
 assert s['cur_nimg']==512000 and s['attempted_iteration']==4000
 assert s['trajectory_config']['seed']==r['seed'] and s['factorial']==r['factorial_identity']
 ema=copy.deepcopy(s['ema']).eval().requires_grad_(False)
 h=rep.module_state_sha256(ema)
 assert h==r['internal_state_sha256']['ema']
 assert all(bool(torch.isfinite(x).all()) for x in ema.state_dict().values())
 p=out/'network-snapshot.pkl'
 rep.atomic_pickle_dump(dict(ema=ema,loss_fn=None,augment_pipe=None,dataset_kwargs=dict(s['trajectory_config']['dataset_kwargs'])),p,overwrite=False)
 rec=dict(status='PASS',seed=r['seed'],arm=r['arm'],source_state_path=r['checkpoint_path'],source_state_sha256=r['checkpoint_sha256'],ema_canonical_sha256=h,snapshot_sha256=sha(p),cur_nimg=512000,attempted_iteration=4000,source_hash_verification='Previously completed source inventory plus received full state SHA verification',exported_on='A100 from already received full state; full-state SHA checked',exported_epoch=time.time(),exporter_sha256=sha(Path(__file__)))
 (out/'export_receipt.json').write_text(json.dumps(rec,sort_keys=True,indent=2)+'\n'); p.chmod(0o444)
 print(i,r['seed'],r['arm'],'EXPORTED',flush=True)
 return rec
if __name__=='__main__':
 rows=[r for r in json.loads((ROOT/'source_inventory/source_inventory.json').read_text())['rows'] if r['included_in_paired_cohort']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool: records=list(pool.map(export,enumerate(rows[:8])))
 (ROOT/'receipts/local_eight_export_complete.json').write_text(json.dumps(dict(status='PASS',count=len(records),records=records),indent=2)+'\n')
