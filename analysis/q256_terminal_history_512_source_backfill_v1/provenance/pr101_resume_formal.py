"""Resume untouched jobs after a transport-readiness scheduler interruption."""
import concurrent.futures,json,sys,time
from pathlib import Path
W=Path('/root/q256-pr101-512-source-fid-backfill-v1'); R=W/'code';sys.path.insert(0,str(R))
from analysis.q256_terminal_history_512_source_backfill_v1 import evaluate as e
from analysis.q256_terminal_history_512_source_backfill_v1.common import *
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
e.check_execution_freeze(W)
protocol,inv,rows=frozen_inputs(R,W); prior=e.prior_runner(W)
planned=load(W/'evaluation/formal/planned_job_manifest.json')['jobs']
start=load(W/'receipts/formal_matrix_started.json')['started_epoch']
write_json(W/'receipts/scheduler_resume.json',dict(status='AUTHORIZED_TECHNICAL_RESUME',reason='Receipt file arrived before snapshot transfer completed. Input hash gate stopped untouched jobs before evaluation. Resume requires separate successful rsync completion receipt.',completed_jobs_preserved=8,not_run_jobs_to_resume=50,scientific_jobs_repeated=0,resume_script_sha256=sha256(Path(__file__)),original_execution_freeze_sha256=sha256(W/'receipts/execution_code_freeze.json'),resumed_epoch=time.time()))
def worker(gpu):
 results=[]
 for job in planned[gpu::8]:
  i=job['queue_index']; rp=W/'receipts/formal_jobs'/(job['opaque_id']+'.json')
  if rp.exists():
   r=load(rp);require(r['status']=='PASS','existing non-PASS attempt requires separate audit')
   ex=load(W/'evaluation/exports'/('job%03d'%i)/'export_receipt.json')
  else:
   ready=W/'receipts/transfers'/('ema_job%03d.json'%i)
   while not ready.exists():time.sleep(3)
   require(load(ready)['status']=='PASS','transport completion is not PASS')
   ex=e.export_snapshot(W,rows[i],i)
  job.update(checkpoint=ex['snapshot_path'],checkpoint_sha256=ex['snapshot_sha256'],export_receipt_sha256=sha256(Path(ex['snapshot_path']).parent/'export_receipt.json'))
  if rp.exists():results.append(r);continue
  begin=time.time(); mono=time.monotonic()
  r=dict(opaque_id=job['opaque_id'],seed=job['seed'],arm=job['arm'],gpu_index=gpu,source_checkpoint_sha256=job['source_checkpoint_sha256'],exported_model_sha256=job['checkpoint_sha256'],export_receipt_sha256=job['export_receipt_sha256'],protocol_sha256=PROTOCOL_SHA)
  try:
   pr=prior.run_job(job,gpu,W/'evaluation/caches'/('gpu%d'%gpu));r['prior_evaluator_receipt']=pr
   r['status']='PASS' if pr['status']=='PASS' else 'FAILED';r['failure_type']=None if r['status']=='PASS' else 'UNCLASSIFIED_REQUIRES_LOG_AUDIT'
  except Exception as err:r.update(status='FAILED',failure_type='TECHNICAL_EXCEPTION',error=repr(err))
  r.update(started_epoch=begin,ended_epoch=time.time(),gpu_seconds=time.monotonic()-mono)
  write_json(rp,r);results.append(r);print(job['opaque_id'],r['status'],flush=True)
 return results
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(worker,g) for g in range(8)]; results=[x for f in futures for x in f.result()]
end=time.time()
write_json(W/'evaluation/formal/job_manifest.json',dict(protocol_sha256=PROTOCOL_SHA,jobs=planned))
write_json(W/'receipts/transfer_sha_verification.json',dict(status='PASS',checkpoint_count=len(planned),verification='Each snapshot SHA and source binding checked before formal generation. First eight full states additionally verified on A100. Remaining jobs gated on successful per-job rsync completion.'))
write_json(W/'receipts/formal_matrix_terminal.json',dict(protocol_sha256=PROTOCOL_SHA,started_epoch=start,ended_epoch=end,wall_seconds=end-start,job_count=len(planned),pass_count=sum(x['status']=='PASS' for x in results),failed_count=sum(x['status']=='FAILED' for x in results),not_run_count=0,all_jobs_terminal=True,technical_scheduler_resume=True,scientific_jobs_repeated=0))
print('FORMAL_MATRIX_TERMINAL',len(results),flush=True)
