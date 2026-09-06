"""Verify, export, smoke, evaluate, and seal retained PR101 states. Never trains."""
import argparse
import concurrent.futures
import copy
import hashlib
import importlib
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from analysis.q256_terminal_history_512_source_backfill_v1.common import (
    EXPERIMENT_ID, PROTOCOL_SHA, check_hash, frozen_inputs, load, require,
    sha256, source_path, write_json)

def prior_runner(work):
    prior=importlib.import_module('analysis.q256_terminal_history_n30_matpool_v1.firstwave_eval_pool')
    prior.WORK_ROOT=work/'evaluation/formal'
    prior.EVALUATOR=work/'evaluator'
    prior.TRAIN_RUNTIME_BASE=work/'runtime-base'
    prior.TRAIN_RUNTIME_ENV=work/'runtime-env'
    prior.DATASET=work/'assets/dataset/cifar10-32x32.zip'
    prior.PROTOCOL_SHA256=PROTOCOL_SHA
    return prior

def script_manifest():
    files=list((REPO/'analysis'/EXPERIMENT_ID).glob('*.py'))
    files += [REPO/'analysis/q256_terminal_history_n30_matpool_v1/firstwave_eval_pool.py',
              REPO/'scripts/export_q256_replay_milestone_snapshots.py',
              REPO/'training/reproducibility.py', REPO/'training/schedule_switch.py']
    return {str(p.relative_to(REPO)):sha256(p) for p in sorted(files)}

def verify_transfer(work):
    protocol,inv,rows=frozen_inputs(REPO,work)
    checked=[]
    for i,r in enumerate(rows):
        p=work/'source_snapshots'/('job%03d'%i)/'network-snapshot.pkl'
        ex=load(p.parent/'export_receipt.json')
        require(ex['status']=='PASS' and ex['source_state_sha256']==r['checkpoint_sha256'] and
                ex['ema_canonical_sha256']==r['internal_state_sha256']['ema'],'source-export binding mismatch')
        require((ex['seed'],ex['arm'],ex['cur_nimg'],ex['attempted_iteration'])==
                (r['seed'],r['arm'],512000,4000),'export identity mismatch')
        check_hash(p,ex['snapshot_sha256'])
        p.chmod(0o444)
        checked.append(dict(seed=r['seed'],arm=r['arm'],path=str(p),sha256=ex['snapshot_sha256'],
                            source_state_sha256=r['checkpoint_sha256']))
    for rel,h in protocol['assets'].items(): check_hash(work/rel,h)
    check_hash(REPO/protocol['joint_analysis']['endpoint_file'],protocol['joint_analysis']['endpoint_file_sha256'])
    receipt=dict(status='PASS',protocol_sha256=PROTOCOL_SHA,checkpoint_count=len(checked),
                 checked_checkpoints=checked,checked_assets=protocol['assets'],
                 verified_at_epoch=time.time(),no_new_training=True)
    write_json(work/'receipts/transfer_sha_verification.json',receipt)
    print('TRANSFER_SHA_PASS',len(checked),flush=True)

def verify_environment(work):
    import platform
    import numpy as np
    import scipy
    import torch
    protocol,_,rows=frozen_inputs(REPO,work)
    for rel,h in protocol['assets'].items(): check_hash(work/rel,h)
    prior=prior_runner(work)
    check_hash(prior.EVALUATOR/'ct_eval.py',protocol['evaluator']['ct_eval_py_sha256'])
    prior.DATASET.parent.mkdir(parents=True,exist_ok=True)
    if not prior.DATASET.exists(): prior.DATASET.symlink_to(work/'assets/cifar10-32x32-eval.zip')
    check_hash(prior.DATASET,protocol['dataset_binding']['eval_zip_sha256'])
    probe=dict(python=platform.python_version(),torch=torch.__version__,torch_cuda=torch.version.cuda,
               numpy=np.__version__,scipy=scipy.__version__)
    require(probe==protocol['runtime']['target'],'runtime target mismatch: '+str(probe))
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total',
                                     '--format=csv,noheader'],text=True).splitlines()
    require(len(devices)==8 and all('A100' in d for d in devices),'8 A100 hardware contract failed')
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                         '--format=csv,noheader'],text=True).strip(),'GPUs are occupied')
    # Cache filenames encode the dataset path. Rebind names for the new path,
    # preserving the exact frozen cache bytes; never regenerate real features.
    cache=work/'assets/cache-template'
    for rel,h in protocol['assets'].items():
        if 'cache-template' in rel: check_hash(work/rel,h)
    aliases=[]
    dataset_kwargs=dict(class_name='training.dataset.ImageFolderDataset',path=str(prior.DATASET),
        use_labels=False,xflip=False,cache=True,resolution=32,max_size=None)
    for p in sorted((cache/'gan-metrics').glob('*.pkl')):
        with p.open('rb') as f: stats=pickle.load(f)
        require(stats['num_items']==50000 and stats['num_features']==2048,'frozen real-feature shape mismatch')
        require(bool(stats['capture_all']) != bool(stats['capture_mean_cov']),'ambiguous real-cache capture mode')
        kwargs={'capture_all':True} if stats['capture_all'] else {'capture_mean_cov':True}
        args=dict(dataset_kwargs=dataset_kwargs,
            detector_url='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',
            detector_kwargs=dict(return_features=True),stats_kwargs=kwargs)
        tag='cifar10-32x32-inception-2015-12-05-'+hashlib.md5(repr(sorted(args.items())).encode('utf-8')).hexdigest()+'.pkl'
        aliases.append(dict(source_name=p.name,target_name=tag,sha256=sha256(p),capture_mode=kwargs))
    require(len(aliases)==2,'expected two frozen real-feature caches')
    for lane in range(8):
        dst=work/'evaluation/caches'/('gpu%d'%lane)
        require(not dst.exists(),'cache already provisioned')
        shutil.copytree(cache,dst,copy_function=os.link)
        for a in aliases:
            target=dst/'gan-metrics'/a['target_name']
            if not target.exists(): os.link(cache/'gan-metrics'/a['source_name'],target)
            check_hash(target,a['sha256'])
    write_json(work/'receipts/environment_parity.json',dict(status='PASS',runtime_probe=probe,
        gpu_inventory=devices,evaluator_ct_eval_sha256=sha256(prior.EVALUATOR/'ct_eval.py'),
        dataset_sha256=sha256(prior.DATASET),protocol_sha256=PROTOCOL_SHA,
        runtime_provisioning='Exact pinned versions; see runtime setup logs and pip freeze',
        prior_evaluator_reused=True,real_feature_cache_aliases=aliases,formal_job_count=len(rows),
        cpu_thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        evaluator_source_file_sha256={str(p.relative_to(prior.EVALUATOR)):sha256(p) for p in sorted(prior.EVALUATOR.rglob('*.py'))},
        pip_freeze_sha256=sha256(work/'receipts/runtime-pip-freeze.txt')))
    print('ENVIRONMENT_PARITY_PASS',flush=True)

def export_snapshot(work,row,index):
    original=work/('source_snapshots_local' if index<8 else 'source_snapshots')/('job%03d'%index)
    while not (original/'export_receipt.json').exists(): time.sleep(3)
    r=load(original/'export_receipt.json')
    require(r['status']=='PASS' and r['source_state_sha256']==row['checkpoint_sha256'] and
            r['ema_canonical_sha256']==row['internal_state_sha256']['ema'],'source export mismatch')
    require((r['seed'],r['arm'],r['cur_nimg'],r['attempted_iteration'])==
            (row['seed'],row['arm'],512000,4000),'export metadata mismatch')
    check_hash(original/'network-snapshot.pkl',r['snapshot_sha256'])
    out=work/'evaluation/exports'/('job%03d'%index)
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'network-snapshot.pkl'
    os.link(original/'network-snapshot.pkl',snapshot)
    snapshot.chmod(0o444)
    r.update(snapshot_path=str(snapshot),protocol_sha256=PROTOCOL_SHA,
             destination_verification='Single pre-job snapshot SHA verified',verified_at_epoch=time.time())
    write_json(out/'export_receipt.json',r)
    return r

def waive_smoke_and_freeze(work):
    require(load(work/'receipts/environment_parity.json')['status']=='PASS','environment gate missing')
    a=REPO/'analysis'/EXPERIMENT_ID/'budget_amendment.json'
    require(load(a)['smoke_status']=='NOT_RUN_USER_WAIVED','explicit waiver missing')
    write_json(work/'receipts/smoke_attempt00.json',dict(status='NOT_RUN_USER_WAIVED',gpu_seconds=0,
               user_authorized=True,amendment_sha256=sha256(a)))
    write_json(work/'receipts/execution_code_freeze.json',dict(protocol_sha256=PROTOCOL_SHA,
        amendment_sha256=sha256(a),pipeline_amendment_sha256=sha256(a.with_name('pipeline_amendment.json')),file_sha256=script_manifest(),frozen_at_epoch=time.time(),
        environment_parity_sha256=sha256(work/'receipts/environment_parity.json')))
    print('SMOKE_WAIVED_CODE_FROZEN',flush=True)

def smoke(work):
    import numpy as np
    _,_,rows=frozen_inputs(REPO,work)
    require(load(work/'receipts/environment_parity.json')['status']=='PASS','environment gate missing')
    prior=prior_runner(work)
    export=export_snapshot(work,rows[0],0)
    out=work/'evaluation/smoke/attempt00'
    require(not out.exists(),'smoke attempt already exists'); out.mkdir(parents=True)
    script=Path(__file__).with_name('smoke_entry.py')
    command=[str(prior.TRAIN_RUNTIME_ENV/'bin/python'),'-m','torch.distributed.run','--standalone',
       '--nproc_per_node=1',str(script),'--resume',export['snapshot_path'],'--outdir',str(out),'--nosubdir',
       '--data',str(prior.DATASET),'--cond=False','--arch=ddpmpp','--precond=ct','--dropout=0.2',
       '--augment=0','--xflip=False','--fp16=False','--cache=True','--workers=1','--eval-batch=512',
       '--metric-generator-batch=128','--nfe=1','--metrics=source_smoke_kid64,source_smoke_fid64',
       '--metric-repeats=1','--sample-seeds=0-63','--seed=20260730','--retain-generated-artifacts']
    env=prior.runtime_env(0,work/'evaluation/caches/gpu0',53150)
    env['Q256_FROZEN_EVALUATOR']=str(prior.EVALUATOR)
    start=time.time(); monotonic_start=time.monotonic()
    with (work/'logs/smoke-attempt00.log').open('xb') as log:
        rc=subprocess.call(command,cwd=prior.EVALUATOR,env=env,stdout=log,stderr=subprocess.STDOUT)
    end=time.time()
    r=dict(status='FAILED',exit_code=rc,started_epoch=start,ended_epoch=end,gpu_seconds=time.monotonic()-monotonic_start,
           sample_count=64,source_state_sha256=export['source_state_sha256'],
           exported_model_sha256=export['snapshot_sha256'],protocol_sha256=PROTOCOL_SHA,command=command)
    try:
        require(rc==0,'smoke evaluator process failed')
        a=out/'generated-features-source_smoke_kid64-repeat00.npy'
        b=out/'generated-features-source_smoke_fid64-repeat00.npy'
        require(sha256(a)==sha256(b),'smoke FID/KID features differ')
        x=np.load(a,allow_pickle=False)
        require(x.shape==(64,2048) and x.dtype==np.float32 and np.isfinite(x).all(),'smoke feature shape/finiteness')
        opts=load(out/'training_options.json')
        require(opts['sample_seeds']==list(range(64)) and opts['seed']==20260730,'smoke seed contract mismatch')
        require(not opts['network_kwargs']['use_fp16'],'smoke not FP32')
        binding=load(out/'model-transfer-binding.json')
        require(binding['status']=='PASS' and binding['evaluation_model_canonical_sha256']==export['ema_canonical_sha256'],
                'smoke evaluator weights not bound to source EMA')
        require('Exiting...' in (out/'log.txt').read_text(),'smoke completion marker missing')
        for metric in ('source_smoke_kid64','source_smoke_fid64'):
            lines=(out/('metric-'+metric+'.jsonl')).read_text().splitlines()
            require(len(lines)==1,'smoke metric row count mismatch')
            import json
            require(math.isfinite(float(json.loads(lines[0])['results'][metric])),'smoke metric nonfinite')
        aliases=load(work/'receipts/environment_parity.json')['real_feature_cache_aliases']
        allowed={a[k] for a in aliases for k in ('source_name','target_name')}
        actual={p.name for p in (work/'evaluation/caches/gpu0/gan-metrics').glob('*.pkl')}
        require(actual==allowed,'unexpected real-feature cache generated during smoke')
        for a in aliases: check_hash(work/'evaluation/caches/gpu0/gan-metrics'/a['target_name'],a['sha256'])
        r.update(status='PASS',generated_features_sha256=sha256(a),shared_features=True,
                 job_directory_uniqueness=len({str(work/'evaluation/formal/jobs'/('job%03d_attempt00'%i)) for i in range(len(rows))})==len(rows))
    except Exception as e: r['error']=str(e)
    write_json(work/'receipts/smoke_attempt00.json',r)
    require(r['status']=='PASS','smoke failed; see retained log and receipt')
    write_json(work/'receipts/execution_code_freeze.json',dict(protocol_sha256=PROTOCOL_SHA,
        file_sha256=script_manifest(),frozen_at_epoch=time.time(),
        environment_parity_sha256=sha256(work/'receipts/environment_parity.json')))
    print('SMOKE_PASS: 64 samples, shared feature SHA, source-to-export binding',flush=True)

def check_execution_freeze(work):
    r=load(work/'receipts/execution_code_freeze.json')
    require(r['protocol_sha256']==PROTOCOL_SHA,'execution code freeze protocol mismatch')
    require(script_manifest()==r['file_sha256'],'execution source set changed after freeze')
    for rel,h in r['file_sha256'].items(): check_hash(REPO/rel,h)
    check_hash(work/'receipts/environment_parity.json',r['environment_parity_sha256'])
    env=load(work/'receipts/environment_parity.json')
    for rel,h in env['evaluator_source_file_sha256'].items(): check_hash(work/'evaluator'/rel,h)

def formal(work):
    import torch
    torch.set_num_threads(2)
    protocol,_,rows=frozen_inputs(REPO,work)
    require(load(work/'receipts/smoke_attempt00.json')['status']=='NOT_RUN_USER_WAIVED','user smoke waiver missing')
    check_execution_freeze(work)
    prior=prior_runner(work)
    for d in ('jobs','receipts','logs'): (prior.WORK_ROOT/d).mkdir(parents=True,exist_ok=True)
    require(not (work/'receipts/formal_matrix_started.json').exists(),'formal matrix already started')
    jobs=[dict(seed=row['seed'],arm=row['arm'],queue_index=i,opaque_id='job%03d_attempt00'%i,
          source_checkpoint_sha256=row['checkpoint_sha256'],gpu=i%8) for i,row in enumerate(rows)]
    write_json(work/'evaluation/formal/planned_job_manifest.json',dict(protocol_sha256=PROTOCOL_SHA,jobs=jobs))
    started=time.time(); matrix_monotonic_start=time.monotonic()
    write_json(work/'receipts/formal_matrix_started.json',dict(started_epoch=started,job_count=len(jobs),
               protocol_sha256=PROTOCOL_SHA,no_new_training=True))
    def worker(gpu):
        results=[]
        for job in jobs[gpu::8]:
            ex=export_snapshot(work,rows[job['queue_index']],job['queue_index'])
            job.update(checkpoint=ex['snapshot_path'],checkpoint_sha256=ex['snapshot_sha256'],
                       export_receipt_sha256=sha256(Path(ex['snapshot_path']).parent/'export_receipt.json'))
            begin=time.time(); begin_monotonic=time.monotonic()
            r=dict(opaque_id=job['opaque_id'],seed=job['seed'],arm=job['arm'],gpu_index=gpu,
                   source_checkpoint_sha256=job['source_checkpoint_sha256'],exported_model_sha256=job['checkpoint_sha256'],
                   export_receipt_sha256=job['export_receipt_sha256'],protocol_sha256=PROTOCOL_SHA)
            try:
                prior_r=prior.run_job(job,gpu,work/'evaluation/caches'/('gpu%d'%gpu))
                r['prior_evaluator_receipt']=prior_r
                r['status']='PASS' if prior_r['status']=='PASS' else 'FAILED'
                r['failure_type']=None if r['status']=='PASS' else 'UNCLASSIFIED_REQUIRES_LOG_AUDIT'
            except Exception as e:
                r.update(status='FAILED',failure_type='TECHNICAL_EXCEPTION',error=repr(e))
            r.update(started_epoch=begin,ended_epoch=time.time(),gpu_seconds=time.monotonic()-begin_monotonic)
            write_json(work/'receipts/formal_jobs'/(job['opaque_id']+'.json'),r)
            results.append(r)
            print(job['opaque_id'],r['status'],flush=True)
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,gpu) for gpu in range(8)]
        results=[r for f in futures for r in f.result()]
    ended=time.time()
    write_json(work/'evaluation/formal/job_manifest.json',dict(protocol_sha256=PROTOCOL_SHA,jobs=jobs))
    write_json(work/'receipts/transfer_sha_verification.json',dict(status='PASS',checkpoint_count=len(jobs),verification='Per-job source-export and received snapshot SHA verified before generation',source_state_SHA='Complete source inventory; first eight received full states additionally verified locally'))
    write_json(work/'receipts/formal_matrix_terminal.json',dict(protocol_sha256=PROTOCOL_SHA,
       started_epoch=started,ended_epoch=ended,wall_seconds=time.monotonic()-matrix_monotonic_start,job_count=len(jobs),
       pass_count=sum(r['status']=='PASS' for r in results),failed_count=sum(r['status']=='FAILED' for r in results),
       not_run_count=0,all_jobs_terminal=True))
    print('FORMAL_MATRIX_TERMINAL',len(results),flush=True)

def seal(work):
    import numpy as np
    protocol,_,rows=frozen_inputs(REPO,work)
    check_execution_freeze(work)
    terminal=load(work/'receipts/formal_matrix_terminal.json')
    jobs=load(work/'evaluation/formal/job_manifest.json')['jobs']
    require(terminal['all_jobs_terminal'] and len(jobs)==len(rows),'matrix not terminal')
    records=[]
    for job,row in zip(jobs,rows):
        require((job['seed'],job['arm'])==(row['seed'],row['arm']),'job-to-source identity mismatch')
        rp=work/'receipts/formal_jobs'/(job['opaque_id']+'.json'); r=load(rp)
        require(r['status'] in ('PASS','FAILED','NOT_RUN'),'nonterminal job status')
        require((r['seed'],r['arm'])==(row['seed'],row['arm']) and r['protocol_sha256']==PROTOCOL_SHA,
                'receipt source identity/protocol mismatch')
        require(r['source_checkpoint_sha256']==row['checkpoint_sha256'] and
                r['exported_model_sha256']==job['checkpoint_sha256'],'receipt checkpoint binding mismatch')
        ex_path=Path(job['checkpoint']).parent/'export_receipt.json'
        check_hash(ex_path,job['export_receipt_sha256']); ex=load(ex_path)
        require(ex['source_state_sha256']==row['checkpoint_sha256'] and
                ex['ema_canonical_sha256']==row['internal_state_sha256']['ema'],'source-export chain mismatch')
        record=dict(opaque_id=job['opaque_id'],seed=row['seed'],arm=row['arm'],status=r['status'],
                    evaluation_receipt_sha256=sha256(rp),source_checkpoint_sha256=row['checkpoint_sha256'],
                    exported_model_sha256=job['checkpoint_sha256'],source_to_export_binding='PASS')
        if r['status']=='PASS':
            prior_r=r['prior_evaluator_receipt']; out=work/'evaluation/formal/jobs'/job['opaque_id']
            for rel,info in prior_r['artifact_hashes'].items():
                if not rel.endswith(('.npy','.pkl')): check_hash(out/rel,info['sha256'])
            feature=out/'generated-features-kid50k_full-repeat00.npy'
            x=np.load(feature,mmap_mode='r',allow_pickle=False)
            require(x.shape==(50000,2048) and x.dtype==np.float32,'formal feature count/type mismatch')
            require(all(np.isfinite(x[i:i+1024]).all() for i in range(0,len(x),1024)),'formal features nonfinite')
            feature_sha=prior_r['artifact_hashes'][feature.name]['sha256']
            require(feature_sha==prior_r['artifact_hashes']['generated-features-fid50k_full-repeat00.npy']['sha256'],'formal shared-feature mismatch')
            options=load(out/'training_options.json')
            samples=np.load(out/'generated-samples.npy',mmap_mode='r',allow_pickle=False)
            require(samples.shape[0]==50000,'formal generated sample count mismatch')
            require(options['sample_seeds']==list(range(50000)) and options['seed']==20260730 and
                    options['mid_t']==[] and not options['network_kwargs']['use_fp16'],'formal options mismatch')
            record.update(shared_feature_binding='PASS',generated_features_sha256=feature_sha,
                          metric_artifact_sha256=prior_r['metric_artifact_sha256'])
        else:
            failed_dir=work/'evaluation/formal/jobs'/job['opaque_id']
            record['retained_failure_artifact_sha256']={str(p.relative_to(work)):sha256(p) for p in failed_dir.rglob('*') if p.is_file()}
        launcher_log=work/'evaluation/formal/logs'/(job['opaque_id']+'.launcher.log')
        record['launcher_log_sha256']=sha256(launcher_log) if launcher_log.is_file() else None
        records.append(record)
    total=sum(load(work/'receipts/formal_jobs'/(j['opaque_id']+'.json'))['gpu_seconds'] for j in jobs)
    smoke_r=load(work/'receipts/smoke_attempt00.json')
    cost=dict(formal_gpu_seconds=total,formal_gpu_hours=total/3600,smoke_gpu_seconds=smoke_r['gpu_seconds'],
              total_a100_gpu_hours=(total+smoke_r['gpu_seconds'])/3600,formal_wall_seconds=terminal['wall_seconds'],
              formal_started_epoch=terminal['started_epoch'],formal_ended_epoch=terminal['ended_epoch'],
              accounting='Sum of elapsed seconds while each evaluation attempt held one exclusive GPU; includes generation and metric computation, excludes CPU inventory/export/transfer.')
    out=work/'results'/EXPERIMENT_ID
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'GPU_COST_REPORT.json',cost)
    seal_record=dict(status='PASS',all_formal_jobs_terminal=True,protocol_sha256=PROTOCOL_SHA,
        smoke_status='NOT_RUN_USER_WAIVED',integrity_scope='Full source inventory SHA, single received snapshot SHA, per-job evaluator artifact hashes; no repeated bulk binary SHA at seal',source_valid_pairs=protocol['cohort']['n_source_pairs'],formal_job_count=len(jobs),jobs=records,
        sealed_at_epoch=time.time(),no_new_training=True,scalar_analysis_not_yet_run=True,
        execution_code_freeze_sha256=sha256(work/'receipts/execution_code_freeze.json'),
        formal_matrix_terminal_sha256=sha256(work/'receipts/formal_matrix_terminal.json'))
    write_json(out/'integrity_verification.json',seal_record)
    print('INTEGRITY_SEAL_PASS',len(records),flush=True)

def main():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['verify-transfer','environment','smoke','waive-smoke','formal','seal'])
    p.add_argument('--work-root',type=Path,required=True); a=p.parse_args()
    {'verify-transfer':verify_transfer,'environment':verify_environment,'smoke':smoke,'waive-smoke':waive_smoke_and_freeze,'formal':formal,'seal':seal}[a.stage](a.work_root)

if __name__=='__main__': main()
