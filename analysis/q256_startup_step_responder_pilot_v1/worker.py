"""Fixed seed/GPU worker. Sequential arms, verified recovery, bounded dispatch."""
from __future__ import annotations
import argparse,contextlib,csv,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
import torch
from . import protocol as p
from training import startup_quality as q
from scripts.run_m1_training_slot import runtime_environment,scientific_failure,truncate_attempt_csv

TERMINAL=set(p.SPEC['terminal_statuses'])

@contextlib.contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def gpu_lock(config,logical):
    expected=config['gpu_mapping'][str(logical)]
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
    found=dict((int(r.split(',')[0]),r.split(',')[1].strip()) for r in rows)
    if found.get(expected['local_gpu'])!=expected['uuid']:raise RuntimeError('GPU UUID differs from allocation')
    return expected


def check_idle(uuid):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(r.split(',')[0].strip()==uuid for r in rows.splitlines()):raise RuntimeError('GPU already has a compute process')


def projection(ledger):
    now=time.time();cost=ledger['engineering_used_gpuh']+ledger.get('evaluation_used_gpuh',0);remaining=0
    for job in ledger.get('evaluation_running',{}).values():
        elapsed=(now-job['started_wall'])/3600;cost+=elapsed;remaining+=max(0,job['estimate_gpuh']-elapsed)
    for job in ledger['jobs'].values():
        if job['status']=='RUNNING':
            elapsed=job.get('previous_gpuh',0)+(now-job['started_wall'])/3600
            cost+=elapsed;remaining+=max(0,job['estimate_gpuh']-elapsed)
        else:
            cost+=job.get('actual_gpuh',0)
            if job['status']=='PENDING':remaining+=job['estimate_gpuh']
    reserves=ledger['engineering_remaining_gpuh']+ledger['evaluation_remaining_gpuh']
    # Other host's entire allocation remains reserved, including its live and unstarted work.
    exempt=ledger.get('budget_exempt',False)
    other=0 if exempt or ledger.get('cap_scope')=='paid_matpool_only' else 90-ledger['allocation_cap_gpuh']
    paid_forecast=0 if exempt else cost+remaining+reserves+other
    return dict(local_completed_and_live_gpuh=cost,local_remaining_training_gpuh=remaining,
                local_remaining_evaluation_and_engineering_gpuh=reserves,
                other_host_reserved_envelope_gpuh=other,
                global_conservative_projected_gpuh=paid_forecast,
                owned_unmetered_projected_gpuh=cost+remaining+reserves if exempt else 0,
                cap_scope=ledger.get('cap_scope','legacy_global'))


def budget_update(config,key,action,**fields):
    path=Path(config['budget_ledger'])
    with locked(str(path)+'.lock'):
        ledger=json.loads(path.read_text());job=ledger['jobs'][key]
        if action=='start':
            view=projection(ledger)
            if not ledger.get('budget_exempt',False) and (ledger.get('status')=='INCOMPLETE_BUDGET' or view['global_conservative_projected_gpuh']>90):
                ledger['status']='INCOMPLETE_BUDGET';p.write(path,ledger);return False
            if job['status'] not in ('PENDING','TECHNICAL_FAILURE'):raise RuntimeError('budget slot already dispatched/terminal')
            job.update(status='RUNNING',started_wall=time.time(),previous_gpuh=job.get('actual_gpuh',0),**fields)
        elif action=='finish':job.update(**fields)
        elif action=='tick':
            job.update(**fields)
            progress=job.get('current_attempt',job.get('start_attempt',0))-job.get('start_attempt',0)
            if progress>=200:
                # Forecast from this process's wall time, never inherited checkpoint elapsed.
                elapsed=(time.time()-job['started_wall'])/3600
                observed_full=elapsed/progress*8000
                job['estimate_gpuh']=max(job['estimate_gpuh'],job.get('previous_gpuh',0)+elapsed+(8000-job['current_attempt'])*elapsed/progress)
                for pending in ledger['jobs'].values():
                    if pending['status']=='PENDING':pending['estimate_gpuh']=max(pending['estimate_gpuh'],observed_full)
        ledger['projection']=projection(ledger)
        if not ledger.get('budget_exempt',False) and ledger['projection']['global_conservative_projected_gpuh']>90:ledger['status']='INCOMPLETE_BUDGET'
        p.write(path,ledger)
        return True


def verify_freeze(config):
    gate=json.loads(Path(config['freeze_receipt']).read_text())
    if gate.get('status')!='FROZEN' or gate.get('preflight_status')!='PASS':raise RuntimeError('formal freeze/preflight missing')
    for name,digest in gate['source_hashes'].items():
        if p.digest(p.ROOT/name)!=digest:raise RuntimeError('frozen implementation file changed: '+name)
    if p.digest(config['protocol_json'])!=gate['protocol_sha256']:raise RuntimeError('protocol changed')
    if p.digest(config['old_control_bindings'])!=gate['old_control_bindings_sha256']:raise RuntimeError('old controls changed')
    if config['gpu_mapping']!=gate['gpu_mapping']:raise RuntimeError('GPU mapping changed')


def latest(run,m):
    candidates=[]
    for path in [run/'training-state-latest.pt',*run.glob('training-state-kimg*.pt')]:
        if not path.exists():continue
        try:
            seal=json.loads(Path(str(path)+'.sha256.json').read_text())
            if p.digest(path)!=seal['sha256']:raise RuntimeError('file hash differs')
            state=torch.load(path,map_location='cpu',weights_only=False);q.validate_state(state,m)
            candidates.append((state['attempted_iteration'],path));del state
        except Exception as exc:
            p.write(run/(path.name+'.recovery-rejected.json'),dict(status='REJECTED',reason=repr(exc)))
    return max(candidates,key=lambda x:x[0]) if candidates else (0,None)


def trim_for_resume(run,attempt):
    stamp=time.time_ns()
    for name in ('train_summary.csv','factorial_training_telemetry_v1.csv'):
        path=run/name
        shutil.copy2(path,run/(name+f'.before-recovery-{stamp}'));truncate_attempt_csv(path,attempt)
    path=run/'startup_quality_telemetry.jsonl';shutil.copy2(path,run/(path.name+f'.before-recovery-{stamp}'))
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    kept=[x for x in rows if x['attempted_iteration']<=attempt]
    path.write_text(''.join(json.dumps(x)+'\n' for x in kept))


def run_arm(config,seed,arm,mapping):
    run=Path(config['output_root'])/f'seed{seed}'/arm;run.mkdir(parents=True,exist_ok=True)
    mp=Path(config['manifest_root'])/f'seed{seed}-{arm}.json';m=json.loads(mp.read_text());q.validate_manifest(m)
    if m!=p.manifest(config,seed,arm,run):raise RuntimeError('manifest drift')
    status=run/'outcome.json';key=f'seed{seed}-{arm}'
    if status.exists():
        prior=json.loads(status.read_text())
        if prior['status'] in TERMINAL and prior['status']!='TECHNICAL_FAILURE':return prior
    attempt,resume=latest(run,m)
    if attempt==8000:
        result=dict(seed=seed,arm=arm,status='PASS',attempted_iteration=8000,endpoint=resume.name,checkpoint_sha256=p.digest(resume))
        p.write(status,result);return result
    if list(run.glob('process-*.log')) and resume is None:
        result=dict(seed=seed,arm=arm,status='TECHNICAL_FAILURE',reason='logs exist without a verifiable own checkpoint; manual audit required')
        p.write(status,result);return result
    if resume:trim_for_resume(run,attempt)
    check_idle(mapping['uuid'])
    if not budget_update(config,key,'start',start_attempt=attempt):
        result=dict(seed=seed,arm=arm,status='INCOMPLETE_BUDGET');p.write(status,result);return result
    cmd=p.command(config,m,mp,resume)
    log=run/f'process-{time.time_ns()}.log';env=runtime_environment(mapping['uuid'],Path(config['runtime_python']))
    started=time.monotonic();wall=time.time()
    with log.open('x') as f:
        proc=subprocess.Popen(cmd,cwd=p.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        p.write(status,dict(seed=seed,arm=arm,status='RUNNING',pid=proc.pid,started_wall=wall,command=cmd,start_attempt=attempt))
        while proc.poll() is None:
            time.sleep(15)
            progress=attempt
            telemetry=run/'startup_quality_telemetry.jsonl'
            if telemetry.exists():
                with telemetry.open('rb') as tf:
                    tf.seek(0,2);size=tf.tell();tf.seek(max(0,size-8192))
                    lines=tf.read().splitlines()
                if lines:
                    try:progress=json.loads(lines[-1])['attempted_iteration']
                    except (ValueError,KeyError):pass
            budget_update(config,key,'tick',pid=proc.pid,heartbeat_wall=time.time(),current_attempt=progress)
        rc=proc.returncode
    elapsed=(time.monotonic()-started)/3600
    result=dict(seed=seed,arm=arm,returncode=rc,process_gpuh=elapsed,started_wall=wall,ended_wall=time.time(),start_attempt=attempt,
                status='SCIENTIFIC_FAILURE' if scientific_failure(log) else 'TECHNICAL_FAILURE')
    if rc==0:
        endpoint=run/'training-state-kimg001024.pt'
        try:
            state=torch.load(endpoint,map_location='cpu',weights_only=False);meta=q.validate_state(state,m)
            if state['attempted_iteration']!=8000 or meta['ema_512_init_count']!=1:raise RuntimeError('invalid endpoint')
            if p.digest(endpoint)!=json.loads(Path(str(endpoint)+'.sha256.json').read_text())['sha256']:raise RuntimeError('endpoint seal differs')
            result.update(status='PASS',attempted_iteration=8000,successful_optimizer_steps=state['successful_optimizer_steps'],checkpoint_sha256=p.digest(endpoint),endpoint=endpoint.name);del state
        except Exception as exc:result.update(error=repr(exc))
    ledger=json.loads(Path(config['budget_ledger']).read_text());previous=ledger['jobs'][key].get('previous_gpuh',0)
    budget_update(config,key,'finish',status=result['status'],actual_gpuh=previous+elapsed,returncode=rc)
    p.write(status,result);return result


def selected_arms(config,seed):
    if config.get('host')=='extra':
        if seed!=57 or config.get('assigned_arms')!={'57':['D_startup_up5']} or config.get('placement_amendment')!='replacement_gpu_for_unstarted_seed57_D_v4':
            raise RuntimeError('replacement host may run only the authorized unstarted seed57 D arm')
        return ('D_startup_up5',)
    return p.ARMS if seed%2==0 else p.ARMS[::-1]


def main():
    a=argparse.ArgumentParser();a.add_argument('--deployment',type=Path,required=True);a.add_argument('--seed',type=int,choices=p.SEEDS,required=True);a.add_argument('--gpu',type=int,required=True);args=a.parse_args()
    config=json.loads(args.deployment.read_text());verify_freeze(config)
    if args.seed not in config['assigned_seeds'] or args.gpu!=args.seed-50:raise RuntimeError('fixed seed/GPU mapping differs')
    mapping=gpu_lock(config,args.gpu)
    with locked(Path(config['lock_root'])/(mapping['uuid']+'.lock')):
        for arm in selected_arms(config,args.seed):
            result=run_arm(config,args.seed,arm,mapping);print(json.dumps(result),flush=True)
            # A technical failure ends this slot pending audit. Do not overwrite/restart it.
    return 0

if __name__=='__main__':raise SystemExit(main())
