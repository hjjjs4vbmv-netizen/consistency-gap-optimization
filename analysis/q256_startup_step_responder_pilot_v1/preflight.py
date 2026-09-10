"""One bounded GPU preflight; never computes quality metrics."""
from __future__ import annotations
import argparse,csv,json,os,subprocess,time
from pathlib import Path
import torch
from . import protocol as p
from training import startup_quality as q, reproducibility as repro
from scripts.run_m1_training_slot import runtime_environment, equal_state
from analysis.q256_history_component_chase_v1.preflight import runtime

KEYS=['batch_sha256','t_sha256','base_r_sha256','target_r_sha256','denominator_r_sha256','loss','raw_grad_norm','raw_grad_nonfinite_count','update_norm','model_norm','ema_norm','grad_scale_before','grad_scale_after','successful_optimizer_steps']


def assets(config):
    from training.startup_update import DATA_SHA256,TRANSFER_SHA256
    if runtime()!=p.ENVIRONMENT:raise RuntimeError('runtime differs: '+str(runtime()))
    for key,sha in [('dataset',DATA_SHA256),('transfer',TRANSFER_SHA256)]:
        if p.digest(config[key])!=sha:raise RuntimeError(key+' hash differs')
    published={(r['seed'],r['path'],r['block']):r for r in p.old_controls()}
    bindings=json.loads(Path(config['old_control_bindings']).read_text())
    if len(bindings)!=48:raise RuntimeError('old controls incomplete')
    for x in bindings:
        old=published[(x['seed'],x['arm'],x['block'])]
        if (x['FID'],x['KID'])!=(old['FID'],old['KID']):raise RuntimeError('old metrics differ')
        if p.digest(Path(config['reference_root'])/x['receipt'])!=x['receipt_sha256']:raise RuntimeError('old receipt hash differs')
        if old.get('receipt_sha256') and old['receipt_sha256']!=x['receipt_sha256']:raise RuntimeError('published control receipt differs')
    return dict(status='PASS',runtime=runtime(),dataset_sha256=DATA_SHA256,transfer_sha256=TRANSFER_SHA256,old_control_count=48)


def checkpoint(run,m):
    file=run/'training-state-latest.pt'
    seal=json.loads(Path(str(file)+'.sha256.json').read_text())
    if p.digest(file)!=seal['sha256']:raise RuntimeError('checkpoint file hash differs')
    state=torch.load(file,map_location='cpu',weights_only=False);q.validate_state(state,m)
    if any(run.glob('update-success-*.pt')) or 'startup_engineering' in state:raise RuntimeError('engineering tensors/metadata leaked')
    return state


def compare_states(a,b):
    fields=['optimizer_state','gradscaler_state','rank_states','loss_fn_state','attempted_iteration','successful_optimizer_steps','cur_nimg','cur_tick','tick_start_nimg','snapshot_grid_z','snapshot_grid_c','snapshot_grid_size']
    for key in fields:
        if not equal_state(a[key],b[key]):raise RuntimeError('pause/resume differs: '+key)
    for key in ['net','ema']:
        if repro.module_state_sha256(a[key])!=repro.module_state_sha256(b[key]):raise RuntimeError('pause/resume differs: '+key)
    if a['startup_quality']['controller']!=b['startup_quality']['controller']:raise RuntimeError('controller resume differs')


def run(config,gpu):
    root=Path(config['preflight_root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'started.json').exists():raise RuntimeError('preflight already attempted; audit preserved results before further execution')
    asset=assets(config);p.write(root/'assets.json',asset,True)
    p.write(root/'started.json',dict(started=time.time(),gpu=gpu),True)
    costs=[];outputs={};env=runtime_environment(gpu,Path(config['runtime_python']))
    def execute(arm,name,resume=None,stop=None):
        directory=root/name
        m=p.manifest(config,50,arm,directory,True);mp=root/(name+'.manifest.json');p.write(mp,m,True)
        cmd=p.command(config,m,mp,resume,stop)
        if sum(x['process_gpuh'] for x in costs) + .25 > 3:raise RuntimeError('preflight budget exhausted')
        log=root/(name+('-resume' if resume else '')+f'-{stop or 64}.log')
        if log.exists():raise RuntimeError('no overwriting a preflight attempt')
        start=time.monotonic()
        with log.open('x') as f:rc=subprocess.run(cmd,cwd=p.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        costs.append(dict(name=name,command=cmd,returncode=rc,process_gpuh=(time.monotonic()-start)/3600))
        p.write(root/'gpu_ledger.json',costs)
        if rc:raise RuntimeError('preflight process failed: '+str(log))
        return directory,m
    for arm in ('A_replay',)+p.ARMS:
        directory,m=execute(arm,arm+'-64');state=checkpoint(directory,m)
        if state['attempted_iteration']!=64:raise RuntimeError('not 64 attempts')
        rows=[json.loads(x) for x in (directory/'startup_quality_telemetry.jsonl').read_text().splitlines()]
        for row in rows:
            step=row['successful_optimizer_steps']
            expected=q.MULTIPLIERS[arm] if not row['step_skipped'] and step<=5 else 1.
            if row['applied_multiplier']!=expected or row['restored_lr']!=[1e-4]:raise RuntimeError('multiplier/LR boundary differs')
        if not {5,6}.issubset({r['successful_optimizer_steps'] for r in rows}):raise RuntimeError('step boundaries unavailable')
        outputs[arm]=directory;del state
    old=list(csv.DictReader((Path(config['reference_root'])/'seed50/factorial_training_telemetry_v1.csv').open()))[:64]
    replay=list(csv.DictReader((outputs['A_replay']/'factorial_training_telemetry_v1.csv').open()))
    matches={k:sum(x[k]==y[k] for x,y in zip(replay,old)) for k in KEYS}
    p.write(root/'A_replay.json',dict(rows=len(replay),fields=matches),True)
    if len(replay)!=64 or any(x!=64 for x in matches.values()):raise RuntimeError('original A replay differs')
    for arm in p.ARMS:
        # One trajectory is paused at each prescribed success boundary, preserving its own identity.
        directory,m=execute(arm,arm+'-split',stop=3)
        for stop in (5,6,None):
            checkpoint(directory,m)
            directory,m=execute(arm,arm+'-split',directory/'training-state-latest.pt',stop)
        split=checkpoint(directory,m)
        full_m=p.manifest(config,50,arm,outputs[arm],True);full=checkpoint(outputs[arm],full_m)
        compare_states(full,split);del full,split
    result=dict(status='PASS',assets=asset,original_A_fields=matches,pause_resume_success_boundaries=[3,5,6],
                process_gpuh=sum(x['process_gpuh'] for x in costs),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.ROOT,text=True).strip())
    p.write(root/'preflight_receipt.json',result,True);print(json.dumps(result),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--deployment',type=Path,required=True);a.add_argument('--gpu',type=int,default=0);a.add_argument('--assets-only',action='store_true');args=a.parse_args()
    c=json.loads(args.deployment.read_text())
    if args.assets_only:print(json.dumps(assets(c)))
    else:
        try:run(c,args.gpu)
        except BaseException as exc:
            p.write(Path(c['preflight_root'])/'failure.json',dict(status='FAILED',error=repr(exc)));raise
