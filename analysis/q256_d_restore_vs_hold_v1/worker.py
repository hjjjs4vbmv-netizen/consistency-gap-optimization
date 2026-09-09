"""Exactly one DD suffix per seed, native production loss, no automatic scientific retries."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import torch
from scripts import run_m1_training_slot as legacy
from training import d_restore as dd, schedule_switch as sw
from analysis.q256_history_component_chase_v1.worker import compare_telemetry
from . import protocol as p, budget


def checkpoint(run,manifest):
    paths=[run/'training-state-latest.pt']+sorted(run.glob('training-state-kimg*.pt'),reverse=True)
    for path in paths:
        if path.is_file():
            state=torch.load(path,map_location='cpu',weights_only=False)
            dd.validate_resumed_state(state,manifest)
            attempt=state['attempted_iteration']; del state
            return path,attempt
    return None,4000


def run(config,seed,gpu,dry_run=False):
    if seed not in p.SEEDS or seed not in config['assigned_seeds']: raise RuntimeError('seed outside allocation')
    inventory=json.loads(Path(config['source_inventory']).read_text())
    source=next(r for r in inventory if r['seed']==seed)
    root=Path(config['output_root'])/f'S{seed-49:02d}'/'DD'/'suffix'
    manifest=p.manifest(seed,source['prefix_path'],root,source['binding'],da_branch_init=source['da_branch_init_path'])
    manifest_path=root/'formal_run_manifest.json'
    command=p.command(python=config['runtime_python'],dataset=config['dataset'],output=root,seed=seed,
                      resume=source['prefix_path'],switch_manifest=manifest_path)
    if dry_run: return {'seed':seed,'manifest':manifest,'command':command,'status':'DRY_RUN'}
    gate=json.loads(Path(config['preflight_receipt']).read_text())
    if gate['status']!='PASS': raise RuntimeError('preflight is not PASS')
    root.mkdir(parents=True,exist_ok=True)
    with budget.locked(root/'worker.lock'):
        status_path=root/'stage_status.json'
        if status_path.exists():
            prior=json.loads(status_path.read_text())
            if prior['status'] in {'PASS','SCIENTIFIC_FAILURE'}: return prior
        if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
            raise RuntimeError('immutable DD manifest changed')
        legacy.write_json(manifest_path,manifest); sw.load_run_manifest(manifest_path)
        resume,start=checkpoint(root,manifest)
        terminal=root/'training-state-kimg001024.pt'
        if start==8000:
            raise RuntimeError('terminal exists; reconcile receipt without training again')
        if resume is None and list(root.glob('train-attempt-*.log')):
            raise RuntimeError('technical interruption without a full DD state; no fresh restart')
        if resume:
            for name in ('train_summary.csv','factorial_training_telemetry_v1.csv','schedule_switch_training_telemetry_v1.csv'):
                path=root/name
                if path.exists():
                    shutil.copy2(path,path.with_name(path.name+f'.before-recovery-{time.time_ns()}'))
                    legacy.truncate_attempt_csv(path,start)
        command=p.command(python=config['runtime_python'],dataset=config['dataset'],output=root,seed=seed,
                          resume=resume or source['prefix_path'],switch_manifest=manifest_path)
        index=len(list(root.glob('train-attempt-*.log')))+1; log=root/f'train-attempt-{index:02d}.log'
        key=f'S{seed-49:02d}/DD/train'
        legacy.write_json(status_path,{'status':'RUNNING','seed':seed,'command':command,'start_attempt':start})
        record=budget.run(command,cwd=p.ROOT,env=legacy.runtime_environment(gpu,Path(config['runtime_python'])),
                          log=log,ledger=config['budget_ledger'],key=key,max_gpuh=config.get('training_lease_gpuh',3.5),seed=seed)
        record.update(seed=seed,command=command,start_attempt=start,end_attempt=start,processed_attempts=0,log=str(log))
        if record['status']=='BUDGET_PAUSED' and record['exit_code'] is None:
            legacy.write_json(status_path,record); return record
        try:
            if record['status']!='BUDGET_PAUSED':
                record['status']='SCIENTIFIC_FAILURE' if legacy.scientific_failure(log) else 'TECHNICAL_FAILURE'
                if record['exit_code']==0:
                    state=torch.load(terminal,map_location='cpu',weights_only=False)
                    dd.validate_terminal_state(state,manifest)
                    record.update(status='PASS',end_attempt=8000,endpoint=str(terminal),
                                  successful_optimizer_steps=state['successful_optimizer_steps'])
                    del state
                    record['paired_telemetry']=compare_telemetry(root/'schedule_switch_training_telemetry_v1.csv',source['da_telemetry_path'])
                    first=next(__import__('csv').DictReader((root/'schedule_switch_training_telemetry_v1.csv').open()))
                    if float(first['target_gap_scale'])!=1 or float(first['denominator_gap_scale'])!=1.1:
                        raise RuntimeError('first actual DD loss factors differ')
            latest,end=checkpoint(root,manifest); record['end_attempt']=end; record['processed_attempts']=end-start
        except Exception as exc:
            if record['status']!='SCIENTIFIC_FAILURE': record['status']='TECHNICAL_FAILURE'
            record['error']=repr(exc)
        finally:
            budget.finish(config['budget_ledger'],key,record)
            legacy.write_json(root/f'train-attempt-{index:02d}.json',record)
            legacy.write_json(status_path,record)
        return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',type=Path,required=True); parser.add_argument('--seed',type=int,choices=p.SEEDS,required=True)
    parser.add_argument('--gpu',type=int,required=True); parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args(); result=run(json.loads(a.deployment.read_text()),a.seed,a.gpu,a.dry_run)
    print(json.dumps(result,indent=2)); return 0 if result['status'] in {'PASS','SCIENTIFIC_FAILURE','DRY_RUN'} else 1


if __name__=='__main__': raise SystemExit(main())
