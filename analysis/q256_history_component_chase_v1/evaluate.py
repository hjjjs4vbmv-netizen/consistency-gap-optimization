"""Separate endpoint export/evaluation; never reads quality into training dispatch."""
from __future__ import annotations
import argparse
import csv
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from scripts import run_m1_evaluation_job as frozen_command
from scripts import validate_m1_evaluation_job as validation
from scripts.classify_m1_readout import classify_fixed_input
from scripts.run_m1_training_slot import runtime_environment
from training import history_component as hc, reproducibility, schedule_switch
from . import protocol as p
from .preflight import runtime as runtime_probe
from .worker import locked, write


def validate_export_identity(manifest,seed,branch):
    if (manifest.get('experiment_protocol')!=p.EXPERIMENT_ID
            or manifest.get('seed')!=seed or manifest.get('branch')!=branch):
        raise RuntimeError('export path does not match the formal training identity')


def export_readouts(config, seed, branch, gpu):
    started=time.monotonic()
    run=Path(config['output_root'])/f'S{seed-49:02d}'/branch/'suffix'
    status=json.loads((run/'stage_status.json').read_text())
    if status['status'] in {'NO_ENDPOINT','SCIENTIFIC_FAILURE'}: return None
    if status['status']!='PASS': raise RuntimeError('training is not scientifically complete')
    target=Path(config['evaluation_output'])/'readouts'/f'seed{seed}-{branch}'
    target.mkdir(parents=True,exist_ok=True)
    manifest=schedule_switch.load_run_manifest(run/'formal_run_manifest.json')
    validate_export_identity(manifest,seed,branch)
    source=run/'training-state-kimg001024.pt'
    existing=target/'export.json'
    if existing.exists():
        receipt=json.loads(existing.read_text())
        if receipt['source_state']!=str(source.resolve()) or receipt['seed']!=seed or receipt['path']!=branch:
            raise RuntimeError('existing export identity mismatch')
        for row in receipt['readouts'].values():
            if schedule_switch.sha256_file(row['snapshot'])!=row['snapshot_sha256']:
                raise RuntimeError('existing readout snapshot changed')
        return receipt
    state=torch.load(source,map_location='cpu',weights_only=False)
    hc.validate_terminal_state(state,manifest)
    receipt={'protocol':p.EXPERIMENT_ID,'seed':seed,'path':branch,'source_state':str(source.resolve()),
             'source_attempt':8000,'source_nimg':1024000,'source_state_sha256':schedule_switch.sha256_file(source),
             'training_implementation_commit':json.loads(Path(config['preflight_receipt']).read_text())['implementation_commit'],
             'readouts':{}}
    for readout in p.READOUT_BLOCKS:
        snapshot=hc.evaluator_snapshot(state,readout)
        before=reproducibility.module_state_sha256(snapshot['ema'])
        observation=classify_fixed_input(snapshot['ema'],torch.device('cuda',gpu))
        write(target/f'{readout}-fixed-input.json',observation)
        if observation['classification']!='FINITE_READOUT':
            raise RuntimeError('fixed-input readout invalid; retain observation and do not replace checkpoint')
        snapshot['ema'].cpu()
        if reproducibility.module_state_sha256(snapshot['ema'])!=before:
            raise RuntimeError('export/classification altered readout state')
        path=target/f'{readout}.pkl'
        if path.exists():
            with path.open('rb') as handle: previous=pickle.load(handle)
            if reproducibility.module_state_sha256(previous['ema'])!=before:
                raise RuntimeError('partial export snapshot differs; preserve it for diagnosis')
            del previous
        else:
            reproducibility.atomic_pickle_dump(snapshot,path,overwrite=False)
        receipt['readouts'][readout]={'snapshot':str(path.resolve()),'snapshot_sha256':schedule_switch.sha256_file(path),
                                    'module_state_sha256':before,'fixed_input':observation}
        del snapshot
    del state
    receipt['export_process_gpuh']=(time.monotonic()-started)/3600
    write(existing,receipt)
    return receipt


def validate_result(slot, snapshot, directory, dataset, exit_code):
    if exit_code!=0: raise RuntimeError(f'evaluator process exit {exit_code}')
    if 'Exiting...' not in (directory/'log.txt').read_text(errors='replace'):
        raise RuntimeError('evaluator did not finish')
    options=json.loads((directory/'training_options.json').read_text())
    start,end=slot['sample_seed_start'],slot['sample_seed_end']
    expected={'sample_seeds':list(range(start,end+1)),'seed':20260730,
              'metrics':['kid50k_full','fid50k_full'],'metric_repeats':1,
              'metric_generator_batch':128,'retain_generated_artifacts':True,'mid_t':[]}
    for key,value in expected.items():
        if options.get(key)!=value: raise RuntimeError('evaluator option mismatch: '+key)
    if options['network_kwargs'].get('use_fp16') is not False: raise RuntimeError('evaluator is not FP32')
    if Path(options['resume_pkl']).resolve()!=Path(snapshot).resolve(): raise RuntimeError('wrong snapshot')
    if Path(options['dataset_kwargs']['path']).resolve()!=Path(dataset).resolve(): raise RuntimeError('wrong reference dataset')
    samples=np.load(directory/'generated-samples.npy',mmap_mode='r')
    if samples.shape[0]!=50000: raise RuntimeError('not a FID50k block')
    del samples
    results={}; hashes=[]
    for metric in ('kid50k_full','fid50k_full'):
        row=validation.read_metric(directory/f'metric-{metric}.jsonl',metric)
        if row['status']!='SEALED_PASS': raise RuntimeError('invalid '+metric+': '+str(row))
        feature=directory/f'generated-features-{metric}-repeat00.npy'
        if np.load(feature,mmap_mode='r').shape[0]!=50000: raise RuntimeError('wrong feature count')
        hashes.append(schedule_switch.sha256_file(feature))
        results[metric]=row['value']
    if hashes[0]!=hashes[1]: raise RuntimeError('FID and KID did not share generated features')
    return {'FID':results['fid50k_full'],'KID':results['kid50k_full'],'feature_sha256':hashes[0]}


def verify_evaluator(config):
    actual_runtime=runtime_probe()
    if actual_runtime!=p.EXPECTED_RUNTIME:
        raise RuntimeError('evaluation runtime differs from the frozen numeric environment')
    repo=Path(config['evaluator_repo'])
    # cwd works with the archived ECT host's old Git as well as current Git.
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=repo,text=True).strip()
    if head!=p.EVALUATOR_COMMIT or dirty: raise RuntimeError('frozen evaluator checkout differs or is dirty')
    if schedule_switch.sha256_file(repo/'ct_eval.py')!=validation.EVALUATOR_CT_EVAL_SHA256:
        raise RuntimeError('frozen evaluator entrypoint differs')
    validation.verify_evaluation_dataset(Path(config['evaluation_dataset']))
    return head


def admit_evaluation(config,slot):
    path=Path(config['budget_ledger'])
    with locked(str(path)+'.lock'):
        budget=json.loads(path.read_text()); scope=budget.get('scoped_seeds',list(p.SEEDS))
        if slot['seed'] not in scope: raise RuntimeError('evaluation outside allocated node scope')
        scoped_jobs=budget.get('scoped_job_ids')
        if scoped_jobs is not None:
            canonical={s['job_id'] for s in p.evaluation_slots() if s['seed'] in scope}
            if len(scoped_jobs)!=len(set(scoped_jobs)) or not set(scoped_jobs)<=canonical:
                raise RuntimeError('invalid allocated evaluation job scope')
            if slot['job_id'] not in scoped_jobs:
                raise RuntimeError('evaluation outside allocated job scope')
        dedicated=budget.get('evaluation_only') is True
        if dedicated:
            if config.get('evaluation_only') is not True or budget.get('stages'):
                raise RuntimeError('dedicated evaluation requires a separate allocated ledger')
            if not budget.get('allocation_id'):
                raise RuntimeError('dedicated evaluation lacks the global budget allocation record')
            # export_readouts has already required this path's actual PASS/8000
            # endpoint. Other prespecified paths may still be training elsewhere.
        else:
            for row in p.training_queue():
                if row['seed'] not in scope: continue
                for phase in ('prefix','suffix'):
                    stage=budget['stages'].get(f"{row['slot']}/{row['path']}/{phase}",{})
                    if stage.get('status') not in {'PASS','SCIENTIFIC_FAILURE','NO_ENDPOINT'}:
                        raise RuntimeError('this node still has incomplete training; evaluation waits')
        records=budget.setdefault('evaluation_jobs',{})
        if records.get(slot['job_id'],{}).get('status')=='RUNNING':
            raise RuntimeError('evaluation lease already active')
        training=sum(a['process_gpuh'] for stage in budget['stages'].values() for a in stage.get('attempts',[]))
        attempts=[a for row in records.values() for a in row.get('attempts',[])]
        evaluation=sum(a['process_gpuh'] for a in attempts)
        completed=[a['process_gpuh'] for a in attempts if a['status']=='PASS']
        rate=sum(completed)/len(completed) if completed else .05
        engineering=(budget.get('evaluation_setup_process_gpuh',0) if dedicated else
                     json.loads(Path(config['preflight_receipt']).read_text())['short_checks']['process_gpuh'])
        export_cost=sum(json.loads(x.read_text()).get('export_process_gpuh',0)
                        for x in (Path(config['evaluation_output'])/'readouts').glob('seed*/export.json'))
        remaining=sum(s['seed'] in scope and (scoped_jobs is None or s['job_id'] in scoped_jobs) and records.get(s['job_id'],{}).get('status') not in {'PASS','NO_ENDPOINT'} for s in p.evaluation_slots())
        active=sum(max(0,time.time()-r['started_unix'])/3600 for r in records.values() if r.get('status')=='RUNNING')
        projected=training+engineering+export_cost+evaluation+active+remaining*rate
        budget['evaluation_forecast_gpuh']=projected
        if projected>budget['cap_gpuh'] or budget['status']=='INCOMPLETE_BUDGET':
            budget['status']='INCOMPLETE_BUDGET';write(path,budget);return False
        records[slot['job_id']]={**records.get(slot['job_id'],{}),'status':'RUNNING','started_unix':time.time()}
        write(path,budget);return True


def account_evaluation(config,result):
    path=Path(config['budget_ledger'])
    with locked(str(path)+'.lock'):
        budget=json.loads(path.read_text())
        row=budget.setdefault('evaluation_jobs',{}).setdefault(result['job_id'],{})
        # Cost/status only: no FID or KID enters any dispatch ledger.
        row['status']=result['status']
        if 'process_gpuh' in result:
            row.setdefault('attempts',[]).append({'status':result['status'],'process_gpuh':result['process_gpuh']})
        write(path,budget)


def run_job(config,slot,exports,gpu):
    root=Path(config['evaluation_output'])
    result_path=root/'receipts'/f"{slot['job_id']}.json"
    with locked(str(result_path)+'.lock'):
        if result_path.exists():
            prior=json.loads(result_path.read_text())
            if any(prior[k]!=slot[k] for k in ('seed','path','readout','block','job_id')):
                raise RuntimeError('existing evaluation receipt identity mismatch')
            if prior['status'] in {'PASS','NO_ENDPOINT'}: return prior
        if exports is None:
            result={**slot,'status':'NO_ENDPOINT','reason':'own training scientific failure'}
            write(result_path,result);account_evaluation(config,result);return result
        if not admit_evaluation(config,slot):
            result={**slot,'status':'BUDGET_PAUSED'};write(result_path,result);return result
        snapshot=exports['readouts'][slot['readout']]['snapshot']
        directory=root/'jobs'/slot['job_id']
        directory.mkdir(parents=True,exist_ok=True)
        attempt=len(list(directory.glob('attempt-*')))
        output=directory/f'attempt-{attempt:02d}'; output.mkdir()
        wire={**{k:str(v) for k,v in slot.items()},'slot_id':slot['job_id'],'metrics':'kid50k_full,fid50k_full',
              'nfe':'1','precision':'fp32','metric_seed':'20260730'}
        cmd=frozen_command.build_command(wire,snapshot,Path(config['evaluation_dataset']),output,
                Path(config['evaluator_repo']),Path(config['runtime_python']),46000+gpu)
        cmd=[s.replace('--desc=m1-','--desc=history-component-') for s in cmd]
        python=Path(config['runtime_python'])
        prefix=python.resolve().parent.parent
        libraries=runtime_environment(gpu,python)['LD_LIBRARY_PATH'].split(':')
        env=frozen_command.runtime_env(prefix,prefix,Path(config['evaluation_cache']),gpu,
                46000+gpu,runtime_library_paths=libraries)
        started=time.monotonic()
        with (output/'process.log').open('xb') as log:
            process=subprocess.Popen(cmd,cwd=config['evaluator_repo'],env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None: time.sleep(5)
        result={**slot,'status':'TECHNICAL_FAILURE','attempt':attempt,'command':cmd,'gpu':gpu,
                'process_gpuh':(time.monotonic()-started)/3600,'exit_code':process.returncode,
                'evaluator_commit':p.EVALUATOR_COMMIT,'checkpoint':exports['source_state'],
                'checkpoint_sha256':exports['source_state_sha256'],'readout_snapshot':exports['readouts'][slot['readout']]}
        try:
            result.update(validate_result(slot,snapshot,output,config['evaluation_dataset'],process.returncode),status='PASS')
        except Exception as exc: result['error']=repr(exc)
        write(output/'attempt_receipt.json',result)
        write(result_path,result)
        account_evaluation(config,result)
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=p.SEEDS,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--path',choices=('CA','DA'),help='Evaluate one completed path without waiting for the other')
    args=parser.parse_args(); config=json.loads(args.deployment.read_text())
    verify_evaluator(config)
    device=frozen_command.gpu_resource_probe(args.gpu)
    with locked(f"/tmp/ect-history-component-{device['uuid']}.lock"):
        for branch in ((args.path,) if args.path else ('CA','DA')):
            exports=export_readouts(config,args.seed,branch,args.gpu)
            for slot in p.evaluation_slots():
                if slot['seed']!=args.seed or slot['path']!=branch: continue
                result=run_job(config,slot,exports,args.gpu)
                print(json.dumps({'job_id':slot['job_id'],'status':result['status']}),flush=True)
                if result['status'] in {'TECHNICAL_FAILURE','BUDGET_PAUSED'}: return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
