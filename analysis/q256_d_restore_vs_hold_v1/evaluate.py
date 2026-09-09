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
from training import d_restore as hc, reproducibility, schedule_switch
from . import protocol as p
from analysis.q256_history_component_chase_v1.preflight import runtime as runtime_probe
from .budget import locked
from scripts.run_m1_training_slot import write_json as write
from . import budget


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
            write(result_path,result);return result
        snapshot=exports['readouts'][slot['readout']]['snapshot']
        directory=root/'jobs'/slot['job_id']
        directory.mkdir(parents=True,exist_ok=True)
        attempt=len(list(directory.glob('attempt-*')))
        output=directory/f'attempt-{attempt:02d}'; output.mkdir()
        wire={**{k:str(v) for k,v in slot.items()},'slot_id':slot['job_id'],'metrics':'kid50k_full,fid50k_full',
              'nfe':'1','precision':'fp32','metric_seed':'20260730'}
        cmd=frozen_command.build_command(wire,snapshot,Path(config['evaluation_dataset']),output,
                Path(config['evaluator_repo']),Path(config['runtime_python']),46000+gpu)
        cmd=[s.replace('--desc=m1-','--desc=d-restore-') for s in cmd]
        python=Path(config['runtime_python'])
        prefix=python.resolve().parent.parent
        libraries=runtime_environment(gpu,python)['LD_LIBRARY_PATH'].split(':')
        env=frozen_command.runtime_env(prefix,prefix,Path(config['evaluation_cache']),gpu,
                46000+gpu,runtime_library_paths=libraries)
        key=slot['job_id']
        process_record=budget.run(cmd,cwd=config['evaluator_repo'],env=env,log=output/'process.log',
            ledger=config['budget_ledger'],key=key,max_gpuh=config.get('evaluation_lease_gpuh',.25),seed=slot['seed'])
        if process_record['status']=='BUDGET_PAUSED':
            result={**slot,**process_record}
            if process_record['exit_code'] is not None:budget.finish(config['budget_ledger'],key,result)
            write(result_path,result);return result
        result={**slot,'status':'TECHNICAL_FAILURE','attempt':attempt,'command':cmd,'gpu':gpu,
                'process_gpuh':process_record['process_gpuh'],'exit_code':process_record['exit_code'],
                'evaluator_commit':p.EVALUATOR_COMMIT,'checkpoint':exports['source_state'],
                'checkpoint_sha256':exports['source_state_sha256'],'readout_snapshot':exports['readouts'][slot['readout']]}
        try:
            result.update(validate_result(slot,snapshot,output,config['evaluation_dataset'],process_record['exit_code']),status='PASS')
        except Exception as exc: result['error']=repr(exc)
        write(output/'attempt_receipt.json',result)
        write(result_path,result)
        budget.finish(config['budget_ledger'],slot['job_id'],result)
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=p.SEEDS,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--path',choices=('DD',),help='Evaluate one completed path without waiting for the other')
    parser.add_argument('--export-only',action='store_true')
    args=parser.parse_args(); config=json.loads(args.deployment.read_text())
    if args.export_only:
        export_readouts(config,args.seed,'DD',args.gpu); return 0
    verify_evaluator(config)
    device=frozen_command.gpu_resource_probe(args.gpu)
    with locked(f"/tmp/ect-d-restore-{device['uuid']}.lock"):
        for branch in ((args.path,) if args.path else ('DD',)):
            export_path=Path(config['evaluation_output'])/'readouts'/f'seed{args.seed}-DD'/'export.json'
            stage=json.loads((Path(config['output_root'])/f'S{args.seed-49:02d}'/'DD/suffix/stage_status.json').read_text())
            if stage['status'] in {'SCIENTIFIC_FAILURE','NO_ENDPOINT'}:
                exports=None
            else:
                if not export_path.exists():
                    key=f'S{args.seed-49:02d}/DD/export'
                    log=Path(config['evaluation_output'])/f'export-seed{args.seed}-{time.time_ns()}.log'
                    log.parent.mkdir(parents=True,exist_ok=True)
                    cmd=[config['runtime_python'],'-m','analysis.q256_d_restore_vs_hold_v1.evaluate',
                         '--deployment',str(args.deployment),'--seed',str(args.seed),'--gpu','0','--export-only']
                    rec=budget.run(cmd,cwd=p.ROOT,env=runtime_environment(args.gpu,Path(config['runtime_python'])),log=log,
                                   ledger=config['budget_ledger'],key=key,max_gpuh=.10,seed=args.seed)
                    if rec['status']=='EXITED':rec['status']='PASS' if rec['exit_code']==0 else 'TECHNICAL_FAILURE'
                    if rec['exit_code'] is not None:budget.finish(config['budget_ledger'],key,rec)
                    if rec['status']!='PASS':return 1
                exports=json.loads(export_path.read_text())
            for slot in p.evaluation_slots():
                if slot['seed']!=args.seed or slot['path']!=branch: continue
                result=run_job(config,slot,exports,args.gpu)
                print(json.dumps({'job_id':slot['job_id'],'status':result['status']}),flush=True)
                if result['status'] in {'TECHNICAL_FAILURE','BUDGET_PAUSED'}: return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
