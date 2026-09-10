"""Endpoint-only evaluation behind the complete frozen training matrix."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
import torch
from . import protocol as p
from .worker import locked,gpu_lock,check_idle,verify_freeze,projection
from training import startup_quality as q,m1,reproducibility as repro
from scripts.run_m1_training_slot import runtime_environment
from scripts.run_m1_evaluation_job import build_command
from analysis.q256_d_restore_vs_hold_v1.evaluate import validate_result,verify_evaluator


def training_gate(config):
    path=Path(config['training_matrix_frozen']);matrix=json.loads(path.read_text())
    rows=matrix['outcomes'];expected={(s,a) for s in p.SEEDS for a in p.ARMS}
    if len(rows)!=16 or {(x['seed'],x['arm']) for x in rows}!=expected:raise RuntimeError('complete training matrix required')
    if any(r['status'] not in p.SPEC['terminal_statuses'] for r in rows):raise RuntimeError('training outcomes not terminal')
    if matrix['protocol_sha256']!=p.digest(config['protocol_json']):raise RuntimeError('training matrix protocol differs')
    return {(r['seed'],r['arm']):r for r in rows}


def export(config,seed,arm):
    run=Path(config['output_root'])/f'seed{seed}'/arm
    manifest=json.loads((Path(config['manifest_root'])/f'seed{seed}-{arm}.json').read_text())
    source=run/'training-state-kimg001024.pt';state=torch.load(source,map_location='cpu',weights_only=False);q.validate_state(state,manifest)
    if state['attempted_iteration']!=8000 or state['startup_quality']['ema_512_init_count']!=1:raise RuntimeError('not a final quality endpoint')
    root=Path(config['evaluation_output'])/'readouts'/f'seed{seed}-{arm}';root.mkdir(parents=True,exist_ok=True)
    outputs={}
    for name in ('ONLINE','E_KEEP','E_512'):
        value=m1.evaluator_snapshot(state,name);path=root/(name+'.pkl');modelhash=repro.module_state_sha256(value['ema'])
        if not path.exists():repro.atomic_pickle_dump(value,path,overwrite=False)
        else:
            import pickle
            with path.open('rb') as f:old=pickle.load(f)
            if repro.module_state_sha256(old['ema'])!=modelhash:raise RuntimeError('existing export differs')
        outputs[name]=dict(path=str(path),sha256=p.digest(path),module_sha256=modelhash)
    p.write(root/'export.json',dict(seed=seed,arm=arm,source_sha256=p.digest(source),readouts=outputs),True)
    return outputs


def evaluate_seed(config,seed,logical):
    verify_freeze(config);matrix=training_gate(config);verify_evaluator(config);mapping=gpu_lock(config,logical)
    root=Path(config['evaluation_output']);root.mkdir(parents=True,exist_ok=True)
    with locked(Path(config['lock_root'])/(mapping['uuid']+'.lock')):
        for arm in (p.ARMS if seed%2==0 else p.ARMS[::-1]):
            outcome=matrix[(seed,arm)];outputs=export(config,seed,arm) if outcome['status']=='PASS' else None
            for slot in [x for x in p.slots() if x['seed']==seed and x['arm']==arm]:
                receipt=root/'receipts'/(slot['job_id']+'.json')
                if receipt.exists():continue
                if outputs is None:
                    p.write(receipt,{**slot,'status':'NO_ENDPOINT','training_status':outcome['status']},True);continue
                ledgerpath=Path(config['budget_ledger']);lease=8.5/48
                with locked(str(ledgerpath)+'.lock'):
                    ledger=json.loads(ledgerpath.read_text());view=projection(ledger)
                    if ledger['status']=='INCOMPLETE_BUDGET' or view['global_conservative_projected_gpuh']>90 or ledger['evaluation_remaining_gpuh']<lease:
                        ledger['status']='INCOMPLETE_BUDGET';p.write(ledgerpath,ledger)
                        p.write(receipt,{**slot,'status':'INCOMPLETE_BUDGET'},True);continue
                    ledger['evaluation_remaining_gpuh']-=lease
                    ledger.setdefault('evaluation_running',{})[slot['job_id']]=dict(started_wall=time.time(),estimate_gpuh=lease)
                    p.write(ledgerpath,ledger)
                check_idle(mapping['uuid']);directory=root/'jobs'/slot['job_id']
                if directory.exists():raise RuntimeError('evaluation output exists without a receipt; audit required')
                directory.mkdir(parents=True)
                wire={**{k:str(v) for k,v in slot.items()},'slot_id':slot['job_id'],'metrics':'kid50k_full,fid50k_full'}
                snapshot=outputs['E_512']['path']
                cmd=build_command(wire,snapshot,Path(config['evaluation_dataset']),directory,Path(config['evaluator_repo']),Path(config['runtime_python']),46000+logical)
                env=runtime_environment(mapping['uuid'],Path(config['runtime_python']));env['DNNLIB_CACHE_DIR']=config['evaluation_cache'];env['HOME']=config.get('runtime_home',env['HOME'])
                started=time.monotonic();wall=time.time()
                with (root/(slot['job_id']+'.process.log')).open('x') as f:
                    rc=subprocess.run(cmd,cwd=config['evaluator_repo'],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
                elapsed=(time.monotonic()-started)/3600
                result={**slot,'status':'TECHNICAL_FAILURE','returncode':rc,'process_gpuh':elapsed,'started_wall':wall,'ended_wall':time.time(),
                        'evaluator_commit':p.EVALUATOR_COMMIT,'checkpoint_sha256':outcome['checkpoint_sha256'],'snapshot_sha256':outputs['E_512']['sha256']}
                try:result.update(validate_result(slot,snapshot,directory,config['evaluation_dataset'],rc),status='PASS')
                except Exception as exc:result['error']=repr(exc)
                p.write(receipt,result,True)
                with locked(str(ledgerpath)+'.lock'):
                    ledger=json.loads(ledgerpath.read_text());ledger['evaluation_running'].pop(slot['job_id'])
                    ledger['evaluation_used_gpuh']=ledger.get('evaluation_used_gpuh',0)+elapsed
                    # Return unused reserved evaluation capacity; completed slots are never re-evaluated.
                    ledger['projection']=projection(ledger)
                    if ledger['projection']['global_conservative_projected_gpuh']>90:ledger['status']='INCOMPLETE_BUDGET'
                    p.write(ledgerpath,ledger)
                print(json.dumps(result),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--deployment',type=Path,required=True);a.add_argument('--seed',type=int,choices=p.SEEDS,required=True);a.add_argument('--gpu',type=int,required=True);args=a.parse_args()
    c=json.loads(args.deployment.read_text())
    if args.seed not in c['assigned_seeds'] or args.gpu!=args.seed-50:raise RuntimeError('seed/GPU allocation differs')
    evaluate_seed(c,args.seed,args.gpu)
