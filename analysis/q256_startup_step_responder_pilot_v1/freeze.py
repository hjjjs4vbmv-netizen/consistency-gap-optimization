"""Freeze node manifests, commands, source hashes and budget after PASS preflight."""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
from . import protocol as p
from .preflight import assets


def freeze(config,receipt,implementation_commit,training_estimate):
    root=Path(config['root']);assets(config)
    preflight=json.loads(Path(receipt).read_text())
    if preflight['status']!='PASS':raise RuntimeError('preflight is not PASS')
    if not 0<training_estimate<10:raise RuntimeError('invalid measured forecast')
    # Source list comes from the actual audited implementation, independent of .git deployment layout.
    source_names=json.loads((root/'control/source_files.json').read_text())
    hashes={n:p.digest(p.ROOT/n) for n in source_names}
    expected=json.loads((root/'control/expected_source_hashes.json').read_text())
    if hashes!=expected:raise RuntimeError('deployment source hashes differ from implementation')
    cmds=[];jobs={}
    from .worker import selected_arms
    for s in config['assigned_seeds']:
        for arm in selected_arms(config,s):
            output=Path(config['output_root'])/f'seed{s}'/arm
            mp=Path(config['manifest_root'])/f'seed{s}-{arm}.json';m=p.manifest(config,s,arm,output);p.write(mp,m,True)
            cmd=p.command(config,m,mp);cmds.append(dict(seed=s,arm=arm,command=cmd,manifest_sha256=p.digest(mp)))
            jobs[f'seed{s}-{arm}']=dict(status='PENDING',estimate_gpuh=training_estimate,actual_gpuh=0)
    owned=config['host']=='ect'
    fraction=len(config['assigned_seeds'])/8 if owned else 1.0
    used=preflight['process_gpuh'] if config['host']=='matpool' else 0.
    engineering=config.get('engineering_reserve_gpuh',2.5*fraction)
    evaluation=config.get('evaluation_reserve_gpuh',8.5*fraction)
    if used>engineering:raise RuntimeError('engineering process budget exceeds reserved host envelope')
    budget=dict(protocol_id=p.PROTOCOL,cap_gpuh=None if config.get('budget_exempt',owned) else 90,allocation_cap_gpuh=config.get('allocation_cap_gpuh',90),
        budget_exempt=config.get('budget_exempt',owned),cap_scope='owned_ect_unmetered' if owned else config.get('cap_scope','paid_matpool_only'),
        evaluation_reserve_gpuh=evaluation,engineering_reserve_gpuh=engineering,
        engineering_used_gpuh=used,engineering_remaining_gpuh=engineering-used,
        evaluation_used_gpuh=0,evaluation_remaining_gpuh=evaluation,
        jobs=jobs,status='READY',allocation=config['host'],created_wall=time.time())
    from .worker import projection
    budget['projection']=projection(budget)
    if budget['projection']['global_conservative_projected_gpuh']>90:budget['status']='INCOMPLETE_BUDGET'
    p.write(config['budget_ledger'],budget,True)
    p.write(root/'control/frozen_commands.json',cmds,True)
    p.write(root/'control/training_queue.json',[x for x in p.queue() if x['seed'] in config['assigned_seeds']],True)
    p.write(root/'control/evaluation_slots.json',[x for x in p.slots() if x['seed'] in config['assigned_seeds']],True)
    result=dict(status='FROZEN',protocol_id=p.PROTOCOL,preflight_status='PASS',preflight_receipt_sha256=p.digest(receipt),
        preflight_execution_commit=preflight['implementation_commit'],implementation_commit=implementation_commit,
        source_hashes=hashes,protocol_sha256=p.digest(config['protocol_json']),
        old_control_bindings_sha256=p.digest(config['old_control_bindings']),gpu_mapping=config['gpu_mapping'],
        runtime=preflight['assets']['runtime'],forecast_training_gpuh=training_estimate,commands_sha256=p.digest(root/'control/frozen_commands.json'))
    p.write(config['freeze_receipt'],result,True)
    print(json.dumps(dict(status=result['status'],budget_status=budget['status'],projection=budget['projection'])),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--deployment',type=Path,required=True);a.add_argument('--preflight',type=Path,required=True);a.add_argument('--implementation-commit',required=True);a.add_argument('--training-estimate',type=float,required=True);args=a.parse_args()
    freeze(json.loads(args.deployment.read_text()),args.preflight,args.implementation_commit,args.training_estimate)
