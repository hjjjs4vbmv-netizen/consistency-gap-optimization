"""Fixed seeds50/51, fixed source states, 16-attempt audits; no FID or adaptive matrix."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import torch
from training import history_component as hc, m1, schedule_switch as sw, d_restore as dd
from scripts import run_m1_training_slot as legacy
from analysis.q256_history_component_chase_v1.short_checks import compare_states as compare_old_states, EXACT_KEYS
from . import protocol as p, budget, audit_compare


def compare_dd_states(left,right):
    a=torch.load(left,map_location='cpu',weights_only=False)
    b=torch.load(right,map_location='cpu',weights_only=False)
    for k in ('net','ema','ema_512'):
        if not m1._equal_state(a[k].state_dict(),b[k].state_dict()):raise RuntimeError('DD resume module mismatch: '+k)
    for k in (*EXACT_KEYS,'d_restore'):
        if not m1._equal_state(a[k],b[k]):raise RuntimeError('DD resume state mismatch: '+k)
    return {'status':'PASS','exact':True,'scope':'16-attempt continuous versus 8+8 same-state resume'}


def execute(config,output,gpu):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(Path(config['source_inventory']).read_text())
    report={'status':'RUNNING','checks':{},'audit_comparisons':{},'quality_evaluations':0,'processes':[]}
    def run(name,cmd,cwd=p.ROOT):
        target=output/name;target.mkdir(parents=True,exist_ok=True)
        done=target/'completed.json'
        stop=next((x.split('=')[1] for x in cmd if x.startswith('--stop-after-attempts=')), 'audit')
        # Split-resume has two distinct bounded calls in the same output directory.
        done=target/f'completed-{stop}.json'
        if done.exists():
            previous=json.loads(done.read_text())
            report['processes'].append(previous)
            return target/'run/training-state-latest.pt'
        log=target/f'process-{len(list(target.glob("process-*.log")))}.log'
        key='engineering/'+name+'/'+log.stem
        rec=budget.run(cmd,cwd=cwd,env=legacy.runtime_environment(gpu,Path(config['runtime_python'])),log=log,
                       ledger=config['engineering_ledger'],key=key,max_gpuh=.12)
        if rec['status']=='EXITED':rec['status']='PASS' if rec['exit_code']==0 else 'TECHNICAL_FAILURE'
        if rec['exit_code'] is not None:budget.finish(config['engineering_ledger'],key,rec)
        report['processes'].append({'name':name,'command':cmd,**rec})
        legacy.write_json(output/'short_checks.json',report)
        if rec['status']!='PASS':raise RuntimeError('bounded engineering process did not complete: '+str(log))
        legacy.write_json(done,report['processes'][-1])
        return target/'run/training-state-latest.pt'
    def dd_command(name,stop,resume=None,seed=50):
        src=next(r for r in inventory if r['seed']==seed);run_dir=output/name/'run';run_dir.mkdir(parents=True,exist_ok=True)
        manifest=p.manifest(seed,src['prefix_path'],run_dir,src['binding'],engineering=True,da_branch_init=src['da_branch_init_path'])
        path=run_dir/'engineering_manifest.json';legacy.write_json(path,manifest)
        return p.command(python=config['runtime_python'],dataset=config['dataset'],output=run_dir,seed=seed,
                         resume=resume or src['prefix_path'],switch_manifest=path,engineering_stop=stop)
    try:
        a=run('DD_continuous',dd_command('DD_continuous',4016))
        b=run('DD_split',dd_command('DD_split',4008))
        b=run('DD_split',dd_command('DD_split',4016,resume=b))
        report['checks']['DD_split_resume']=compare_dd_states(a,b)
        for target in (a,b):
            run_dir=target.parent;manifest=json.loads((run_dir/'engineering_manifest.json').read_text())
            state=torch.load(target,map_location='cpu',weights_only=False)
            dd.validate_resumed_state(state,manifest)
            if state['d_restore']['ema_512_init_count']!=1:raise RuntimeError('EMA reinitialized')
        report['checks']['once_only_E512']={'status':'PASS'}
        # Same original DA branch under original code and new code, identical source and computation state.
        src=next(r for r in inventory if r['seed']==50)
        old_states=[]
        for label,repo in [('DA_original',Path(config['pr108_repo'])),('DA_new_code',p.ROOT)]:
            run_dir=output/label/'run';run_dir.mkdir(parents=True,exist_ok=True)
            manifest=p.old.manifest(50,'DA',src['prefix_path'],run_dir,engineering=True)
            path=run_dir/'manifest.json';legacy.write_json(path,manifest)
            cmd=p.old.command(python=config['runtime_python'],dataset=config['dataset'],transfer=None,output=run_dir,
                              seed=50,history='D',phase='suffix',resume=src['prefix_path'],switch_manifest=path,engineering_stop=4016)
            cmd[5]=str(repo/'ct_train.py');old_states.append(run(label,cmd,cwd=repo))
        report['checks']['old_DA_unchanged']=compare_old_states(*old_states,shadow=True)
        # Four variants, two seeds and two exactly reconstructed source boundaries, fixed 16 attempts each.
        for seed in (50,51):
            for boundary in ('ECT_initial','D_prefix_512'):
                artifacts={}
                for variant in ('A','D','A_scalar','A_lr'):
                    name=f'B/seed{seed}/{boundary}/{variant}';run_dir=output/name/'run';run_dir.mkdir(parents=True,exist_ok=True)
                    if boundary=='D_prefix_512':cmd=dd_command(name,4016,seed=seed)
                    else:
                        cmd=p.old.command(python=config['runtime_python'],dataset=config['dataset'],transfer=config['transfer'],
                            output=run_dir,seed=seed,history='D',phase='prefix',engineering_stop=16)
                        cmd=[v for v in cmd if not v.startswith('--planned-pause-protocol=')]
                    cmd=cmd[:5]+[str(p.ROOT/'analysis/q256_d_restore_vs_hold_v1/audit_entry.py'),
                                 '--variant',variant,'--output',str(output/name),'--']+cmd[6:]
                    state=run(name,cmd)
                    if boundary=='ECT_initial':
                        current=json.loads((state.parent/'initial_state_receipt_v1.json').read_text())
                        source=next(r for r in inventory if r['seed']==seed)
                        reference=Path(source['prefix_path']).parent/'initial_state_receipt_v1.json'
                        hc.validate_initial_receipt(current,json.loads(reference.read_text()))
                    artifacts[variant]=output/name/'first_effective_batch.pt'
                comparison=audit_compare.compare(artifacts)
                dest=output/'B'/f'seed{seed}'/boundary/'comparison.json';legacy.write_json(dest,comparison)
                report['audit_comparisons'][f'{seed}/{boundary}']=str(dest)
        report['status']='PASS'
        report['full_scalar_quality_arm']='NOT_STARTED; numerical divergence never expands the matrix'
    except Exception as exc:
        report.update(status='FAILED',error=repr(exc));raise
    finally:
        report['process_gpuh']=sum(r['process_gpuh'] for r in report['processes'])
        report['implementation_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.ROOT,text=True).strip()
        legacy.write_json(output/'short_checks.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,required=True)
    a=parser.parse_args();execute(json.loads(a.deployment.read_text()),a.output,a.gpu)
