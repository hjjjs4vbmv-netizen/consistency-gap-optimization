"""Assemble all fixed slots and outcomes. Missing technical work cannot become scientific failure."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import protocol as p, analyze


def collect(deployments,output):
    configs=[json.loads(Path(x).read_text()) for x in deployments]
    seeds=[s for c in configs for s in c['assigned_seeds']]
    if sorted(seeds)!=list(p.SEEDS):raise ValueError('expected disjoint complete 16-seed deployment')
    slots=[];outcomes=[];cost=[]
    for seed in p.SEEDS:
        c=next(c for c in configs if seed in c['assigned_seeds'])
        stage=Path(c['output_root'])/f'S{seed-49:02d}/DD/suffix/stage_status.json'
        state=json.loads(stage.read_text()) if stage.exists() else {'status':'PENDING'}
        outcomes.append({'seed':seed,'DA':'PASS','DD':state['status'],'AA':'NO_ENDPOINT' if seed in (58,65) else 'PASS',
                         **{k:state.get(k) for k in ('end_attempt','successful_optimizer_steps','process_gpuh','error')}})
        for slot in (r for r in p.evaluation_slots() if r['seed']==seed):
            path=Path(c['evaluation_output'])/'receipts'/(slot['job_id']+'.json')
            if path.exists():
                row=json.loads(path.read_text())
                for k in ('seed','path','readout','block','job_id'):
                    if row[k]!=slot[k]:raise ValueError('receipt identity mismatch')
            elif state['status']=='SCIENTIFIC_FAILURE':
                row={**slot,'status':'NO_ENDPOINT','reason':'prespecified DD scientific failure'}
            else:row=slot
            slots.append(row)
    for c in configs:
        b=json.loads(Path(c['budget_ledger']).read_text())
        cost.append({'assigned_seeds':c['assigned_seeds'],'cap_gpuh':b['cap_gpuh'],
                     'process_gpuh':sum(a['process_gpuh'] for r in b['processes'].values() for a in r.get('attempts',[])),
                     'active_process_count':sum(r['status']=='RUNNING' for r in b['processes'].values())})
    result=analyze.summarize(slots)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,data in [('DD_80',slots),('DA_DD_160',p.import_controls()+slots),('AA_auxiliary_80',p.import_controls('AA')),
                      ('outcomes_16',outcomes),('statistics',result),('node_costs',cost)]:
        (output/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    for name,data in [('outcomes_16',outcomes),('metrics_160',p.import_controls()+slots),('per_seed',result['per_seed'])]:
        if data:p.write_csv(output/(name+'.csv'),data)
    return result


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--deployment',action='append',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);x=a.parse_args();result=collect(x.deployment,x.output);print(result['status'])
