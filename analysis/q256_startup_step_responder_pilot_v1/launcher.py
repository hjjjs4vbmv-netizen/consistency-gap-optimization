"""Launch the eight fixed logical GPU workers across an explicitly supplied SSH inventory."""
import argparse,json,shlex,subprocess
from . import protocol as p


def commands(inventory):
    rows=[]
    for slot in p.queue():
        if slot['order']!=0:continue
        host=inventory[slot['host']]
        worker=[host['runtime_python'],'-m','analysis.q256_startup_step_responder_pilot_v1.worker',
                '--deployment',host['deployment'],'--seed',str(slot['seed']),'--gpu',str(slot['logical_gpu'])]
        script='cd '+shlex.quote(host['code'])+'\nnohup '+shlex.join(worker)+' > '+shlex.quote(host['log_root']+'/worker-seed'+str(slot['seed'])+'.log')+' 2>&1 < /dev/null &'
        rows.append(dict(seed=slot['seed'],logical_gpu=slot['logical_gpu'],host=slot['host'],command=host['ssh_argv']+[script]))
    assert len(rows)==8
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inventory',required=True);parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    with open(a.inventory) as f:rows=commands(json.load(f))
    if a.execute:
        for row in rows:subprocess.run(row['command'],check=True)
    else:print(json.dumps(rows,indent=2))
