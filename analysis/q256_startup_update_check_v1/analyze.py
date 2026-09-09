"""Offline CPU analysis. Never performs training or loads a CUDA tensor onto GPU."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.q256_startup_update_check_v1.protocol import ARMS

def compare(a,b,key):
    aa,bb=a[key],b[key]
    if a['parameter_order']!=b['parameter_order']:raise ValueError('parameter order mismatch')
    na=nb=dot=error=0.
    for x,y in zip(aa,bb):
        if x is None or y is None:
            if x is not y:raise ValueError('gradient support mismatch')
            continue
        x=x.double().reshape(-1);y=y.double().reshape(-1)
        na+=float(x.square().sum());nb+=float(y.square().sum());dot+=float(x.dot(y));error+=float((x-y).square().sum())
    return dict(candidate_norm=math.sqrt(na),reference_norm=math.sqrt(nb),absolute_l2_error=math.sqrt(error),
                norm_ratio=math.sqrt(na/nb) if nb else None,
                cosine=dot/math.sqrt(na*nb) if na and nb else None,
                relative_l2_error=math.sqrt(error/nb) if na and nb else None)
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def main(root,output):
    output.mkdir(parents=True,exist_ok=True)
    status=json.loads((root/'control/run_status.json').read_text())
    runs={}
    for row in status:
        directory=root/'runs'/row['slot'];p=directory/'startup_attempts.jsonl'
        records=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        runs[(row['seed'],row['arm'])]=records
    comparisons=[];pairing=[];attempt_ratios=[]
    for seed in (50,51):
        initial=[]
        for arm in ARMS:
            p=root/f'runs/seed{seed}-{arm}/initial_state_receipt_v1.json'
            if p.exists():initial.append(json.loads(p.read_text()))
        pairing.append(dict(seed=seed,initial_receipts=len(initial),
                            all_four_initial_hashes_equal=len(initial)==4 and all(x['hashes']==initial[0]['hashes'] for x in initial)))
        for arm,ref in [('D_compensate','A'),('A_mimic','D'),('D','A')]:
            ca={r['attempted_index']:r for r in runs[(seed,arm)]};cb={r['attempted_index']:r for r in runs[(seed,ref)]}
            for attempt in sorted(ca.keys()&cb.keys()):
                a,b=ca[attempt],cb[attempt]
                bn=float(b['update_norm']);an=float(a['update_norm'])
                attempt_ratios.append(dict(seed=seed,candidate=arm,reference=ref,alignment='same_attempt',attempt=attempt,
                    norm_ratio=an/bn if bn else None,skip_equal=a['skip']==b['skip'],clocks_equal=a['clocks_after']==b['clocks_after'],
                    random_fields_equal=all(a[k]==b[k] for k in ('batch_sha256','t_sha256','base_r_sha256','rng_before','rng_after'))))
            for step in (1,5,6):
                ap=root/f'runs/seed{seed}-{arm}/update-success-{step:02d}.pt'
                bp=root/f'runs/seed{seed}-{ref}/update-success-{step:02d}.pt'
                if not ap.exists() or not bp.exists():
                    comparisons.append(dict(seed=seed,candidate=arm,reference=ref,successful_step=step,status='TENSOR_UNAVAILABLE'));continue
                a=torch.load(ap,map_location='cpu',weights_only=False);b=torch.load(bp,map_location='cpu',weights_only=False)
                ai=a['metadata']['attempted_index'];bi=b['metadata']['attempted_index']
                ar,br=ca[ai],cb[bi]
                row=dict(seed=seed,candidate=arm,reference=ref,successful_step=step,alignment='same_successful_step',
                    candidate_attempt=ai,reference_attempt=bi,also_same_attempt=ai==bi,status='MEASURED',
                    random_fields_equal=all(ar[k]==br[k] for k in ('batch_sha256','t_sha256','base_r_sha256','rng_before','rng_after')),
                    clocks_equal=ar['clocks_after']==br['clocks_after'],
                    updates=compare(a,b,'updates'),parameters_before=compare(a,b,'parameters_before'),gradients=compare(a,b,'gradients'))
                row['same_state_local_test']=(row['parameters_before']['absolute_l2_error']==0 and row['random_fields_equal'])
                row['interpretation']='common-state local comparison' if row['same_state_local_test'] else 'diverged trajectory comparison; not a common-state counterfactual'
                comparisons.append(row);del a,b
    dump(output/'comparisons.json',comparisons);dump(output/'initial_pairing.json',pairing);dump(output/'same_attempt_norm_ratios.json',attempt_ratios)
    checks=[]
    for status_row in status:
        records=runs[(status_row['seed'],status_row['arm'])];errors=[]
        for r in records:
            if r['optimizer_step_called']==r['skip']:errors.append('skip/call mismatch')
            if r['skip'] and r['clocks_before']!=r['clocks_after']:errors.append('skip advanced clock')
            if len(set(r['clocks_after']))>1:errors.append('clock disagreement')
            if set(r['clocks_after'])!={r['successful_optimizer_steps']}:errors.append('success counter disagreement')
            for p in r['actual_step_plan']:
                expected=(1.1 if r['arm']=='D_compensate' else 1/1.1 if r['arm']=='A_mimic' else 1) if p['step']<=5 else 1
                if p['multiplier']!=expected or p['base_lr']!=1e-4:errors.append('LR mismatch')
            if r['optimizer_step_called'] and any(v!=1e-4 for v in r['restored_lr']):errors.append('LR not restored')
        checks.append(dict(**status_row,engineering_errors=sorted(set(errors)),engineering_pass=len(records)==64 and not errors))
    dump(output/'engineering_checks.json',checks)
    with (output/'key_updates.csv').open('w') as f:
        rows=[{k:r[k] for k in ['seed','candidate','reference','successful_step','candidate_attempt','reference_attempt','also_same_attempt','same_state_local_test']}|r['updates'] for r in comparisons if r['status']=='MEASURED']
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for seed in (50,51):
        for arm in ARMS:
            rr=runs[(seed,arm)];label=f'{seed} {arm}';success=[r for r in rr if r['optimizer_step_called']]
            axs[0,0].plot([r['successful_optimizer_steps'] for r in success],[r['actual_step_plan'][0]['multiplier'] for r in success],label=label,alpha=.8)
            axs[0,1].plot([r['attempted_index'] for r in rr],[float(r['update_norm']) for r in rr],label=label,alpha=.8)
            axs[1,1].step([r['attempted_index'] for r in rr],[r['scale_after'] for r in rr],where='post',label=label)
            skipped=[r for r in rr if r['skip']]
            axs[1,1].scatter([r['attempted_index'] for r in skipped],[r['scale_after'] for r in skipped],marker='x',s=20)
        for arm,ref in [('D_compensate','A'),('A_mimic','D'),('D','A')]:
            rr=[r for r in comparisons if r['seed']==seed and r['candidate']==arm and r['reference']==ref and r['status']=='MEASURED']
            axs[1,0].plot([r['successful_step'] for r in rr],[r['updates']['relative_l2_error'] for r in rr],'-o',label=f'{seed} {arm}/{ref}')
    for ax,title,x,y in zip(axs.flat,['Applied LR at actual RAdam.step','Actual parameter update norm','Update vector relative L2 error','AMP scale; x = skipped attempt'],
                            ['Successful step','Attempt','Successful step','Attempt'],['LR multiplier','L2 norm','Relative L2 error','GradScaler scale']):
        ax.set(title=title,xlabel=x,ylabel=y);ax.grid(alpha=.2)
    axs[0,0].set_xlim(.5,8.5);axs[0,1].set_yscale('symlog',linthresh=1e-5);axs[1,1].set_yscale('log',base=2)
    axs[0,0].legend(fontsize=7,ncol=2);axs[1,0].legend(fontsize=7)
    fig.suptitle('Engineering-only startup check: 2 seeds x 4 arms x 64 attempts')
    fig.savefig(output/'startup_diagnostics.png',dpi=180);fig.savefig(output/'startup_diagnostics.pdf');plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
