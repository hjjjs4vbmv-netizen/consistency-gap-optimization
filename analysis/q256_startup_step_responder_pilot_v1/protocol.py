"""Immutable scientific matrix and commands; no computation on import."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'q256_startup_step_responder_pilot_v1'
BASE = 'd9e0021f7e6a8049a642f81bd22390aec0b383ac'
EVALUATOR_COMMIT = 'd6aba02fb88e9db0993623895eb2228ed717d810'
SEEDS = tuple(range(50,58))
ARMS = ('D_startup_up5','A_startup_down5')
ENVIRONMENT = dict(python='3.11.13',torch='2.6.0+cu124',torch_cuda='12.4',cudnn=90100,numpy='2.1.2',scipy='1.16.1')
SPEC = dict(protocol_id=PROTOCOL, experiment_class='responder_enriched_pilot',
    experiment_identity='responder-enriched exploratory mechanism pilot', base_commit=BASE,
    seeds=list(SEEDS),arms=list(ARMS),initialization='fresh_transfer',base_lr=0.0001,
    startup_success_steps=5, switch_kimg=512,final_kimg=1024,
    primary='S=(C-M)/2',simple_effects={'C':'Y_D_startup_up5-Y_DA','M':'Y_A_startup_down5-Y_AA'},
    Y='mean(log(FID_B0),log(FID_B1),log(FID_B2))',directions={'S':'>0','C':'>0','M':'<0'},
    evaluation=dict(kimg=1024,readout='E_512',precision='fp32',nfe=1,metrics=['fid50k_full','kid50k_full'],
        metric_seed=20260730,metric_repeats=1,evaluator_commit=EVALUATOR_COMMIT,
        blocks={'B0':[0,49999],'B1':[50000,99999],'B2':[100000,149999]}),
    statistics=dict(unit='training_seed',primary='S',ci='nominal two-sided 95% t',
        simple_effect_tests='two-sided paired t, Holm across C and M',
        equivalence='TOST alpha .05 with bounds +/-log(1.03)',sign_flip='all 2^n paired sign flips'),
    budget=dict(cap_gpuh=90,evaluation_reserve_gpuh=8.5,engineering_reserve_gpuh=2.5,
        accounting='actual process wall time; rental idle recorded separately'),
    allocation={'matpool':list(range(50,56)),'ect':[56,57]},
    preflight=dict(seed=50,max_attempts=64,gpuh_target=3,no_fid=True),
    terminal_statuses=['PASS','SCIENTIFIC_FAILURE','NO_ENDPOINT','TECHNICAL_FAILURE','INCOMPLETE_BUDGET'],
    no_protocol_adaptation=True, no_training_fid=True)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if immutable and path.exists():
        if path.read_text()!=text:raise RuntimeError('immutable artifact changed: '+str(path))
        return
    if immutable:
        with path.open('x') as f:f.write(text)
    else:
        import os,tempfile
        fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)


def queue():
    return [dict(seed=s,logical_gpu=s-50,host='matpool' if s<56 else 'ect',
                 local_gpu=s-50 if s<56 else s-56,arm=a,order=i,status='PENDING')
            for s in SEEDS for i,a in enumerate(ARMS if s%2==0 else ARMS[::-1])]


def slots():
    return [dict(seed=q['seed'],path=q['arm'],arm=q['arm'],readout='E_512',block=b,
                 sample_seed_start=limits[0],sample_seed_end=limits[1],metric_seed=20260730,
                 precision='fp32',nfe=1,job_id=f"seed{q['seed']}-{q['arm']}-E_512-{b}",status='PENDING')
            for q in queue() for b,limits in SPEC['evaluation']['blocks'].items()]


def old_controls():
    source=ROOT/'analysis/q256_history_component_chase_v1/results/quality_slots_320.json'
    rows=[r for r in json.loads(source.read_text()) if r['seed'] in SEEDS and r['path'] in ('AA','DA') and r['readout']=='E_512']
    if len(rows)!=48 or {(r['seed'],r['path'],r['block']) for r in rows}!={(s,a,b) for s in SEEDS for a in ('AA','DA') for b in ('B0','B1','B2')}:
        raise RuntimeError('old controls matrix differs')
    if any(r['status']!='PASS' for r in rows):raise RuntimeError('unexpected old control outcome')
    return rows


def manifest(config,seed,arm,output,preflight=False):
    from training.startup_quality import FIXED,MULTIPLIERS,validate_manifest
    from training.startup_update import DATA_SHA256,TRANSFER_SHA256
    receipt=Path(config['reference_root'])/f'seed{seed}'/'initial_state_receipt_v1.json'
    bindings=json.loads(Path(config['old_control_bindings']).read_text())
    m=dict(FIXED,seed=seed,arm=arm,native_prefix_arm='D' if arm==ARMS[0] else 'A',
        lr_multiplier=MULTIPLIERS[arm],transfer_sha256=TRANSFER_SHA256,dataset_sha256=DATA_SHA256,
        reference_initial_receipt=dict(path=str(receipt),sha256=digest(receipt)),
        immutable_output_root=str(Path(output).resolve()),
        old_control_bindings=[x for x in bindings if x['seed']==seed],preflight_only=preflight)
    return validate_manifest(m)


def command(config,m,path,resume=None,stop_success=None):
    denominator=1.1 if m['native_prefix_arm']=='D' else 1.0
    cmd=[config['runtime_python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(ROOT/'ct_train.py'),
         f"--data={config['dataset']}",f"--outdir={m['immutable_output_root']}",'--nosubdir',
         '--cond=False','--arch=ddpmpp','--precond=ect','--batch=128','--batch-gpu=16',
         '--optim=RAdam','--lr=0.0001','--dropout=0.2','--augment=0','--xflip=False',
         '--mean=-1.1','--std=2.0','--mapping=sigmoid','--global-gap-scale=1.0',
         '--factorial-protocol=q256_target_weight_v1','--target-gap-scale=1.0',f'--denominator-gap-scale={denominator}',
         '-q','256','-k','8','-b','1','-c','0','--double=10000','--ema_beta=0.9993',f"--seed={m['seed']}",
         '--fp16=True','--tf32=False','--ls=1.0','--enable_amp=True','--bench=False','--cache=True','--workers=1',
         '--metrics=none','--duration=1.024','--tick=10','--snap=0','--dump=0','--ckpt=10','--sample_every=26',
         '--eval_every=50','--mid_t=0.821','--adaptive-update-kimg=0.5','--immutable-checkpoint-kimg=512,1024',
         f'--startup-quality-manifest={path}']
    cmd += [f'--resume={resume}'] if resume else [f"--transfer={config['transfer']}"]
    if stop_success is not None:cmd += [f'--quality-preflight-stop-success={stop_success}']
    return cmd
