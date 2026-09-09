"""Frozen engineering matrix and native training command. No GPU work on import."""
from pathlib import Path
PROTOCOL='q256_startup_update_check_v1'
ARMS=('A','D','D_compensate','A_mimic')
BASE='092ab9ab54e3f2dd780654b752bf85643d92af59'
EXPECTED_ENV=dict(python='3.11.13',torch='2.6.0+cu124',cuda='12.4',cudnn=90100,numpy='2.1.2',scipy='1.16.1')

def native_cli(manifest):
    if manifest['seed'] not in (50,51) or manifest['arm'] not in ARMS:
        raise ValueError('outside frozen matrix')
    denominator=1.1 if manifest['arm'] in ('D','D_compensate') else 1.0
    return [f"--data={manifest['dataset']}",f"--outdir={manifest['output']}",'--nosubdir',
        '--cond=False','--arch=ddpmpp','--precond=ect','--batch=128','--batch-gpu=16',
        '--optim=RAdam','--lr=0.0001','--dropout=0.2','--augment=0','--xflip=False',
        '--mean=-1.1','--std=2.0','--mapping=sigmoid','--global-gap-scale=1.0',
        '--factorial-protocol=q256_target_weight_v1','--target-gap-scale=1.0',
        f'--denominator-gap-scale={denominator}','-q','256','-k','8','-b','1','-c','0',
        '--double=10000','--ema_beta=0.9993',f"--seed={manifest['seed']}",
        '--fp16=True','--tf32=False','--ls=1.0','--enable_amp=True','--bench=False',
        '--cache=True','--workers=1','--metrics=none','--duration=1.024','--tick=10',
        '--snap=0','--dump=0','--ckpt=10','--sample_every=26','--eval_every=50',
        '--mid_t=0.821','--adaptive-update-kimg=0.5','--immutable-checkpoint-kimg=512',
        '--stop-after-attempts=64',f'--planned-pause-protocol={PROTOCOL}',
        f"--transfer={manifest['transfer']}",f"--startup-check-manifest={manifest['manifest_path']}"]
