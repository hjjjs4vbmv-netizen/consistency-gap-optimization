"""Immutable science; host allocation is separately bound before each lane starts."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from training import startup_windows as w
from analysis.startup_window_experiments_v1.protocol import (
    ROOT, ENVIRONMENT, EVALUATOR_COMMIT, BLOCKS, DATA_SHA256, TRANSFER_SHA256,
    digest, write, command,
)

PROTOCOL_ID = w.F12
SEEDS = tuple(range(401, 413))
RESERVE_SEEDS = tuple(range(413, 417))
ENGINEERING_SEEDS = (99401,)
ARMS = w.ARMS128
TERMINAL = ('PASS', 'SCIENTIFIC_FAILURE', 'TECHNICAL_FAILURE', 'NO_ENDPOINT')
EQUIVALENCE_BOUND = math.log(1.03)


def specification():
    return dict(protocol_id=PROTOCOL_ID, base_commit='fcc94d26b1085011ac935084f7c1cf4c149e2947',
        cohort=list(SEEDS), reserve_cohort=list(RESERVE_SEEDS), q=256, arms=list(ARMS),
        cohort_identity='prospectively_fixed_response_unselected_fresh_training_seeds',
        initialization='fresh_transfer_same_seed_common_initial_state', reused_arms=[],
        final_kimg=1024, switch_kimg=512, batch=128, microbatch=16, world_size=1,
        attempted_iterations=8000, attempted_images=1024000,
        windows={a:dict(zip(('start_success_step','end_success_step','lr_multiplier'),w.WINDOWS[a])) for a in ARMS},
        statistics=dict(unit='training_seed', Y='mean of three log-FID50k values',
            primary='H=Y_DA-Y_AA', test='two-sided paired t', ci95=True, ci90=True,
            secondary=['M=Y_A_startup_down5-Y_AA','C=Y_D_startup_up5-Y_DA','S=(C-M)/2'],
            secondary_multiplicity='Holm family of three M,C,S; nominal confidence intervals labeled',
            equivalence=dict(test='TOST', alpha=.05, bounds_log=[-EQUIVALENCE_BOUND,EQUIVALENCE_BOUND]),
            primary_missing='all four paths and all three NFE1 blocks valid',
            sensitivity='available AA/DA, AA/A_down, DA/D_up pairs; S needs all four; descriptive only',
            reserve_rule='after all initial 48 paths reach final completion statuses, add all 413-416 iff complete four-arm seed count <9; no metric values inspected; at most one addition',
            conclusion='report direction/significance and equivalence separately; if final n<9 flag underpowered'),
        evaluation=dict(endpoint_kimg=1024, readout='E_512', precision='fp32',
            nfe_primary=1, nfe_descriptive=2, nfe2_mid_t=[.821], blocks=BLOCKS,
            samples_per_block=50000, metric_seed=20260730, metrics=['kid50k_full','fid50k_full'],
            evaluator_commit=EVALUATOR_COMMIT, timing='after each lane finishes its training jobs',
            unblind='once, only after activated cohort is terminal and primary completion/backup decision is sealed'),
        engineering=dict(seeds=list(ENGINEERING_SEEDS), max_attempts=32, no_quality_evaluation=True,
            criteria='configuration, actual LR/success clocks, finite state and exact own-state resume; observed norm-direction is descriptive and never excludes a seed'),
        execution=dict(max_concurrent_gpus=24, gpu_per_trajectory=1,
            allocation='immutable per-lane GPU UUID/host permit; newly supplied hosts fill unassigned lanes',
            retry='no effect-driven reruns; technical continuation only from verified own state with original records retained',
            hard_timeouts_seconds=dict(training=43200,evaluation=7200,engineering=3600)),
        environment=ENVIRONMENT,
        exclusions='all failures retained; no response-based replacement, outlier removal, interim quality look or additional contrast')


def queue(config=None, include_reserve=False):
    # Two lanes per seed: AA/DA share one GPU; the two startup arms share another.
    # Within each pair, order alternates by seed, independently of all outcomes.
    rows=[]
    seeds=SEEDS+RESERVE_SEEDS if include_reserve else SEEDS
    for i,seed in enumerate(seeds):
        for pair,arms in enumerate((ARMS[:2],ARMS[2:])):
            ordered=arms if i%2==0 else arms[::-1]
            lane=2*i+pair
            for order,arm in enumerate(ordered):
                rows.append(dict(protocol_id=PROTOCOL_ID, seed=seed, arm=arm,
                    logical_gpu=lane, order=order, reserve=seed in RESERVE_SEEDS))
    return rows


def manifest(config, group, seed, arm, output, mode='formal'):
    if group!=PROTOCOL_ID:
        raise ValueError('F12 independent identity required')
    start,end,factor=w.WINDOWS[arm]
    reference=Path(config['reference_root'])/'q256'/f'seed{seed}'/'initial_state_receipt_v1.json'
    return w.validate_manifest(dict(protocol_id=PROTOCOL_ID,q=256,seed=seed,arm=arm,
        cohort=list(SEEDS),reserve_cohort=list(RESERVE_SEEDS), mode=mode,
        initialization='fresh_transfer',base_lr=1e-4,final_kimg=1024,switch_kimg=512,
        target_gap_scale=1.0,global_gap_scale=1.0,diagnostic_success_steps=16,
        native_prefix_arm=w.native_prefix(arm),start_success_step=start,end_success_step=end,lr_multiplier=factor,
        max_attempts=32 if mode=='engineering' else 0 if mode=='initialization' else 8000,
        reference_initial_receipt=None if mode=='initialization' else dict(path=str(reference),sha256=digest(reference)),
        dataset_sha256=DATA_SHA256,transfer_sha256=TRANSFER_SHA256,
        immutable_output_root=str(Path(output).resolve()),old_control_bindings=[]))


def evaluation_slots(rows, nfe=1):
    if nfe not in (1,2):
        raise ValueError('only prespecified NFE1 and NFE2')
    return [dict(**r,block=b,sample_seed_start=limits[0],sample_seed_end=limits[1],
        metric_seed=20260730,readout='E_512',precision='fp32',nfe=nfe,
        job_id=f"{PROTOCOL_ID}-seed{r['seed']}-{r['arm']}-E_512-NFE{nfe}-{b}")
        for r in rows for b,limits in BLOCKS.items()]


def evaluation_command(config, slot, snapshot, target):
    from scripts.run_m1_evaluation_job import build_command
    wire={**{k:str(v) for k,v in slot.items()},'nfe':'1','slot_id':slot['job_id'],
          'metrics':'kid50k_full,fid50k_full'}
    cmd=build_command(wire,str(snapshot),Path(config['evaluation_dataset']),Path(target),
        Path(config['evaluator_repo']),Path(config['runtime_python']),47000+slot['logical_gpu'])
    if slot['nfe']==2:
        cmd=[('--nfe=2' if value=='--nfe=1' else value) for value in cmd]
        cmd.append('--mid_t=0.821')
    return cmd
