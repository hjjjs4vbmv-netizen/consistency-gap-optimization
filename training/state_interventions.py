"""One-time late RAdam restart and cross-history state transplant."""
import copy
import json
from pathlib import Path

import torch

SEEDS = (55, 56, 59, 60, 61, 62)
M2 = 'm2_restart_timing_q256_v2'
SWAP = 'optimizer_donor_swap_q256_v1'
PROTOCOLS = (M2, SWAP)
BRANCHES = {'L_A': ('A', 'A'), 'L_B': ('B', 'A'),
            'X_A_from_B': ('A', 'A'), 'X_B_from_A': ('B', 'A')}


def enabled(manifest):
    return manifest is not None and manifest.get('experiment_protocol') in PROTOCOLS


def boundary(manifest):
    return 768000 if manifest.get('experiment_protocol') == M2 else 512000


def validate_manifest(manifest):
    protocol, branch = manifest['experiment_protocol'], manifest['branch']
    if branch not in (('L_A', 'L_B') if protocol == M2 else ('X_A_from_B', 'X_B_from_A')):
        raise ValueError('intervention protocol/branch mismatch')
    if manifest['seed'] not in SEEDS:
        raise ValueError('seed outside fixed intervention cohort')
    if protocol == SWAP and not manifest.get('donor_state', {}).get('path', '').startswith('/'):
        raise ValueError('SWAP requires its original donor path')


def verify_late_source(state, manifest):
    from training import m1, schedule_switch
    original = state.get('m1', {})
    arm = manifest['origin_arm']
    if original.get('protocol_id') != m1.PROTOCOL_ID or original.get('branch') != 'K_' + arm:
        raise ValueError('late restart requires an original K state, never R/repair')
    if state.get('cur_nimg') != 768000 or state.get('attempted_iteration') != 6000:
        raise ValueError('late source is not exactly 768 kimg / attempt 6000')
    if state['trajectory_config']['seed'] != manifest['seed']:
        raise ValueError('late source seed mismatch')
    original_manifest = dict(manifest, experiment_protocol=m1.PROTOCOL_ID,
                             branch='K_' + arm,
                             source_state={'path': original['source_path']})
    m1.validate_resumed_state(state, original_manifest)
    schedule_switch.verify_switched_state(state, original_manifest)
    if state['rank_states'][0]['sampler_state']['consumed_samples'] != 768000:
        raise ValueError('late source sampler cursor mismatch')
    if len(state['rank_states']) != 1 or state['trajectory_config']['network_kwargs'].get('use_fp16') is not True:
        raise ValueError('late source must retain original single-rank FP16 trajectory')


def metadata(manifest, successful_steps):
    late = manifest['experiment_protocol'] == M2
    return dict(protocol_id=manifest['experiment_protocol'], branch=manifest['branch'],
                seed=manifest['seed'], source_path=manifest['source_state']['path'],
                initialized_at_nimg=boundary(manifest), reset_count=int(late),
                transplant_applied_count=int(not late),
                donor_path=None if late else manifest['donor_state']['path'],
                initialized_emas=[] if late else ['E_512'],
                shadow_update_enabled=True, successful_steps_at_init=successful_steps,
                successful_steps_since_init=0)


def named_optimizer_state(model, optimizer_state):
    # The audited training constructor passes net.parameters() as a single group.
    groups = optimizer_state['param_groups']
    parameters = list(model.named_parameters())
    if len(groups) != 1 or len(groups[0]['params']) != len(parameters):
        raise ValueError('optimizer does not match the audited single-group constructor')
    ids = groups[0]['params']
    if len(set(ids)) != len(ids) or set(optimizer_state['state']) != set(ids):
        raise ValueError('duplicate, missing or unexplained optimizer parameter states')
    result = {}
    for (name, parameter), index in zip(parameters, ids):
        state = optimizer_state['state'][index]
        if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError(f'unexpected RAdam state keys for {name}')
        if any(state[key].shape != parameter.shape for key in ('exp_avg', 'exp_avg_sq')):
            raise ValueError(f'moment shape mismatch: {name}')
        if state['step'].numel() != 1:
            raise ValueError(f'non-scalar RAdam step: {name}')
        result[name] = (parameter, state)
    return result


def transplant(optimizer, model, donor_model, donor_optimizer_state):
    donor = named_optimizer_state(donor_model, donor_optimizer_state)
    receiver_names = dict(model.named_parameters())
    receiver_parameters = [p for group in optimizer.param_groups for p in group['params']]
    if (len(optimizer.param_groups) != 1 or set(receiver_names) != set(donor)
            or len(receiver_parameters) != len(receiver_names)
            or {id(p) for p in receiver_parameters} != {id(p) for p in receiver_names.values()}):
        raise ValueError('receiver/donor parameter mapping is not one-to-one')
    replacements = {}
    for name, parameter in receiver_names.items():
        donor_parameter, state = donor[name]
        if donor_parameter.shape != parameter.shape or donor_parameter.dtype != parameter.dtype:
            raise ValueError(f'receiver/donor parameter signature mismatch: {name}')
        own = optimizer.state[parameter]
        if set(own) != set(state):
            raise ValueError(f'receiver optimizer state mismatch: {name}')
        replacement = {}
        for key, value in state.items():
            if own[key].dtype != value.dtype:
                raise ValueError(f'receiver/donor state dtype mismatch: {name}/{key}')
            replacement[key] = value.detach().to(device=own[key].device).clone()
        replacements[parameter] = replacement
    optimizer.state.clear()
    optimizer.state.update(replacements)
    return sorted(receiver_names)


def apply(optimizer, model, manifest):
    from training import reproducibility, schedule_switch
    from training.m1_optimizer_split import state_summary
    if optimizer.__class__.__name__ != 'RAdam':
        raise ValueError('state interventions require RAdam')
    groups = copy.deepcopy(optimizer.state_dict()['param_groups'])
    summary = {'receiver_before': state_summary(optimizer)}
    if manifest['experiment_protocol'] == M2:
        optimizer.state.clear()
    else:
        rng = reproducibility.capture_rng_state()
        try:
            donor = torch.load(manifest['donor_state']['path'], map_location='cpu', weights_only=False)
        finally:
            reproducibility.restore_rng_state(rng)
        donor_arm = 'B' if manifest['origin_arm'] == 'A' else 'A'
        donor_manifest = dict(manifest, origin_arm=donor_arm,
                              source_state=manifest['donor_state'])
        schedule_switch.verify_source_state(donor, donor_manifest)
        summary['parameter_mapping'] = transplant(optimizer, model, donor['net'], donor['optimizer_state'])
        summary['donor_successful_optimizer_steps'] = donor['successful_optimizer_steps']
    if optimizer.state_dict()['param_groups'] != groups:
        raise RuntimeError('intervention changed receiver parameter groups')
    summary['after'] = state_summary(optimizer)
    return summary


def record_attempt(run_dir, optimizer, telemetry, rng_before):
    steps = sorted({float(state['step']) for state in optimizer.state.values()})
    record = dict(telemetry, optimizer_step_values=steps,
                  ema_attempt=telemetry['attempted_iteration'],
                  input_rng_before=rng_before)
    with (Path(run_dir) / 'first64.jsonl').open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
