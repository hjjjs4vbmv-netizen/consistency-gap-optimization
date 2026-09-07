"""Compare actual bounded trajectories and complete restart states."""
import csv
import json
from pathlib import Path

import torch

from training import m1, state_interventions as interventions

TIMING = {'elapsed_sec', 'gpu_hours_cumulative'}
IDENTITY = {'experiment_protocol', 'branch', 'schema'}


def observations(directory):
    return {row['attempted_iteration']: row for row in
            map(json.loads, (Path(directory) / 'first64.jsonl').read_text().splitlines())}


def compare_inputs(own, intervention, end):
    left, right = observations(own), observations(intervention)
    for attempt, row in right.items():
        if attempt > end:
            continue
        for key in ('input_rng_before', 'batch_sha256', 't_sha256', 'base_r_sha256'):
            if row[key] != left[attempt][key]:
                raise ValueError(f'input alignment differs at {attempt}: {key}')


def compare_observations(continuous, split):
    left, right = observations(continuous), observations(split)
    if left.keys() != right.keys():
        raise ValueError('observer attempt sets differ')
    for attempt, row in left.items():
        if any(value != right[attempt][key] for key, value in row.items() if key not in TIMING):
            raise ValueError(f'observer resume differs at {attempt}')


def read_rows(path):
    with Path(path).open() as handle:
        return {int(row['attempted_iteration']): row for row in csv.DictReader(handle)}


def compare_telemetry(original, current, start, end, *, same_branch=False):
    left, right = read_rows(original), read_rows(current)
    ignored = TIMING if same_branch else TIMING | IDENTITY
    for attempt in range(start + 1, end + 1):
        if attempt not in left or attempt not in right:
            raise ValueError(f'missing replay row at attempt {attempt}')
        differences = [key for key in left[attempt] if key not in ignored
                       and left[attempt][key] != right[attempt].get(key)]
        if differences:
            raise ValueError(f'replay mismatch at {attempt}: {differences}')


def compare_states(left_path, right_path):
    left = torch.load(left_path, map_location='cpu', weights_only=False)
    right = torch.load(right_path, map_location='cpu', weights_only=False)
    keys = ('optimizer_state', 'gradscaler_state', 'rank_states', 'loss_fn_state',
            'attempted_iteration', 'successful_optimizer_steps', 'cur_nimg',
            'cur_tick', 'tick_start_nimg', 'snapshot_grid_z', 'snapshot_grid_c',
            'snapshot_grid_size', 'factorial', 'trajectory_config', 'trajectory_config_sha256')
    for key in keys:
        if not m1._equal_state(left[key], right[key]):
            raise ValueError(f'continuous vs resume state mismatch: {key}')
    for key in ('net', 'ema', 'ema_512'):
        if not m1._equal_state(left[key].state_dict(), right[key].state_dict()):
            raise ValueError(f'continuous vs resume module mismatch: {key}')
    return True


def check_initialization(path, manifest, *, noop=False):
    source = torch.load(manifest['source_state']['path'], map_location='cpu', weights_only=False)
    state = torch.load(path, map_location='cpu', weights_only=False)
    late = manifest['experiment_protocol'] == interventions.M2
    for current, original in (('net', 'net'), ('ema', 'ema'),
                              ('ema_512', 'ema_512' if late else 'net')):
        if not m1._equal_state(state[current].state_dict(), source[original].state_dict()):
            raise ValueError(f'initialization changed {current}')
    for key in ('gradscaler_state', 'rank_states', 'loss_fn_state', 'cur_nimg',
                'attempted_iteration', 'successful_optimizer_steps', 'cur_tick',
                'tick_start_nimg', 'snapshot_grid_z', 'snapshot_grid_c', 'snapshot_grid_size', 'factorial'):
        if not m1._equal_state(state[key], source[key]):
            raise ValueError(f'initialization changed receiver {key}')
    if not m1._equal_state(state['optimizer_state']['param_groups'], source['optimizer_state']['param_groups']):
        raise ValueError('initialization changed receiver parameter groups')
    if noop:
        if not m1._equal_state(state['optimizer_state'], source['optimizer_state']):
            raise ValueError('own-state operation changed optimizer')
    elif late:
        if state['optimizer_state']['state']:
            raise ValueError('late reset did not clear optimizer state')
    else:
        donor = torch.load(manifest['donor_state']['path'], map_location='cpu', weights_only=False)
        expected = interventions.named_optimizer_state(donor['net'], donor['optimizer_state'])
        actual = interventions.named_optimizer_state(state['net'], state['optimizer_state'])
        for name in expected:
            if not m1._equal_state(expected[name][1], actual[name][1]):
                raise ValueError(f'transplanted donor state differs: {name}')
