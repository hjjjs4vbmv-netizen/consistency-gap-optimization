"""Isolated 64-attempt startup engineering protocol; never a quality result."""
from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from training import reproducibility as repro, history_component

PROTOCOL = 'q256_startup_update_check_v1'
ARMS = ('A', 'D', 'D_compensate', 'A_mimic')
DATA_SHA256 = '9818e4b801a52eac437485bc8a69e40b54e9ae9c5d1427467343c91de868f1b3'
TRANSFER_SHA256 = '4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_pause(*, seed, attempts, total_kimg, resume_state_dump,
                   schedule_switch_manifest, manifest):
    if (type(seed) is not int or seed not in (50, 51) or type(attempts) is not int
            or attempts != 64 or total_kimg != 1024 or resume_state_dump is not None
            or schedule_switch_manifest is not None or not isinstance(manifest, dict)
            or manifest.get('protocol') != PROTOCOL or manifest.get('arm') not in ARMS
            or manifest.get('seed') != seed or manifest.get('attempts') != 64
            or manifest.get('initialization') != 'fresh_transfer'
            or manifest.get('engineering_only') is not True):
        raise ValueError('startup check requires fixed seed50/51, fresh initialization, 64 attempts, 1024 kimg and engineering manifest')


def read_manifest(path):
    return json.loads(Path(path).read_text()) if path is not None else None


def validate_config(manifest, config, transfer):
    ref_path = manifest['reference_receipt']
    if file_hash(ref_path) != manifest['reference_receipt_sha256']:
        raise ValueError('archived initial receipt hash mismatch')
    ref = json.loads(Path(ref_path).read_text())
    if ref['attempted_iteration'] != 0 or ref['processed_nimg'] != 0:
        raise ValueError('reference is not the original training start')
    # The historical comparator permits only loss factors and filesystem paths.
    from training.schedule_switch import trajectory_configs_compatible
    if not trajectory_configs_compatible(ref['trajectory_config'], config, {'final_kimg': 1024}):
        raise ValueError('startup scientific configuration differs from original receipt')
    factor = 1.1 if manifest['arm'] in ('D', 'D_compensate') else 1.0
    if (config['loss_kwargs']['target_gap_scale'] != 1.0
            or config['loss_kwargs']['denominator_gap_scale'] != factor):
        raise ValueError('startup arm/native loss mismatch')
    if file_hash(config['dataset_kwargs']['path']) != DATA_SHA256:
        raise ValueError('original training archive hash mismatch')
    if not transfer or file_hash(transfer) != TRANSFER_SHA256:
        raise ValueError('original transfer hash mismatch')


def rho(step, beta2):
    infinity = 2 / (1 - beta2) - 1
    return infinity - 2 * step * beta2 ** step / (1 - beta2 ** step)


def clocks(optimizer, *, active_only=False):
    return [int(optimizer.state.get(p, {}).get('step', 0))
            for group in optimizer.param_groups for p in group['params']
            if not active_only or p.grad is not None]


def step_plan(optimizer, arm):
    if type(optimizer) is not torch.optim.RAdam or arm not in ARMS:
        raise ValueError('native RAdam and a fixed startup arm are required')
    active = clocks(optimizer, active_only=True)
    if not active or len(set(active)) != 1:
        raise RuntimeError('active parameter clocks disagree (no clock alignment is allowed): ' + repr(active))
    plans = []
    for group in optimizer.param_groups:
        if tuple(group['betas']) != (.9, .999) or group['eps'] != 1e-8 or group['weight_decay'] != 0:
            raise ValueError('RAdam hyperparameters differ from frozen protocol')
        next_step = active[0] + 1
        value = rho(next_step, group['betas'][1])
        multiplier = (1.1 if arm == 'D_compensate' else 1 / 1.1 if arm == 'A_mimic' else 1.0) if value <= 5 else 1.0
        plans.append(dict(step=next_step, rho=value, branch='rectified' if value > 5 else 'non_adaptive',
                          multiplier=multiplier, base_lr=group['lr'], lr=group['lr'] * multiplier))
    return plans


@contextmanager
def temporary_lr(optimizer, arm):
    """Enter only from an actual optimizer.step call. Never touch grads/moments."""
    rates = [group['lr'] for group in optimizer.param_groups]
    plans = step_plan(optimizer, arm)
    try:
        for group, plan in zip(optimizer.param_groups, plans):
            group['lr'] = plan['lr']
        yield plans
    finally:
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate


class Observer:
    def __init__(self, manifest, net, optimizer, run_dir):
        self.manifest, self.net, self.optimizer = manifest, net, optimizer
        self.directory = Path(run_dir)
        current = json.loads((self.directory / 'initial_state_receipt_v1.json').read_text())
        reference = json.loads(Path(manifest['reference_receipt']).read_text())
        check = history_component.validate_initial_receipt(current, reference)
        if current['attempted_iteration'] != 0 or any(clocks(optimizer)) or optimizer.state:
            raise ValueError('startup observer rejects initialized optimizer history')
        repro.atomic_json_dump({**check, **manifest}, self.directory / 'startup_initial_check.json')
        self.order = [dict(name=n, shape=list(p.shape), dtype=str(p.dtype)) for n, p in net.named_parameters()]
        repro.atomic_json_dump(self.order, self.directory / 'parameter_order.json')
        original_step = optimizer.step
        self.calls = 0

        def actual_step(*args, **kwargs):
            self.calls += 1
            self.event['optimizer_step_called'] = True
            self.event['clocks_at_step_entry'] = clocks(optimizer)
            try:
                with temporary_lr(optimizer, manifest['arm']) as plans:
                    self.event['actual_step_plan'] = plans
                    result = original_step(*args, **kwargs)
                self.event['clocks_at_step_exit'] = clocks(optimizer)
                return result
            except BaseException as exc:
                self.event['error'] = repr(exc)
                raise
            finally:
                self.event['restored_lr'] = [g['lr'] for g in optimizer.param_groups]
                repro.atomic_json_dump(self.event, self.directory / f'step-call-{self.event["attempted_index"]:03d}.json')

        optimizer.step = actual_step

    def begin(self, attempted, successful):
        self.event = dict(seed=self.manifest['seed'], arm=self.manifest['arm'], protocol=PROTOCOL,
                          engineering_only=True, attempted_index=attempted + 1,
                          successful_steps_before=successful, optimizer_step_called=False,
                          clocks_before=clocks(self.optimizer), actual_step_plan=[],
                          rng_before=repro.state_sha256(repro.capture_rng_state()))
        self.gradient = None
        repro.atomic_json_dump(self.event, self.directory / f'attempt-begin-{attempted + 1:03d}.json')

    def before_step(self, nonfinite):
        # A finite gradient means AMP may call step. Snapshot only the fixed 1/5/6 boundaries.
        if nonfinite == 0:
            plan = step_plan(self.optimizer, self.manifest['arm'])
            if plan[0]['step'] in (1, 5, 6):
                self.gradient = [p.grad.detach().cpu().clone() if p.grad is not None else None
                                 for p in self.net.parameters()]

    def after_step(self, before, scale_before, scale_after, skipped):
        self.event.update(clocks_after=clocks(self.optimizer), scale_before=scale_before,
                          scale_after=scale_after, skip=bool(skipped),
                          successful_optimizer_steps=self.event['successful_steps_before'] + int(not skipped),
                          rng_after=repro.state_sha256(repro.capture_rng_state()))
        if self.event['optimizer_step_called'] == bool(skipped):
            raise RuntimeError('AMP skip and actual optimizer invocation disagree')
        if skipped and self.event['clocks_before'] != self.event['clocks_after']:
            raise RuntimeError('AMP skip advanced RAdam clocks')
        if self.gradient is not None and not skipped:
            step = self.event['successful_optimizer_steps']
            target = self.directory / f'update-success-{step:02d}.pt'
            repro.atomic_torch_save(dict(metadata=copy.deepcopy(self.event), parameter_order=self.order,
                parameters_before=[p.detach().cpu() for p in before], gradients=self.gradient,
                updates=[(p.detach() - old).cpu() for p, old in zip(self.net.parameters(), before)]), target)
            self.event['tensor_file'] = target.name
        self.gradient = None

    def record(self, telemetry, first_moment, second_moment):
        row = {**telemetry, **self.event,
               'native_loss_arm': telemetry['arm'],
               'first_moment_norm': first_moment, 'second_moment_norm': second_moment}
        with (self.directory / 'startup_attempts.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
