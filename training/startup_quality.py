"""Independent formal startup controller. The PR110 engineering guards stay intact."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from training import startup_update as engineering, reproducibility as repro, history_component

PROTOCOL = 'q256_startup_step_responder_pilot_v1'
ARMS = ('D_startup_up5', 'A_startup_down5')
MULTIPLIERS = {'D_startup_up5': 1.1, 'A_startup_down5': 1 / 1.1, 'A_replay': 1.0}
FIXED = dict(protocol_id=PROTOCOL, experiment_class='responder_enriched_pilot',
             initialization='fresh_transfer', startup_success_steps=5, switch_kimg=512,
             final_kimg=1024, base_lr=0.0001, continuation_arm='A')


def validate_manifest(m):
    if not isinstance(m, dict) or any(m.get(k) != v for k, v in FIXED.items()):
        raise ValueError('formal startup protocol identity/configuration mismatch')
    preflight = m.get('preflight_only', False)
    if type(preflight) is not bool:
        raise ValueError('invalid preflight flag')
    if type(m.get('seed')) is not int or m['seed'] not in range(50, 58):
        raise ValueError('seed outside frozen cohort')
    if m.get('arm') not in (ARMS + ('A_replay',) if preflight else ARMS):
        raise ValueError('arm outside fixed matrix')
    if preflight and m['seed'] != 50:
        raise ValueError('preflight is seed50 only')
    if m.get('lr_multiplier') != MULTIPLIERS[m['arm']]:
        raise ValueError('startup multiplier differs')
    native = 'D' if m['arm'] == ARMS[0] else 'A'
    if m.get('native_prefix_arm') != native:
        raise ValueError('native prefix arm differs')
    for key in ('reference_initial_receipt', 'immutable_output_root', 'old_control_bindings'):
        if not m.get(key):
            raise ValueError('missing binding: ' + key)
    for key, expected in [('dataset_sha256', engineering.DATA_SHA256),
                          ('transfer_sha256', engineering.TRANSFER_SHA256)]:
        if m.get(key) != expected:
            raise ValueError('original source hash differs: ' + key)
    return m


def read_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text())) if path else None


def validate_config(m, config, transfer, run_dir):
    if Path(run_dir).resolve() != Path(m['immutable_output_root']).resolve():
        raise ValueError('immutable output identity differs')
    ref = m['reference_initial_receipt']
    if engineering.file_hash(ref['path']) != ref['sha256']:
        raise ValueError('reference initial receipt changed')
    receipt = json.loads(Path(ref['path']).read_text())
    if receipt['seed'] != m['seed'] or receipt['attempted_iteration'] != 0 or receipt['processed_nimg'] != 0:
        raise ValueError('wrong reference seed/start')
    from training.schedule_switch import trajectory_configs_compatible
    if not trajectory_configs_compatible(receipt['trajectory_config'], config, {'final_kimg': 1024}):
        raise ValueError('scientific configuration differs from original start')
    factors = config['loss_kwargs']
    if factors['target_gap_scale'] != 1.0 or factors['denominator_gap_scale'] != (1.1 if m['native_prefix_arm'] == 'D' else 1.0):
        raise ValueError('native prefix factors differ')
    if engineering.file_hash(config['dataset_kwargs']['path']) != m['dataset_sha256']:
        raise ValueError('dataset changed')
    if transfer and engineering.file_hash(transfer) != m['transfer_sha256']:
        raise ValueError('transfer changed')


def validate_state(state, m):
    validate_manifest(m)
    if 'startup_engineering' in state:
        raise ValueError('engineering checkpoint forbidden')
    meta = state.get('startup_quality')
    if not isinstance(meta, dict) or meta.get('manifest') != m:
        raise ValueError('checkpoint seed/arm/protocol/output identity mismatch')
    required = ('net', 'ema', 'optimizer_state', 'gradscaler_state', 'rank_states',
                'loss_fn_state', 'trajectory_config', 'trajectory_config_sha256',
                'cur_nimg', 'cur_tick', 'tick_start_nimg', 'snapshot_grid_z',
                'snapshot_grid_c', 'snapshot_grid_size', 'successful_optimizer_steps', 'attempted_iteration')
    if any(k not in state for k in required):
        raise ValueError('incomplete full-state checkpoint')
    attempts = state['attempted_iteration']; success = state['successful_optimizer_steps']
    if type(attempts) is not int or not 0 <= success <= attempts <= 8000 or state['cur_nimg'] != attempts * 128:
        raise ValueError('invalid training counters')
    controller = meta['controller']
    if controller != dict(successful_steps=success, inactive=success >= 6):
        raise ValueError('controller counters differ')
    steps = [int(v.get('step', 0)) for v in state['optimizer_state']['state'].values()]
    if (success and (not steps or set(steps) != {success})) or (not success and steps):
        raise ValueError('checkpoint optimizer clock mismatch')
    initialized = meta['ema_512_init_count']
    if initialized not in (0, 1) or (state['cur_nimg'] > 512000 and initialized != 1):
        raise ValueError('E_512 initialization count invalid')
    if initialized and (state['cur_nimg'] < 512000 or 'ema_512' not in state):
        raise ValueError('E_512 boundary/state missing')
    expected = 'A' if initialized else m['native_prefix_arm']
    if state['factorial']['arm'] != expected:
        raise ValueError('checkpoint native loss phase differs')
    if repro.state_sha256(state['trajectory_config']) != state['trajectory_config_sha256']:
        raise ValueError('checkpoint configuration hash differs')
    return meta


def use_native_A(loss_fn):
    from training.loss import resolve_target_weight_factorial
    loss_fn.factorial = resolve_target_weight_factorial('q256_target_weight_v1', 1.0, 1.0,
                                                       adj='sigmoid', global_gap_scale=1.0, q=256)


class Controller:
    """Runs only on actual optimizer.step, including after GradScaler filtering."""
    def __init__(self, optimizer, arm, state=None):
        if type(optimizer) is not torch.optim.RAdam or arm not in MULTIPLIERS:
            raise ValueError('native RAdam and fixed arm required')
        self.optimizer, self.arm = optimizer, arm
        self.successful_steps = 0
        self.inactive = False
        self.called = False
        self.multiplier = 1.0
        self.last_clock = 0
        if state is not None:
            if state != dict(successful_steps=state['successful_steps'], inactive=state['successful_steps'] >= 6):
                raise ValueError('invalid controller state')
            self.successful_steps = state['successful_steps']; self.inactive = state['inactive']
        self.last_clock = self.successful_steps
        original = optimizer.step
        def actual_step(*args, **kwargs):
            self.called = True
            rates = [g['lr'] for g in optimizer.param_groups]
            if any(x != 0.0001 for x in rates):
                raise RuntimeError('base LR differs')
            if self.inactive:
                result = original(*args, **kwargs)
                self.successful_steps += 1
                self.last_clock = self.successful_steps
                return result
            clocks = engineering.clocks(optimizer, active_only=True)
            if not clocks or set(clocks) != {self.successful_steps}:
                raise RuntimeError('active RAdam clocks disagree; fail closed')
            step = self.successful_steps + 1
            for g in optimizer.param_groups:
                if tuple(g['betas']) != (.9, .999) or g['eps'] != 1e-8 or g['weight_decay'] != 0:
                    raise ValueError('RAdam settings changed')
                rho = engineering.rho(step, g['betas'][1])
                if (step <= 5 and rho > 5) or (step == 6 and rho <= 5):
                    raise RuntimeError('RAdam rectification boundary differs')
            self.multiplier = MULTIPLIERS[arm] if step <= 5 else 1.0
            try:
                for g, lr in zip(optimizer.param_groups, rates):
                    g['lr'] = lr * self.multiplier
                result = original(*args, **kwargs)
                if set(engineering.clocks(optimizer, active_only=True)) != {step}:
                    raise RuntimeError('optimizer did not advance all active clocks exactly once')
                self.successful_steps = step; self.last_clock = step
                self.inactive = step >= 6
                return result
            finally:
                for g, lr in zip(optimizer.param_groups, rates):
                    g['lr'] = lr
        optimizer.step = actual_step

    def state_dict(self):
        return dict(successful_steps=self.successful_steps, inactive=self.inactive)

    def begin(self):
        self.called = False; self.multiplier = 1.0

    def record(self, row, directory):
        skipped = bool(row['step_skipped'])
        if skipped == self.called or row['successful_optimizer_steps'] != self.successful_steps:
            raise RuntimeError('AMP skip/success controller mismatch')
        keys = ('attempted_iteration', 'successful_optimizer_steps', 'step_skipped',
                'update_norm', 'batch_sha256', 't_sha256', 'base_r_sha256',
                'grad_scale_before', 'grad_scale_after')
        telemetry = {k: row[k] for k in keys}
        telemetry.update(optimizer_clock=self.last_clock,
                         clock_evidence='actual_all_active_parameters' if not self.inactive or self.successful_steps == 6 else 'successful_call_counter',
                         applied_multiplier=self.multiplier if self.called else 1.0,
                         base_lr=0.0001, restored_lr=[g['lr'] for g in self.optimizer.param_groups],
                         inactive=self.inactive)
        with (Path(directory) / 'startup_quality_telemetry.jsonl').open('a') as f:
            f.write(json.dumps(telemetry, allow_nan=False) + '\n')
            f.flush()


def seal_checkpoint(path, manifest, attempted):
    path=Path(path)
    repro.atomic_json_dump(dict(protocol_id=PROTOCOL, seed=manifest['seed'],arm=manifest['arm'],
        checkpoint=path.name,sha256=engineering.file_hash(path),attempted_iteration=attempted),
        str(path)+'.sha256.json',overwrite=True)
