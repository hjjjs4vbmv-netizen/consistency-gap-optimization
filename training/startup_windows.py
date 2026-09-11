"""Frozen fresh-q128 and delayed-q256 protocols, independent of PR110/111."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from training import reproducibility as repro, startup_update as engineering
from training.loss import Q128_STARTUP_NATIVE_PROTOCOL, resolve_target_weight_factorial

Q128 = 'q128_startup_fresh8_v1'
Q256 = 'q256_startup_delayed6_10_v1'
METADATA_KEY = 'startup_windows'
BASE_LR = 1e-4
ARMS128 = ('AA', 'DA', 'A_startup_down5', 'D_startup_up5')
ARMS256 = ('A_delayed_down6_10',)
ENGINEERING_SEEDS = (99001, 99002)
WINDOWS = {
    'AA': (1, 5, 1.0), 'DA': (1, 5, 1.0),
    'A_startup_down5': (1, 5, 1 / 1.1),
    'D_startup_up5': (1, 5, 1.1),
    'A_delayed_down6_10': (6, 10, 1 / 1.1),
    'A_replay': (1, 5, 1.0),
}
TEMPLATE = Path(__file__).resolve().parents[1] / 'analysis/startup_window_experiments_v1/trajectory_template.json'


def native_protocol(q):
    if q == 128:
        return Q128_STARTUP_NATIVE_PROTOCOL
    if q == 256:
        return 'q256_target_weight_v1'
    raise ValueError('only the two frozen q values are supported')


def native_prefix(arm):
    return 'D' if arm in ('DA', 'D_startup_up5') else 'A'


def validate_manifest(m):
    if not isinstance(m, dict) or m.get('protocol_id') not in (Q128, Q256):
        raise ValueError('independent startup window protocol identity required')
    q = 128 if m['protocol_id'] == Q128 else 256
    fixed = dict(q=q, initialization='fresh_transfer', base_lr=BASE_LR,
                 switch_kimg=512, final_kimg=1024, target_gap_scale=1.0,
                 global_gap_scale=1.0, diagnostic_success_steps=16)
    if any(m.get(k) != v for k, v in fixed.items()):
        raise ValueError('frozen window scientific configuration differs')
    mode = m.get('mode')
    if mode not in ('formal', 'engineering', 'initialization'):
        raise ValueError('explicit execution mode required')
    cohort = m.get('cohort')
    if (not isinstance(cohort, list) or len(cohort) != 8 or
            any(type(s) is not int for s in cohort) or cohort != sorted(set(cohort))):
        raise ValueError('exactly eight ordered training seeds required')
    if (q == 256 and cohort != list(range(50, 58))) or (q == 128 and min(cohort) < 301):
        raise ValueError('cohort identity differs')
    allowed_seeds = (ENGINEERING_SEEDS if q == 128 else (50, 51)) if mode == 'engineering' else cohort
    if mode == 'initialization':
        allowed_seeds = [*cohort, *ENGINEERING_SEEDS]
    if type(m.get('seed')) is not int or m['seed'] not in allowed_seeds:
        raise ValueError('seed outside the frozen execution cohort')
    allowed_arms = ARMS128 if q == 128 else ARMS256 + (('A_replay',) if mode == 'engineering' else ())
    if m.get('arm') not in allowed_arms:
        raise ValueError('arm outside the fixed matrix')
    start, end, multiplier = WINDOWS[m['arm']]
    if (m.get('start_success_step'), m.get('end_success_step'), m.get('lr_multiplier')) != (start, end, multiplier):
        raise ValueError('successful-update window differs')
    if m.get('native_prefix_arm') != native_prefix(m['arm']):
        raise ValueError('native loss prefix differs')
    expected_max = (128 if q == 128 else 64) if mode == 'engineering' else 0 if mode == 'initialization' else 8000
    if m.get('max_attempts') != expected_max:
        raise ValueError('attempt budget differs; engineering cannot become formal')
    if not m.get('immutable_output_root'):
        raise ValueError('immutable output identity required')
    if (m.get('dataset_sha256'), m.get('transfer_sha256')) != (engineering.DATA_SHA256, engineering.TRANSFER_SHA256):
        raise ValueError('original asset identities required')
    if mode == 'initialization':
        if q != 128 or m['arm'] != 'AA' or m.get('reference_initial_receipt') is not None:
            raise ValueError('only new q128 AA initialization receipts may be created')
    elif not m.get('reference_initial_receipt'):
        raise ValueError('own canonical initialization binding required')
    controls = m.get('old_control_bindings')
    if q == 128:
        if controls != []:
            raise ValueError('fresh q128 cannot bind historical quality controls')
    else:
        expected = {(m['seed'], a, b) for a in ('AA', 'A_startup_down5') for b in ('B0', 'B1', 'B2')}
        if not isinstance(controls, list) or len(controls) != 6 or {(r['seed'], r['arm'], r['block']) for r in controls} != expected:
            raise ValueError('q256 requires exact AA and early-window readout blocks')
    return m


def read_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text())) if path else None


def validate_config(m, config, transfer, run_dir):
    validate_manifest(m)
    if Path(run_dir).resolve() != Path(m['immutable_output_root']).resolve():
        raise ValueError('immutable output identity differs')
    expected = copy.deepcopy(json.loads(TEMPLATE.read_text())['trajectory_config'])
    expected.update(seed=m['seed'], rank_seed=m['seed'])
    expected['loss_kwargs'].update(q=float(m['q']), factorial_protocol=native_protocol(m['q']),
                                  target_gap_scale=1.0,
                                  denominator_gap_scale=1.1 if m['native_prefix_arm'] == 'D' else 1.0)
    expected['dataset_kwargs']['path'] = config['dataset_kwargs']['path']
    if repro.canonical_json_data(expected) != repro.canonical_json_data(config):
        raise ValueError('actual trajectory differs from the frozen complete configuration')
    if engineering.file_hash(config['dataset_kwargs']['path']) != m['dataset_sha256']:
        raise ValueError('original training data changed')
    if transfer and engineering.file_hash(transfer) != m['transfer_sha256']:
        raise ValueError('original transfer changed')
    ref = m.get('reference_initial_receipt')
    if ref:
        if engineering.file_hash(ref['path']) != ref['sha256']:
            raise ValueError('canonical initialization receipt changed')
        r = json.loads(Path(ref['path']).read_text())
        if (r['seed'], r['attempted_iteration'], r['processed_nimg']) != (m['seed'], 0, 0):
            raise ValueError('canonical receipt is not this seed at initialization')
        if r['trajectory_config']['loss_kwargs']['q'] != m['q']:
            raise ValueError('canonical initialization q differs')


def validate_actual_loss(loss_fn, m, suffix=False):
    expected = resolve_target_weight_factorial(native_protocol(m['q']), 1.0,
        1.0 if suffix or m['native_prefix_arm'] == 'A' else 1.1,
        adj='sigmoid', global_gap_scale=1.0, q=m['q'])
    if loss_fn.q != m['q'] or loss_fn.factorial != expected:
        raise ValueError('effective loss q/factors differ from the manifest')


def use_native_A(loss_fn, q):
    if loss_fn.q != q:
        raise ValueError('suffix cannot change the actual q')
    loss_fn.factorial = resolve_target_weight_factorial(native_protocol(q), 1.0, 1.0,
                                                       adj='sigmoid', global_gap_scale=1.0, q=q)


def transition_to_suffix(loss_fn, net, previous_ema, init_count, q, cur_nimg):
    """Idempotent state-level boundary operation; optimizer and RNG are not inputs."""
    if cur_nimg != 512000 or init_count not in (0, 1) or loss_fn.q != q:
        raise ValueError('invalid suffix boundary/state')
    if init_count == 1:
        if previous_ema is None or loss_fn.factorial['arm'] != 'A':
            raise ValueError('initialized suffix is incomplete')
        return previous_ema, 1
    if previous_ema is not None:
        raise ValueError('uninitialized suffix already has an EMA')
    from training.history_component import initialize_ema_512
    result = initialize_ema_512(net)
    use_native_A(loss_fn, q)
    return result, 1


def validate_initial(current, reference, m):
    from training.history_component import validate_initial_receipt
    if current['trajectory_config']['loss_kwargs']['q'] != m['q']:
        raise ValueError('actual initial q differs')
    return {**validate_initial_receipt(current, reference), 'protocol_id': m['protocol_id'],
            'reference_kind': 'new_same_seed_canonical' if m['q'] == 128 else 'archived_original_seed'}


def validate_state(state, m):
    validate_manifest(m)
    if 'startup_engineering' in state or 'startup_quality' in state or m['mode'] == 'initialization':
        raise ValueError('foreign or initialization-only checkpoint forbidden')
    meta = state.get(METADATA_KEY)
    if not isinstance(meta, dict) or meta.get('manifest') != m:
        raise ValueError('checkpoint protocol/seed/arm/output identity differs')
    required = ('net', 'ema', 'optimizer_state', 'gradscaler_state', 'rank_states',
                'loss_fn_state', 'trajectory_config', 'trajectory_config_sha256',
                'cur_nimg', 'cur_tick', 'tick_start_nimg', 'snapshot_grid_z',
                'snapshot_grid_c', 'snapshot_grid_size', 'successful_optimizer_steps', 'attempted_iteration')
    if any(k not in state for k in required):
        raise ValueError('incomplete recoverable checkpoint')
    attempts, success = state['attempted_iteration'], state['successful_optimizer_steps']
    if any(type(x) is not int for x in (attempts, success)) or not 0 <= success <= attempts <= m['max_attempts'] or state['cur_nimg'] != attempts * 128:
        raise ValueError('checkpoint counters differ')
    Controller.validate_controller_state(meta['controller'], m, success)
    clocks = [int(v.get('step', 0)) for v in state['optimizer_state']['state'].values()]
    if (success and (not clocks or set(clocks) != {success})) or (not success and clocks):
        raise ValueError('checkpoint RAdam clocks differ')
    initialized = meta['ema_512_init_count']
    if initialized not in (0, 1) or (state['cur_nimg'] > 512000 and initialized != 1):
        raise ValueError('E_512 initialization count differs')
    if initialized and (state['cur_nimg'] < 512000 or 'ema_512' not in state):
        raise ValueError('E_512 state/boundary missing')
    config = state['trajectory_config']
    if config['loss_kwargs']['q'] != m['q'] or config['loss_kwargs']['factorial_protocol'] != native_protocol(m['q']):
        raise ValueError('checkpoint actual q/protocol differs')
    expected_factor = resolve_target_weight_factorial(native_protocol(m['q']), 1.0,
        1.0 if initialized or m['native_prefix_arm'] == 'A' else 1.1,
        adj='sigmoid', global_gap_scale=1.0, q=m['q'])
    if state['factorial'] != expected_factor:
        raise ValueError('checkpoint native loss phase differs')
    if repro.state_sha256(config) != state['trajectory_config_sha256']:
        raise ValueError('checkpoint configuration digest differs')
    stage = state['loss_fn_state']['stage']
    if state['loss_fn_state']['ratio'] != 1 - 1 / m['q'] ** (stage + 1):
        raise ValueError('checkpoint effective schedule q differs')
    return meta


class Controller:
    """Changes only temporary LR inside real optimizer.step; AMP skips do not enter."""
    def __init__(self, optimizer, manifest, state=None):
        if type(optimizer) is not torch.optim.RAdam:
            raise ValueError('native torch RAdam required')
        self.optimizer, self.manifest = optimizer, manifest
        self.start, self.end, self.factor = (manifest[k] for k in ('start_success_step', 'end_success_step', 'lr_multiplier'))
        self.successful_steps = 0
        self.exposure = {name: dict(count=0, sum_update_norm=0.0, net_displacement_norm=0.0) for name in ('1_5', '6_10')}
        self.accumulator = None
        if state is not None:
            self.validate_controller_state(state, manifest, state['successful_steps'])
            self.successful_steps = state['successful_steps']
            self.exposure = copy.deepcopy(state['update_exposure'])
            device = optimizer.param_groups[0]['params'][0].device
            self.accumulator = [p.to(device).clone() for p in state['exposure_accumulator']] if state['exposure_accumulator'] is not None else None
        for group in optimizer.param_groups:
            if tuple(group['betas']) != (.9, .999) or group['eps'] != 1e-8 or group['weight_decay'] != 0 or group['lr'] != BASE_LR:
                raise ValueError('frozen native RAdam configuration differs')
        original = optimizer.step

        def actual_step(*args, **kwargs):
            self.called = True
            rates = [g['lr'] for g in optimizer.param_groups]
            if any(lr != BASE_LR for lr in rates):
                raise RuntimeError('base LR changed')
            step = self.successful_steps + 1
            self.clock_checked = step <= 16
            if self.clock_checked:
                clocks = engineering.clocks(optimizer, active_only=True)
                if not clocks or set(clocks) != {self.successful_steps}:
                    raise RuntimeError('active RAdam clocks disagree')
            self.multiplier = self.factor if self.start <= step <= self.end else 1.0
            try:
                for group, lr in zip(optimizer.param_groups, rates):
                    group['lr'] = lr * self.multiplier
                result = original(*args, **kwargs)
                if self.clock_checked and set(engineering.clocks(optimizer, active_only=True)) != {step}:
                    raise RuntimeError('actual RAdam step did not advance all active clocks once')
                self.successful_steps = step
                return result
            finally:
                for group, lr in zip(optimizer.param_groups, rates):
                    group['lr'] = lr

        optimizer.step = actual_step
        self.begin()

    @staticmethod
    def validate_controller_state(state, m, success):
        required = {'successful_steps', 'inactive', 'window', 'update_exposure', 'exposure_accumulator'}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError('incomplete window controller state')
        window = {k: m[k] for k in ('start_success_step', 'end_success_step', 'lr_multiplier')}
        if state['successful_steps'] != success or state['inactive'] != (success > m['end_success_step']) or state['window'] != window:
            raise ValueError('controller window/counters differ')
        for name, start in (('1_5', 1), ('6_10', 6)):
            record = state['update_exposure'][name]
            if record['count'] != min(5, max(0, success - start + 1)):
                raise ValueError('actual update exposure count differs')
            if any(not math.isfinite(record[k]) or record[k] < 0 for k in ('sum_update_norm', 'net_displacement_norm')):
                raise ValueError('non-finite actual update exposure')
        if (state['exposure_accumulator'] is not None) != (success in (1, 2, 3, 4, 6, 7, 8, 9)):
            raise ValueError('unfinished update exposure accumulator missing')

    def state_dict(self):
        return dict(successful_steps=self.successful_steps, inactive=self.successful_steps > self.end,
            window={k: self.manifest[k] for k in ('start_success_step', 'end_success_step', 'lr_multiplier')},
            update_exposure=copy.deepcopy(self.exposure),
            exposure_accumulator=None if self.accumulator is None else [p.detach().cpu().clone() for p in self.accumulator])

    def begin(self):
        self.called = False
        self.multiplier = 1.0
        self.clock_checked = False
        self.next_success = self.successful_steps + 1

    @torch.no_grad()
    def observe_update(self, parameters, before, update_norm):
        step = self.successful_steps
        if not self.called or step > 10:
            return
        name = '1_5' if step <= 5 else '6_10'
        delta = [(p.detach() - old).to(torch.float64) for p, old in zip(parameters, before)]
        if self.accumulator is None:
            self.accumulator = delta
        else:
            for total, value in zip(self.accumulator, delta):
                total.add_(value)
        record = self.exposure[name]
        record['count'] += 1
        record['sum_update_norm'] += float(update_norm)
        record['net_displacement_norm'] = math.sqrt(sum(float(p.square().sum()) for p in self.accumulator))
        if step in (5, 10):
            self.accumulator = None

    def record(self, row, directory):
        if bool(row['step_skipped']) == self.called or row['successful_optimizer_steps'] != self.successful_steps:
            raise RuntimeError('AMP/controller successful-update count differs')
        keys = ('attempted_iteration', 'successful_optimizer_steps', 'step_skipped', 'update_norm',
                'batch_sha256', 't_sha256', 'base_r_sha256', 'grad_scale_before', 'grad_scale_after')
        event = {k: row[k] for k in keys}
        rho = engineering.rho(self.next_success, .999)
        event.update(protocol_id=self.manifest['protocol_id'], q=self.manifest['q'],
            optimizer_clock=self.successful_steps, proposed_success_step=self.next_success,
            rho=rho, radam_branch='rectified' if rho > 5 else 'non_adaptive',
            clock_evidence='all_active_parameters' if self.clock_checked else 'successful_call_counter',
            applied_multiplier=self.multiplier if self.called else 1.0,
            base_lr=BASE_LR, restored_lr=[g['lr'] for g in self.optimizer.param_groups],
            inactive=self.successful_steps > self.end)
        if self.next_success <= 16:
            event['update_exposure'] = copy.deepcopy(self.exposure)
        with (Path(directory) / 'startup_window_telemetry.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
            f.flush()


def seal_checkpoint(path, manifest, attempted):
    path = Path(path)
    repro.atomic_json_dump(dict(protocol_id=manifest['protocol_id'], q=manifest['q'], seed=manifest['seed'],
        arm=manifest['arm'], mode=manifest['mode'], checkpoint=path.name,
        sha256=engineering.file_hash(path), attempted_iteration=attempted), str(path) + '.sha256.json', overwrite=True)
