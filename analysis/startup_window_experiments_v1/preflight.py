"""Bounded engineering trajectories and exact replay checks; never computes FID."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import torch

from . import protocol as p
from training import startup_windows as w, reproducibility as repro
from scripts.run_m1_training_slot import runtime_environment, equal_state
from analysis.q256_history_component_chase_v1.preflight import runtime

EVALUATION_DATA_SHA256 = '08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372'
PARITY_REQUIRED = ('batch_sha256', 't_sha256', 'base_r_sha256', 'target_r_sha256',
    'denominator_r_sha256', 'loss', 'raw_grad_norm', 'raw_grad_nonfinite_count',
    'update_norm', 'model_norm', 'ema_norm', 'grad_scale_before', 'grad_scale_after',
    'successful_optimizer_steps')
TIMING_FIELDS = {'elapsed_sec', 'gpu_hours_cumulative', 'peak_vram_gb', 'gpu_memory_gb', 'timestamp', 'wall_time'}


def assets(config):
    actual = runtime()
    if actual != p.ENVIRONMENT:
        raise RuntimeError('runtime differs: ' + repr(actual))
    expected = dict(dataset=p.DATA_SHA256, transfer=p.TRANSFER_SHA256,
                    evaluation_dataset=EVALUATION_DATA_SHA256)
    hashes = {name: p.digest(config[name]) for name in expected}
    if hashes != expected:
        raise RuntimeError('original assets differ: ' + repr(hashes))
    return dict(status='PASS', runtime=actual, hashes=hashes)


def load_state(directory, manifest):
    path = Path(directory) / 'training-state-latest.pt'
    seal = json.loads(Path(str(path) + '.sha256.json').read_text())
    if p.digest(path) != seal['sha256']:
        raise RuntimeError('checkpoint seal differs')
    state = torch.load(path, map_location='cpu', weights_only=False)
    w.validate_state(state, manifest)
    return state


def compare_states(first, second):
    keys = ('optimizer_state', 'gradscaler_state', 'rank_states', 'loss_fn_state',
            'attempted_iteration', 'successful_optimizer_steps', 'cur_nimg', 'cur_tick',
            'tick_start_nimg', 'snapshot_grid_z', 'snapshot_grid_c', 'snapshot_grid_size')
    for key in keys:
        if not equal_state(first[key], second[key]):
            raise RuntimeError('pause/resume differs: ' + key)
    for key in ('net', 'ema'):
        if repro.module_state_sha256(first[key]) != repro.module_state_sha256(second[key]):
            raise RuntimeError('pause/resume module differs: ' + key)
    if not equal_state(first[w.METADATA_KEY]['controller'], second[w.METADATA_KEY]['controller']):
        raise RuntimeError('pause/resume controller or accumulated exposure differs')


def compare_rows(first, second):
    if len(first) != len(second) or not first:
        raise RuntimeError('replay attempt count differs')
    common = set(first[0]) & set(second[0])
    if not set(PARITY_REQUIRED).issubset(common):
        raise RuntimeError('required parity fields absent')
    checked = set(PARITY_REQUIRED)
    for key in common - TIMING_FIELDS:
        try:
            float(first[0][key])
        except (ValueError, TypeError):
            continue
        checked.add(key)
    differences = {key: sum(a[key] != b[key] for a, b in zip(first, second)) for key in sorted(checked)}
    if any(differences.values()):
        raise RuntimeError('shared scientific replay fields differ: ' + repr({k: n for k, n in differences.items() if n}))
    return dict(rows=len(first), exact_fields=sorted(checked))


def read_rows(directory):
    with (Path(directory) / 'factorial_training_telemetry_v1.csv').open() as f:
        return list(csv.DictReader(f))


def check_telemetry(directory, manifest):
    rows = [json.loads(line) for line in (Path(directory) / 'startup_window_telemetry.jsonl').read_text().splitlines()]
    success = 0
    for attempt, row in enumerate(rows, 1):
        success += not row['step_skipped']
        multiplier = manifest['lr_multiplier'] if not row['step_skipped'] and manifest['start_success_step'] <= success <= manifest['end_success_step'] else 1.0
        if (row['attempted_iteration'] != attempt or row['successful_optimizer_steps'] != success
                or row['applied_multiplier'] != multiplier or row['restored_lr'] != [1e-4]
                or row['optimizer_clock'] != success or row['q'] != manifest['q']):
            raise RuntimeError('window/attempt/LR/clock telemetry mismatch')
        if not row['step_skipped'] and success <= 16:
            if row['clock_evidence'] != 'all_active_parameters':
                raise RuntimeError('missing actual active clock evidence')
            if row['radam_branch'] != ('non_adaptive' if success <= 5 else 'rectified'):
                raise RuntimeError('native RAdam branch boundary differs')
    if len(rows) != manifest['max_attempts'] or success < 16:
        raise RuntimeError('bounded engineering run did not reach the required diagnostic steps')
    return dict(attempts=len(rows), successful_updates=success, skipped=sum(r['step_skipped'] for r in rows),
                q=manifest['q'], update_exposure=next(r['update_exposure'] for r in reversed(rows) if 'update_exposure' in r))


def run(config, group, seed, gpu, label, continue_validated_runs=False):
    root = Path(config['root']) / 'engineering' / label
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'receipt.json').exists():
        raise RuntimeError('engineering sequence is already complete')
    existing = (root / 'started.json').exists()
    if existing and not continue_validated_runs:
        raise RuntimeError('engineering attempt already exists; preserve it and use an audited new label')
    binding = next(r for r in config['allocation'] if r['host'] == config['host'] and r['local_gpu'] == gpu)
    physical = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    found = {int(line.split(',')[0]): line.split(',')[1].strip() for line in physical.splitlines()}
    if found.get(gpu) != binding['uuid']:
        raise RuntimeError('physical GPU differs from the new allocation')
    env = runtime_environment(binding['uuid'], Path(config['runtime_python']))
    costs = json.loads((root / 'gpu_ledger.json').read_text()) if existing else []
    if not existing:
        p.write(root / 'started.json', dict(group=group, seed=seed, gpu=binding, started_wall=time.time()), True)
    else:
        p.write(root / f'validation-continuation-{time.time_ns()}.json', dict(
            reason='reuse only sealed completed engineering runs; exclude runtime accounting from scientific parity and complete missing checks',
            source_sha256=p.digest(__file__), prior_gpu_ledger_sha256=p.digest(root / 'gpu_ledger.json')), True)
    p.write(root / 'assets.json', assets(config), True)

    def execute(arm, name, mode='engineering', resume=None, stop=None):
        directory = root / name
        if mode == 'initialization':
            directory = Path(config['reference_root']) / 'q128' / f'seed{seed}'
        manifest = p.manifest(config, group, seed, arm, directory, mode)
        mp = root / (name + '.manifest.json')
        p.write(mp, manifest, True)
        command = p.command(config, manifest, mp, resume, stop)
        log = root / (name + f'-success-{stop or 0}-' + ('resume' if resume else 'fresh') + '.log')
        if log.exists():
            if continue_validated_runs and mode == 'engineering' and resume is None and name.endswith('-full'):
                state = load_state(directory, manifest)
                if state['attempted_iteration'] != manifest['max_attempts']:
                    raise RuntimeError('only sealed completed engineering runs can be reused')
                del state
                return directory, manifest
            raise RuntimeError('engineering logs cannot be overwritten')
        budget = .75 if group == w.Q128 else .375
        if sum(r['process_gpuh'] for r in costs) >= budget:
            raise RuntimeError('engineering share exhausted before dispatch')
        running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if any(line.split(',')[0].strip() == binding['uuid'] and int(line.split(',')[1]) != os.getpid() for line in running.splitlines()):
            raise RuntimeError('assigned engineering GPU already has another compute process')
        started = time.monotonic()
        with log.open('x') as f:
            process = subprocess.Popen(command, cwd=p.ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            p.write(root / 'running.json', dict(pid=process.pid, command=command, started_wall=time.time(), log=str(log)))
            while process.poll() is None:
                time.sleep(10)
                p.write(root / 'heartbeat.json', dict(pid=process.pid, process_alive=True, wall=time.time(), log=str(log)))
                if time.monotonic() - started > 1800:
                    p.write(root / 'timeout.json', dict(pid=process.pid, hard_timeout_seconds=1800, wall=time.time()))
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            rc = process.returncode
        costs.append(dict(name=name, mode=mode, command=command, returncode=rc,
                          process_gpuh=(time.monotonic() - started) / 3600))
        p.write(root / 'gpu_ledger.json', costs)
        if rc:
            raise RuntimeError('engineering process failed; preserved log: ' + str(log))
        return directory, manifest

    if group == w.Q128:
        reference = Path(config['reference_root']) / 'q128' / f'seed{seed}' / 'initial_state_receipt_v1.json'
        if not reference.exists():
            execute('AA', 'canonical-initialization', mode='initialization')
        arms = w.ARMS128
    else:
        arms = ('A_replay', 'A_delayed_down6_10')
    outputs, checks = {}, {}
    for arm in arms:
        directory, manifest = execute(arm, arm + '-full')
        state = load_state(directory, manifest)
        if state['attempted_iteration'] != manifest['max_attempts']:
            raise RuntimeError('engineering budget mismatch')
        del state
        checks[arm] = check_telemetry(directory, manifest)
        outputs[arm] = (directory, manifest)
    parity = {}
    if group == w.Q256:
        with (Path(config['reference_root']) / 'q256' / f'seed{seed}' / 'factorial_training_telemetry_v1.csv').open() as f:
            archived = list(csv.DictReader(f))[:64]
        replay = read_rows(outputs['A_replay'][0])
        parity['archived_A_all_shared_scientific_fields'] = compare_rows(replay, archived)
        delayed = read_rows(outputs['A_delayed_down6_10'][0])
        sixth = next(i for i, row in enumerate(delayed) if int(row['successful_optimizer_steps']) == 6)
        parity['delayed_before_sixth_success'] = compare_rows(delayed[:sixth], replay[:sixth])
    # Representative full-state pauses at every relevant window edge.
    pause_arm = 'D_startup_up5' if group == w.Q128 and seed == 99001 else 'A_delayed_down6_10' if group == w.Q256 and seed == 50 else None
    if pause_arm:
        directory, manifest = execute(pause_arm, pause_arm + '-split', stop=5)
        for stop in (6, 10, 11, None):
            del_state = load_state(directory, manifest)
            del del_state
            directory, manifest = execute(pause_arm, pause_arm + '-split', resume=directory / 'training-state-latest.pt', stop=stop)
        full = load_state(*outputs[pause_arm])
        split = load_state(directory, manifest)
        compare_states(full, split)
        del full, split
    receipt = dict(status='PASS', protocol_id=group, seed=seed, gpu=binding, checks=checks,
        parity=parity, pause_boundaries=[5, 6, 10, 11] if pause_arm else [],
        process_gpuh=sum(r['process_gpuh'] for r in costs), completed_wall=time.time(), no_quality_evaluation=True)
    p.write(root / 'receipt.json', receipt, True)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--group', choices=(w.Q128, w.Q256), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--continue-validated-runs', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.deployment.read_text())
    try:
        run(config, args.group, args.seed, args.gpu, args.label, args.continue_validated_runs)
    except BaseException as exc:
        failed = Path(config['root']) / 'engineering' / args.label / 'failure.json'
        if failed.exists():
            failed = failed.with_name(f'failure-{time.time_ns()}.json')
        p.write(failed, dict(status='FAILED', error=repr(exc)), True)
        raise


if __name__ == '__main__':
    main()
