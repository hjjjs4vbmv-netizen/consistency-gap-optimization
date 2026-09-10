"""One independent trajectory at a time, with endpoint-only three-block evaluation."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import torch

from . import protocol as p
from training import startup_windows as w, reproducibility as repro, m1
from scripts.run_m1_training_slot import runtime_environment, scientific_failure, truncate_attempt_csv
from scripts.run_m1_evaluation_job import build_command
from analysis.q256_d_restore_vs_hold_v1.evaluate import verify_evaluator, validate_result


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_freeze(config, deployment_path):
    gate_path = Path(config['control_root']) / 'freeze.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('status') != 'FROZEN' or gate.get('engineering_status') != 'PASS':
        raise RuntimeError('formal execution requires completed engineering and freeze')
    cluster = json.loads((Path(config['control_root']) / 'cluster_ready.json').read_text())
    if (cluster.get('status') != 'READY' or cluster.get('training_count') != 40
            or cluster.get('evaluation_block_count') != 120
            or cluster.get('freeze_receipts', {}).get(config['host']) != p.digest(gate_path)):
        raise RuntimeError('all three host freezes and the complete global matrix must be ready')
    if p.digest(deployment_path) != gate['deployment_sha256']:
        raise RuntimeError('deployment changed after freeze')
    for relative, expected in gate['source_hashes'].items():
        if p.digest(p.ROOT / relative) != expected:
            raise RuntimeError('frozen source changed: ' + relative)
    for relative, expected in gate['control_hashes'].items():
        if p.digest(Path(config['control_root']) / relative) != expected:
            raise RuntimeError('frozen queue/protocol/commands changed: ' + relative)
    return gate


def verify_gpu(mapping):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    physical = {int(line.split(',')[0]): line.split(',')[1].strip() for line in lines}
    if physical.get(mapping['local_gpu']) != mapping['uuid']:
        raise RuntimeError('GPU UUID differs from the actual frozen allocation')
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines()
    if any(line.split(',')[0].strip() == mapping['uuid'] and int(line.split(',')[1]) != os.getpid() for line in lines):
        raise RuntimeError('assigned GPU has another compute process')


def run_directory(config, row):
    return Path(config['output_root']) / row['protocol_id'] / f"seed{row['seed']}" / row['arm']


def key(row):
    return f"{row['protocol_id']}-seed{row['seed']}-{row['arm']}"


def latest(directory, manifest):
    candidates = []
    for path in [directory / 'training-state-latest.pt', *directory.glob('training-state-kimg*.pt'),
                 directory / 'training-state-branch-init-kimg000512.pt']:
        if not path.exists():
            continue
        try:
            seal = json.loads(Path(str(path) + '.sha256.json').read_text())
            candidates.append((seal['attempted_iteration'], 'branch-init' in path.name, path, seal))
        except Exception as exc:
            p.write(directory / (path.name + '.recovery-rejected.json'), dict(error=repr(exc)))
    for attempt, _, path, seal in sorted(candidates, key=lambda value: (value[0], value[1]), reverse=True):
        try:
            if p.digest(path) != seal['sha256']:
                raise RuntimeError('checkpoint hash differs')
            state = torch.load(path, map_location='cpu', weights_only=False)
            w.validate_state(state, manifest)
            if state['attempted_iteration'] != attempt:
                raise RuntimeError('sidecar counters differ')
            del state
            return attempt, path
        except Exception as exc:
            p.write(directory / (path.name + '.recovery-rejected.json'), dict(error=repr(exc)))
    return 0, None


def trim_for_resume(directory, attempt):
    stamp = time.time_ns()
    receipts = []
    for name in ('train_summary.csv', 'factorial_training_telemetry_v1.csv', 'startup_window_telemetry.jsonl'):
        path = directory / name
        backup = directory / (name + f'.before-recovery-{stamp}')
        shutil.copy2(path, backup)
        receipts.append(dict(path=backup.name, sha256=p.digest(backup)))
        if path.suffix == '.csv':
            truncate_attempt_csv(path, attempt)
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['attempted_iteration'] <= attempt))
    p.write(directory / f'recovery-{stamp}.json', dict(resume_attempt=attempt, preserved=receipts), True)


def progress(directory):
    path = directory / 'startup_window_telemetry.jsonl'
    if not path.exists():
        return 0
    with path.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 16384))
        rows = handle.read().splitlines()
    if not rows:
        return 0
    try:
        return json.loads(rows[-1])['attempted_iteration']
    except (ValueError, KeyError):
        return 0


def process(config, command, mapping, log, heartbeat, *, training_dir=None, timeout=28800):
    verify_gpu(mapping)
    env = runtime_environment(mapping['uuid'], Path(config['runtime_python']))
    env.update(DNNLIB_CACHE_DIR=config['evaluation_cache'], PYTHONDONTWRITEBYTECODE='1',
               PYTHONPYCACHEPREFIX=str(Path(config['control_root']) / 'isolated_pycache'))
    cwd = config['code'] if training_dir is not None else config['evaluator_repo']
    started, wall = time.monotonic(), time.time()
    timed_out = False
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            attempt = progress(training_dir) if training_dir is not None else None
            elapsed = time.monotonic() - started
            p.write(heartbeat, dict(status='RUNNING', pid=child.pid, started_wall=wall,
                heartbeat_wall=time.time(), elapsed_process_gpuh=elapsed / 3600,
                attempted_iteration=attempt, command=command, log=str(log), gpu=mapping,
                expected_hard_timeout_seconds=timeout))
            if elapsed > timeout:
                timed_out = True
                p.write(log.with_suffix('.timeout.json'), dict(pid=child.pid, timeout=timeout, wall=time.time()))
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(15)
        rc = child.returncode
    return dict(returncode=rc, process_gpuh=(time.monotonic() - started) / 3600,
                started_wall=wall, ended_wall=time.time(), timeout=timed_out, command=command, log=str(log))


def train(config, row, mapping, heartbeat):
    directory = run_directory(config, row)
    directory.mkdir(parents=True, exist_ok=True)
    mp = Path(config['control_root']) / 'manifests' / (key(row) + '.json')
    manifest = w.read_manifest(mp)
    if manifest != p.manifest(config, row['protocol_id'], row['seed'], row['arm'], directory):
        raise RuntimeError('formal manifest drift')
    outcome_path = directory / 'outcome.json'
    if outcome_path.exists():
        old = json.loads(outcome_path.read_text())
        if old['status'] in ('PASS', 'SCIENTIFIC_FAILURE', 'NO_ENDPOINT'):
            return old
    attempt, resume = latest(directory, manifest)
    if attempt == 8000:
        result = dict(**row, status='PASS', endpoint=str(resume), checkpoint_sha256=p.digest(resume),
                      attempted_iteration=8000, recovered_completed_endpoint=True,
                      process_gpuh=sum(json.loads(path.read_text()).get('session_process_gpuh', 0)
                                       for path in directory.glob('process-receipt-*.json')))
        p.write(outcome_path, result)
        return result
    if list(directory.glob('process-*.log')) and resume is None:
        result = dict(**row, status='TECHNICAL_FAILURE', reason='prior logs without a verified own state; start-over requires documented technical audit')
        p.write(outcome_path, result)
        return result
    if resume is not None:
        trim_for_resume(directory, attempt)
    log = directory / f'process-{time.time_ns()}.log'
    observation = process(config, p.command(config, manifest, mp, resume), mapping, log, heartbeat, training_dir=directory)
    prior_gpuh = sum(json.loads(path.read_text()).get('session_process_gpuh', 0)
                     for path in directory.glob('process-receipt-*.json'))
    text = log.read_text(errors='replace')
    scientific = scientific_failure(log) or any(marker in text for marker in (
        'non-finite RAdam moment state', 'non-finite E_512 state'))
    result = dict(**row, **observation, status='SCIENTIFIC_FAILURE' if scientific else 'TECHNICAL_FAILURE', start_attempt=attempt)
    result.update(session_process_gpuh=observation['process_gpuh'], process_gpuh=prior_gpuh + observation['process_gpuh'])
    if observation['returncode'] == 0:
        endpoint = directory / 'training-state-kimg001024.pt'
        try:
            state = torch.load(endpoint, map_location='cpu', weights_only=False)
            meta = w.validate_state(state, manifest)
            if state['attempted_iteration'] != 8000 or meta['ema_512_init_count'] != 1:
                raise RuntimeError('not a complete own E_512 endpoint')
            if p.digest(endpoint) != json.loads(Path(str(endpoint) + '.sha256.json').read_text())['sha256']:
                raise RuntimeError('endpoint seal differs')
            result.update(status='PASS', endpoint=str(endpoint), checkpoint_sha256=p.digest(endpoint),
                attempted_iteration=8000, successful_optimizer_steps=state['successful_optimizer_steps'])
            del state
        except Exception as exc:
            result['error'] = repr(exc)
    p.write(directory / f'process-receipt-{time.time_ns()}.json', result, True)
    p.write(outcome_path, result)
    return result


def evaluate(config, row, mapping, outcome, heartbeat):
    directory = run_directory(config, row)
    root = directory / 'evaluation'
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'E_512.pkl'
    if outcome['status'] == 'PASS':
        verify_evaluator(config)
        source = Path(outcome['endpoint'])
        if p.digest(source) != outcome['checkpoint_sha256']:
            raise RuntimeError('own endpoint changed before export')
        manifest = w.read_manifest(Path(config['control_root']) / 'manifests' / (key(row) + '.json'))
        state = torch.load(source, map_location='cpu', weights_only=False)
        w.validate_state(state, manifest)
        exported = m1.evaluator_snapshot(state, 'E_512')
        module_hash = repro.module_state_sha256(exported['ema'])
        if not snapshot.exists():
            repro.atomic_pickle_dump(exported, snapshot, overwrite=False)
        else:
            import pickle
            with snapshot.open('rb') as handle:
                previous = pickle.load(handle)
            if repro.module_state_sha256(previous['ema']) != module_hash:
                raise RuntimeError('existing E_512 export differs')
            del previous
        p.write(root / 'export.json', dict(readout='E_512', checkpoint_sha256=outcome['checkpoint_sha256'],
            snapshot_sha256=p.digest(snapshot), module_sha256=module_hash), True)
        del state, exported
    results = []
    for slot in p.evaluation_slots([row]):
        receipt = root / (slot['block'] + '.json')
        if receipt.exists():
            results.append(json.loads(receipt.read_text()))
            continue
        if outcome['status'] != 'PASS':
            result = dict(**slot, status='NO_ENDPOINT', training_status=outcome['status'])
        else:
            target = root / slot['block']
            if target.exists():
                raise RuntimeError('partial evaluation without receipt requires audit; no overwrite')
            target.mkdir()
            wire = {**{k: str(v) for k, v in slot.items()}, 'slot_id': slot['job_id'], 'metrics': 'kid50k_full,fid50k_full'}
            command = build_command(wire, str(snapshot), Path(config['evaluation_dataset']), target,
                                    Path(config['evaluator_repo']), Path(config['runtime_python']), 47000 + row['logical_gpu'])
            observation = process(config, command, mapping, root / (slot['block'] + '.process.log'), heartbeat, timeout=7200)
            result = dict(**slot, **observation, status='TECHNICAL_FAILURE', evaluator_commit=p.EVALUATOR_COMMIT,
                checkpoint_sha256=outcome['checkpoint_sha256'], snapshot_sha256=p.digest(snapshot))
            try:
                result.update(validate_result(slot, str(snapshot), target, config['evaluation_dataset'], observation['returncode']), status='PASS')
            except Exception as exc:
                result['error'] = repr(exc)
        p.write(receipt, result, True)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--logical-gpu', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.deployment.read_text())
    gate = verify_freeze(config, args.deployment)
    mapping = next(r for r in config['allocation'] if r['logical_gpu'] == args.logical_gpu and r['host'] == config['host'])
    control = Path(config['control_root'])
    rows = [row for row in p.queue(config) if row['logical_gpu'] == args.logical_gpu]
    heartbeat = control / f'worker-{args.logical_gpu}.json'
    ledger_path = control / f'ledger-{args.logical_gpu}.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(jobs={}, payment_class=mapping['payment_class'])
    with locked(control / (mapping['uuid'] + '.lock')):
        for index, row in enumerate(rows):
            verify_freeze(config, args.deployment)
            previous = ledger['jobs'].get(key(row))
            if (previous and previous['training_status'] in ('PASS', 'SCIENTIFIC_FAILURE', 'NO_ENDPOINT')
                    and len(previous['evaluation_statuses']) == 3
                    and all(status in ('PASS', 'NO_ENDPOINT') for status in previous['evaluation_statuses'])):
                continue
            spent = sum(job.get('process_gpuh', 0) for job in ledger['jobs'].values())
            complete_times = [job['training_gpuh'] for job in ledger['jobs'].values() if job.get('training_status') == 'PASS' and job.get('training_gpuh', 0) > 0]
            estimate = max([3.3923, *complete_times]) + 3 * .17924
            planned = spent + (len(rows) - index) * estimate + gate['engineering_process_gpuh'] / 8
            if planned > config['planning_gpuh'] / 8:
                p.write(heartbeat, dict(status='COST_REVIEW_REQUIRED', projected_slot_gpuh=planned,
                    fixed_remaining_jobs=rows[index:], reason='projection exceeds the prepared per-slot share; report full matrix cost before more dispatch'))
                return
            outcome = train(config, row, mapping, heartbeat)
            evaluations = [] if outcome['status'] == 'TECHNICAL_FAILURE' else evaluate(config, row, mapping, outcome, heartbeat)
            used = outcome.get('process_gpuh', 0) + sum(result.get('process_gpuh', 0) for result in evaluations)
            ledger['jobs'][key(row)] = dict(training_status=outcome['status'], training_gpuh=outcome.get('process_gpuh', 0),
                process_gpuh=used, evaluation_statuses=[result['status'] for result in evaluations])
            ledger['updated_wall'] = time.time()
            p.write(ledger_path, ledger)
            if outcome['status'] == 'TECHNICAL_FAILURE' or any(r['status'] == 'TECHNICAL_FAILURE' for r in evaluations):
                p.write(heartbeat, dict(status='TECHNICAL_ATTENTION', job=row, remaining_jobs=rows[index + 1:]))
                return
        p.write(heartbeat, dict(status='COMPLETE', jobs=rows, ledger=str(ledger_path), completed_wall=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import sys
        try:
            deployment = Path(sys.argv[sys.argv.index('--deployment') + 1])
            logical = int(sys.argv[sys.argv.index('--logical-gpu') + 1])
            config = json.loads(deployment.read_text())
            p.write(Path(config['control_root']) / f'worker-{logical}.json',
                    dict(status='EXCEPTION', error=repr(exc), wall=time.time()))
        except Exception:
            pass
        raise
