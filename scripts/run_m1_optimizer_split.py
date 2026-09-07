"""One free-GPU worker for a fixed 32-cell diagnostic queue on ECT."""

import argparse
import csv
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts import run_m1_training_slot as runner
from training.m1_optimizer_split import OPERATIONS, SEEDS


def check_replay(original, actual, start, end):
    def rows(path):
        with path.open() as handle:
            return {int(r['attempted_iteration']): r for r in csv.DictReader(handle)}
    reference, observed = rows(original), rows(actual)
    ignored = {'elapsed_sec', 'gpu_hours_cumulative'}
    for attempt in range(start + 1, end + 1):
        differences = [k for k, v in reference[attempt].items()
                       if k not in ignored and observed[attempt].get(k) != v]
        if differences:
            raise ValueError(f'replay differs at attempt {attempt}: {differences}')



BASE = Path('/data/raw/ECT/m1_optimizer_restart_ema')
SOURCE = Path('/data/raw/ECT/q256_terminal_history_n30_full_archive_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--wait-idle', action='store_true')
    args = parser.parse_args()
    output_root = BASE / 'optimizer_split_diagnostic'
    output_root.mkdir(exist_ok=True)
    gpu_lock = Path(f'/tmp/m1-gpu-{args.gpu}.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed in SEEDS:
        for arm in ('A', 'B'):
            for operation in OPERATIONS:
                cell = f'seed{seed}/{arm}/{operation}'
                output = output_root / cell
                if (output / 'status.json').exists():
                    continue
                while True:
                    if (output_root / 'STOP.txt').exists():
                        raise RuntimeError((output_root / 'STOP.txt').read_text())
                    if (output / 'status.json').exists():
                        break
                    processes = subprocess.check_output([
                        'nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                        '--format=csv,noheader'], text=True).strip()
                    if not processes:
                        break
                    if not args.wait_idle:
                        raise RuntimeError(f'GPU {args.gpu} is occupied by {processes}')
                    time.sleep(30)
                claim = output_root / cell.replace('/', '_')
                with claim.with_suffix('.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if (output / 'status.json').exists():
                        continue
                    run_cell(seed, arm, operation, output, args.gpu)


def run_cell(seed, arm, operation, output, gpu):
    output.mkdir(parents=True, exist_ok=False)
    source = runner.source_path(SOURCE / 'training', seed, arm)
    branch = ('R_' if operation == 'R' else 'K_') + arm
    manifest = runner.manifest_value(seed=seed, branch=branch, source=source, run_dir=output)
    manifest.update(analysis_role='POST_OUTCOME_OPTIMIZER_DIAGNOSTIC',
                    optimizer_operation=operation, stop_after_attempts=4064)
    path = output / 'formal_run_manifest.json'
    runner.write_json(path, manifest)
    python = Path(sys.executable)
    command = runner.training_command(python, SOURCE / 'assets/cifar10-32x32-training-original.zip',
                                      seed, path, source)
    command[command.index(str(runner.REPO_ROOT / 'ct_train.py'))] = str(
        runner.REPO_ROOT / 'scripts/m1_optimizer_split_train.py')
    command.append('--stop-after-attempts=4064')
    env = runner.runtime_environment(gpu, python)
    env.update(M1_SPLIT_OUTPUT=str(output), M1_SPLIT_OPERATION=operation,
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    status = dict(seed=seed, arm=arm, operation=operation, gpu=gpu, status='RUNNING',
                  started_utc=runner.utc_now(), command=command)
    runner.write_json(output / 'status.json', status)
    print(f'START {seed} {arm} {operation} gpu={gpu}', flush=True)
    with (output / 'process.log').open('x') as log:
        result = subprocess.run(command, cwd=runner.REPO_ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    status.update(exit_code=result.returncode, ended_utc=runner.utc_now())
    telemetry = output / 'schedule_switch_training_telemetry_v1.csv'
    rows = list(csv.DictReader(telemetry.open())) if telemetry.exists() else []
    status['last_attempt'] = int(rows[-1]['attempted_iteration']) if rows else None
    status['status'] = ('COMPLETE_64' if not result.returncode and status['last_attempt'] == 4064 else
                        'NUMERICAL_FAILURE' if runner.scientific_failure(output / 'process.log') else
                        'TECHNICAL_FAILURE')
    runner.write_json(output / 'status.json', status)
    if operation in ('K', 'R') and rows:
        roots = {64: BASE / 'archive/new4_26139/runs',
                 55: BASE / 'archive/hub2_26065/runs',
                 60: BASE / 'archive/old4_29026/runs', 56: BASE / 'runs'}
        original = roots[seed] / f'S{seed-49:02d}' / branch
        try:
            check_replay(original / telemetry.name, telemetry, 4000, status['last_attempt'])
        except ValueError as error:
            status.update(status='REPLAY_MISMATCH', error=str(error))
            runner.write_json(output / 'status.json', status)
            (BASE / 'optimizer_split_diagnostic/STOP.txt').write_text(str(error))
            raise
        status['original_replay'] = 'MATCH'
        runner.write_json(output / 'status.json', status)
    print(json.dumps(status), flush=True)
    if status['status'] == 'TECHNICAL_FAILURE':
        (BASE / 'optimizer_split_diagnostic/STOP.txt').write_text(str(output))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
