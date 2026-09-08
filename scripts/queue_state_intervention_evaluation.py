"""Evaluate each fixed seed's 36 new slots after its four training branches."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from scripts import run_state_intervention_evaluation as job


def seed_slots(seed, branch=None):
    if seed not in job.experiment.SEEDS:
        raise ValueError('seed is outside the fixed cohort')
    if branch is not None and branch not in job.experiment.BRANCHES:
        raise ValueError('branch is outside the new training branches')
    return [row for row in job.slots.build_slots() if row['seed'] == seed
            and row['mode'] == 'NEW' and (branch is None or row['branch'] == branch)]


def training_ready(runs_root, seed):
    waiting = False
    for branch in job.experiment.BRANCHES:
        protocol = job.experiment.M2 if branch.startswith('L_') else job.experiment.SWAP
        path = runs_root / protocol / f'seed{seed}' / branch / 'branch_status.json'
        if not path.exists():
            waiting = True
            continue
        record = json.loads(path.read_text())
        if record.get('seed') != seed or record.get('branch') != branch:
            raise ValueError(f'training identity mismatch: {path}')
        if record['status'] == 'RUNNING':
            waiting = True
        elif record['status'] not in ('COMPLETE', 'NUMERICAL_FAILURE'):
            raise RuntimeError(f'training needs review: {path}')
    return not waiting


def no_endpoint(slot, args):
    directory = job.slots.state_directory(slot, args.source_root, args.runs_root)
    path = directory / 'branch_status.json'
    record = json.loads(path.read_text())
    if record.get('seed') != slot['seed'] or record.get('branch') != slot['branch']:
        raise ValueError('source branch identity mismatch')
    if slot['budget_kimg'] != 1024 or record['status'] != 'NUMERICAL_FAILURE':
        return None
    if (directory / 'training-state-kimg001024.pt').exists():
        raise RuntimeError('numerical failure has an endpoint; inspect before classification')
    receipt = dict(slot, schema='ect.state-interventions.evaluation-job/v1',
        status='NO_ENDPOINT', reason='training NUMERICAL_FAILURE without endpoint',
        branch_status_path=str(path.resolve()),
        branch_status_sha256=job.validation.sha256_file(path))
    receipt['planned_metrics'] = receipt.pop('metrics')
    return receipt


def completed_receipt(slot, args):
    path = args.output / 'receipts' / f"{slot['slot_id']}.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if (any(record.get(k) != v for k, v in slot.items() if k != 'metrics') or
            record.get('planned_metrics') != slot['metrics'] or
            record.get('schema') != 'ect.state-interventions.evaluation-job/v1'):
        raise ValueError('existing receipt belongs to a different slot')
    if record['status'] == 'NO_ENDPOINT':
        if record != no_endpoint(slot, args):
            raise ValueError('NO_ENDPOINT no longer matches its training evidence')
        return True
    if record['status'] != 'PASS':
        raise RuntimeError(f'prior evaluation needs review: {path}')
    snapshot, export_path, _ = job.load_readout(slot, args)
    if job.validation.sha256_file(export_path) != record['export_receipt_sha256']:
        raise ValueError('completed job export receipt changed')
    directory = args.output / 'jobs' / slot['slot_id']
    if job.inspect_output(slot, directory, snapshot, args.dataset) != record['metrics']:
        raise ValueError('completed metrics differ from receipt')
    names = ['training_options.json', 'generated-samples.npy']
    for metric in job.validation.METRICS:
        names.extend([f'generated-features-{metric}-repeat00.npy', f'metric-{metric}.jsonl'])
    for name in names:
        if job.validation.sha256_file(directory / name) != record['artifact_sha256'][name]:
            raise ValueError(f'completed artifact changed: {name}')
    return True


def export_readout(slot, args):
    directory = args.readouts / slot['readout_id']
    if (directory / 'receipt.json').exists():
        job.load_readout(slot, args)
        return
    if directory.exists():
        raise FileExistsError(f'incomplete prior readout export: {directory}')
    env = job.original.runtime_environment(args.gpu, args.runtime_python)
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    log = args.output / 'export-logs' / f"{slot['readout_id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        subprocess.run([str(args.runtime_python), '-m', 'scripts.export_state_intervention_readout',
            '--slot-id', slot['slot_id'], '--source-root', str(args.source_root),
            '--runs-root', str(args.runs_root), '--output-root', str(args.readouts)],
            cwd=Path(__file__).resolve().parents[1], env=env, stdout=stream,
            stderr=subprocess.STDOUT, check=True)


def run_lane(args):
    branch = getattr(args, 'branch', None)
    rows = seed_slots(args.training_seed, branch)
    suffix = f'-{branch}' if branch else ''
    lane_path = args.output / 'lanes' / f'seed{args.training_seed}{suffix}.json'
    lane_path.parent.mkdir(parents=True, exist_ok=True)
    with lane_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if lane_path.exists():
            raise FileExistsError(f'prior lane record exists; inspect before restarting: {lane_path}')
        lane = dict(seed=args.training_seed, gpu=args.gpu, planned=len(rows),
                    started_utc=job.original.utc_now(), status='WAIT_TRAINING')
        if branch:
            lane['branch'] = branch
        job.original.write_json(lane_path, lane)
        try:
            while not training_ready(args.runs_root, args.training_seed):
                time.sleep(60)
            # The training controller may still hold its shared lock while exiting.
            with Path(f'/tmp/m1-gpu-{args.gpu}.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            while subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                    '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(60)
            lane['status'] = 'EVALUATING'
            for slot in rows:
                lane['current_slot'] = slot['slot_id']
                job.original.write_json(lane_path, lane)
                if completed_receipt(slot, args):
                    continue
                missing = no_endpoint(slot, args)
                if missing is not None:
                    job.validation.atomic_json(args.output / 'receipts' / f"{slot['slot_id']}.json", missing)
                    continue
                export_readout(slot, args)
                job.execute(SimpleNamespace(**vars(args), slot_id=slot['slot_id'], prepare_only=False))
            lane.update(status='COMPLETE', ended_utc=job.original.utc_now())
        except Exception as exc:
            lane.update(status='TECHNICAL_UNRESOLVED', error=f'{type(exc).__name__}: {exc}',
                        ended_utc=job.original.utc_now())
            raise
        finally:
            job.original.write_json(lane_path, lane)


def main():
    parser = argparse.ArgumentParser()
    for flag in ('readouts', 'output', 'source-root', 'runs-root', 'runtime-python',
                 'evaluator', 'dataset', 'cache'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--training-seed', type=int, choices=job.experiment.SEEDS, required=True)
    parser.add_argument('--branch', choices=job.experiment.BRANCHES,
                        help='Evaluate only this explicitly assigned new branch')
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    run_lane(parser.parse_args())


if __name__ == '__main__':
    main()
