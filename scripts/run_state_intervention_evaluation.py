"""Run one NEW fixed slot after its seed's four training branches finish."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import subprocess

from scripts import run_m1_evaluation_job as worker
from scripts import run_m1_training_slot as original
from scripts import state_intervention_evaluation_slots as slots
from scripts import validate_m1_evaluation_job as validation
from scripts.inspect_frozen_evaluation import inspect_output
from training import state_interventions as experiment


def load_readout(slot, args):
    receipt_path = args.readouts / slot['readout_id'] / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    expected = {k: slot[k] for k in
                ('protocol_id', 'seed', 'branch', 'budget_kimg', 'readout', 'readout_id')}
    expected.update(schema='ect.state-interventions.readout-export/v1', status='EXPORTED',
                    source_cur_nimg=slot['budget_kimg'] * 1000,
                    source_attempted_iteration=slot['budget_kimg'] * 1000 // 128)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('export receipt does not match the fixed evaluation slot')
    if (receipt['source_readout_sha256'] != receipt['snapshot_readout_sha256'] or
            receipt['fixed_input_observation']['classification'] != 'FINITE_READOUT'):
        raise ValueError('readout export needs review before evaluation')
    source = slots.state_directory(slot, args.source_root, args.runs_root)
    paths = dict(snapshot=receipt_path.parent / 'readout.pkl',
                 source_state=source / f"training-state-kimg{slot['budget_kimg']:06d}.pt",
                 branch_manifest=source / 'formal_run_manifest.json',
                 branch_status=source / 'branch_status.json')
    for name, path in paths.items():
        if (path.resolve() != Path(receipt[name + '_path']).resolve() or
                validation.sha256_file(path) != receipt[name + '_sha256']):
            raise ValueError(f'export source binding mismatch: {name}')
    return paths['snapshot'].resolve(), receipt_path, receipt


def require_training_finished(runs_root, seed):
    for branch in experiment.BRANCHES:
        protocol = experiment.M2 if branch.startswith('L_') else experiment.SWAP
        path = runs_root / protocol / f'seed{seed}' / branch / 'branch_status.json'
        row = json.loads(path.read_text())
        if (row.get('seed') != seed or row.get('branch') != branch or
                row.get('status') not in ('COMPLETE', 'NUMERICAL_FAILURE')):
            raise RuntimeError(f'assigned training is not resolved: {path}')


def verify_evaluator(path):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=all'], cwd=path, text=True)
    if (head != slots.original.EVALUATOR_COMMIT or dirty or
            validation.sha256_file(path / 'ct_eval.py') != validation.EVALUATOR_CT_EVAL_SHA256):
        raise ValueError('evaluator is not the clean frozen implementation')
    return dict(commit=head, clean=True)


def prepare(args):
    slot = slots.get_slot(args.slot_id, new_only=True)
    if slot['seed'] != args.training_seed:
        raise ValueError('slot does not belong to the assigned seed lane')
    snapshot, export_path, export = load_readout(slot, args)
    evaluator = verify_evaluator(args.evaluator)
    dataset_hash = validation.verify_evaluation_dataset(args.dataset)
    worker.validate_cache_root(args.cache)
    job = args.output / 'jobs' / slot['slot_id']
    command = worker.build_command({k: str(v) for k, v in slot.items()},
        str(snapshot), args.dataset, job, args.evaluator, args.runtime_python, 53000 + args.gpu)
    record = dict(slot, schema='ect.state-interventions.evaluation-job/v1',
        host=socket.gethostname(), gpu=args.gpu, command=command,
        snapshot_path=str(snapshot), snapshot_sha256=export['snapshot_sha256'],
        export_receipt_path=str(export_path.resolve()),
        export_receipt_sha256=validation.sha256_file(export_path),
        source_state_path=export['source_state_path'],
        source_state_sha256=export['source_state_sha256'],
        evaluation_dataset_sha256=dataset_hash, evaluator=evaluator)
    record['planned_metrics'] = record.pop('metrics')
    record['execution_source_files'] = {
        str(Path(module.__file__).relative_to(Path(__file__).resolve().parents[1])):
        validation.sha256_file(Path(module.__file__))
        for module in (worker, original, validation, slots)}
    record['execution_source_files']['scripts/run_state_intervention_evaluation.py'] = validation.sha256_file(Path(__file__))
    record['execution_source_files']['scripts/inspect_frozen_evaluation.py'] = validation.sha256_file(Path(__file__).with_name('inspect_frozen_evaluation.py'))
    return slot, job, snapshot, record


def execute(args):
    slot, job, snapshot, record = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(record, status='PREPARED_NO_GENERATION')))
        return 0
    require_training_finished(args.runs_root, args.training_seed)
    with Path(f'/tmp/m1-gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if occupied:
            raise RuntimeError(f'assigned GPU still occupied: {occupied}')
        record['gpu_probe'] = worker.gpu_resource_probe(args.gpu)
        worker.disk_resource_probe(args.output)
        receipt_path = args.output / 'receipts' / f"{slot['slot_id']}.json"
        if receipt_path.exists() or job.exists():
            raise FileExistsError('prior evaluation attempt exists; inspect it, do not overwrite')
        env = original.runtime_environment(args.gpu, args.runtime_python)
        env.update(DNNLIB_CACHE_DIR=str(args.cache), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        job.mkdir(parents=True)
        record.update(status='RUNNING', started_utc=original.utc_now())
        validation.atomic_json(receipt_path, record)
        try:
            record['canary'] = worker.run_canary(record['command'], str(snapshot),
                args.runtime_python, args.evaluator, env)
            code, timeout, elapsed = worker.run_process(
                record['command'], args.evaluator, env, job / 'process.log')
            record.update(exit_code=code, timeout=timeout, elapsed_seconds=elapsed)
            if code or timeout:
                raise RuntimeError('evaluator exited unsuccessfully; inspect process.log')
            record['metrics'] = inspect_output(slot, job, snapshot, args.dataset)
            record['artifact_sha256'] = {
                path.name: validation.sha256_file(path) for path in (
                    job / 'training_options.json', job / 'generated-samples.npy',
                    *[job / f'generated-features-{m}-repeat00.npy' for m in validation.METRICS],
                    *[job / f'metric-{m}.jsonl' for m in validation.METRICS])}
            record['status'] = 'PASS'
        except Exception as exc:
            # Nonfinite outputs require technical review, not an automatic scientific verdict.
            record.update(status='TECHNICAL_UNRESOLVED', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['ended_utc'] = original.utc_now()
            original.write_json(receipt_path, record)
    print(json.dumps({'slot_id': slot['slot_id'], 'status': record['status']}))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot-id', required=True)
    for flag in ('readouts', 'output', 'source-root', 'runs-root', 'runtime-python',
                 'evaluator', 'dataset', 'cache'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--training-seed', type=int, choices=experiment.SEEDS, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--prepare-only', action='store_true')
    return execute(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
