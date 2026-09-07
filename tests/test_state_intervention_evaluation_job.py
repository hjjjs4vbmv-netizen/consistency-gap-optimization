import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from scripts import run_state_intervention_evaluation as runner
from scripts import state_intervention_evaluation_slots as slots
from scripts.inspect_frozen_evaluation import inspect_output


class EvaluationJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.slot = slots.get_slot('seed59-L_A-kimg001024-E_512-B0', new_only=True)

    def test_frozen_command_and_reuse_isolation(self):
        command = runner.worker.build_command(
            {k: str(v) for k, v in self.slot.items()}, 'snapshot.pkl', self.root / 'data.zip',
            self.root / 'job', self.root / 'evaluator', self.root / 'python', 53000)
        for value in ('--nfe=1', '--fp16=False', '--sample-seeds=0-49999',
                      '--metric-generator-batch=128', '--metric-repeats=1',
                      '--metrics=kid50k_full,fid50k_full', '--seed=20260730'):
            self.assertIn(value, command)
        with self.assertRaises(ValueError):
            slots.get_slot('seed59-K_A-kimg001024-E_512-B0', new_only=True)
        wrong = dict(self.slot, sample_seed_end=49998, nfe='1')
        with self.assertRaises(runner.validation.ValidationError):
            runner.worker.build_command(wrong, 'x', self.root, self.root, self.root, self.root, 53000)

    def test_export_identity_and_file_binding(self):
        args = SimpleNamespace(readouts=self.root / 'readouts', source_root=self.root,
                               runs_root=self.root / 'runs')
        source = slots.state_directory(self.slot, args.source_root, args.runs_root)
        source.mkdir(parents=True)
        out = args.readouts / self.slot['readout_id']
        out.mkdir(parents=True)
        files = dict(snapshot=out / 'readout.pkl',
            source_state=source / 'training-state-kimg001024.pt',
            branch_manifest=source / 'formal_run_manifest.json',
            branch_status=source / 'branch_status.json')
        receipt = {k: self.slot[k] for k in
                   ('protocol_id', 'seed', 'branch', 'budget_kimg', 'readout', 'readout_id')}
        receipt.update(schema='ect.state-interventions.readout-export/v1', status='EXPORTED',
            source_cur_nimg=1024000, source_attempted_iteration=8000,
            source_readout_sha256='weights', snapshot_readout_sha256='weights',
            fixed_input_observation=dict(classification='FINITE_READOUT'))
        for name, path in files.items():
            path.write_bytes(name.encode())
            receipt[name + '_path'] = str(path)
            receipt[name + '_sha256'] = runner.validation.sha256_file(path)
        receipt_path = out / 'receipt.json'
        receipt_path.write_text(json.dumps(receipt))
        self.assertEqual(runner.load_readout(self.slot, args)[0], files['snapshot'])
        receipt['seed'] = 60
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            runner.load_readout(self.slot, args)
        receipt['seed'] = 59
        receipt_path.write_text(json.dumps(receipt))
        files['snapshot'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            runner.load_readout(self.slot, args)

    def test_all_four_training_branches_must_finish(self):
        paths = []
        for branch in runner.experiment.BRANCHES:
            protocol = runner.experiment.M2 if branch.startswith('L_') else runner.experiment.SWAP
            path = self.root / protocol / 'seed59' / branch / 'branch_status.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(seed=59, branch=branch, status='COMPLETE')))
            paths.append(path)
        runner.require_training_finished(self.root, 59)
        for state in ('RUNNING', 'TECHNICAL_UNRESOLVED'):
            row = json.loads(paths[-1].read_text())
            paths[-1].write_text(json.dumps(dict(row, status=state)))
            with self.assertRaises(RuntimeError):
                runner.require_training_finished(self.root, 59)
        paths[-1].unlink()
        with self.assertRaises(FileNotFoundError):
            runner.require_training_finished(self.root, 59)

    def test_real_output_arrays_and_metric_rules(self):
        snapshot, dataset = self.root / 'snapshot.pkl', self.root / 'data.zip'
        options = dict(sample_seeds=list(range(50000)), seed=20260730,
            metrics=['kid50k_full', 'fid50k_full'], metric_repeats=1,
            metric_generator_batch=128, retain_generated_artifacts=True, mid_t=[],
            network_kwargs=dict(use_fp16=False), resume_pkl=str(snapshot),
            dataset_kwargs=dict(path=str(dataset)))
        (self.root / 'training_options.json').write_text(json.dumps(options))
        (self.root / 'log.txt').write_text('Exiting...')
        np.save(self.root / 'generated-samples.npy', np.zeros((50000, 1), dtype=np.uint8))
        for metric, value in (('kid50k_full', -0.001), ('fid50k_full', 1e9)):
            array = np.lib.format.open_memmap(
                self.root / f'generated-features-{metric}-repeat00.npy',
                mode='w+', dtype=np.float32, shape=(50000, 2048))
            del array
            (self.root / f'metric-{metric}.jsonl').write_text(json.dumps(
                dict(metric=metric, num_gpus=1, results={metric: value})) + '\n')
        self.assertEqual(inspect_output(self.slot, self.root, snapshot, dataset),
                         {'kid50k_full': -0.001, 'fid50k_full': 1e9})
        feature = np.load(self.root / 'generated-features-fid50k_full-repeat00.npy', mmap_mode='r+')
        feature[123, 4] = 1
        feature.flush()
        with self.assertRaisesRegex(ValueError, 'features differ'):
            inspect_output(self.slot, self.root, snapshot, dataset)
        feature[123, 4] = 0
        feature.flush()
        del feature
        (self.root / 'metric-fid50k_full.jsonl').write_text(json.dumps(
            dict(metric='fid50k_full', num_gpus=1, results={'fid50k_full': float('nan')})))
        with self.assertRaises(FloatingPointError):
            inspect_output(self.slot, self.root, snapshot, dataset)


if __name__ == '__main__':
    unittest.main()
