import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts import queue_state_intervention_evaluation as queue


class EvaluationQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(runs_root=self.root / 'runs', source_root=self.root,
                                    output=self.root / 'evaluation')

    def write_training(self, branch, state):
        protocol = queue.job.experiment.M2 if branch.startswith('L_') else queue.job.experiment.SWAP
        path = self.args.runs_root / protocol / 'seed59' / branch / 'branch_status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(seed=59, branch=branch, status=state)))
        return path

    def test_fixed_lanes_cover_only_216_new_slots_once(self):
        rows = [r for seed in queue.job.experiment.SEEDS for r in queue.seed_slots(seed)]
        self.assertEqual(len(rows), 216)
        self.assertEqual(len({r['slot_id'] for r in rows}), 216)
        self.assertEqual({r['mode'] for r in rows}, {'NEW'})
        for seed in queue.job.experiment.SEEDS:
            self.assertEqual(len(queue.seed_slots(seed)), 36)
        with self.assertRaises(ValueError):
            queue.seed_slots(57)

    def test_waits_for_training_but_exposes_technical_failure(self):
        self.assertFalse(queue.training_ready(self.args.runs_root, 59))
        for branch in queue.job.experiment.BRANCHES:
            self.write_training(branch, 'COMPLETE')
        self.assertTrue(queue.training_ready(self.args.runs_root, 59))
        self.write_training('X_B_from_A', 'RUNNING')
        self.assertFalse(queue.training_ready(self.args.runs_root, 59))
        self.write_training('L_A', 'TECHNICAL_UNRESOLVED')
        with self.assertRaises(RuntimeError):
            queue.training_ready(self.args.runs_root, 59)

    def test_branch_dispatch_partitions_twenty_swap_slots_without_overlap(self):
        rows = [r for seed in (55, 56) for branch in ('X_A_from_B', 'X_B_from_A')
                for r in queue.seed_slots(seed, branch)]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r['slot_id'] for r in rows}), 20)
        self.assertTrue(all(r['budget_kimg'] == 1024 and r['mode'] == 'NEW' for r in rows))
        for seed in (55, 56):
            assigned = {r['slot_id'] for r in rows if r['seed'] == seed}
            retained = [r for r in queue.seed_slots(seed) if r['slot_id'] not in assigned]
            self.assertEqual(len(retained), 26)
            self.assertFalse(any(r['branch'].startswith('X_') for r in retained))
        with self.assertRaises(ValueError):
            queue.seed_slots(55, 'R_B')

    def test_scientific_failure_records_all_five_without_metrics(self):
        path = self.write_training('L_A', 'NUMERICAL_FAILURE')
        rows = [r for r in queue.seed_slots(59) if r['branch'] == 'L_A']
        self.assertEqual(len(rows), 5)
        for slot in rows:
            receipt = queue.no_endpoint(slot, self.args)
            self.assertEqual(receipt['status'], 'NO_ENDPOINT')
            self.assertNotIn('metrics', receipt)
            queue.job.validation.atomic_json(
                self.args.output / 'receipts' / f"{slot['slot_id']}.json", receipt)
            self.assertTrue(queue.completed_receipt(slot, self.args))
        path.with_name('training-state-kimg001024.pt').write_bytes(b'unexpected endpoint')
        with self.assertRaises(RuntimeError):
            queue.completed_receipt(rows[0], self.args)

    def test_prior_attempt_and_wrong_identity_are_not_silently_skipped(self):
        slot = queue.seed_slots(59)[0]
        path = self.args.output / 'receipts' / f"{slot['slot_id']}.json"
        path.parent.mkdir(parents=True)
        record = dict(slot, schema='ect.state-interventions.evaluation-job/v1', status='RUNNING')
        record['planned_metrics'] = record.pop('metrics')
        path.write_text(json.dumps(record))
        with self.assertRaises(RuntimeError):
            queue.completed_receipt(slot, self.args)
        record.update(seed=60, status='PASS')
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            queue.completed_receipt(slot, self.args)


if __name__ == '__main__':
    unittest.main()
