import unittest
from pathlib import Path
import json
from tempfile import TemporaryDirectory

from scripts import collect_state_intervention_results as collect
from scripts import state_intervention_evaluation_slots as slots
from scripts import summarize_state_intervention_results as summary


class ResultBindingTests(unittest.TestCase):
    def test_identity_rejects_wrong_seed_block_version_and_status(self):
        slot = slots.build_slots()[0]
        record = dict(slot, status='PASS')
        collect.check_identity(record, slot)
        for key, value in dict(seed=56, sample_seed_end=49998, nfe=2,
                               evaluator_commit='wrong', status='RUNNING').items():
            with self.assertRaises(ValueError):
                collect.check_identity(dict(record, **{key: value}), slot)
        legacy = dict(record, seed=str(slot['seed']))
        collect.check_identity(legacy, slot, legacy=True)

    def test_archive_translation_keeps_node_origin(self):
        root = Path('/data/task')
        base = root / 'archive/cloud28453'
        self.assertEqual(collect.archived_path('/root/final_state_interventions_q256_v2/evaluation/x',
                                              base, root), base / 'evaluation/x')
        self.assertEqual(collect.archived_path('/data/task/evaluation/y', base, root),
                         root / 'evaluation/y')
        with self.assertRaises(ValueError):
            collect.archived_path('/another/experiment/x', base, root)

    def test_full_plan_and_auxiliary_readouts(self):
        rows = [dict(r, status='PASS', fid50k_full=1 + r['seed'] / 100,
                     kid50k_full=-0.001) for r in slots.build_slots()]
        result = summary.summarize(rows)
        self.assertEqual(result['E_512_three_blocks']['fid50k_full']['swap_common']['n'], 6)
        self.assertNotIn('m2_same_chase', result['E_KEEP_B0']['fid50k_full'])
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                summary.summarize(invalid)
        rows[0]['status'] = 'NO_ENDPOINT'
        self.assertEqual(summary.summarize(rows)['ONLINE_B0']['fid50k_full']['m2_same_endpoint']['n'], 5)

    def test_public_training_metadata_removes_nested_private_locations(self):
        value = {'seed': 55, 'source_path': '/private/source', 'nested':
                 {'donor_path': '/private/donor', 'step_values': [3990]}, 'host': 'private-host'}
        self.assertEqual(summary.public_metadata(value), {'seed': 55, 'nested': {'step_values': [3990]}})

    def test_initial_source_inventory_binds_exact_36_and_rejects_changed_prefix_size(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / 'task'
            root.mkdir()
            inventory, rows = [], []
            for seed in slots.experiment.SEEDS:
                for arm in ('A', 'B'):
                    p = collect.sources.prefix(root.parent, seed, arm)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b'original')
                    inventory.append(dict(path=str(p), bytes=p.stat().st_size))
                for branch in ('K_A', 'K_B', 'R_A', 'R_B'):
                    p = collect.sources.original_run(root.parent, seed, branch) / 'training-state-kimg000768.pt'
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b'bound768')
                    rows.extend(dict(seed=seed, branch=branch, budget_kimg=768,
                                     source_state_sha256=collect.digest(p)) for _ in range(4))
            (root / 'sources_inventory.json').write_text(json.dumps(inventory))
            result = collect.bind_initial_sources(root, dict(rows=rows))
            self.assertEqual(len(result['sources']), 36)
            self.assertEqual(sum(r['budget_kimg'] == 512 for r in result['sources']), 12)
            Path(inventory[0]['path']).write_bytes(b'changed size')
            with self.assertRaisesRegex(ValueError, 'inventory size'):
                collect.bind_initial_sources(root, dict(rows=rows))


if __name__ == '__main__':
    unittest.main()
