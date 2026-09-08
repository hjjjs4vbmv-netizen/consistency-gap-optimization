import copy
from collections import Counter
import unittest

import torch

from scripts import state_intervention_evaluation_slots as slots
from scripts.export_state_intervention_readout import prepare_readout, validate_branch_status
from training import m1, reproducibility, schedule_switch


def fixture(slot):
    arm = 'A' if slot['branch'] in ('K_A', 'R_A', 'L_A', 'X_A_from_B') else 'B'
    manifest = dict(experiment_protocol=slot['protocol_id'], seed=slot['seed'],
        branch=slot['branch'], origin_arm=arm, continuation_arm='A',
        run_kind='formal', source_state={'path': '/original/source'},
        donor_state={'path': '/original/donor'}, m1_shadow_update=True)
    budget = slot['budget_kimg']
    meta = m1.initial_metadata(manifest, int(m1.optimizer_intervention(slot['branch']) == 'reset'))
    state = dict(m1=meta, cur_nimg=budget * 1000,
        attempted_iteration=budget * 1000 // 128, successful_optimizer_steps=0,
        reproducibility_schema=reproducibility.TRAINING_STATE_SCHEMA,
        trajectory_config=dict(seed=slot['seed'], total_kimg=1024,
            dataset_kwargs={'path': '/original/data'}, network_kwargs={'use_fp16': True}),
        rank_states=[{'sampler_state': {'consumed_samples': budget * 1000}}],
        schedule_switch=schedule_switch.state_metadata(manifest), factorial={'arm': arm},
        optimizer_state={}, gradscaler_state={}, loss_fn_state={}, cur_tick=0,
        tick_start_nimg=0, trajectory_config_sha256='test')
    for index, key in enumerate(('net', 'ema', 'ema_512')):
        state[key] = torch.nn.Linear(1, 1, bias=False)
        state[key].weight.data.fill_(index + 1)
    return state, manifest


class EvaluationPreparationTests(unittest.TestCase):
    def test_original_terminal_recheck_without_exit_code_can_export(self):
        for seed in (55, 56):
            row = slots.get_slot(f'seed{seed}-R_B-kimg000768-ONLINE-B0')
            record = dict(seed=seed, branch='R_B', status='PASS', resume_attempt=8000)
            before = record.copy()
            validate_branch_status(record, row)
            self.assertEqual(record, before)
            state, manifest = fixture(row)
            snapshot, _ = prepare_readout(state, manifest, row)
            self.assertEqual(snapshot['ema'].weight.item(), 1)

    def test_terminal_recheck_does_not_accept_failed_or_wrong_sources(self):
        row = slots.get_slot('seed55-R_B-kimg000768-ONLINE-B0')
        record = dict(seed=55, branch='R_B', status='PASS', resume_attempt=8000)
        for change in ({'seed': 56}, {'branch': 'R_A'}, {'status': 'TECHNICAL_FAILURE'},
                       {'resume_attempt': 7999}, {'exit_code': 1}, {'exit_code': None}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_branch_status(dict(record, **change), row)
        new = slots.get_slot('seed55-L_A-kimg001024-ONLINE-B0')
        with self.assertRaises(ValueError):
            validate_branch_status(dict(record, branch='L_A', status='COMPLETE'), new)
        validate_branch_status(dict(record, branch='L_A', status='COMPLETE', exit_code=0), new)

    def test_fixed_matrix_and_reuse_never_become_new_jobs(self):
        rows = slots.build_slots()
        self.assertEqual(len(rows), 336)
        self.assertEqual(len({r['slot_id'] for r in rows}), 336)
        self.assertEqual(Counter(r['mode'] for r in rows), {'NEW': 216, 'REUSE': 120})
        self.assertEqual(Counter((r['budget_kimg'], r['mode']) for r in rows),
                         {(768, 'NEW'): 96, (1024, 'NEW'): 120, (1024, 'REUSE'): 120})
        self.assertFalse(any(r['budget_kimg'] == 768 and r['readout'] == 'E_KEEP' for r in rows))
        for row in rows:
            self.assertEqual(row['sample_seed_end'] - row['sample_seed_start'] + 1, 50000)
        reused = next(r for r in rows if r['mode'] == 'REUSE')
        with self.assertRaises(ValueError):
            slots.get_slot(reused['slot_id'], new_only=True)

    def test_every_new_readout_preserves_its_distinct_weights(self):
        rows = [r for r in slots.build_slots() if r['mode'] == 'NEW' and r['block'] == 'B0']
        for row in rows:
            with self.subTest(row=row['slot_id']):
                state, manifest = fixture(row)
                before = copy.deepcopy(state)
                rng = torch.get_rng_state()
                snapshot, digest = prepare_readout(state, manifest, row)
                expected = {'ONLINE': 1, 'E_KEEP': 2, 'E_512': 3}[row['readout']]
                self.assertEqual(snapshot['ema'].weight.item(), expected)
                self.assertEqual(reproducibility.module_state_sha256(snapshot['ema']), digest)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key in ('net', 'ema', 'ema_512'):
                    self.assertTrue(m1._equal_state(state[key].state_dict(), before[key].state_dict()))
                snapshot['ema'].weight.fill_(99)
                self.assertEqual(m1.readout_module(state, row['readout']).weight.item(), expected)

    def test_wrong_milestone_seed_protocol_and_sampler_are_rejected(self):
        row = next(r for r in slots.build_slots() if r['budget_kimg'] == 768)
        for damage in ('attempt', 'seed', 'protocol', 'sampler'):
            state, manifest = fixture(row)
            if damage == 'attempt': state['attempted_iteration'] = 5999
            if damage == 'seed': manifest['seed'] = 56
            if damage == 'protocol': manifest['experiment_protocol'] = 'repair'
            if damage == 'sampler': state['rank_states'][0]['sampler_state']['consumed_samples'] = 767999
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                prepare_readout(state, manifest, row)


if __name__ == '__main__':
    unittest.main()
