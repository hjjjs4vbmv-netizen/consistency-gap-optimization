import copy
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from pathlib import Path

import torch

from training import m1, reproducibility, schedule_switch, state_interventions as intervention


def initialized_model(reverse=False):
    model = torch.nn.Module()
    for name in (('b', 'a') if reverse else ('a', 'b')):
        model.register_parameter(name, torch.nn.Parameter(torch.tensor([1., 2.])))
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
    for _ in range(8):
        for name, parameter in model.named_parameters():
            parameter.grad = torch.full_like(parameter, .1 if name == 'a' else .3)
        optimizer.step()
    return model, optimizer


class StateInterventionTests(unittest.TestCase):
    def test_formal_runner_preserves_scientific_failure_without_retry(self):
        from scripts import run_state_interventions as runner
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), gpu=0, code_commit='test',
                                   source_root=Path('/sources'), dataset=Path('/data'))
            manifest = {'source_state': {'path': '/original'}}
            with mock.patch.object(runner, 'wait_sources'), \
                 mock.patch.object(runner.sources, 'manifest', return_value=manifest), \
                 mock.patch.object(runner.schedule_switch, 'load_run_manifest'), \
                 mock.patch.object(runner.sources, 'training_command', return_value=['train']), \
                 mock.patch.object(runner.old, 'runtime_environment', return_value={}), \
                 mock.patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run, \
                 mock.patch.object(runner.old, 'scientific_failure', return_value=True):
                first = runner.run_branch(args, 55, 'L_A')
                second = runner.run_branch(args, 55, 'L_A')
            self.assertEqual(first['status'], 'NUMERICAL_FAILURE')
            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 1)

    def test_first64_observation_does_not_consume_rng_or_change_state(self):
        model, optimizer = initialized_model()
        before = copy.deepcopy(optimizer.state_dict())
        rng = reproducibility.capture_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            intervention.record_attempt(directory, optimizer,
                {'attempted_iteration': 4001}, reproducibility.state_sha256(rng))
            self.assertIn('8.0', (Path(directory) / 'first64.jsonl').read_text())
        self.assertTrue(m1._equal_state(before, optimizer.state_dict()))
        self.assertTrue(m1._equal_state(rng, reproducibility.capture_rng_state()))

    def test_late_runner_requests_only_future_immutable_milestones(self):
        from scripts.state_intervention_sources import training_command
        for branch, expected in (('L_A', '896,1024'), ('X_A_from_B', '640,768,896,1024')):
            command = training_command(Path('/python'), Path('/data'), 55,
                                       Path('/manifest'), Path('/source'), branch)
            self.assertIn('--immutable-checkpoint-kimg=' + expected, command)
            self.assertEqual(sum(v.startswith('--immutable-checkpoint-kimg=') for v in command), 1)

    def test_transplant_matches_names_not_integer_indices(self):
        receiver, optimizer = initialized_model()
        donor, donor_optimizer = initialized_model(reverse=True)
        before = copy.deepcopy(receiver.state_dict())
        groups = copy.deepcopy(optimizer.state_dict()['param_groups'])
        original = copy.deepcopy(donor_optimizer.state_dict())
        rng = torch.get_rng_state()
        intervention.transplant(optimizer, receiver, donor, original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(groups, optimizer.state_dict()['param_groups'])
        for name, parameter in receiver.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
            donor_parameter = dict(donor.named_parameters())[name]
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                actual = optimizer.state[parameter][key]
                expected = donor_optimizer.state[donor_parameter][key]
                self.assertTrue(torch.equal(actual, expected), (name, key))
                self.assertNotEqual(actual.data_ptr(), expected.data_ptr())
        self.assertTrue(m1._equal_state(original, donor_optimizer.state_dict()))

    def test_own_state_reload_matches_next_update(self):
        model, optimizer = initialized_model()
        clone = copy.deepcopy(model)
        own = torch.optim.RAdam(clone.parameters(), lr=1e-4)
        own.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        intervention.transplant(own, clone, model, optimizer.state_dict())
        for current, opt in ((model, optimizer), (clone, own)):
            for p in current.parameters():
                p.grad = torch.ones_like(p)
            opt.step()
        self.assertTrue(m1._equal_state(model.state_dict(), clone.state_dict()))
        self.assertTrue(m1._equal_state(optimizer.state_dict(), own.state_dict()))

    def test_malformed_donor_is_rejected_before_receiver_mutation(self):
        model, optimizer = initialized_model()
        for damage in ('missing', 'unknown', 'shape'):
            state = copy.deepcopy(optimizer.state_dict())
            if damage == 'missing':
                del state['state'][0]
            elif damage == 'unknown':
                state['state'][0]['unexpected'] = 1
            else:
                state['state'][0]['exp_avg'] = torch.zeros(3)
            before = copy.deepcopy(optimizer.state_dict())
            with self.assertRaises(ValueError):
                intervention.transplant(optimizer, model, model, state)
            self.assertTrue(m1._equal_state(before, optimizer.state_dict()))

    def test_late_reset_preserves_groups_and_parameters(self):
        model, optimizer = initialized_model()
        weights = copy.deepcopy(model.state_dict())
        groups = copy.deepcopy(optimizer.state_dict()['param_groups'])
        intervention.apply(optimizer, model, {'experiment_protocol': intervention.M2})
        self.assertFalse(optimizer.state)
        self.assertEqual(groups, optimizer.state_dict()['param_groups'])
        self.assertTrue(m1._equal_state(weights, model.state_dict()))

    def test_metadata_keeps_global_counter_separate_from_optimizer_step(self):
        manifest = dict(experiment_protocol=intervention.SWAP, branch='X_A_from_B',
                        seed=55, source_state={'path': '/receiver'},
                        donor_state={'path': '/donor'})
        metadata = intervention.metadata(manifest, 3989)
        metadata['successful_steps_since_init'] = 7
        state = dict(m1=metadata, ema_512=object(), successful_optimizer_steps=3996)
        self.assertEqual(m1.validate_resumed_state(state, manifest), metadata)
        state['m1']['transplant_applied_count'] = 2
        with self.assertRaises(RuntimeError):
            m1.validate_resumed_state(state, manifest)

    def test_late_init_keeps_768_and_does_not_relabel_curriculum_switch(self):
        manifest = dict(experiment_protocol=intervention.M2, branch='L_A', seed=55,
                        origin_arm='A', continuation_arm='A', run_kind='formal',
                        source_state={'path': '/original-K-768'})
        metadata = intervention.metadata(manifest, 5988)
        self.assertEqual(metadata['initialized_at_nimg'], 768000)
        self.assertEqual(metadata['initialized_emas'], [])
        self.assertEqual(schedule_switch.state_metadata(manifest)['switch_kimg'], 512)
        from tests.test_m1_training_state import M1TrainingStateTests
        state = M1TrainingStateTests().source_state()
        state.update(m1=metadata, cur_nimg=768000, attempted_iteration=6000,
                     successful_optimizer_steps=5988, ema_512=copy.deepcopy(state['net']),
                     schedule_switch=schedule_switch.state_metadata(manifest))
        with tempfile.TemporaryDirectory() as directory:
            saved = m1.save_branch_init_state(state, directory)
            self.assertEqual(Path(saved).name, 'training-state-kimg000768.pt')
            with self.assertRaises(FileExistsError):
                m1.save_branch_init_state(state, directory)


if __name__ == '__main__':
    unittest.main()
