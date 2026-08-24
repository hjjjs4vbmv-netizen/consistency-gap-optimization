import copy
import hashlib
import itertools
import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from torch_utils.misc import InfiniteSampler
from training import reproducibility
from training.ct_training_loop import (
    canonical_processed_nimg,
    copy_module_state_exact,
    enforce_generic_exact_finite,
    enforce_generic_exact_finite_before_sanitization,
)
from training.phema import PowerFunctionEMA


def take(iterator, count):
    return list(itertools.islice(iterator, count))


class TinyStateModule(torch.nn.Module):
    def __init__(self, width=3, *, extra=False, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(width, dtype=dtype))
        self.register_buffer('running', torch.ones(width, dtype=dtype))
        if extra:
            self.register_buffer('extra', torch.zeros(1, dtype=dtype))


class ExactModuleTransferTest(unittest.TestCase):
    def test_exact_module_identity_copies_every_tensor(self):
        source = TinyStateModule()
        destination = copy.deepcopy(source)
        with torch.no_grad():
            source.weight.add_(10)
            source.running.mul_(7)
        copy_module_state_exact(source, destination, label='test')
        self.assertTrue(torch.equal(source.weight, destination.weight))
        self.assertTrue(torch.equal(source.running, destination.running))

    def test_missing_extra_shape_and_dtype_fail_closed(self):
        incompatible = [
            (TinyStateModule(extra=True), TinyStateModule()),
            (TinyStateModule(), TinyStateModule(extra=True)),
            (TinyStateModule(width=4), TinyStateModule(width=3)),
            (TinyStateModule(dtype=torch.float64), TinyStateModule()),
        ]
        for source, destination in incompatible:
            with self.subTest(
                source=source.state_dict(), destination=destination.state_dict()
            ), self.assertRaises(RuntimeError):
                copy_module_state_exact(source, destination, label='test')

    def test_only_an_explicit_content_bound_source_extra_is_allowed(self):
        source = TinyStateModule(extra=True)
        destination = TinyStateModule()
        with torch.no_grad():
            source.weight.add_(10)
            source.running.mul_(7)
        extra = source.extra.detach().cpu().contiguous()
        policy = {
            'extra': {
                'shape': list(extra.shape),
                'dtype': str(extra.dtype),
                'tensor_bytes_sha256': hashlib.sha256(
                    extra.numpy().tobytes()
                ).hexdigest(),
                'reason': 'unit-test-only unused source tensor',
            }
        }
        copy_module_state_exact(
            source,
            destination,
            label='test',
            allowed_source_extras=policy,
        )
        self.assertTrue(torch.equal(source.weight, destination.weight))
        self.assertTrue(torch.equal(source.running, destination.running))

        invalid = copy.deepcopy(policy)
        invalid['extra']['tensor_bytes_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'source-extra identity mismatch'):
            copy_module_state_exact(
                source,
                TinyStateModule(),
                label='test',
                allowed_source_extras=invalid,
            )

    def test_donor_extras_are_allowed_but_every_target_is_required(self):
        source = TinyStateModule(extra=True)
        destination = TinyStateModule()
        copy_module_state_exact(
            source,
            destination,
            label='donor',
            allow_unlisted_source_extras=True,
        )
        self.assertTrue(torch.equal(source.weight, destination.weight))
        self.assertTrue(torch.equal(source.running, destination.running))

        with self.assertRaisesRegex(RuntimeError, 'missing'):
            copy_module_state_exact(
                TinyStateModule(),
                TinyStateModule(extra=True),
                label='donor',
                allow_unlisted_source_extras=True,
            )


class ProcessedNimgContractTest(unittest.TestCase):
    def test_integral_float_is_canonicalized_for_csv_and_resume(self):
        for value in (0, 4096, 4096.0, np.int64(4096)):
            with self.subTest(value=value):
                result = canonical_processed_nimg(value)
                self.assertEqual(result, int(value))
                self.assertIs(type(result), int)

    def test_non_integral_or_nonfinite_progress_fails_closed(self):
        for value in (True, -1, 0.5, float('nan'), float('inf'), 'invalid'):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                canonical_processed_nimg(value)

    def test_generic_exact_rejects_before_nonfinite_sanitization(self):
        parameter = torch.nn.Parameter(torch.ones([]))
        parameter.grad = torch.tensor(float('inf'))
        with self.assertRaisesRegex(FloatingPointError, 'raw gradient'):
            enforce_generic_exact_finite_before_sanitization(
                [torch.zeros([])], [parameter], torch.device('cpu')
            )
        self.assertTrue(torch.isinf(parameter.grad))

        parameter.grad = torch.zeros([])
        with self.assertRaisesRegex(FloatingPointError, 'loss'):
            enforce_generic_exact_finite_before_sanitization(
                [torch.tensor(float('nan'))],
                [parameter],
                torch.device('cpu'),
            )

        enforce_generic_exact_finite(
            'loss/gradient before sanitization',
            {'loss': 0, 'raw gradient': 0},
        )

    def test_generic_exact_rejects_nonfinite_model_and_power_ema_state(self):
        for name in ('optimizer update/model', 'EMA', 'PowerEMA'):
            with self.subTest(name=name), self.assertRaisesRegex(
                FloatingPointError, name.split('/')[-1]
            ):
                enforce_generic_exact_finite(
                    'state update', {name: 1}
                )


class InfiniteSamplerReplayTest(unittest.TestCase):
    def test_zero_cursor_is_identical_to_default_sequence(self):
        dataset = list(range(19))
        default = InfiniteSampler(dataset, seed=17, window_size=0.5)
        explicit = InfiniteSampler(
            dataset, seed=17, window_size=0.5, start_sample=0
        )
        self.assertEqual(take(iter(default), 200), take(iter(explicit), 200))

    def test_each_rank_resume_matches_baseline_suffix(self):
        dataset = list(range(23))
        for rank in range(3):
            with self.subTest(rank=rank):
                baseline_sampler = InfiniteSampler(
                    dataset,
                    rank=rank,
                    num_replicas=3,
                    seed=91,
                    window_size=0.5,
                )
                baseline = take(iter(baseline_sampler), 180)
                state = baseline_sampler.state_dict(consumed_samples=73)
                resumed_sampler = InfiniteSampler(
                    dataset,
                    rank=rank,
                    num_replicas=3,
                    seed=91,
                    window_size=0.5,
                )
                resumed_sampler.load_state_dict(state)
                self.assertEqual(
                    take(iter(resumed_sampler), 80), baseline[73:153]
                )

    def test_prefetch_requests_do_not_define_committed_cursor(self):
        dataset = list(range(29))
        baseline_sampler = InfiniteSampler(dataset, seed=5, window_size=0.5)
        baseline = take(iter(baseline_sampler), 160)

        prefetched_iterator = iter(
            InfiniteSampler(dataset, seed=5, window_size=0.5)
        )
        take(prefetched_iterator, 64)  # only 32 are declared consumed below
        committed_state = InfiniteSampler(
            dataset, seed=5, window_size=0.5
        ).state_dict(consumed_samples=32)
        resumed = InfiniteSampler(dataset, seed=5, window_size=0.5)
        resumed.load_state_dict(committed_state)
        self.assertEqual(take(iter(resumed), 80), baseline[32:112])

    def test_state_identity_and_cursor_are_fail_closed(self):
        sampler = InfiniteSampler(list(range(11)), seed=3)
        state = sampler.state_dict(consumed_samples=4)
        for field, value in [
            ('dataset_size', 12),
            ('seed', 4),
            ('rank', 1),
            ('consumed_samples', None),
        ]:
            invalid = dict(state)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                sampler.load_state_dict(invalid)


class RngReplayTest(unittest.TestCase):
    def test_python_numpy_and_cpu_torch_roundtrip(self):
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        state = reproducibility.capture_rng_state()
        expected = (
            random.random(),
            np.random.standard_normal(8),
            torch.randn(8),
        )
        random.random()
        np.random.standard_normal(23)
        torch.randn(23)
        reproducibility.restore_rng_state(state)
        actual = (
            random.random(),
            np.random.standard_normal(8),
            torch.randn(8),
        )
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_current_device_cpu_rng_roundtrip(self):
        random.seed(501)
        np.random.seed(502)
        torch.manual_seed(503)
        state = reproducibility.capture_current_device_rng_state('cpu')
        expected = (random.random(), np.random.randn(4), torch.randn(4))
        random.random()
        np.random.randn(9)
        torch.randn(9)
        reproducibility.restore_current_device_rng_state(state, 'cpu')
        actual = (random.random(), np.random.randn(4), torch.randn(4))
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))


class PowerEmaStateTest(unittest.TestCase):
    def test_profiles_roundtrip_without_changing_contract(self):
        source = TinyStateModule()
        tracker = PowerFunctionEMA(source, stds=(0.01, 0.05, 0.1))
        with torch.no_grad():
            source.weight.add_(3)
        tracker.update(cur_nimg=128, batch_size=128)
        state = tracker.state_dict()
        self.assertEqual(state['stds'], (0.01, 0.05, 0.1))
        self.assertEqual(len(state['emas']), 3)

        restored = PowerFunctionEMA(
            TinyStateModule(), stds=(0.01, 0.05, 0.1)
        )
        restored.load_state_dict(state)
        for expected, actual in zip(tracker.emas, restored.emas):
            for key in expected.state_dict():
                self.assertTrue(torch.equal(
                    expected.state_dict()[key], actual.state_dict()[key]
                ))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_all_cuda_rng_states_roundtrip(self):
        torch.cuda.manual_seed_all(404)
        state = reproducibility.capture_rng_state()
        expected = [
            torch.randn(8, device=torch.device('cuda', index))
            for index in range(torch.cuda.device_count())
        ]
        for index in range(torch.cuda.device_count()):
            torch.randn(31, device=torch.device('cuda', index))
        reproducibility.restore_rng_state(state)
        actual = [
            torch.randn(8, device=torch.device('cuda', index))
            for index in range(torch.cuda.device_count())
        ]
        for first, second in zip(actual, expected):
            self.assertTrue(torch.equal(first, second))


class StateDigestAndAtomicWriteTest(unittest.TestCase):
    def test_canonical_config_hash_survives_json_roundtrip(self):
        value = {
            'tuple': (1, np.int64(2), (3.5,)),
            'list': [True, None, 'value'],
        }
        canonical = reproducibility.canonical_json_data(value)
        roundtripped = json.loads(json.dumps(canonical))
        self.assertEqual(canonical, roundtripped)
        self.assertEqual(
            reproducibility.state_sha256(canonical),
            reproducibility.state_sha256(roundtripped),
        )

    def test_state_digest_is_order_independent_for_dicts_and_tensor_exact(self):
        first = {'b': torch.tensor([1.0, 2.0]), 'a': [3, 4]}
        second = {'a': [3, 4], 'b': torch.tensor([1.0, 2.0])}
        changed = {'a': [3, 4], 'b': torch.tensor([1.0, 2.0001])}
        self.assertEqual(
            reproducibility.state_sha256(first),
            reproducibility.state_sha256(second),
        )
        self.assertNotEqual(
            reproducibility.state_sha256(first),
            reproducibility.state_sha256(changed),
        )

    def test_numbered_artifact_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'numbered.json')
            reproducibility.atomic_json_dump({'value': 1}, path)
            with self.assertRaises(FileExistsError):
                reproducibility.atomic_json_dump({'value': 2}, path)
            with open(path, 'rt', encoding='utf-8') as handle:
                self.assertIn('"value": 1', handle.read())

    def test_failed_latest_write_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'latest.bin')
            reproducibility._atomic_write(  # pylint: disable=protected-access
                path, lambda handle: handle.write(b'old'), overwrite=True
            )

            def fail_after_write(handle):
                handle.write(b'new')
                raise RuntimeError('injected serialization failure')

            with self.assertRaises(RuntimeError):
                reproducibility._atomic_write(  # pylint: disable=protected-access
                    path, fail_after_write, overwrite=True
                )
            with open(path, 'rb') as handle:
                self.assertEqual(handle.read(), b'old')
            self.assertEqual(
                [name for name in os.listdir(directory) if '.tmp-' in name],
                [],
            )


if __name__ == '__main__':
    unittest.main()
