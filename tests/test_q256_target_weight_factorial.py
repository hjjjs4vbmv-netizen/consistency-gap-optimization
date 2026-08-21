import contextlib
import io
import unittest

import torch

from training import reproducibility
from training.loss import (
    ECMLoss,
    TARGET_WEIGHT_FACTORIAL_ARMS,
    TARGET_WEIGHT_FACTORIAL_PROTOCOL,
    compute_target_weight_times,
    resolve_target_weight_factorial,
)
from training.schedules import get_schedule


GRADIENT_SCALING_RTOL = 2e-6
GRADIENT_SCALING_ATOL = 1e-6


def make_loss(**kwargs):
    defaults = dict(q=256, k=8, b=1, c=0, adj='sigmoid')
    defaults.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        loss_fn = ECMLoss(**defaults)
    loss_fn.update_schedule(0)
    return loss_fn


def normalized_native_schedule_metadata(loss_fn):
    """Project native and factorized configurations onto shared semantics.

    Native g=1.10 is represented by ``global_sigmoid`` while the factorial
    protocol deliberately keeps the base schedule at ``sigmoid`` and selects
    target/denominator sources independently.  Comparing the raw metadata
    dictionaries would therefore compare implementation shape, not the
    realized native A/B semantics.
    """
    metadata = loss_fn.schedule_metadata()
    factorial = loss_fn.factorial
    if factorial['enabled']:
        target_gap_scale = factorial['target_gap_scale']
        denominator_gap_scale = factorial['denominator_gap_scale']
    elif metadata['name'] == 'sigmoid':
        target_gap_scale = 1.0
        denominator_gap_scale = 1.0
    elif metadata['name'] == 'global_sigmoid':
        target_gap_scale = metadata['global_gap_scale']
        denominator_gap_scale = metadata['global_gap_scale']
    else:  # pragma: no cover - helper is intentionally native-A/B-only.
        raise AssertionError(
            f'unsupported native parity schedule: {metadata["name"]!r}'
        )
    return {
        'base_schedule': 'sigmoid',
        'q': float(metadata['q']),
        'k': float(metadata['k']),
        'b': float(metadata['b']),
        'stage': int(metadata['stage']),
        'ratio': float(metadata['ratio']),
        'target_gap_scale': float(target_gap_scale),
        'denominator_gap_scale': float(denominator_gap_scale),
    }


def realized_native_trace(calls):
    """Extract the sampled native pair and its realized per-sample gap."""
    if len(calls) != 2:
        raise AssertionError(f'native A/B loss must make two calls, got {len(calls)}')
    source, target = calls
    denominator = source['t'] - target['t']
    return {
        'sampled_t': source['t'].detach().clone(),
        'sampled_r': target['t'].detach().clone(),
        'target_input': target['x'].detach().clone(),
        'target_output': target['output'].detach().clone(),
        'realized_denominator': denominator.detach().clone(),
    }


class RecordingDropoutNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.625))
        self.calls = []

    def forward(self, x, t, labels=None, augment_labels=None):
        del labels, augment_labels
        mask = (torch.rand_like(x) > 0.25).to(x.dtype)
        output = (x * self.weight + t) * mask
        self.calls.append({
            'x': x.detach().clone(),
            't': t.detach().clone(),
            'output': output.detach().clone(),
            'mask': mask.detach().clone(),
            'grad_enabled': torch.is_grad_enabled(),
        })
        return output


class MultiParameterRecordingNet(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([0.625, -0.375], device=device).reshape(1, 2, 1, 1)
        )
        self.bias = torch.nn.Parameter(
            torch.tensor([0.125, -0.25], device=device).reshape(1, 2, 1, 1)
        )
        self.calls = []

    def forward(self, x, t, labels=None, augment_labels=None):
        del labels, augment_labels
        mask = (torch.rand_like(x) > 0.25).to(x.dtype)
        output = (x * self.weight + self.bias + t) * mask
        self.calls.append({
            'x': x.detach().clone(),
            't': t.detach().clone(),
            'output': output.detach().clone(),
            'mask': mask.detach().clone(),
            'grad_enabled': torch.is_grad_enabled(),
        })
        return output


class SyntheticBoundarySchedule:
    name = 'sigmoid'

    def compute_r(self, t, stage):
        del stage
        ratios = torch.tensor(
            [0.9, 0.2, 0.0, 0.95], dtype=t.dtype, device=t.device
        ).reshape_as(t)
        return t * ratios

    def preclip_gap_scale(self, t):
        return torch.ones_like(t)

    def runtime_metrics(self):
        return {
            'loss_ema': None,
            'loss_reference': None,
            'correction': 0.0,
            'signal_updates': 0,
            'adaptive_active': False,
        }


def run_loss(loss_fn, seed=20260819):
    net = RecordingDropoutNet()
    images = torch.linspace(-1, 1, 4 * 2 * 3 * 3).reshape(4, 2, 3, 3)
    torch.manual_seed(seed)
    per_sample = loss_fn(net=net, images=images)
    reduced = per_sample.mean()
    reduced.backward()
    calls = net.calls
    has_schedule_metadata = hasattr(loss_fn.schedule, 'metadata')
    return {
        'images': images.detach().clone(),
        'loss': per_sample.detach().clone(),
        'reduced': reduced.detach().clone(),
        'gradient': net.weight.grad.detach().clone(),
        'calls': calls,
        'native_trace': realized_native_trace(calls),
        'schedule_metadata': (
            loss_fn.schedule_metadata() if has_schedule_metadata else None
        ),
        'normalized_schedule_metadata': (
            normalized_native_schedule_metadata(loss_fn)
            if has_schedule_metadata else None
        ),
        'telemetry': loss_fn.factorial_runtime_metrics(),
        'post_rng': torch.get_rng_state().clone(),
    }


def factorized(arm):
    factors = {
        derived: pair for pair, derived in TARGET_WEIGHT_FACTORIAL_ARMS.items()
    }[arm]
    return make_loss(
        factorial_protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
        target_gap_scale=factors[0],
        denominator_gap_scale=factors[1],
    )


def run_cuda_loss(loss_fn, seed=20260819):
    device = torch.device('cuda')
    net = MultiParameterRecordingNet(device)
    images = torch.linspace(
        -1, 1, 4 * 2 * 3 * 3, device=device
    ).reshape(4, 2, 3, 3)
    torch.cuda.manual_seed_all(seed)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        per_sample = loss_fn(net=net, images=images)
        reduced = per_sample.mean()
    reduced.backward()
    calls = net.calls
    return {
        'loss': per_sample.detach().clone(),
        'reduced': reduced.detach().clone(),
        'gradients': {
            name: parameter.grad.detach().clone()
            for name, parameter in net.named_parameters()
        },
        'calls': calls,
        'native_trace': realized_native_trace(calls),
        'schedule_metadata': loss_fn.schedule_metadata(),
        'normalized_schedule_metadata': (
            normalized_native_schedule_metadata(loss_fn)
        ),
        'telemetry': loss_fn.factorial_runtime_metrics(),
        'post_rng': torch.cuda.get_rng_state().clone(),
    }


def run_gradient_identity_loss(loss_fn, seed=20260819):
    """Run one deterministic CPU arm for the frozen gradient manipulation check."""
    device = torch.device('cpu')
    net = MultiParameterRecordingNet(device)
    images = torch.linspace(
        -1, 1, 4 * 2 * 3 * 3, device=device
    ).reshape(4, 2, 3, 3)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in net.named_parameters()
    }
    torch.manual_seed(seed)
    per_sample = loss_fn(net=net, images=images)
    reduced = per_sample.mean()
    reduced.backward()
    source_t = net.calls[0]['t']
    base_r = loss_fn.schedule.compute_r(t=source_t, stage=loss_fn.stage)
    r_target, r_denominator, _, delta_denominator = (
        compute_target_weight_times(
            source_t,
            base_r,
            target_gap_scale=loss_fn.factorial['target_gap_scale'],
            denominator_gap_scale=(
                loss_fn.factorial['denominator_gap_scale']
            ),
        )
    )
    return {
        'images': images.detach().clone(),
        'initial_parameters': initial_parameters,
        'gradients': {
            name: parameter.grad.detach().clone()
            for name, parameter in net.named_parameters()
        },
        'calls': net.calls,
        'r_target': r_target.detach().clone(),
        'r_denominator': r_denominator.detach().clone(),
        'delta_denominator': delta_denominator.detach().clone(),
        'telemetry': loss_fn.factorial_runtime_metrics(),
        'post_rng': torch.get_rng_state().clone(),
    }


def gradient_scaling_residual(reference, scaled):
    """Return auditable residuals for G_scaled = G_reference / 1.10."""
    max_absolute_residual = 0.0
    max_relative_residual = 0.0
    max_expected_absolute = 0.0
    for name, reference_gradient in reference['gradients'].items():
        expected = reference_gradient / 1.1
        observed = scaled['gradients'][name]
        residual = (observed - expected).abs()
        max_absolute_residual = max(
            max_absolute_residual, float(residual.max())
        )
        max_expected_absolute = max(
            max_expected_absolute, float(expected.abs().max())
        )
        nonzero = expected != 0
        if bool(nonzero.any()):
            max_relative_residual = max(
                max_relative_residual,
                float((residual[nonzero] / expected[nonzero].abs()).max()),
            )

    expected_denominator = reference['delta_denominator'] * 1.1
    denominator_residual = (
        scaled['delta_denominator'] - expected_denominator
    ).abs()
    denominator_nonzero = expected_denominator != 0
    return {
        'gradient_max_absolute_residual': max_absolute_residual,
        'gradient_max_relative_residual': max_relative_residual,
        'gradient_max_expected_absolute': max_expected_absolute,
        'denominator_max_absolute_residual': float(
            denominator_residual.max()
        ),
        'denominator_max_relative_residual': float(
            (
                denominator_residual[denominator_nonzero]
                / expected_denominator[denominator_nonzero].abs()
            ).max()
        ),
    }


class FactorialConfigurationTest(unittest.TestCase):
    def test_all_four_arm_labels_are_derived_from_explicit_factors(self):
        for factors, expected_arm in TARGET_WEIGHT_FACTORIAL_ARMS.items():
            with self.subTest(arm=expected_arm):
                resolved = resolve_target_weight_factorial(
                    TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                    factors[0],
                    factors[1],
                    adj='sigmoid',
                    global_gap_scale=1.0,
                    q=256,
                    c=0,
                )
                self.assertTrue(resolved['enabled'])
                self.assertEqual(resolved['arm'], expected_arm)

    def test_protocol_fails_closed_on_partial_or_off_protocol_config(self):
        invalid = [
            dict(protocol='none', target_gap_scale=1.0),
            dict(protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                 target_gap_scale=1.0, denominator_gap_scale=None),
            dict(protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                 target_gap_scale=1.2, denominator_gap_scale=1.0),
            dict(protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                 target_gap_scale=1.0, denominator_gap_scale=1.0, q=128),
            dict(protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                 target_gap_scale=1.0, denominator_gap_scale=1.0,
                 adj='global_sigmoid'),
            dict(protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                 target_gap_scale=1.0, denominator_gap_scale=1.0,
                 global_gap_scale=1.1),
        ]
        for kwargs in invalid:
            defaults = dict(
                denominator_gap_scale=None,
                adj='sigmoid',
                global_gap_scale=1.0,
                q=256,
                c=0,
            )
            defaults.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_target_weight_factorial(**defaults)

    def test_legacy_config_remains_factorial_disabled(self):
        resolved = resolve_target_weight_factorial()
        self.assertFalse(resolved['enabled'])
        self.assertIsNone(resolved['arm'])


class RealizedPerSampleDenominatorTest(unittest.TestCase):
    def test_clamp_is_applied_before_the_realized_denominator(self):
        t = torch.ones([3, 1, 1, 1], dtype=torch.float64)
        base_r = torch.tensor([0.9, 0.2, 0.0], dtype=torch.float64).reshape_as(t)
        r_target, r_weight, delta_target, delta_weight = (
            compute_target_weight_times(
                t,
                base_r,
                target_gap_scale=1.0,
                denominator_gap_scale=1.1,
            )
        )
        self.assertTrue(torch.equal(r_target, base_r))
        self.assertTrue(torch.equal(delta_target, t - base_r))
        self.assertTrue(torch.equal(r_weight[-1], torch.zeros_like(r_weight[-1])))
        self.assertTrue(torch.equal(delta_weight[-1], torch.ones_like(delta_weight[-1])))
        scalar_approximation = (t - base_r) * 1.1
        self.assertFalse(torch.equal(delta_weight, scalar_approximation))
        ratios = delta_weight.flatten() / (t - base_r).flatten()
        self.assertGreater(float(ratios.max() - ratios.min()), 0.09)

    def test_nonpositive_or_nonfinite_gaps_fail(self):
        t = torch.ones([2, 1, 1, 1])
        invalid_base_times = [
            t.clone(),
            torch.tensor([0.5, float('nan')]).reshape_as(t),
            torch.tensor([0.5, -0.1]).reshape_as(t),
            torch.tensor([0.5, 1.1]).reshape_as(t),
        ]
        for base_r in invalid_base_times:
            with self.subTest(base_r=base_r), self.assertRaises(FloatingPointError):
                compute_target_weight_times(
                    t,
                    base_r,
                    target_gap_scale=1.0,
                    denominator_gap_scale=1.1,
                )

        for invalid_t in [torch.zeros_like(t), torch.full_like(t, float('inf'))]:
            with self.subTest(t=invalid_t), self.assertRaises(FloatingPointError):
                compute_target_weight_times(
                    invalid_t,
                    torch.zeros_like(invalid_t),
                    target_gap_scale=1.0,
                    denominator_gap_scale=1.1,
                )


class CanonicalParityTest(unittest.TestCase):
    def assert_run_equal(self, first, second):
        self.assertTrue(torch.equal(first['loss'], second['loss']))
        self.assertTrue(torch.equal(first['reduced'], second['reduced']))
        self.assertTrue(torch.equal(first['gradient'], second['gradient']))
        for field in (
            'sampled_t', 'sampled_r', 'target_input', 'target_output',
            'realized_denominator',
        ):
            self.assertTrue(torch.equal(
                first['native_trace'][field], second['native_trace'][field]
            ), field)
        self.assertEqual(
            first['normalized_schedule_metadata'],
            second['normalized_schedule_metadata'],
        )
        self.assertTrue(torch.equal(first['post_rng'], second['post_rng']))

    def assert_factorial_denominator_is_bound(self, result):
        telemetry = result['telemetry']
        self.assertIsNotNone(telemetry)
        self.assertEqual(
            telemetry['target_gap_scale'],
            telemetry['denominator_gap_scale'],
        )
        denominator = result['native_trace']['realized_denominator']
        self.assertTrue(bool((denominator > 0).all()))
        self.assertEqual(
            telemetry['denominator_delta_sha256'],
            reproducibility.state_sha256(denominator),
        )

    def test_A_is_bitwise_equal_to_native_sigmoid(self):
        canonical = run_loss(make_loss())
        factorial = run_loss(factorized('A'))
        self.assert_run_equal(canonical, factorial)
        self.assertEqual(canonical['schedule_metadata']['name'], 'sigmoid')
        self.assertEqual(factorial['schedule_metadata']['name'], 'sigmoid')
        self.assert_factorial_denominator_is_bound(factorial)

    def test_B_is_bitwise_equal_to_native_global_sigmoid_g110(self):
        canonical = run_loss(
            make_loss(adj='global_sigmoid', global_gap_scale=1.1)
        )
        factorial = run_loss(factorized('B'))
        self.assert_run_equal(canonical, factorial)
        self.assertEqual(
            canonical['schedule_metadata']['name'], 'global_sigmoid'
        )
        self.assertEqual(
            canonical['schedule_metadata']['global_gap_scale'], 1.1
        )
        self.assertEqual(factorial['schedule_metadata']['name'], 'sigmoid')
        self.assertEqual(factorial['telemetry']['target_gap_scale'], 1.1)
        self.assertEqual(factorial['telemetry']['denominator_gap_scale'], 1.1)
        self.assert_factorial_denominator_is_bound(factorial)

    def test_same_target_factorial_identities_are_elementwise_exact(self):
        runs = {arm: run_loss(factorized(arm)) for arm in 'ABCD'}
        for first, second in [('A', 'D'), ('B', 'C')]:
            with self.subTest(pair=(first, second)):
                first_target = runs[first]['calls'][1]
                second_target = runs[second]['calls'][1]
                for key in ('t', 'x', 'output', 'mask'):
                    self.assertTrue(torch.equal(
                        first_target[key], second_target[key]
                    ))
        self.assertFalse(torch.equal(runs['A']['loss'], runs['D']['loss']))
        self.assertFalse(torch.equal(runs['B']['loss'], runs['C']['loss']))

    def test_clip_free_denominator_gradient_scaling_identities(self):
        runs = {
            arm: run_gradient_identity_loss(factorized(arm))
            for arm in 'ABCD'
        }
        for arm in 'BCD':
            self.assertTrue(torch.equal(
                runs['A']['images'], runs[arm]['images']
            ))
            self.assertEqual(
                runs['A']['initial_parameters'].keys(),
                runs[arm]['initial_parameters'].keys(),
            )
            for name in runs['A']['initial_parameters']:
                self.assertTrue(torch.equal(
                    runs['A']['initial_parameters'][name],
                    runs[arm]['initial_parameters'][name],
                ))
            self.assertTrue(torch.equal(
                runs['A']['post_rng'], runs[arm]['post_rng']
            ))

        for first, second in [('A', 'D'), ('C', 'B')]:
            with self.subTest(identity=f'G_{second}=G_{first}/1.10'):
                for key in ('t', 'x', 'output', 'mask'):
                    self.assertTrue(torch.equal(
                        runs[first]['calls'][1][key],
                        runs[second]['calls'][1][key],
                    ), key)
                self.assertTrue(torch.equal(
                    runs[first]['r_target'], runs[second]['r_target']
                ))
                self.assertEqual(
                    runs[second]['telemetry'][
                        'denominator_scaled_to_zero_count'
                    ],
                    0,
                )
                self.assertTrue(bool(
                    (runs[second]['r_denominator'] > 0).all()
                ))
                torch.testing.assert_close(
                    runs[second]['delta_denominator'],
                    runs[first]['delta_denominator'] * 1.1,
                    rtol=GRADIENT_SCALING_RTOL,
                    atol=GRADIENT_SCALING_ATOL,
                )
                for name, reference_gradient in (
                    runs[first]['gradients'].items()
                ):
                    torch.testing.assert_close(
                        runs[second]['gradients'][name],
                        reference_gradient / 1.1,
                        rtol=GRADIENT_SCALING_RTOL,
                        atol=GRADIENT_SCALING_ATOL,
                    )

                residual = gradient_scaling_residual(
                    runs[first], runs[second]
                )
                allowed_at_max = (
                    GRADIENT_SCALING_ATOL
                    + GRADIENT_SCALING_RTOL
                    * residual['gradient_max_expected_absolute']
                )
                self.assertLessEqual(
                    residual['gradient_max_absolute_residual'],
                    allowed_at_max,
                )

    def test_target_branch_is_stop_gradient_and_dropout_is_paired(self):
        result = run_loss(factorized('C'))
        self.assertTrue(result['calls'][0]['grad_enabled'])
        self.assertFalse(result['calls'][1]['grad_enabled'])
        self.assertTrue(torch.equal(
            result['calls'][0]['mask'], result['calls'][1]['mask']
        ))

    def test_same_batch_gradient_rerun_and_arm_order_are_bitwise(self):
        forward = {arm: run_loss(factorized(arm)) for arm in 'ABCD'}
        reverse = {arm: run_loss(factorized(arm)) for arm in 'DCBA'}
        for arm in 'ABCD':
            with self.subTest(arm=arm):
                self.assert_run_equal(forward[arm], reverse[arm])

    def test_runtime_telemetry_is_versioned_and_factor_specific(self):
        result = run_loss(factorized('D'))
        telemetry = result['telemetry']
        self.assertEqual(telemetry['schema'], 'ect.q256.target-weight-runtime/v1')
        self.assertEqual(telemetry['arm'], 'D')
        self.assertEqual(telemetry['target_gap_scale'], 1.0)
        self.assertEqual(telemetry['denominator_gap_scale'], 1.1)
        self.assertEqual(telemetry['sample_count'], 4)
        self.assertEqual(telemetry['nonfinite_count'], 0)
        self.assertEqual(telemetry['nonpositive_denominator_count'], 0)
        self.assertGreater(telemetry['denominator_delta_min'], 0)

    def test_factorial_uses_native_schedule_clamp_path(self):
        t = torch.tensor([1e-4, 0.1, 1.0, 10.0]).reshape(-1, 1, 1, 1)
        base = get_schedule('sigmoid', q=256).compute_r(t=t, stage=0)
        native = get_schedule(
            'global_sigmoid', q=256, global_gap_scale=1.1
        ).compute_r(t=t, stage=0)
        target, weight, _, _ = compute_target_weight_times(
            t,
            base,
            target_gap_scale=1.1,
            denominator_gap_scale=1.1,
        )
        self.assertTrue(torch.equal(target, native))
        self.assertIs(target, weight)

    def test_real_loss_wires_the_realized_per_sample_denominator(self):
        loss_fn = factorized('D')
        loss_fn.schedule = SyntheticBoundarySchedule()
        result = run_loss(loss_fn)
        source, target = result['calls']
        t = source['t']
        base_r = loss_fn.schedule.compute_r(t, stage=0)
        _, _, _, denominator = compute_target_weight_times(
            t,
            base_r,
            target_gap_scale=1.0,
            denominator_gap_scale=1.1,
        )
        target_mask = target['t'] > 0
        effective_target = torch.nan_to_num(target['output'])
        effective_target = (
            target_mask * effective_target
            + (~target_mask) * result['images']
        )
        numerator = (
            (source['output'] - effective_target)
            .square()
            .reshape(t.shape[0], -1)
            .sum(dim=1)
            .sqrt()
        )
        expected = numerator / denominator.flatten()
        self.assertTrue(torch.equal(result['loss'], expected))
        scalar_approximation = numerator / (
            (t - base_r).flatten() * 1.1
        )
        self.assertFalse(torch.equal(expected, scalar_approximation))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_cuda_amp_A_and_B_match_native_full_forward_gradients_and_rng(self):
        comparisons = [
            (make_loss(), factorized('A')),
            (
                make_loss(adj='global_sigmoid', global_gap_scale=1.1),
                factorized('B'),
            ),
        ]
        for canonical_loss, factorial_loss in comparisons:
            canonical = run_cuda_loss(canonical_loss)
            factorial_result = run_cuda_loss(factorial_loss)
            with self.subTest(arm=factorial_loss.factorial['arm']):
                self.assertTrue(torch.equal(
                    canonical['loss'], factorial_result['loss']
                ))
                self.assertTrue(torch.equal(
                    canonical['reduced'], factorial_result['reduced']
                ))
                for field in (
                    'sampled_t', 'sampled_r', 'target_input', 'target_output',
                    'realized_denominator',
                ):
                    self.assertTrue(torch.equal(
                        canonical['native_trace'][field],
                        factorial_result['native_trace'][field],
                    ), field)
                self.assertEqual(
                    canonical['normalized_schedule_metadata'],
                    factorial_result['normalized_schedule_metadata'],
                )
                self.assert_factorial_denominator_is_bound(factorial_result)
                self.assertEqual(
                    canonical['gradients'].keys(),
                    factorial_result['gradients'].keys(),
                )
                for name in canonical['gradients']:
                    self.assertTrue(torch.equal(
                        canonical['gradients'][name],
                        factorial_result['gradients'][name],
                    ))
                self.assertEqual(
                    len(canonical['calls']), len(factorial_result['calls'])
                )
                for first, second in zip(
                    canonical['calls'], factorial_result['calls']
                ):
                    for key in ('x', 't', 'output', 'mask'):
                        self.assertTrue(torch.equal(first[key], second[key]))
                    self.assertEqual(
                        first['grad_enabled'], second['grad_enabled']
                    )
                self.assertTrue(torch.equal(
                    canonical['post_rng'], factorial_result['post_rng']
                ))


if __name__ == '__main__':
    unittest.main()
