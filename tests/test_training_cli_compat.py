import contextlib
import io
import unittest

import click
import dnnlib
import torch
from click.testing import CliRunner

import ct_train
from torch_utils import misc
from training.loss import ECMLoss, TARGET_WEIGHT_FACTORIAL_PROTOCOL
from training.networks_edm2 import MPConv


def parse_train_args(*extra_args):
    args = ['--outdir', 'out', '--data', 'dataset', *extra_args]
    with ct_train.main.make_context('ct_train.py', args) as ctx:
        return dict(ctx.params)


class TrainingCliCompatibilityTest(unittest.TestCase):
    def test_edm2_constant_inherits_reference_dtype_and_device(self):
        reference = torch.empty(1, dtype=torch.float64)
        value = misc.const_like(reference, [1, 2])
        self.assertEqual(value.dtype, reference.dtype)
        self.assertEqual(value.device, reference.device)

    def test_no_new_option_keeps_legacy_sigmoid_default(self):
        params = parse_train_args()
        self.assertEqual(params['mapping'], 'sigmoid')
        self.assertNotIn('schedule', params)
        self.assertEqual(params['adaptive_loss_ema_beta'], 0.9)
        self.assertEqual(params['adaptive_update_kimg'], 0.5)
        self.assertEqual(params['adaptive_warmup_updates'], 2)
        self.assertEqual(params['adaptive_max_adjust'], 0.05)
        self.assertEqual(params['adaptive_min_gap'], 1e-3)
        self.assertEqual(params['global_gap_scale'], 1.0)
        loss_kwargs = ct_train.make_loss_kwargs(dnnlib.EasyDict(params))
        self.assertNotIn('factorial_protocol', loss_kwargs)
        self.assertNotIn('target_gap_scale', loss_kwargs)
        self.assertNotIn('denominator_gap_scale', loss_kwargs)
        self.assertNotIn('wt', loss_kwargs)

    def test_edm2_img64_s_preset_is_the_frozen_donor_recipe(self):
        params = dnnlib.EasyDict(parse_train_args('--preset', 'edm2-img64-s'))
        ct_train.apply_config_preset(params)
        self.assertEqual(params.arch, 'edm2')
        self.assertEqual(params.cbase, 192)
        self.assertEqual(params.betas, (0.9, 0.99))
        self.assertEqual(params.lr_ref_batches, 2000)
        self.assertEqual(params.lr, 0.001)
        self.assertEqual(params.dropout, 0.4)
        self.assertEqual(params.dropres, 16)
        self.assertEqual(params.augment, 0)
        self.assertFalse(params.xflip)
        self.assertEqual(params.mean, -0.8)
        self.assertEqual(params.std, 1.6)
        self.assertEqual(params.q, 4)
        self.assertEqual(params.k, 8)
        self.assertEqual(params.b, 1)
        self.assertEqual(params.c, 0.06)
        self.assertEqual(params.wt, 'snrpk')
        self.assertEqual(params.power_ema_stds, (0.01, 0.05, 0.1))
        loss_kwargs = ct_train.make_loss_kwargs(params)
        self.assertEqual(loss_kwargs.wt, 'snrpk')

    def test_edm2_img64_s_requires_the_formal_dataset_contract(self):
        class Dataset:
            resolution = 64
            num_channels = 3
            label_dim = 1000

        opts = dnnlib.EasyDict(
            preset='edm2-img64-s', cond=True, xflip=False
        )
        ct_train.validate_edm2_img64_dataset(opts, Dataset())
        for field, value in (
            ('resolution', 32),
            ('num_channels', 1),
            ('label_dim', 10),
        ):
            dataset = Dataset()
            setattr(dataset, field, value)
            with self.subTest(field=field), self.assertRaises(click.ClickException):
                ct_train.validate_edm2_img64_dataset(opts, dataset)
        with self.assertRaises(click.ClickException):
            ct_train.validate_edm2_img64_dataset(
                dnnlib.EasyDict(
                    preset='edm2-img64-s', cond=True, xflip=True
                ),
                Dataset(),
            )

    def test_snrpk_weight_matches_edm2_formula(self):
        with contextlib.redirect_stdout(io.StringIO()):
            loss_fn = ECMLoss(wt='snrpk')
        t = torch.tensor([0.25, 0.5, 1.0])
        expected = (t.square() + 0.5 ** 2) / (t * 0.5).square()
        self.assertTrue(torch.equal(loss_fn.snrplusk_wt(t), expected))

    def test_snrpk_rejects_factorial_denominator_protocol(self):
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            ECMLoss(
                q=256,
                c=0,
                factorial_protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                target_gap_scale=1.0,
                denominator_gap_scale=1.0,
                wt='snrpk',
            )

    def test_edm2_teacher_disables_forced_weight_normalization(self):
        class RecordingNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mpconv = MPConv(1, 1, kernel=[])
                self.scale = torch.nn.Parameter(torch.ones([]))
                self.force_wn_calls = []

            def forward(self, x, _sigma, _labels=None, **_kwargs):
                self.force_wn_calls.append(self.mpconv.force_wn)
                return x * self.scale

        net = RecordingNet()
        loss_fn = ECMLoss(q=4, adj='const', wt='snrpk')
        loss_fn(net=net, images=torch.randn(2, 1, 2, 2))
        self.assertEqual(net.force_wn_calls, [True, False])
        self.assertTrue(net.mpconv.force_wn)

    def test_legacy_force_wn_attribute_is_not_touched(self):
        class LegacyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.force_wn = True
                self.scale = torch.nn.Parameter(torch.ones([]))
                self.calls = []

            def forward(self, x, _sigma, _labels=None, **_kwargs):
                self.calls.append(self.force_wn)
                return x * self.scale

        net = LegacyNet()
        loss_fn = ECMLoss(q=4, adj='const')
        loss_fn(net=net, images=torch.randn(2, 1, 2, 2))
        self.assertEqual(net.calls, [True, True])

    def test_zero_disables_numbered_snapshot_and_state_dump(self):
        params = parse_train_args('--snap', '0', '--dump', '0')
        self.assertEqual(params['snap'], 0)
        self.assertEqual(params['dump'], 0)

    def test_image_outputs_can_be_explicitly_disabled(self):
        params = parse_train_args(
            '--startup-preview', 'False',
            '--sample_every', '0',
            '--eval_every', '0',
        )
        self.assertFalse(params['startup_preview'])
        self.assertEqual(params['sample_every'], 0)
        self.assertEqual(params['eval_every'], 0)

    def test_legacy_image_output_defaults_are_unchanged(self):
        params = parse_train_args()
        self.assertTrue(params['startup_preview'])
        self.assertEqual(params['sample_every'], 10)
        self.assertEqual(params['eval_every'], 50)

    def test_legacy_mapping_option_is_preserved(self):
        self.assertEqual(parse_train_args('--mapping=const')['mapping'], 'const')
        self.assertEqual(parse_train_args('--mapping=sigmoid')['mapping'], 'sigmoid')

    def test_q_requires_a_value_strictly_greater_than_one(self):
        for value in ['1', '0', '-2']:
            with self.subTest(value=value), self.assertRaises(click.BadParameter):
                parse_train_args('-q', value)
        self.assertEqual(parse_train_args('-q', '1.01')['q'], 1.01)

    def test_schedule_and_mapping_are_equivalent_names(self):
        for schedule in [
            'const', 'sigmoid', 'global_sigmoid', 'adaptive_v1',
            'local_tbin_v1', 'local_tbin_v2', 'local_tbin_v3'
        ]:
            with self.subTest(schedule=schedule):
                legacy = parse_train_args('--mapping', schedule)
                current = parse_train_args('--schedule', schedule)
                self.assertEqual(legacy, current)

    def test_hyphenated_adaptive_name_is_canonicalized(self):
        self.assertEqual(
            parse_train_args('--schedule', 'adaptive-v1')['mapping'],
            'adaptive_v1',
        )
        self.assertEqual(
            parse_train_args('--schedule', 'local-tbin-v1')['mapping'],
            'local_tbin_v1',
        )
        self.assertEqual(
            parse_train_args('--schedule', 'local-tbin-v2')['mapping'],
            'local_tbin_v2',
        )
        self.assertEqual(
            parse_train_args('--schedule', 'global-sigmoid')['mapping'],
            'global_sigmoid',
        )
        self.assertEqual(
            parse_train_args('--schedule', 'local-tbin-v3')['mapping'],
            'local_tbin_v3',
        )

    def test_help_exposes_both_option_names(self):
        result = CliRunner().invoke(ct_train.main, ['--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('--schedule', result.output)
        self.assertIn('--mapping', result.output)

    def test_adaptive_parameters_are_complete_in_loss_config(self):
        params = parse_train_args(
            '--schedule', 'adaptive_v1',
            '--adaptive-loss-ema-beta', '0.8',
            '--adaptive-update-kimg', '0.25',
            '--adaptive-warmup-updates', '3',
            '--adaptive-max-adjust', '0.04',
            '--adaptive-min-gap', '0.002',
        )
        loss_kwargs = ct_train.make_loss_kwargs(dnnlib.EasyDict(params))
        self.assertEqual(loss_kwargs.adj, 'adaptive_v1')
        self.assertEqual(loss_kwargs.adaptive_loss_ema_beta, 0.8)
        self.assertEqual(params['adaptive_update_kimg'], 0.25)
        self.assertEqual(loss_kwargs.adaptive_warmup_updates, 3)
        self.assertEqual(loss_kwargs.adaptive_max_adjust, 0.04)
        self.assertEqual(loss_kwargs.adaptive_min_gap, 0.002)

    def test_global_gap_scale_reaches_factorized_schedules(self):
        for schedule in ['global_sigmoid', 'local_tbin_v3']:
            with self.subTest(schedule=schedule):
                params = parse_train_args(
                    '--schedule', schedule,
                    '--global-gap-scale', '1.032',
                )
                loss_kwargs = ct_train.make_loss_kwargs(dnnlib.EasyDict(params))
                self.assertEqual(loss_kwargs.adj, schedule)
                self.assertEqual(loss_kwargs.global_gap_scale, 1.032)
                with contextlib.redirect_stdout(io.StringIO()):
                    loss_fn = ECMLoss(**loss_kwargs)
                self.assertEqual(loss_fn.schedule.global_gap_scale, 1.032)

    def test_explicit_sigmoid_disables_adaptive_v1(self):
        params = parse_train_args('--schedule', 'sigmoid')
        loss_kwargs = ct_train.make_loss_kwargs(dnnlib.EasyDict(params))
        self.assertEqual(loss_kwargs.adj, 'sigmoid')
        with contextlib.redirect_stdout(io.StringIO()):
            loss_fn = ECMLoss(**loss_kwargs)
        loss_fn.update_schedule(3)
        t = torch.tensor([0.01, 0.1, 1.0, 10.0])
        self.assertTrue(torch.equal(
            loss_fn.schedule.compute_r(t=t, stage=loss_fn.stage),
            loss_fn.t_to_r_sigmoid(t),
        ))

    def test_all_factorial_arms_are_explicitly_persisted(self):
        arms = {
            'A': (1.0, 1.0),
            'B': (1.1, 1.1),
            'C': (1.1, 1.0),
            'D': (1.0, 1.1),
        }
        for arm, (target, denominator) in arms.items():
            with self.subTest(arm=arm):
                params = parse_train_args(
                    '--schedule', 'sigmoid',
                    '-q', '256',
                    '-c', '0',
                    '--factorial-protocol', TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                    '--target-gap-scale', str(target),
                    '--denominator-gap-scale', str(denominator),
                )
                loss_kwargs = ct_train.make_loss_kwargs(
                    dnnlib.EasyDict(params)
                )
                self.assertEqual(
                    loss_kwargs.factorial_protocol,
                    TARGET_WEIGHT_FACTORIAL_PROTOCOL,
                )
                self.assertEqual(loss_kwargs.target_gap_scale, target)
                self.assertEqual(
                    loss_kwargs.denominator_gap_scale, denominator
                )

    def test_factorial_cli_fails_closed_on_partial_factors(self):
        params = parse_train_args(
            '-q', '256',
            '-c', '0',
            '--factorial-protocol', TARGET_WEIGHT_FACTORIAL_PROTOCOL,
            '--target-gap-scale', '1.1',
        )
        with self.assertRaises(ValueError):
            ct_train.make_loss_kwargs(dnnlib.EasyDict(params))


if __name__ == '__main__':
    unittest.main()
