import os
import tempfile
import unittest
from unittest import mock

import click
import dnnlib
import numpy as np
import torch

import ct_eval


class _Dataset:
    resolution = 2
    num_channels = 3
    label_dim = 4


class _Net(torch.nn.Module):
    img_resolution = 2
    img_channels = 3
    label_dim = 4

    def forward(self, x, _sigma, _labels=None):
        return torch.tanh(x / 80)


class _IdentityNet(_Net):
    def forward(self, x, _sigma, _labels=None):
        return x


class _Stats:
    def __init__(self, features):
        self.features = features

    def get_all(self):
        return self.features


class ImageNet64EvaluatorTests(unittest.TestCase):
    def feature_opts(self, **overrides):
        values = {
            'arch': 'edm2',
            'preset': 'edm2-img64-s',
            'cond': True,
            'resume': 'checkpoint.pkl',
            'metrics': [],
            'retain_generated_artifacts': False,
            'seed': 20260730,
            'fp16': False,
            'engineering_feature_count': None,
            'nfe': '2',
            'mid_t': (1.526,),
        }
        values.update(overrides)
        return dnnlib.EasyDict(values)

    def test_feature_contract_is_fail_closed(self):
        seeds = list(range(50_000))
        ct_eval.validate_imagenet64_feature_contract(
            self.feature_opts(), seeds, world_size=1, resolution=64,
            num_channels=3, label_dim=1000,
        )
        with self.assertRaisesRegex(click.ClickException, 'sample-seeds=0-49999'):
            ct_eval.validate_imagenet64_feature_contract(
                self.feature_opts(), seeds[:-1], world_size=1,
                resolution=64, num_channels=3, label_dim=1000,
            )
        with self.assertRaisesRegex(click.ClickException, 'metrics=none'):
            ct_eval.validate_imagenet64_feature_contract(
                self.feature_opts(metrics=['fid50k_full']),
                seeds,
                world_size=1,
                resolution=64,
                num_channels=3,
                label_dim=1000,
            )
        with self.assertRaisesRegex(click.ClickException, 'mid_t=1.526'):
            ct_eval.validate_imagenet64_feature_contract(
                self.feature_opts(mid_t=(0.821,)),
                seeds,
                world_size=1,
                resolution=64,
                num_channels=3,
                label_dim=1000,
            )
        with self.assertRaisesRegex(click.ClickException, 'fp16=False'):
            ct_eval.validate_imagenet64_feature_contract(
                self.feature_opts(fp16=True), seeds, world_size=1,
                resolution=64, num_channels=3, label_dim=1000,
            )

        ct_eval.validate_imagenet64_feature_contract(
            self.feature_opts(engineering_feature_count=4), list(range(4)),
            world_size=1, resolution=64, num_channels=3, label_dim=1000,
        )

    def test_generator_supports_edm2_without_round_sigma(self):
        seeds = [3, 7]
        latents = ct_eval.metric_utils.make_seeded_latents(seeds, (3, 2, 2))
        generated = ct_eval.generator_fn(
            _Net(), latents, mid_t=[1.526], sample_seeds=seeds
        )
        self.assertEqual(generated.shape, latents.shape)
        self.assertTrue(torch.isfinite(generated).all())

    def test_nfe2_noise_is_the_next_draw_from_each_sample_seed(self):
        seeds = [3, 7]
        shape = (3, 2, 2)
        latents = ct_eval.metric_utils.make_seeded_latents(seeds, shape)
        next_draws = []
        for seed in seeds:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            torch.randn(shape, generator=generator, dtype=torch.float64)
            next_draws.append(
                torch.randn(shape, generator=generator, dtype=torch.float64)
            )
        next_draws = torch.stack(next_draws)

        nfe1 = ct_eval.generator_fn(
            _IdentityNet(), latents, mid_t=[], sample_seeds=seeds
        )
        nfe2 = ct_eval.generator_fn(
            _IdentityNet(), latents, mid_t=[1.526], sample_seeds=seeds
        )

        torch.testing.assert_close(nfe1, latents * 80, rtol=0, atol=0)
        torch.testing.assert_close(
            nfe2, latents * 80 + next_draws * 1.526, rtol=0, atol=0
        )

    def test_feature_only_skips_preview_images_and_metric_reporting(self):
        features = np.zeros((3, 8), dtype=np.float32)

        def construct(**kwargs):
            if kwargs.get('class_name') == 'training.dataset.ImageFolderDataset':
                return _Dataset()
            return _Net()

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(ct_eval.dnnlib.util, 'construct_class_by_name', side_effect=construct), \
                mock.patch.object(ct_eval.metric_utils, 'compute_feature_stats_for_generator', return_value=_Stats(features)) as extract, \
                mock.patch.object(ct_eval, 'setup_snapshot_image_grid', side_effect=AssertionError('preview dataset image forbidden')), \
                mock.patch.object(ct_eval, 'save_image_grid', side_effect=AssertionError('preview output forbidden')), \
                mock.patch.object(ct_eval.metric_main, 'calc_metric', side_effect=AssertionError('quality metric forbidden')), \
                mock.patch.object(ct_eval.dist, 'get_world_size', return_value=1), \
                mock.patch.object(ct_eval.dist, 'get_rank', return_value=0), \
                mock.patch.object(ct_eval.dist, 'print0'):
            output = os.path.join(tmpdir, 'features.npy')
            with mock.patch.object(ct_eval, 'IMAGENET64_FEATURE_COUNT', 3), \
                    mock.patch.object(ct_eval, 'IMAGENET64_FEATURE_DIM', 8):
                ct_eval.evaluation(
                    run_dir=tmpdir,
                    dataset_kwargs={
                        'class_name': 'training.dataset.ImageFolderDataset'
                    },
                    network_kwargs={'class_name': 'training.networks_edm2.Precond'},
                    seed=20260730,
                    resume_pkl=None,
                    mid_t=[1.526],
                    metrics=[],
                    sample_seeds=[0, 1, 2],
                    feature_only=True,
                    feature_output=output,
                    balanced_class_labels=4,
                    device=torch.device('cpu'),
                )

        extract.assert_called_once()
        self.assertEqual(
            extract.call_args.kwargs['detector_url'],
            ct_eval.metric_utils.OFFICIAL_EDM2_INCEPTION_URL,
        )
        retained = np.load(output, allow_pickle=False)
        np.testing.assert_array_equal(retained, features)


if __name__ == '__main__':
    unittest.main()
