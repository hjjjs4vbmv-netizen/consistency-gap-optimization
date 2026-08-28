import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from metrics import metric_main, metric_utils
from metrics import frechet_inception_distance, kernel_inception_distance


class _DummyDataset:
    def __len__(self):
        return 16

    def get_label(self, _index):
        return np.zeros([0], dtype=np.float32)


class _DummyNet(torch.nn.Module):
    img_channels = 3
    img_resolution = 2
    label_dim = 0


class _ConditionalDummyNet(_DummyNet):
    label_dim = 4


class _DummyDetector:
    def __call__(self, images, **_kwargs):
        return images.to(torch.float32).flatten(1)[:, :8]


def _generator(_net, latents, _labels, **_kwargs):
    return torch.tanh(latents / 80)


class MetricArtifactRetentionTests(unittest.TestCase):
    def test_retains_exact_features_and_samples_without_losing_moments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, 'features.npy')
            samples_path = os.path.join(tmpdir, 'samples.npy')
            opts = metric_utils.MetricOptions(
                generator_fn=_generator,
                G=_DummyNet(),
                dataset_kwargs={},
                device=torch.device('cpu'),
                sample_seeds=[7, 8, 9],
                generated_features_path=features_path,
                generated_samples_path=samples_path,
            )
            with mock.patch.object(metric_utils.dnnlib.util, 'construct_class_by_name', return_value=_DummyDataset()), \
                    mock.patch.object(metric_utils, 'get_feature_detector', return_value=_DummyDetector()):
                stats = metric_utils.compute_feature_stats_for_generator(
                    opts,
                    detector_url='unused',
                    detector_kwargs={},
                    batch_size=2,
                    batch_gen=1,
                    capture_mean_cov=True,
                    max_items=3,
                )

            features = np.load(features_path, allow_pickle=False)
            samples = np.load(samples_path, allow_pickle=False)
            np.testing.assert_array_equal(features, stats.get_all())
            self.assertEqual(features.shape, (3, 8))
            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(samples.shape, (3, 3, 2, 2))
            self.assertEqual(samples.dtype, np.uint8)
            mean, covariance = stats.get_mean_cov()
            self.assertEqual(mean.shape, (8,))
            self.assertEqual(covariance.shape, (8, 8))

    def test_retention_rejects_multi_gpu(self):
        opts = metric_utils.MetricOptions(
            generator_fn=_generator,
            G=_DummyNet(),
            dataset_kwargs={},
            num_gpus=2,
            rank=0,
            device=torch.device('cpu'),
            generated_samples_path='unused.npy',
        )
        with self.assertRaisesRegex(ValueError, 'requires num_gpus=1'):
            metric_utils.compute_feature_stats_for_generator(
                opts,
                detector_url='unused',
                detector_kwargs={},
                max_items=1,
            )

    def test_reuses_retained_features_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, 'kid-features.npy')
            destination_path = os.path.join(tmpdir, 'fid-features.npy')
            features = np.arange(24, dtype=np.float32).reshape(3, 8)
            metric_utils._atomic_save_npy(source_path, features)
            opts = metric_utils.MetricOptions(
                G=None,
                dataset_kwargs={},
                device=torch.device('cpu'),
                generated_features_path=destination_path,
                precomputed_generated_features_path=source_path,
                metric_name='fid50k_full',
                precomputed_generated_features_source_metric='kid50k_full',
            )
            with mock.patch.object(
                metric_utils.dnnlib.util,
                'construct_class_by_name',
                side_effect=AssertionError('dataset must not be loaded'),
            ), mock.patch.object(
                metric_utils,
                'get_feature_detector',
                side_effect=AssertionError('detector must not be loaded'),
            ):
                stats = metric_utils.compute_feature_stats_for_generator(
                    opts,
                    detector_url='unused',
                    detector_kwargs={},
                    capture_mean_cov=True,
                    max_items=3,
                )

            np.testing.assert_array_equal(stats.get_all(), features)
            self.assertEqual(Path(source_path).read_bytes(), Path(destination_path).read_bytes())
            mean, covariance = stats.get_mean_cov()
            np.testing.assert_allclose(mean, features.astype(np.float64).mean(axis=0))
            self.assertEqual(covariance.shape, (8, 8))

    def test_reuse_fails_closed_for_unapproved_metric_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, 'features.npy')
            metric_utils._atomic_save_npy(
                source_path,
                np.zeros((3, 8), dtype=np.float32),
            )
            opts = metric_utils.MetricOptions(
                G=None,
                dataset_kwargs={},
                device=torch.device('cpu'),
                precomputed_generated_features_path=source_path,
                metric_name='fid50k_full',
                precomputed_generated_features_source_metric='is50k',
            )
            with self.assertRaisesRegex(ValueError, 'restricted'):
                metric_utils.compute_feature_stats_for_generator(
                    opts,
                    detector_url='unused',
                    detector_kwargs={},
                    max_items=3,
                )

    def test_reuse_rejects_wrong_shape_or_non_float32(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'features.npy')
            opts = metric_utils.MetricOptions(
                generator_fn=_generator,
                G=_DummyNet(),
                dataset_kwargs={},
                device=torch.device('cpu'),
                precomputed_generated_features_path=path,
                metric_name='fid50k_full',
                precomputed_generated_features_source_metric='kid50k_full',
            )
            np.save(path, np.zeros((2, 8), dtype=np.float32), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, 'expected'):
                metric_utils.compute_feature_stats_for_generator(
                    opts,
                    detector_url='unused',
                    detector_kwargs={},
                    max_items=3,
                )
            np.save(path, np.zeros((3, 8), dtype=np.float64), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, 'finite float32'):
                metric_utils.compute_feature_stats_for_generator(
                    opts,
                    detector_url='unused',
                    detector_kwargs={},
                    max_items=3,
                )

    def test_balanced_labels_are_direct_one_hot_without_dataset_lookup(self):
        observed_labels = []

        def generator(_net, latents, labels, **_kwargs):
            observed_labels.append(labels.cpu())
            return torch.tanh(latents / 80)

        opts = metric_utils.MetricOptions(
            generator_fn=generator,
            G=_ConditionalDummyNet(),
            dataset_kwargs={},
            device=torch.device('cpu'),
            sample_seeds=[0, 1, 4, 7],
            balanced_class_labels=4,
        )
        with mock.patch.object(
            metric_utils.dnnlib.util,
            'construct_class_by_name',
            side_effect=AssertionError('balanced labels must not load dataset rows'),
        ), mock.patch.object(
            metric_utils,
            'get_feature_detector',
            return_value=_DummyDetector(),
        ):
            metric_utils.compute_feature_stats_for_generator(
                opts,
                detector_url='unused',
                detector_kwargs={},
                batch_size=2,
                batch_gen=1,
                capture_all=True,
                max_items=4,
            )

        labels = torch.cat(observed_labels)
        self.assertEqual(labels.shape, (4, 4))
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(labels.argmax(dim=1).tolist(), [0, 1, 0, 3])
        torch.testing.assert_close(labels.sum(dim=1), torch.ones(4))

    def test_fid_and_kid_score_only_precomputed_feature_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.RandomState(17)
            real = rng.normal(size=(12, 8)).astype(np.float32)
            generated = rng.normal(size=(6, 8)).astype(np.float32)
            real_path = os.path.join(tmpdir, 'real.npy')
            generated_path = os.path.join(tmpdir, 'generated.npy')
            np.save(real_path, real, allow_pickle=False)
            np.save(generated_path, generated, allow_pickle=False)
            opts = metric_utils.MetricOptions(
                G=None,
                dataset_kwargs={},
                rank=0,
                num_gpus=1,
                device=torch.device('cpu'),
                precomputed_real_features_path=real_path,
                precomputed_generated_features_path=generated_path,
            )
            with mock.patch.object(
                metric_utils.dnnlib.util,
                'construct_class_by_name',
                side_effect=AssertionError('scoring must not load a dataset'),
            ), mock.patch.object(
                metric_utils,
                'get_feature_detector',
                side_effect=AssertionError('scoring must not load a detector'),
            ):
                fid = frechet_inception_distance.compute_fid(
                    opts, max_real=None, num_gen=6
                )
                kid = kernel_inception_distance.compute_kid(
                    opts,
                    max_real=None,
                    num_gen=6,
                    num_subsets=5,
                    max_subset_size=4,
                    random_seed=20260730,
                )

            self.assertTrue(np.isfinite(fid))
            self.assertTrue(np.isfinite(kid))

    def test_formal_scorer_passes_one_generated_feature_path_to_both_metrics(self):
        with mock.patch.object(
            metric_main,
            'calc_metric',
            side_effect=[{'fid': 1.0}, {'kid': 2.0}],
        ) as calc_metric:
            result = metric_main.calc_imagenet64_fid_kid_from_features(
                'generated.npy', 'real.npy', metric_seed=20260730
            )

        self.assertEqual(result.fid, {'fid': 1.0})
        self.assertEqual(result.kid, {'kid': 2.0})
        self.assertEqual(calc_metric.call_count, 2)
        for call in calc_metric.call_args_list:
            self.assertEqual(
                call.kwargs['precomputed_generated_features_path'],
                'generated.npy',
            )
            self.assertEqual(
                call.kwargs['precomputed_real_features_path'],
                'real.npy',
            )
            self.assertEqual(call.kwargs['metric_seed'], 20260730)

        with self.assertRaisesRegex(ValueError, 'metric_seed=20260730'):
            metric_main.calc_imagenet64_fid_kid_from_features(
                'generated.npy', 'real.npy', metric_seed=17
            )

    def test_formal_imagenet_fid_uses_official_unbiased_covariance(self):
        opts = metric_utils.MetricOptions(G=None, dataset_kwargs={})
        with mock.patch.object(
            frechet_inception_distance, 'compute_fid', return_value=1.0,
        ) as compute_fid:
            result = metric_main.imagenet64_fid50k_full(opts)
        self.assertEqual(result, {'imagenet64_fid50k_full': 1.0})
        self.assertTrue(compute_fid.call_args.kwargs['unbiased'])
        self.assertEqual(
            compute_fid.call_args.kwargs['detector_url'],
            metric_utils.OFFICIAL_EDM2_INCEPTION_URL,
        )

    def test_formal_imagenet_kid_uses_official_detector(self):
        opts = metric_utils.MetricOptions(
            G=None, dataset_kwargs={}, metric_seed=20260730,
        )
        with mock.patch.object(
            kernel_inception_distance, 'compute_kid', return_value=2.0,
        ) as compute_kid:
            result = metric_main.imagenet64_kid50k_full(opts)
        self.assertEqual(result, {'imagenet64_kid50k_full': 2.0})
        self.assertEqual(
            compute_kid.call_args.kwargs['detector_url'],
            metric_utils.OFFICIAL_EDM2_INCEPTION_URL,
        )

if __name__ == '__main__':
    unittest.main()
