import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from metrics import metric_utils


class _DummyDataset:
    def __len__(self):
        return 16

    def get_label(self, _index):
        return np.zeros([0], dtype=np.float32)


class _DummyNet(torch.nn.Module):
    img_channels = 3
    img_resolution = 2
    label_dim = 0


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


if __name__ == '__main__':
    unittest.main()
