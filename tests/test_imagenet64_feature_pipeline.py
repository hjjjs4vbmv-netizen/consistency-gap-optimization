import io
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from metrics import metric_utils
from scripts import build_imagenet64_real_features as real_builder
from scripts import score_imagenet64_feature_matrix as scorer
from scripts import summarize_imagenet64_gap_ab_results as summarizer


class _PickledDetector(torch.nn.Module):
    def forward(self, images, return_features=False):
        if not return_features:
            raise AssertionError('official detector requires return_features=True')
        return images.flatten(1).to(torch.float32)


class ImageNet64FeaturePipelineTests(unittest.TestCase):
    def tearDown(self):
        metric_utils._feature_detector_cache.clear()

    def test_official_pickle_detector_is_loaded_without_changing_legacy_pt(self):
        payload = pickle.dumps(_PickledDetector())
        with mock.patch.object(
            metric_utils.dnnlib.util, 'open_url',
            return_value=io.BytesIO(payload),
        ), mock.patch.object(
            metric_utils.torch.jit, 'load',
            side_effect=AssertionError('official pickle must not use torch.jit'),
        ):
            detector = metric_utils.get_feature_detector(
                metric_utils.OFFICIAL_EDM2_INCEPTION_URL,
                device=torch.device('cpu'),
            )
        self.assertIsInstance(detector, _PickledDetector)

        metric_utils._feature_detector_cache.clear()
        legacy = _PickledDetector()
        with mock.patch.object(
            metric_utils.dnnlib.util, 'open_url',
            return_value=io.BytesIO(b'legacy'),
        ), mock.patch.object(
            metric_utils.torch.jit, 'load', return_value=legacy,
        ) as jit_load:
            loaded = metric_utils.get_feature_detector(
                'legacy-inception.pt', device=torch.device('cpu'),
            )
        self.assertIs(loaded, legacy)
        jit_load.assert_called_once()

    def test_unbiased_feature_covariance_matches_numpy(self):
        features = np.arange(24, dtype=np.float32).reshape(6, 4)
        mean, covariance = metric_utils.compute_feature_mean_cov(
            features, unbiased=True,
        )
        np.testing.assert_allclose(mean, features.mean(axis=0))
        np.testing.assert_allclose(
            covariance, np.cov(features, rowvar=False, bias=False),
        )

    def test_matrix_gate_requires_all_one_hundred_twenty_feature_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(FileNotFoundError, 'missing 120/120'):
                scorer.require_complete_matrix(root)
            for job in scorer.feature_jobs(root):
                job['path'].parent.mkdir(parents=True, exist_ok=True)
                job['path'].touch()
            jobs = scorer.require_complete_matrix(root)
        self.assertEqual(len(jobs), 120)

    def test_revised_w2_scope_contains_exactly_seventy_eight_jobs(self):
        jobs = scorer.w2_revised_jobs(Path('features'))
        keys = {
            (job['kimg'], job['seed'], job['method'], job['nfe'])
            for job in jobs
        }
        self.assertEqual(len(keys), 78)
        self.assertIn((8_960, 101, 'IA', 1), keys)
        self.assertIn((8_960, 102, 'IB', 2), keys)
        self.assertNotIn((8_960, 101, 'IB', 1), keys)
        self.assertNotIn((8_960, 103, 'IA', 1), keys)

    def test_local_official_reference_is_shape_checked_and_hashed(self):
        reference = {
            'num_images': 3,
            'fid': {'mu': np.zeros(2), 'sigma': np.eye(2)},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'img64.pkl'
            payload = pickle.dumps(reference)
            path.write_bytes(payload)
            with mock.patch.object(real_builder, 'REAL_COUNT', 3), \
                    mock.patch.object(real_builder, 'FEATURE_DIM', 2):
                fid, digest = real_builder.load_official_reference(path)
        np.testing.assert_array_equal(fid['mu'], reference['fid']['mu'])
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_scorer_requires_exact_local_real_stats(self):
        reference = {
            'num_images': 3,
            'fid': {'mu': np.zeros(2), 'sigma': np.eye(2)},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'img64.pkl'
            path.write_bytes(pickle.dumps(reference))
            with mock.patch.object(scorer, 'REAL_COUNT', 3), \
                    mock.patch.object(scorer, 'FEATURE_DIM', 2):
                mean, covariance = scorer.load_real_stats(path)
                reference['num_images'] = 2
                path.write_bytes(pickle.dumps(reference))
                with self.assertRaisesRegex(ValueError, 'canonical local'):
                    scorer.load_real_stats(path)
        np.testing.assert_array_equal(mean, np.zeros(2))
        np.testing.assert_array_equal(covariance, np.eye(2))

    def test_one_loaded_generated_array_feeds_both_fid_and_kid(self):
        real = np.zeros((6, 3), dtype=np.float32)
        generated = np.ones((4, 3), dtype=np.float32)
        job = dict(
            seed=101, method='IA', iteration=20_000, kimg=2_560,
            nfe=1, path=Path('generated.npy'),
        )
        with mock.patch.object(
            scorer, 'require_complete_matrix', return_value=[job],
        ), mock.patch.object(
            scorer, 'sha256_file',
            side_effect=['stats-sha', 'real-sha', 'generated-sha'],
        ), mock.patch.object(
            scorer, 'load_real_stats',
            return_value=(np.zeros(3), np.eye(3)),
        ), mock.patch.object(
            metric_utils, '_load_precomputed_features',
            side_effect=[real, generated],
        ), mock.patch.object(
            metric_utils, 'compute_feature_mean_cov',
            return_value=(np.ones(3), np.eye(3)),
        ) as compute_mean_cov, mock.patch(
            'metrics.frechet_inception_distance.compute_fid_from_stats',
            return_value=1.0,
        ), mock.patch(
            'metrics.kernel_inception_distance.compute_kid_from_features',
            return_value=2.0,
        ) as compute_kid, mock.patch.object(
            scorer, 'REAL_COUNT', 6,
        ), mock.patch.object(
            scorer, 'GENERATED_COUNT', 4,
        ), mock.patch.object(
            scorer, 'FEATURE_DIM', 3,
        ):
            result = scorer.score_matrix(
                Path('.'), Path('real.npy'), Path('img64.pkl'),
            )

        self.assertEqual(result['results'][0]['fid50k'], 1.0)
        self.assertEqual(result['results'][0]['kid50k'], 2.0)
        self.assertEqual(
            result['results'][0]['generated_features']['sha256'],
            'generated-sha',
        )
        self.assertEqual(result['real_features']['sha256'], 'real-sha')
        self.assertEqual(result['real_stats']['sha256'], 'stats-sha')
        self.assertIs(compute_mean_cov.call_args.args[0], generated)
        self.assertIs(compute_kid.call_args.args[1], generated)

    def test_paired_difference_is_ia_minus_ib_for_fid_and_kid(self):
        results = {
            (7_680, 101, 'IA', 1): {
                'kimg': 7_680, 'seed': 101, 'method': 'IA', 'nfe': 1,
                'fid50k': 9.0, 'kid50k': 0.005,
            },
            (7_680, 101, 'IB', 1): {
                'kimg': 7_680, 'seed': 101, 'method': 'IB', 'nfe': 1,
                'fid50k': 10.0, 'kid50k': 0.006,
            },
        }
        paired = summarizer.paired_differences(results)
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]['delta_fid50k_ia_minus_ib'], -1.0)
        self.assertAlmostEqual(paired[0]['delta_kid50k_ia_minus_ib'], -0.001)


if __name__ == '__main__':
    unittest.main()
