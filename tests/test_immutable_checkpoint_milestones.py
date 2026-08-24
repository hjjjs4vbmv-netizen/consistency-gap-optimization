import tempfile
import unittest
from pathlib import Path

import torch
from click.testing import CliRunner

import ct_train
from training import ct_training_loop


class ImmutableCheckpointMilestoneTests(unittest.TestCase):
    def test_exact_kimg_state_is_a_valid_resume_name(self):
        self.assertEqual(
            ct_train.parse_resume_state_token(
                "/runs/training-state-kimg000384.pt"
            ),
            "kimg000384",
        )
        self.assertEqual(
            ct_train.parse_resume_state_token(
                "/runs/training-state-latest.pt"
            ),
            "latest",
        )

    def test_cli_parser_accepts_frozen_curve(self):
        runner = CliRunner()
        result = runner.invoke(
            ct_train.main,
            ['--help'],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn('--immutable-checkpoint-kimg', result.output)
        self.assertEqual(
            ct_train.parse_immutable_checkpoint_kimg(
                None, None, '384,512,640,768,896,1024'
            ),
            (384, 512, 640, 768, 896, 1024),
        )

    def test_exact_nimg_normalization(self):
        self.assertEqual(
            ct_training_loop.normalize_immutable_checkpoint_nimg(
                (384, 512, 640, 768, 896, 1024),
                total_kimg=1024,
                batch_size=128,
            ),
            (384000, 512000, 640000, 768000, 896000, 1024000),
        )

    def test_imagenet_milestones_map_to_20k_iteration_steps(self):
        milestones = (2560, 5120, 7680, 10240, 12800)
        nimg = ct_training_loop.normalize_immutable_checkpoint_nimg(
            milestones,
            total_kimg=12800,
            batch_size=128,
        )
        self.assertEqual(
            tuple(value // 128 for value in nimg),
            (20000, 40000, 60000, 80000, 100000),
        )

    def test_unreachable_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not reachable'):
            ct_training_loop.normalize_immutable_checkpoint_nimg(
                (385,), total_kimg=1024, batch_size=128
            )

    def test_fixed_global_batch_supports_w2_and_w4_but_rejects_w3(self):
        self.assertEqual(
            ct_training_loop.resolve_batch_layout(128, 32, 2),
            (32, 2),
        )
        self.assertEqual(
            ct_training_loop.resolve_batch_layout(128, 32, 4),
            (32, 1),
        )
        with self.assertRaisesRegex(ValueError, 'not divisible'):
            ct_training_loop.resolve_batch_layout(128, 32, 3)

    def test_inverse_sqrt_lr_uses_global_batch_iteration_count(self):
        schedule = ct_training_loop.learning_rate_schedule
        self.assertEqual(schedule(0, 128, ref_lr=0.001, ref_batches=2000), 0.001)
        self.assertEqual(
            schedule(256000, 128, ref_lr=0.001, ref_batches=2000),
            0.001,
        )
        self.assertAlmostEqual(
            schedule(1024000, 128, ref_lr=0.001, ref_batches=2000),
            0.0005,
        )

    def test_atomic_state_is_immutable_and_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {
                'cur_nimg': 384000,
                'attempted_iteration': 3000,
                'tensor': torch.arange(8),
            }
            path = ct_training_loop.save_immutable_training_state(
                state, directory, 384000
            )
            self.assertEqual(
                Path(path).name, 'training-state-kimg000384.pt'
            )
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(loaded['cur_nimg'], 384000)
            self.assertTrue(torch.equal(loaded['tensor'], state['tensor']))
            with self.assertRaises(FileExistsError):
                ct_training_loop.save_immutable_training_state(
                    state, directory, 384000
                )


if __name__ == '__main__':
    unittest.main()
