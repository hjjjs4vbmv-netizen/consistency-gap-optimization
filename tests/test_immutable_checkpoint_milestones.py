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

    def test_unreachable_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not reachable'):
            ct_training_loop.normalize_immutable_checkpoint_nimg(
                (385,), total_kimg=1024, batch_size=128
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
