"""Regression tests for the non-committing fresh-RAdam update gauge."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from training.schedules import get_schedule

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "radam_update_gauge.py"
SPEC = importlib.util.spec_from_file_location("radam_update_gauge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TinyLoss:
    P_mean = -1.1
    P_std = 0.3
    q = 128.0
    k = 8.0
    b = 1.0
    c = 0.0
    stage = 0
    schedule = get_schedule("sigmoid", q=q, k=k, b=b)


class TinyEDM(torch.nn.Module):
    """A tiny train-mode model with two layer paths and active dropout."""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Linear(1, 1)
        self.dropout = torch.nn.Dropout(p=0.2)
        with torch.no_grad():
            self.encoder.weight.fill_(0.8)
            self.encoder.bias.fill_(0.1)
            self.decoder.weight.fill_(0.6)
            self.decoder.bias.fill_(0.2)

    def forward(self, x, sigma, labels, augment_labels=None):
        del labels, augment_labels
        y = x + sigma.reshape(-1, 1, 1, 1)
        y = self.encoder(y.reshape(-1, 1)).reshape_as(x)
        return self.decoder(self.dropout(y).reshape(-1, 1)).reshape_as(x)


class RAdamUpdateGaugeTests(unittest.TestCase):
    def _run(self, amp=False):
        net = TinyEDM().train()
        images = torch.linspace(-0.8, 0.8, 8).reshape(8, 1, 1, 1)
        labels = torch.empty((8, 0))
        return net, MODULE.run_pair(net, TinyLoss(), images, labels, amp=amp,
                                    initial_scale=128.0, random_seed=1234)

    def test_actual_radam_pair_is_non_committing_and_fresh(self):
        net, (audit, layers) = self._run(amp=False)
        source = audit["source_state_non_committing"]
        self.assertTrue(source["preserved"])
        self.assertEqual(source["parameter_hash_before"], source["parameter_hash_after"])
        self.assertEqual(source["optimizer_state_hash_before"], source["optimizer_state_hash_after"])
        self.assertEqual(source["gradscaler_hash_before"], source["gradscaler_hash_after"])
        self.assertEqual(audit["fresh_radam"]["optimizer_step"], 0)
        self.assertTrue(audit["whole_model"]["gauge_defined"])
        self.assertEqual(len(audit["randomness_contract"]["t_sha256"]), 64)
        self.assertEqual(len(audit["randomness_contract"]["noise_sha256"]), 64)
        self.assertEqual(len(audit["branches"]), 2)
        self.assertFalse(any(branch["step_skipped"] for branch in audit["branches"]))
        for branch in audit["branches"]:
            self.assertEqual(branch["optimizer_step_before"], 0)
            self.assertEqual(branch["optimizer_step_after"], 1)
            self.assertNotEqual(branch["parameter_hash_before"], branch["parameter_hash_after_virtual_step"])
            self.assertNotEqual(branch["optimizer_state_hash_before"], branch["optimizer_state_hash_after_virtual_step"])
        self.assertGreater(len(layers), 1)
        self.assertEqual(net.training, True)

    def test_amp_path_unscales_and_preserves_source_scaler(self):
        _, (audit, _) = self._run(amp=True)
        self.assertTrue(all(branch["amp_unscale_called"] for branch in audit["branches"]))
        self.assertTrue(all(branch["amp_enabled"] for branch in audit["branches"]))
        self.assertTrue(audit["source_state_non_committing"]["preserved"])
        self.assertTrue(all(branch["grad_scale_before"] == 128.0 for branch in audit["branches"]))

    def test_microbatch_accumulation_matches_training_step_shape_and_preserves_rng(self):
        net = TinyEDM().train()
        images = torch.linspace(-0.8, 0.8, 8).reshape(8, 1, 1, 1)
        labels = torch.empty((8, 0))
        rng_before = torch.get_rng_state().clone()
        audit, _ = MODULE.run_pair(net, TinyLoss(), images, labels, amp=False,
                                   random_seed=1, microbatch_size=4)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(audit["randomness_contract"]["accumulation_rounds"], 2)
        self.assertTrue(all(branch["accumulation_rounds"] == 2 for branch in audit["branches"]))
        self.assertTrue(all(branch["optimizer_step_after"] == 1 for branch in audit["branches"]))

    def test_probe_does_not_introduce_autocast_not_used_by_training_loop(self):
        source = Path(SCRIPT).read_text(encoding="utf-8")
        self.assertNotIn("with torch.autocast(", source)

    def test_degenerate_update_is_a_recorded_audit_not_a_crash(self):
        net = TinyEDM().train()
        images = torch.zeros(4, 1, 1, 1)
        labels = torch.empty((4, 0))
        with mock.patch.object(MODULE, "gauge_metrics", side_effect=RuntimeError("simulated skip")):
            audit, layers = MODULE.run_pair(net, TinyLoss(), images, labels, amp=False, random_seed=1)
        self.assertFalse(audit["whole_model"]["gauge_defined"])
        self.assertEqual(audit["whole_model"]["gauge_error"], "simulated skip")
        self.assertEqual(layers, [])

    def test_requested_c_star_scales_u13_to_u1(self):
        one = {"block.weight": torch.tensor([2.0, 0.0], dtype=torch.float64)}
        thirteen = {"block.weight": torch.tensor([1.0, 0.0], dtype=torch.float64)}
        whole, layers = MODULE.gauge_metrics(one, thirteen)
        # Requested: c0_star * d_1.3 ≈ d_1, so c0_star = <d_1.3,d_1>/||d_1.3||^2 = 2.
        self.assertAlmostEqual(whole["c0_star"], 2.0)
        self.assertAlmostEqual(whole["whole_model_residual"], 0.0)
        self.assertAlmostEqual(layers[0]["c0_star"], 2.0)
        self.assertAlmostEqual(layers[0]["layerwise_residual"], 0.0)
        self.assertAlmostEqual(layers[0]["layerwise_residual_with_model_c0_star"], 0.0)

    def test_dataset_hash_supports_files_and_deterministic_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "dataset.zip"
            archive.write_bytes(b"archive contents")
            self.assertEqual(MODULE.dataset_sha256(archive),
                             (MODULE.sha256_file(archive), "sha256_file"))

            first = root / "first"
            second = root / "second"
            for directory in (first, second):
                (directory / "nested").mkdir(parents=True)
                (directory / "nested" / "sample.bin").write_bytes(b"sample contents")
            first_hash, first_algorithm = MODULE.dataset_sha256(first)
            second_hash, second_algorithm = MODULE.dataset_sha256(second)
            self.assertEqual(first_algorithm, "sha256_directory_v1")
            self.assertEqual(second_algorithm, "sha256_directory_v1")
            self.assertEqual(first_hash, second_hash)

            renamed = root / "renamed"
            (renamed / "nested").mkdir(parents=True)
            (renamed / "nested" / "renamed.bin").write_bytes(b"sample contents")
            renamed_hash, _ = MODULE.dataset_sha256(renamed)
            self.assertNotEqual(first_hash, renamed_hash)

            (second / "nested" / "sample.bin").write_bytes(b"changed contents")
            changed_hash, _ = MODULE.dataset_sha256(second)
            self.assertNotEqual(first_hash, changed_hash)

    def test_parse_args_supports_future_checkpoint_age(self):
        args = MODULE.parse_args(["--checkpoint", "state.pkl", "--data", "data.zip",
                                  "--state-kimg", "256", "--batch-size", "64",
                                  "--batch-gpu", "16", "--no-amp"])
        self.assertEqual(args.state_kimg, 256.0)
        self.assertEqual(args.batch_gpu, 16)
        self.assertFalse(args.amp)
        self.assertTrue(math.isclose(args.betas[1], 0.999))


if __name__ == "__main__":
    unittest.main()
