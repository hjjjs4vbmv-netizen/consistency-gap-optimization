"""Regression tests for the stateful non-zero RAdam update audit."""
import importlib.util
import math
import sys
from pathlib import Path
import unittest
from unittest import mock

import torch

from training.schedules import get_schedule

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import radam_update_gauge as gauge  # noqa: E402

SCRIPT = ANALYSIS_DIR / "radam_stateful_update_audit.py"
SPEC = importlib.util.spec_from_file_location("radam_stateful_update_audit", SCRIPT)
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
    """Small stochastic EDM fixture with a non-degenerate dropout path.

    The audit resets the identical dropout RNG state for the ``t`` and ``r``
    forwards.  A single-feature dropout fixture can consequently mask the
    entire sample-wise difference, leaving the zero-at-the-origin ECT loss
    with no backward signal.  Keep several independently masked features so
    this test exercises the shared-RNG contract without that artefact.
    """

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 4)
        self.decoder = torch.nn.Linear(4, 1)
        self.dropout = torch.nn.Dropout(p=0.2)
        with torch.no_grad():
            self.encoder.weight.fill_(0.2)
            self.encoder.bias.copy_(torch.tensor((-0.1, 0.0, 0.1, 0.2)))
            self.decoder.weight.fill_(0.15)
            self.decoder.bias.fill_(0.2)

    def forward(self, x, sigma, labels, augment_labels=None):
        del labels, augment_labels
        y = x + sigma.reshape(-1, 1, 1, 1)
        y = self.encoder(y.reshape(-1, 1))
        return self.decoder(self.dropout(y)).reshape_as(x)


def _warmup_nonzero_state(step: int = 64):
    """Attach finite non-zero RAdam moments to the fresh TinyEDM weights.

    Moments are injected directly so the ECT probe still sees the same finite
    geometry that the fresh-state gauge unit tests rely on.  Warming the net
    with an auxiliary loss moves the tiny model into a non-finite ECT region
    and is therefore avoided.
    """
    torch.manual_seed(0)
    net = TinyEDM().train()
    optimizer = torch.optim.RAdam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    images = torch.linspace(-0.8, 0.8, 8).reshape(8, 1, 1, 1)
    labels = torch.empty((8, 0))
    for parameter in net.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(step)),
            "exp_avg": 0.01 * torch.randn_like(parameter),
            "exp_avg_sq": 0.01 * torch.rand_like(parameter) + 1e-4,
        }
    return net, optimizer, images, labels, TinyLoss()


class StatefulRAdamAuditTests(unittest.TestCase):
    def test_checkpoint_loader_accepts_formal_global_sigmoid_identity(self):
        payload = {"loss_fn": TinyLoss(), "augment_pipe": None}
        payload["loss_fn"].schedule = get_schedule(
            "global_sigmoid", q=128.0, k=8.0, b=1.0, global_gap_scale=1.0,
        )
        checkpoint = REPO_ROOT / "unused-test-checkpoint.pkl"
        with mock.patch.object(MODULE.pickle, "load", return_value=payload):
            with mock.patch.object(Path, "open", mock.mock_open(read_data=b"fixture")):
                loaded = MODULE.load_loss_from_checkpoint(checkpoint)
        self.assertEqual(loaded.schedule.name, "global_sigmoid")
        self.assertEqual(loaded.schedule.global_gap_scale, 1.0)

    def test_checkpoint_loader_still_rejects_unrelated_schedules(self):
        payload = {"loss_fn": TinyLoss(), "augment_pipe": None}
        payload["loss_fn"].schedule = get_schedule("const", q=128.0, k=8.0, b=1.0)
        checkpoint = REPO_ROOT / "unused-test-checkpoint.pkl"
        with mock.patch.object(MODULE.pickle, "load", return_value=payload):
            with mock.patch.object(Path, "open", mock.mock_open(read_data=b"fixture")):
                with self.assertRaisesRegex(SystemExit, "sigmoid.*global_sigmoid"):
                    MODULE.load_loss_from_checkpoint(checkpoint)

    def test_source_commit_uses_repository_working_directory(self):
        expected = "a" * 40
        with mock.patch.object(MODULE.subprocess, "check_output", return_value=expected + "\n") as output:
            self.assertEqual(MODULE._source_commit(), expected)
        output.assert_called_once_with(
            ["git", "rev-parse", "HEAD"], cwd=MODULE.REPO_ROOT, text=True,
            stderr=MODULE.subprocess.DEVNULL,
        )

    def test_refuses_fresh_zero_moments(self):
        net = TinyEDM().train()
        optimizer = torch.optim.RAdam(net.parameters(), lr=1e-3)
        images = torch.linspace(-0.8, 0.8, 4).reshape(4, 1, 1, 1)
        labels = torch.empty((4, 0))
        with self.assertRaisesRegex(RuntimeError, "moments are still zero"):
            MODULE.run_stateful_pair(net, optimizer, TinyLoss(), images, labels, amp=False)

    def test_refuses_partial_or_divergent_optimizer_state(self):
        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        first_parameter = next(net.parameters())
        del optimizer.state[first_parameter]
        with self.assertRaisesRegex(RuntimeError, "missing optimizer state"):
            MODULE.run_stateful_pair(
                net, optimizer, loss, images, labels, amp=False, random_seed=1234,
            )

        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        first_parameter = next(net.parameters())
        optimizer.state[first_parameter]["step"] = torch.tensor(63.0)
        with self.assertRaisesRegex(RuntimeError, "n_K is not uniform"):
            MODULE.run_stateful_pair(
                net, optimizer, loss, images, labels, amp=False, random_seed=1234,
            )

    def test_stateful_pair_is_non_committing_and_reports_core_scalars(self):
        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        source_opt = MODULE.gauge.state_sha256(optimizer.state_dict())
        source_params = MODULE.gauge.module_state_hashes(net)
        # Seed 1234 keeps the tiny ECT geometry finite (same seed as the fresh gauge).
        audit, layers = MODULE.run_stateful_pair(
            net, optimizer, loss, images, labels, amp=False, random_seed=1234,
        )
        self.assertTrue(audit["source_state_non_committing"]["preserved"])
        self.assertEqual(MODULE.gauge.state_sha256(optimizer.state_dict()), source_opt)
        self.assertEqual(MODULE.gauge.module_state_hashes(net), source_params)
        self.assertTrue(audit["stateful_radam"]["moments_nontrivial"])
        self.assertGreater(audit["stateful_radam"]["n_K"], 0)
        state_validation = audit["stateful_radam"]["state_validation"]
        self.assertTrue(state_validation["valid"])
        self.assertEqual(state_validation["initialized_parameter_count"],
                         state_validation["parameter_count"])
        whole = audit["whole_model"]
        self.assertTrue(whole["gauge_defined"])
        for key in ("a_K_star", "R_grad", "s_K_star", "c_K_star", "R_opt",
                    "H_K", "R_opt_minus_R_grad", "R_pred",
                    "on_support_gauge_dispersion_energy",
                    "off_support_candidate_energy_exact",
                    "h_update_minus_moment_weighted_rmse"):
            self.assertIn(key, whole)
            self.assertIsInstance(whole[key], float)
            self.assertTrue(math.isfinite(whole[key]))
        self.assertEqual(whole["residual_convention"],
                         "reference_normalized_candidate_minus_s_star_reference")
        self.assertTrue(whole["H_K_equals_R_opt_identity"])
        self.assertAlmostEqual(whole["H_K"], whole["R_opt"], places=12)
        self.assertAlmostEqual(
            whole["on_support_gauge_dispersion_energy"]
            + whole["off_support_candidate_energy_exact"],
            whole["R_opt"] ** 2, places=12,
        )
        self.assertGreater(len(layers), 1)
        self.assertIn("h_update_weighted_mean", layers[0])
        self.assertIn("h_moment_weighted_mean", layers[0])
        self.assertIn("layer_residual_with_global_c_star", layers[0])
        self.assertEqual(set(layers[0]), set(MODULE.LAYERWISE_FIELDS))

    def test_idealized_predictor_matches_actual_update(self):
        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        audit, _ = MODULE.run_stateful_pair(
            net, optimizer, loss, images, labels, amp=False, random_seed=1234,
        )
        rel = audit["whole_model"]["predicted_vs_actual_relative_l2"]
        # Float32 RAdam vs double predictor: agree to better than 1e-3 relative.
        self.assertLess(rel["1.0"], 1e-3)
        self.assertLess(rel["1.3"], 1e-3)
        self.assertTrue(math.isfinite(
            audit["whole_model"]["h_update_minus_moment_weighted_rmse"]
        ))

    def test_scale_conventions_match_requested_targets(self):
        reference = {"w": torch.tensor([2.0, 0.0], dtype=torch.float64)}
        probe = {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)}
        # The #43/#45 update scale is probe ≈ s * reference => s = 0.5.
        s, c, r_opt, _, _ = MODULE._update_scale_and_residual(reference, probe)
        self.assertAlmostEqual(s, 0.5)
        # c * probe ≈ reference => c = 2 (Arm-C LR multiplier).
        self.assertAlmostEqual(c, 2.0)
        self.assertAlmostEqual(r_opt, 0.0)

    def test_history_gauge_decomposes_on_and_off_support(self):
        reference = {"w": torch.tensor([2.0, 0.0], dtype=torch.float64)}
        candidate = {"w": torch.tensor([1.0, 3.0], dtype=torch.float64)}
        s, c, r_opt, _, _ = MODULE._update_scale_and_residual(reference, candidate)
        summary, layers = MODULE.support_aware_gauge_summary(
            reference, candidate, s_star=s, c_star=c, support_atol=0.0,
        )
        self.assertAlmostEqual(s, 0.5)
        self.assertAlmostEqual(c, 0.2)
        self.assertAlmostEqual(summary["on_support_gauge_dispersion_energy"], 0.0)
        self.assertAlmostEqual(summary["off_support_candidate_energy_exact"], 9 / 4)
        self.assertAlmostEqual(summary["history_gauge_dispersion_H_K"], r_opt)
        self.assertAlmostEqual(
            summary["on_support_gauge_dispersion_energy"]
            + summary["off_support_candidate_energy_exact"], r_opt ** 2,
        )
        self.assertEqual(layers[0]["support_coordinate_count"], 1)

    def test_coordinate_history_and_moment_gauges_are_reported(self):
        reference = {"w": torch.tensor([2.0, 4.0], dtype=torch.float64)}
        candidate = {"w": torch.tensor([1.0, 8.0], dtype=torch.float64)}
        moments_1 = {"w": (torch.tensor([2.0, 2.0]), torch.tensor([1.0, 1.0]))}
        moments_13 = {"w": (torch.tensor([1.0, 4.0]), torch.tensor([1.0, 1.0]))}
        s, c, _, _, _ = MODULE._update_scale_and_residual(reference, candidate)
        summary, layers = MODULE.support_aware_gauge_summary(
            reference, candidate, s_star=s, c_star=c, support_atol=0.0,
            moments_reference=moments_1, moments_candidate=moments_13, eps=0.0,
        )
        self.assertAlmostEqual(layers[0]["h_update_weighted_mean"], 1.7)
        self.assertAlmostEqual(layers[0]["h_update_p50"], 1.25)
        self.assertEqual(layers[0]["h_moment_coordinate_count"], 2)
        self.assertAlmostEqual(summary["h_update_minus_moment_weighted_rmse"], 0.0)

    def test_effective_support_threshold_does_not_replace_exact_decomposition(self):
        reference = {"w": torch.tensor([2.0, 1e-9], dtype=torch.float64)}
        candidate = {"w": torch.tensor([1.0, 1.0], dtype=torch.float64)}
        s, c, r_opt, _, _ = MODULE._update_scale_and_residual(reference, candidate)
        summary, _ = MODULE.support_aware_gauge_summary(
            reference, candidate, s_star=s, c_star=c, support_atol=1e-8,
        )
        self.assertEqual(summary["exact_support_coordinate_count"], 2)
        self.assertEqual(summary["effective_support_coordinate_count"], 1)
        self.assertAlmostEqual(
            summary["on_support_gauge_dispersion_energy"]
            + summary["off_support_candidate_energy_exact"], r_opt ** 2,
        )

    def test_microbatch_accumulation_preserves_rng(self):
        # Regression sweep for the seeds that exposed single-channel dropout
        # degeneracy in the original fixture.
        for seed in range(1190, 1211):
            with self.subTest(seed=seed):
                net, optimizer, images, labels, loss = _warmup_nonzero_state()
                rng_before = torch.get_rng_state().clone()
                audit, _ = MODULE.run_stateful_pair(
                    net, optimizer, loss, images, labels, amp=False,
                    random_seed=seed, microbatch_size=4,
                )
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                self.assertEqual(audit["randomness_contract"]["accumulation_rounds"], 2)
                self.assertTrue(audit["whole_model"]["gauge_defined"])

    def test_probe_does_not_introduce_autocast(self):
        source = Path(SCRIPT).read_text(encoding="utf-8")
        self.assertNotIn("with torch.autocast(", source)

    def test_amp_without_gradscaler_state_fails_closed(self):
        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        with self.assertRaisesRegex(RuntimeError, "GradScaler_K"):
            MODULE.run_stateful_pair(
                net, optimizer, loss, images, labels, amp=True, scaler_state=None,
                random_seed=1234,
            )

    def test_amp_skip_marks_gauge_undefined(self):
        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        real_step = MODULE.virtual_stateful_step

        def skip_one(*args, **kwargs):
            grads, predicted, actual, moments, detail = real_step(*args, **kwargs)
            if kwargs.get("gain") == 1.3:
                detail = dict(detail)
                detail["step_skipped"] = True
                actual = {name: torch.zeros_like(value) for name, value in actual.items()}
            return grads, predicted, actual, moments, detail

        with mock.patch.object(MODULE, "virtual_stateful_step", side_effect=skip_one):
            audit, layers = MODULE.run_stateful_pair(
                net, optimizer, loss, images, labels, amp=False, random_seed=1234,
            )
        self.assertFalse(audit["whole_model"]["gauge_defined"])
        self.assertIn("AMP skipped", audit["whole_model"]["gauge_error"])
        self.assertEqual(layers, [])

    def test_predictor_covers_params_without_grad(self):
        net = TinyEDM().train()
        unused = torch.nn.Linear(1, 1)
        net.unused = unused  # registered parameter with no forward use
        optimizer = torch.optim.RAdam(net.parameters(), lr=1e-3)
        for parameter in net.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(8.0),
                "exp_avg": 0.01 * torch.ones_like(parameter),
                "exp_avg_sq": 0.01 * torch.ones_like(parameter),
            }
        # Only encoder/decoder participate in a fake grad; unused stays None.
        for name, parameter in net.named_parameters():
            if name.startswith("unused"):
                continue
            parameter.grad = torch.ones_like(parameter)
        predicted = MODULE.idealized_radam_update(net, optimizer)
        self.assertIn("unused.weight", predicted)
        self.assertEqual(float(predicted["unused.weight"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
