"""Contracts for the q256 observed/exact-scalar x real/reset audit."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
import unittest

import torch

from training.schedules import get_schedule


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
import radam_stateful_update_audit as AUDIT  # noqa: E402

SUMMARY_PATH = ANALYSIS_DIR / "q256_gradient_state_factorial_summary.py"
SPEC = importlib.util.spec_from_file_location("q256_gradient_state_factorial_summary", SUMMARY_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


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
        return self.decoder(self.dropout(self.encoder(y.reshape(-1, 1)))).reshape_as(x)


def _warmup_nonzero_state(step: int = 64):
    torch.manual_seed(0)
    net = TinyEDM().train()
    optimizer = torch.optim.RAdam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    for parameter in net.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(step)),
            "exp_avg": 0.01 * torch.randn_like(parameter),
            "exp_avg_sq": 0.01 * torch.rand_like(parameter) + 1e-4,
        }
    images = torch.linspace(-0.8, 0.8, 8).reshape(8, 1, 1, 1)
    labels = torch.empty((8, 0))
    return net, optimizer, images, labels, TinyLoss()


def _summary_receipt(seed: int, audit_seed: int, cell: str) -> dict:
    gradient_mode, state_mode = SUMMARY.CELL_KEYS[cell]
    r_opt = {"A": 0.12, "B": 0.24, "C": 0.04, "D": 0.01}[cell]
    exact = cell in {"C", "D"}
    reset = state_mode == "reset"
    whole = {
        "R_grad": 0.0 if exact else 0.03,
        "R_opt": r_opt,
        "update_reference_l2": 2.0,
        "update_probe_l2": 2.2,
        "update_norm_ratio": 1.1,
        "update_cosine": 0.99,
        "absolute_non_scalar_update_residual_l2": 2.0 * r_opt,
        "residual_norm_over_control_update_norm": r_opt,
        "H_K": r_opt,
        "H_equals_R_opt_identity_residual": 0.0,
        "H_K_equals_R_opt_identity": True,
    }
    reset_contract = ({
        "exp_avg_all_zero": True,
        "exp_avg_sq_all_zero": True,
        "per_parameter_step_preserved": True,
        "other_state_preserved": True,
        "param_groups_preserved": True,
    } if reset else None)
    branch = {
        "step_skipped": False,
        "optimizer_step_advanced_exactly_once": True,
        "clone_contract": {"independent": True},
        "gradscaler_preserved": True,
        "reset_contract": reset_contract,
    }
    gate = {
        "valid": True,
        "exact_scalar_reset_identity_pass": True,
    }
    return {
        "schema_version": 1,
        "training_seed": seed,
        "audit_batch_id": audit_seed,
        "audit_seed": audit_seed,
        "cell": cell,
        "gradient_mode": gradient_mode,
        "state_mode": state_mode,
        "reference_gap_scale": 1.0,
        "probe_gap_scale": 1.1,
        "a_star": 1.1,
        "source_state_hash": f"source-{seed}",
        "result_hash": f"result-{seed}-{audit_seed}-{cell}",
        "finite_gate": True,
        "branch_skipped_flag": False,
        "whole_model": whole,
        "branches": {"control": copy.deepcopy(branch), "treatment": copy.deepcopy(branch)},
        "batch_correctness_gate": gate,
        "source_state_non_committing": {"preserved": True},
        "control_control_identity": {
            "real": {"identical": True}, "reset": {"identical": True}},
        "order_invariance_and_rerun": {
            "numerically_invariant": True, "result_hash_identical": True},
        "provenance": {
            "source_state_sha256": f"state-{seed}",
            "checkpoint_sha256": f"checkpoint-{seed}",
            "code_commit": "a" * 40,
            "runner_sha256": "b" * 64,
            "audit_library_sha256": "c" * 64,
        },
    }


class GradientStateFactorialTests(unittest.TestCase):
    def test_global_a_star_is_one_unclamped_float64_fit(self):
        control = {
            "b": torch.tensor([3.0], dtype=torch.float32),
            "a": torch.tensor([1.0, 2.0], dtype=torch.float32),
        }
        treatment = {
            "b": torch.tensor([-6.0], dtype=torch.float32),
            "a": torch.tensor([-2.0, -4.0], dtype=torch.float32),
        }
        a_star, numerator, denominator = AUDIT.global_a_star(control, treatment)
        self.assertEqual(a_star, -2.0)
        self.assertEqual(numerator, -28.0)
        self.assertEqual(denominator, 14.0)

    def test_global_a_star_fails_closed_for_zero_or_nonfinite_input(self):
        with self.assertRaisesRegex(RuntimeError, "near zero"):
            AUDIT.global_a_star({"w": torch.zeros(2)}, {"w": torch.ones(2)})
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            AUDIT.global_a_star({"w": torch.ones(2)},
                                {"w": torch.tensor([1.0, float("nan")])})

    def test_exact_scalar_is_constructed_from_saved_control_without_mutation(self):
        control = {"w": torch.tensor([1.0, -3.0], dtype=torch.float64)}
        source_hash = AUDIT.gauge.state_sha256(control)
        exact = AUDIT.construct_exact_scalar_gradient(control, 1.125)
        self.assertTrue(torch.equal(exact["w"], control["w"] * 1.125))
        self.assertEqual(AUDIT.gauge.state_sha256(control), source_hash)
        self.assertNotEqual(exact["w"].data_ptr(), control["w"].data_ptr())

    def test_clone_independence_and_reset_preserve_all_nonmoment_state(self):
        net, optimizer, _, _, _ = _warmup_nonzero_state()
        clone, clone_optimizer, contract = AUDIT._clone_radam_branch(net, optimizer)
        self.assertTrue(contract["independent"])
        source_hash = AUDIT.gauge.state_sha256(optimizer.state_dict())
        reset = AUDIT.reset_radam_moments_(clone, clone_optimizer)
        self.assertTrue(reset["exp_avg_all_zero"])
        self.assertTrue(reset["exp_avg_sq_all_zero"])
        self.assertTrue(reset["per_parameter_step_preserved"])
        self.assertTrue(reset["other_state_preserved"])
        self.assertTrue(reset["param_groups_preserved"])
        self.assertEqual(AUDIT.gauge.state_sha256(optimizer.state_dict()), source_hash)

    def test_four_cell_run_has_identity_order_rerun_and_source_gates(self):
        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        params_before = AUDIT.gauge.module_state_hashes(net)
        optimizer_before = AUDIT.gauge.state_sha256(optimizer.state_dict())
        gradients_before = AUDIT._source_gradient_buffers_hash(net)
        receipt, layers = AUDIT.run_gradient_state_factorial(
            net, optimizer, loss, images, labels, amp=False,
            random_seed=1234, microbatch_size=4,
        )
        self.assertEqual(set(receipt["cells"]), {
            "observed_real", "observed_reset", "exact_scalar_real", "exact_scalar_reset"})
        self.assertEqual(set(layers), set(receipt["cells"]))
        self.assertTrue(receipt["correctness_gate"]["valid"])
        self.assertTrue(receipt["correctness_gate"]["exact_scalar_reset_identity_pass"])
        self.assertTrue(receipt["correctness_gate"]["control_control_identity_pass"])
        self.assertTrue(receipt["correctness_gate"]["branch_order_invariance_pass"])
        self.assertTrue(receipt["correctness_gate"]["rerun_result_hash_identity_pass"])
        for key in ("exact_scalar_real", "exact_scalar_reset"):
            self.assertLessEqual(receipt["cells"][key]["whole_model"]["R_grad"], 1e-12)
        self.assertEqual(AUDIT.gauge.module_state_hashes(net), params_before)
        self.assertEqual(AUDIT.gauge.state_sha256(optimizer.state_dict()), optimizer_before)
        self.assertEqual(AUDIT._source_gradient_buffers_hash(net), gradients_before)

    def test_summary_rebuild_and_csv_json_schema_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts"
            for seed in (3, 4, 5):
                for audit_seed in range(2026081101, 2026081109):
                    batch = receipts / f"seed{seed}" / f"audit{audit_seed}"
                    batch.mkdir(parents=True)
                    for cell, (gradient_mode, state_mode) in SUMMARY.CELL_KEYS.items():
                        name = f"{gradient_mode}_{state_mode}.json"
                        (batch / name).write_text(
                            json.dumps(_summary_receipt(seed, audit_seed, cell), sort_keys=True),
                            encoding="utf-8")
            first = SUMMARY.build(receipts)
            second = SUMMARY.build(receipts)
            self.assertEqual(first, second)
            self.assertEqual(first["summary"]["status"], "PASS")
            self.assertEqual(len(first["raw"]), 96)
            self.assertEqual(len(first["contrasts"]), 24)
            self.assertEqual(len(first["seed_summary"]), 12)
            out1, out2 = root / "out1", root / "out2"
            SUMMARY.emit(first, out1, make_figures=False)
            SUMMARY.emit(second, out2, make_figures=False)
            for name in ("raw_results.csv", "batch_contrasts.csv", "seed_summary.csv",
                         "summary.json", "report.md"):
                self.assertEqual((out1 / name).read_bytes(), (out2 / name).read_bytes())
            with (out1 / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), SUMMARY.RAW_FIELDS)
            summary_json = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary_json["audit_batches_are_independent_replicates"])
            self.assertFalse(summary_json["four_cell_is_additive_causal_decomposition"])

    def test_preregistration_and_runner_have_frozen_generic_schema(self):
        prereg = json.loads((ANALYSIS_DIR / "q256_gradient_state_factorial_preregistration.json")
                            .read_text(encoding="utf-8"))
        runner = (ANALYSIS_DIR / "q256_gradient_state_factorial.py").read_text(encoding="utf-8")
        self.assertEqual(prereg["formal_source"]["training_seeds"], [3, 4, 5])
        self.assertEqual(len(prereg["audit_rng_seeds"]), 8)
        self.assertEqual([cell["cell"] for cell in prereg["cells"]], list("ABCD"))
        self.assertIn("--reference-gap-scale", runner)
        self.assertIn("--probe-gap-scale", runner)
        self.assertNotIn("1p3", runner)

    def test_old_moment_reset_audit_still_runs(self):
        net, optimizer, images, labels, loss = _warmup_nonzero_state()
        audits, _ = AUDIT.run_moment_reset_manipulation(
            net, optimizer, loss, images, labels, amp=False,
            random_seed=1234, reference_gap_scale=1.0, probe_gap_scale=1.1)
        self.assertEqual(set(audits), {"real", "reset_moments"})
        self.assertEqual(audits["real"]["whole_model"]["R_grad"],
                         audits["reset_moments"]["whole_model"]["R_grad"])


if __name__ == "__main__":
    unittest.main()
