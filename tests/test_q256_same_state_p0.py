from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path

import torch
from click.testing import CliRunner

from ct_train import main as train_cli
from training import reproducibility
from training.ct_training_loop import (
    _FACTORIAL_TELEMETRY_FIELDS,
    _SAME_STATE_TELEMETRY_FIELDS,
    aggregate_factorial_runtime_metrics,
    normalize_immutable_checkpoint_attempts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "q256_same_state_b384_p0_seed3_5"


class SameStateProtocolTest(unittest.TestCase):
    def test_cli_is_explicitly_scoped_to_q256_b384(self):
        result = CliRunner().invoke(train_cli, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--q256-b384-same-state-fork", result.output)
        self.assertIn("--q256-b384-protocol-sha256", result.output)
        self.assertNotIn("--same-state-fork", result.output)
        self.assertNotIn("--same-state-origin-arm", result.output)
        legacy = CliRunner().invoke(train_cli, ["--same-state-fork"])
        self.assertNotEqual(legacy.exit_code, 0)
        self.assertIn("No such option: --same-state-fork", legacy.output)

    def test_attempt_milestones_are_exact_and_bounded(self):
        values = normalize_immutable_checkpoint_attempts(
            (3001, 3004, 3016, 3064, 3128, 3256, 3500),
            total_kimg=448, batch_size=128,
        )
        self.assertEqual(values[-1], 3500)
        with self.assertRaises(ValueError):
            normalize_immutable_checkpoint_attempts(
                (3004, 3001), total_kimg=448, batch_size=128,
            )
        with self.assertRaises(ValueError):
            normalize_immutable_checkpoint_attempts(
                (3501,), total_kimg=448, batch_size=128,
            )

    def test_same_state_telemetry_extends_v1_without_reordering_v1(self):
        for field in _FACTORIAL_TELEMETRY_FIELDS:
            self.assertIn(field, _SAME_STATE_TELEMETRY_FIELDS)
        self.assertEqual(len(_SAME_STATE_TELEMETRY_FIELDS), len(_FACTORIAL_TELEMETRY_FIELDS) + 3)
        self.assertLess(_SAME_STATE_TELEMETRY_FIELDS.index("base_r_sha256"),
                        _SAME_STATE_TELEMETRY_FIELDS.index("input_noise_sha256"))

    def test_exogenous_hashes_aggregate_in_microbatch_order(self):
        base = {
            "schema": "x", "protocol": "q256_target_weight_v1", "arm": "A",
            "target_gap_scale": 1.0, "denominator_gap_scale": 1.0,
            "sample_count": 16, "base_r_zero_count": 0, "target_r_zero_count": 0,
            "target_r_equal_t_count": 0, "target_scaled_to_zero_count": 0,
            "denominator_r_zero_count": 0, "denominator_r_equal_t_count": 0,
            "denominator_scaled_to_zero_count": 0, "nonfinite_count": 0,
            "nonpositive_denominator_count": 0, "target_delta_min": 1.0,
            "target_delta_max": 1.0, "target_delta_mean": 1.0,
            "denominator_delta_min": 1.0, "denominator_delta_max": 1.0,
            "denominator_delta_mean": 1.0,
        }
        hashes = (
            "t_sha256", "base_r_sha256", "target_r_sha256",
            "denominator_r_sha256", "target_delta_sha256",
            "denominator_delta_sha256", "input_noise_sha256",
            "dropout_rng_sha256", "augmentation_rng_sha256",
        )
        batches = []
        for index in range(2):
            item = dict(base)
            item.update({field: f"{field}-{index}" for field in hashes})
            batches.append(item)
        result = aggregate_factorial_runtime_metrics(batches)
        for field in hashes:
            self.assertEqual(
                result[field],
                reproducibility.state_sha256([f"{field}-0", f"{field}-1"]),
            )

    def test_absolute_norm_contrasts_are_exploratory_and_recomputable(self):
        self.assertFalse((RESULT_ROOT / "factorial_contrasts.csv").exists())
        contrast_path = RESULT_ROOT / "exploratory_absolute_norm_contrasts.csv"
        with contrast_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = tuple(reader.fieldnames or ())
        self.assertEqual(len(rows), 960)
        self.assertEqual(
            fields[:6],
            (
                "seed", "horizon", "space", "block", "contrast_formula",
                "contrast_of_l2_norms",
            ),
        )
        formulas = {
            "norm_C_minus_norm_A": lambda y: y["C"] - y["A"],
            "norm_B_minus_norm_D": lambda y: y["B"] - y["D"],
            "norm_D_minus_norm_A": lambda y: y["D"] - y["A"],
            "norm_B_minus_norm_C": lambda y: y["B"] - y["C"],
            "norm_B_minus_norm_C_minus_norm_D_plus_norm_A": (
                lambda y: y["B"] - y["C"] - y["D"] + y["A"]
            ),
        }
        self.assertEqual({row["contrast_formula"] for row in rows}, set(formulas))
        self.assertFalse(any("effect" in row["contrast_formula"] for row in rows))
        self.assertEqual(
            len({
                (
                    row["seed"], row["horizon"], row["space"], row["block"],
                    row["contrast_formula"],
                )
                for row in rows
            }),
            960,
        )
        with (RESULT_ROOT / "matched_horizon_results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            horizon_rows = list(csv.DictReader(handle))
        indexed = {
            (
                row["seed"], row["horizon"], row["space"], row["block"],
                row["arm"],
            ): float(row["value_norm"])
            for row in horizon_rows
        }
        for row in rows:
            prefix = (
                row["seed"], row["horizon"], row["space"], row["block"],
            )
            y = {arm: indexed[(*prefix, arm)] for arm in "ABCD"}
            expected = formulas[row["contrast_formula"]](y)
            self.assertTrue(
                math.isclose(
                    float(row["contrast_of_l2_norms"]), expected,
                    rel_tol=1e-12, abs_tol=1e-12,
                )
            )
        horizon_zero = [row for row in rows if row["horizon"] == "0"]
        self.assertEqual(len(horizon_zero), 120)
        self.assertTrue(
            all(float(row["contrast_of_l2_norms"]) == 0.0 for row in horizon_zero)
        )

    def test_corrected_feedback_table_and_claim_ceiling(self):
        with (RESULT_ROOT / "late_propagation_corrected_feedback.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 36)
        required = {
            "late_median_raw_propagation_gain_G",
            "late_median_corrected_R_over_delta_k",
            "late_median_cos_corrected_R_delta_k",
            "corrected_alignment_sign_h256",
            "corrected_alignment_sign_h500",
            "corrected_feedback_numerically_nonzero",
            "claim_ceiling",
        }
        self.assertTrue(all(required.issubset(row) for row in rows))
        self.assertTrue(all(row["late_horizons"] == "256;500" for row in rows))
        self.assertTrue(
            all(row["corrected_feedback_numerically_nonzero"] == "True" for row in rows)
        )
        summary = json.loads(
            (RESULT_ROOT / "forcing_feedback_summary.json").read_text(
                encoding="utf-8"
            )
        )
        directional = set(
            summary[
                "replicated_directionally_consistent_corrected_feedback_entries"
            ]
        )
        self.assertEqual(
            directional,
            {
                "B:state:theta", "B:state:EMA", "B:state:v",
                "C:state:theta", "C:state:EMA",
                "D:state:theta", "D:state:EMA",
            },
        )
        corrected = summary["corrected_incremental_feedback"]
        self.assertFalse(corrected["any_late_corrected_R_over_delta_k_above_one"])
        report = (RESULT_ROOT / "P0_REPORT.md").read_text(encoding="utf-8")
        self.assertNotIn("Residual/features remain mixed: **False**", report)
        self.assertIn("no declared linear carryover map", report)
        self.assertIn("not `||z_C-z_A||_2`", report)
        self.assertIn("does not establish corrected-feedback dominance", report)
        self.assertIn("no corrected ratio exceeds 1", report)

    def test_compact_sha_manifest(self):
        manifest = RESULT_ROOT / "SHA256SUMS.txt"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            digest = hashlib.sha256((RESULT_ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, name)


if __name__ == "__main__":
    unittest.main()
