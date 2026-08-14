import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_cross_seed_optimizer_geometry_audit as runner
from scripts import summarize_cross_seed_optimizer_geometry as summarize


def receipt(*, r_grad, r_opt, c_star, s_star):
    return {
        "gains": [1.0, 1.3],
        "stateful_radam": {"n_K": 1991, "moments_nontrivial": True, "gradscaler_restored": True},
        "randomness_contract": {
            "same_minibatch": True, "same_t": True, "same_noise": True,
            "same_dropout_rng_state": True,
        },
        "source_state_non_committing": {"preserved": True},
        "branches": [{"step_skipped": False}, {"step_skipped": False}],
        "whole_model": {
            "gauge_defined": True, "H_K_equals_R_opt_identity": True,
            "R_grad": r_grad, "R_opt": r_opt, "c_K_star": c_star, "s_K_star": s_star,
            "on_support_gauge_dispersion_energy": r_opt ** 2,
            "off_support_candidate_energy_exact": 1e-6,
        },
        "provenance": {
            "state_kimg": 256.0,
            "training_state_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
        },
    }


class CrossSeedOptimizerGeometryTests(unittest.TestCase):
    def manifest(self):
        return {
            "schema_version": 1,
            "design": {"state_kimg": 256.0},
            "endpoint_cells": [
                {"training_seed": 3, "run_id": "seed3", "expected_training_state_sha256": "a" * 64, "expected_checkpoint_sha256": "b" * 64, "quality_control": {"fid5k_delta_C_minus_B": 90.494417313, "kid5k_delta_C_minus_B": 0.101760170}},
                {"training_seed": 4, "run_id": "seed4", "expected_training_state_sha256": "a" * 64, "expected_checkpoint_sha256": "b" * 64, "quality_control": {"fid5k_delta_C_minus_B": 4.945933983, "kid5k_delta_C_minus_B": 0.003913724}},
                {"training_seed": 5, "run_id": "seed5", "expected_training_state_sha256": "a" * 64, "expected_checkpoint_sha256": "b" * 64, "quality_control": {"fid5k_delta_C_minus_B": -14.516957145, "kid5k_delta_C_minus_B": -0.017868804}},
            ],
        }

    def write_audits(self, root, *, break_pairing=False):
        values = {3: (0.10, 0.09, 1.033, 0.958), 4: (0.11, 0.08, 1.030, 0.961), 5: (0.12, 0.10, 1.035, 0.956)}
        for seed, metrics in values.items():
            payload = receipt(r_grad=metrics[0], r_opt=metrics[1], c_star=metrics[2], s_star=metrics[3])
            if break_pairing and seed == 4:
                payload["randomness_contract"]["same_noise"] = False
            target = root / f"seed{seed}"
            target.mkdir(parents=True)
            (target / "radam_update_audit_stateful.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_summary_writes_three_seed_and_descriptive_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "matrix.json"
            manifest.write_text(json.dumps(self.manifest()), encoding="utf-8")
            audits = root / "audits"
            self.write_audits(audits)
            output = root / "summary"
            self.assertEqual(summarize.main(["--manifest", str(manifest), "--audit-root", str(audits), "--out", str(output)]), 0)
            with (output / "optimizer_geometry_table.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["training_seed"] for row in rows], ["3", "4", "5", "mean", "sample_sd", "range"])
            self.assertEqual(rows[0]["quality_delta_ctrl_fid5k"], "90.494417313")
            self.assertEqual(rows[2]["quality_delta_ctrl_kid5k"], "-0.017868804")
            self.assertAlmostEqual(float(rows[3]["R_grad"]), 0.11)
            self.assertAlmostEqual(float(rows[5]["R_opt"]), 0.02)
            report = (output / "OPTIMIZER_GEOMETRY_TABLE.md").read_text(encoding="utf-8")
            self.assertIn("does not establish", report)
            self.assertIn("seed 5", report)

    def test_summary_rejects_non_paired_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "matrix.json"
            manifest.write_text(json.dumps(self.manifest()), encoding="utf-8")
            audits = root / "audits"
            self.write_audits(audits, break_pairing=True)
            with self.assertRaisesRegex(SystemExit, "paired-randomness"):
                summarize.main(["--manifest", str(manifest), "--audit-root", str(audits), "--out", str(root / "summary")])

    def test_runner_refuses_missing_inputs_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest()
            manifest["design"].update({
                "reference_arm": "A", "reference_gap_scale": 1.0,
                "candidate_gap_scale": 1.3, "audit_random_seed": 1,
                "batch_size": 128, "batch_gpu": 16, "learning_rate": 0.0001,
                "betas": [0.9, 0.999], "eps": 1e-8,
            })
            manifest["data"] = "/missing/data.zip"
            for cell in manifest["endpoint_cells"]:
                cell["training_state"] = "/missing/training-state.pt"
                cell["checkpoint"] = "/missing/checkpoint.pkl"
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "new-output"
            with self.assertRaisesRegex(SystemExit, "missing input path"):
                runner.main(["--manifest", str(matrix), "--out", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
