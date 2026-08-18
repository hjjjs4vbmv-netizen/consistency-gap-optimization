"""Contract tests for the formal cross-seed moment-reset audit."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = REPO_ROOT / "analysis" / "q256_moment_reset_summary.py"
RUNNER_SCRIPT = REPO_ROOT / "analysis" / "q256_crossseed_moment_reset.py"
SPEC = importlib.util.spec_from_file_location("q256_moment_reset_summary", SUMMARY_SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def _receipt(seed: int, audit_seed: int, condition: str) -> dict:
    r_opt = 0.4 if condition == "real" else 0.2
    metrics = {key: 0.5 for key in SUMMARY.REQUIRED_METRICS}
    metrics.update({
        "R_opt": r_opt,
        "R_grad": 0.1,
        "H_K": r_opt,
        "H_K_squared_minus_R_opt_squared_energy_gap": 0.0,
        "H_K_equals_R_opt_identity": True,
        "source_preserved": True,
        "step_skipped": False,
    })
    reset_contract = {
        "exp_avg_all_zero": True,
        "exp_avg_sq_all_zero": True,
        "per_parameter_step_preserved": True,
        "param_groups_preserved": True,
    } if condition == "reset_moments" else None
    branch = {
        "step_skipped": False,
        "gradient_injection_identical": True,
        "gradscaler_preserved": True,
        "reset_contract": reset_contract,
    }
    return {
        "training_seed": seed,
        "audit_seed": audit_seed,
        "condition": condition,
        "reference_gap_scale": 1.0,
        "probe_gap_scale": 1.1,
        "whole_model": metrics,
        "source_state_non_committing": {"preserved": True},
        "branches": {"reference": copy.deepcopy(branch), "probe": copy.deepcopy(branch)},
        "gradient_contract": {
            "reference": {"gradient_sha256": f"ref-{seed}-{audit_seed}"},
            "probe": {"gradient_sha256": f"probe-{seed}-{audit_seed}"},
        },
        "provenance": {
            "source_state_sha256": f"state-{seed}",
            "checkpoint_sha256": f"checkpoint-{seed}",
            "code_commit": "a" * 40,
            "runner_sha256": "b" * 64,
            "audit_library_sha256": "c" * 64,
        },
    }


class Q256MomentResetTests(unittest.TestCase):
    def test_summary_rebuild_from_compact_receipts_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts"
            for seed in (3, 4, 5):
                for audit_seed in range(8):
                    path = receipts / f"seed{seed}" / f"audit{audit_seed}"
                    path.mkdir(parents=True)
                    for condition in ("real", "reset_moments"):
                        (path / f"{condition}.json").write_text(
                            json.dumps(_receipt(seed, audit_seed, condition)), encoding="utf-8")
            first = SUMMARY.build(receipts)
            second = SUMMARY.build(receipts)
            self.assertEqual(first, second)
            summary, rows, report, provenance = first
            self.assertTrue(summary["preregistered_go_gate"]["go"])
            self.assertEqual(len(rows), 3)
            self.assertEqual(set(provenance), {"3", "4", "5"})
            self.assertIn("not a full-training intervention", report)
            self.assertIn("Cross-seed R_grad vs R_opt_real", report)
            self.assertIn("moment zeroing as a valid memory-neutralization intervention", report)
            for row in rows:
                self.assertEqual(row["median_R_grad"], 0.1)
                self.assertAlmostEqual(row["median_R_opt_real_minus_R_grad"], 0.3)
                self.assertEqual(row["R_opt_real_lt_R_grad_count"], 0)

    def test_formal_runner_freezes_eight_audit_seeds_and_generic_schema(self):
        source = RUNNER_SCRIPT.read_text(encoding="utf-8")
        preregistration = json.loads(
            (REPO_ROOT / "analysis" / "q256_moment_reset_preregistration.json")
            .read_text(encoding="utf-8"))
        self.assertEqual(len(preregistration["audit_rng_seeds"]), 8)
        self.assertEqual(len(set(preregistration["audit_rng_seeds"])), 8)
        self.assertEqual(preregistration["gap_pair"]["reference_gap_scale"], 1.0)
        self.assertEqual(preregistration["gap_pair"]["probe_gap_scale"], 1.1)
        self.assertNotIn("1p3", source)
        self.assertIn("--reference-gap-scale", source)
        self.assertIn("--probe-gap-scale", source)


if __name__ == "__main__":
    unittest.main()
