import importlib.util
import json
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/second_q_ab_q128_learning_curve.frozen.json"
CONFIG_V2_PATH = ROOT / "configs/second_q_ab_q128_learning_curve_v2.frozen.json"
VERDICT_TEMPLATE = ROOT / "configs/role_e_q128_dataset_verdict.template.json"
LAUNCHER_PATH = ROOT / "scripts/run_second_q_ab_q128.py"


def load_launcher() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("second_q_launcher", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SecondQProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load_launcher()
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.config_v2 = json.loads(CONFIG_V2_PATH.read_text(encoding="utf-8"))

    def test_frozen_config_validates(self):
        self.launcher.validate_config(self.config)

    def test_canonical_v2_config_validates(self):
        self.launcher.validate_config(self.config_v2)
        amendment = self.config_v2["amendment"]
        self.assertFalse(amendment["scientific_results_observed_before_amendment"])
        self.assertFalse(amendment["training_started_before_amendment"])
        self.assertEqual(
            self.config_v2["provenance_gate"]["required_preflight_status"],
            "GO_CANONICAL_DATASET",
        )

    def test_v2_priority_cannot_select_budgets(self):
        primary = self.config_v2["evaluation"]["primary"]
        self.assertTrue(primary["execution_priority_is_not_selection"])
        self.assertTrue(primary["all_frozen_budgets_mandatory"])
        self.assertEqual(primary["budgets_kimg"], [512, 640, 768, 896, 1024])

    def test_v2_maps_exactly_one_cell_to_each_of_six_gpus(self):
        execution = self.config_v2["runtime_execution"]
        self.assertEqual(execution["max_concurrent_cells"], 6)
        self.assertEqual(
            set(execution["gpu_assignment"].values()),
            {
                "seed3-armA",
                "seed3-armB",
                "seed4-armA",
                "seed4-armB",
                "seed5-armA",
                "seed5-armB",
            },
        )

    def test_v2_freezes_validation_only_q_scope_amendment(self):
        source = self.config_v2["source_contract"]
        amendment = source["strict_protocol_q_scope_amendment"]
        self.assertEqual(amendment["optimizer_steps_before_amendment"], 0)
        self.assertFalse(amendment["scientific_results_observed_before_amendment"])
        self.assertFalse(amendment["scientific_math_changed"])
        self.assertEqual(amendment["path"], "training/loss.py")
        self.assertEqual(
            self.launcher.sha256_file(ROOT / amendment["path"]),
            amendment["amended_file_sha256"],
        )

    def test_training_command_is_one_requested_cell(self):
        args = types.SimpleNamespace(
            runtime_python=Path("/runtime/python"),
            repo=Path("/repo"),
            dataset=Path("/canonical.zip"),
            transfer=Path("/transfer.pkl"),
        )
        command = self.launcher.training_command(
            args,
            self.config_v2,
            seed=4,
            arm="B",
            run_dir=Path("/runs/seed4/armB"),
        )
        self.assertIn("--seed=4", command)
        self.assertIn("--target-gap-scale=1.1", command)
        self.assertIn("--denominator-gap-scale=1.1", command)
        self.assertIn("--outdir=/runs/seed4/armB", command)

    def test_v2_requires_all_42_ema_snapshot_receipts(self):
        artifact = self.config_v2["artifact_export"]
        self.assertEqual(
            artifact["script"], "scripts/export_second_q_ab_snapshots.py"
        )
        self.assertEqual(artifact["required_snapshot_count"], 42)
        self.assertTrue(artifact["rng_unchanged_required"])

    def test_matrix_is_only_q128_a_b_three_paired_seeds(self):
        self.assertEqual(self.config["training"]["schedule_q"], 128)
        self.assertEqual(self.config["training"]["training_seeds"], [3, 4, 5])
        self.assertEqual(self.config["scope"]["included_arms"], ["A", "B"])
        self.assertEqual(set(self.config["training"]["arms"]), {"A", "B"})

    def test_primary_and_secondary_semantics_are_frozen(self):
        evaluation = self.config["evaluation"]
        self.assertEqual(evaluation["sample_count"], 50000)
        self.assertEqual((evaluation["sample_seed_start"], evaluation["sample_seed_end"]), (0, 49999))
        self.assertEqual(evaluation["metric_seed"], 20260730)
        self.assertEqual(evaluation["primary"]["metric"], "fid50k_full")
        self.assertEqual(evaluation["primary"]["nfe"], 1)
        self.assertEqual(evaluation["secondary"]["nfe2"]["mid_t"], [0.821])

    def test_unresolved_role_e_template_is_rejected(self):
        verdict = json.loads(VERDICT_TEMPLATE.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(self.launcher.ContractError, "not conclusive"):
            self.launcher.validate_verdict(verdict, self.config)

    def test_conclusive_role_e_verdict_is_accepted(self):
        verdict = json.loads(VERDICT_TEMPLATE.read_text(encoding="utf-8"))
        verdict.update(
            verdict="NOT_EQUIVALENT",
            audit_id="role-e-q128-dataset-audit-v1",
            legacy_semantic_manifest_sha256="a" * 64,
            canonical_semantic_manifest_sha256="b" * 64,
            evidence_manifest_sha256="c" * 64,
            signed_off_utc="2026-08-23T03:00:00Z",
        )
        self.assertEqual(self.launcher.validate_verdict(verdict, self.config), "NOT_EQUIVALENT")


if __name__ == "__main__":
    unittest.main()
