import importlib.util
import json
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/second_q_ab_q128_learning_curve.frozen.json"
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

    def test_frozen_config_validates(self):
        self.launcher.validate_config(self.config)

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
