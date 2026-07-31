import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProspectiveFrozenMatricesTest(unittest.TestCase):
    def load(self, name: str) -> dict:
        return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))

    def assert_common_contract(self, matrix: dict, expected_q: int) -> None:
        self.assertEqual(matrix["manifest_kind"], "frozen-logical-evaluation-matrix")
        self.assertEqual(matrix["protocol"], "staged-checkpoint-evaluation-v1")
        self.assertTrue(matrix["runtime_binding"]["required_before_evaluation"])
        self.assertFalse(matrix["runtime_binding"]["versioned_paths"])
        self.assertEqual(matrix["training"]["schedule_q"], expected_q)
        self.assertEqual(matrix["training"]["training_seeds"], [3, 4, 5])
        self.assertEqual(matrix["nfe_modes"], {"1": [], "2": [0.821]})
        self.assertEqual(
            matrix["comparison"]["pairing_key"],
            ["training_seed", "budget_kimg", "nfe", "metric"],
        )
        self.assertEqual(matrix["comparison"]["delta_direction"], "global_only - fixed")
        self.assertEqual(matrix["formal_promotion_policy"]["eligibility"], "provenance_and_integrity_only")
        self.assertEqual(
            matrix["formal_promotion_policy"]["quick_metric_performance"],
            "not_an_eligibility_criterion",
        )
        self.assertEqual(matrix["method_definitions"]["fixed"]["global_gap_scale"], 1.0)
        self.assertEqual(matrix["method_definitions"]["global110"]["global_gap_scale"], 1.1)
        self.assertEqual(matrix["method_definitions"]["fixed"]["schedule_identity"], "sigmoid")
        self.assertEqual(matrix["method_definitions"]["global110"]["schedule_identity"], "global_sigmoid")

        cells = matrix["cells"]
        required = matrix["formal_promotion_policy"]["required_evaluation_checkpoint_ids"]
        self.assertEqual([cell["checkpoint_id"] for cell in cells], required)
        self.assertEqual(len({cell["checkpoint_id"] for cell in cells}), len(cells))
        for cell in cells:
            self.assertEqual(cell["schedule_q"], expected_q)
            self.assertIn(cell["method"], ("fixed", "global110"))
            self.assertIn(cell["training_seed"], (3, 4, 5))
            self.assertNotIn("checkpoint", cell)
            self.assertNotIn("checkpoint_sha256", cell)

    def test_q256_budget_matrix_is_complete_and_has_per_budget_metrics(self):
        matrix = self.load("q256_budget_matrix.frozen.json")
        self.assert_common_contract(matrix, 256)
        self.assertEqual(matrix["training"]["budget_kimg"], [512, 768, 1024])
        self.assertEqual(len(matrix["cells"]), 18)
        self.assertEqual(
            {(cell["budget_kimg"], cell["method"], cell["training_seed"]) for cell in matrix["cells"]},
            {(budget, method, seed) for budget in (512, 768, 1024)
             for method in ("fixed", "global110") for seed in (3, 4, 5)},
        )
        contracts = {contract["budget_kimg"]: contract for contract in matrix["evaluation_contracts"]}
        for budget in (512, 768):
            self.assertEqual(contracts[budget]["stage"], "quick")
            self.assertEqual(contracts[budget]["metric_names"], ["kid5k_full", "fid5k_full"])
            self.assertEqual(contracts[budget]["sample_count"], 5000)
        self.assertEqual(contracts[1024]["stage"], "formal")
        self.assertEqual(contracts[1024]["metric_names"], ["kid50k_full", "fid50k_full"])
        self.assertEqual(contracts[1024]["sample_count"], 50000)
        formal_ids = matrix["formal_promotion_policy"]["required_formal_checkpoint_ids"]
        self.assertEqual(len(formal_ids), 6)
        self.assertTrue(all("-1024k-" in checkpoint_id for checkpoint_id in formal_ids))

    def test_q128_fresh_matrix_is_complete_and_formal(self):
        matrix = self.load("q128_confirmatory_matrix.frozen.json")
        self.assert_common_contract(matrix, 128)
        self.assertEqual(matrix["training"]["budget_kimg"], [256])
        self.assertEqual(len(matrix["cells"]), 6)
        self.assertEqual(
            {(cell["method"], cell["training_seed"]) for cell in matrix["cells"]},
            {(method, seed) for method in ("fixed", "global110") for seed in (3, 4, 5)},
        )
        self.assertEqual(matrix["evaluation_contracts"], [{
            "budget_kimg": 256,
            "stage": "formal",
            "evidence_class": "formal",
            "metric_names": ["kid50k_full", "fid50k_full"],
            "sample_count": 50000,
            "generation_seed_range": "0-49999",
            "metric_seed": 20260730,
        }])
        self.assertEqual(
            matrix["formal_promotion_policy"]["required_formal_checkpoint_ids"],
            matrix["formal_promotion_policy"]["required_evaluation_checkpoint_ids"],
        )


if __name__ == "__main__":
    unittest.main()
