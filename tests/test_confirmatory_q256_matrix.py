import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "staged_evaluation_confirmatory_q256.frozen.json"


class ConfirmatoryQ256MatrixTest(unittest.TestCase):
    def test_matrix_is_complete_and_path_free(self):
        matrix = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(matrix["manifest_kind"], "frozen-logical-checkpoint-matrix")
        self.assertTrue(matrix["runtime_binding"]["required"])
        self.assertFalse(matrix["runtime_binding"]["versioned_paths"])
        self.assertEqual(matrix["training"]["budget_kimg"], 256)
        self.assertEqual(matrix["training"]["schedule_q"], 256)
        self.assertEqual(matrix["training"]["training_seeds"], [3, 4, 5])
        self.assertEqual(
            matrix["comparison"]["pairing_key"],
            ["training_seed", "budget_kimg", "nfe", "metric"],
        )
        self.assertEqual(matrix["comparison"]["delta_direction"], "global_only - fixed")

        cells = matrix["cells"]
        self.assertEqual(len(cells), 6)
        self.assertEqual(
            {(cell["method"], cell["training_seed"]) for cell in cells},
            {(method, seed) for method in ("fixed", "global110") for seed in (3, 4, 5)},
        )

        for cell in cells:
            self.assertNotIn("checkpoint", cell)
            self.assertEqual(cell["budget_kimg"], 256)
            self.assertEqual(cell["schedule_q"], 256)
            self.assertRegex(cell["checkpoint_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(cell["executed_training_source_commit"], r"^[0-9a-f]{40}$")
            self.assertEqual(cell["training_integrity_receipt"]["status"], "passed")
            self.assertTrue(cell["training_integrity_receipt"]["receipt_filename"])

    def test_method_schedule_identities_are_frozen(self):
        matrix = json.loads(MANIFEST.read_text(encoding="utf-8"))
        definitions = matrix["method_definitions"]
        self.assertEqual(definitions["fixed"]["schedule_identity"], "sigmoid")
        self.assertEqual(definitions["fixed"]["global_gap_scale"], 1.0)
        self.assertEqual(definitions["global110"]["schedule_identity"], "global_sigmoid")
        self.assertEqual(definitions["global110"]["global_gap_scale"], 1.1)


if __name__ == "__main__":
    unittest.main()
