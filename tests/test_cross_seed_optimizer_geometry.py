"""Contract tests for the K=256 cross-seed mechanism operation."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("cross_seed_runner", ROOT / "scripts" / "run_cross_seed_optimizer_geometry_audit.py")
SUMMARY = load_module("cross_seed_summary", ROOT / "scripts" / "summarize_cross_seed_optimizer_geometry.py")
EXAMPLE = ROOT / "configs" / "cross_seed_optimizer_geometry_matrix.example.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CrossSeedOptimizerGeometryTests(unittest.TestCase):
    def executable_logical_manifest(self) -> dict:
        payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        payload["dataset"] = {"path": "/data/cifar10-32x32.zip", "sha256": "0" * 64}
        for row in payload["seed_rows"]:
            if row["training_seed"] in (4, 5):
                seed = row["training_seed"]
                row["training_state"] = {
                    "path": f"/data/archive/seed{seed}/training-state-000008.pt",
                    "sha256": str(seed) * 64,
                }
                row["checkpoint"]["path"] = f"/data/archive/seed{seed}/network-snapshot-000008.pkl"
        return payload

    def test_example_is_frozen_at_canonical_k256_protocol(self):
        payload = self.executable_logical_manifest()
        rows = RUNNER.validate_manifest(payload)
        self.assertEqual(sorted(rows), [3, 4, 5])
        self.assertEqual(payload["canonical"]["layer_a"]["probe_rng_seed"], 20260810)
        self.assertEqual(payload["canonical"]["layer_b"]["n_steps"], 20)
        self.assertEqual(payload["canonical"]["layer_b"]["eval_step"], 19)
        self.assertEqual(rows[3]["row_kind"], "existing_artifact")
        self.assertEqual(rows[4]["row_kind"], "new_independent_training_trajectory")
        self.assertEqual(rows[5]["row_kind"], "new_independent_training_trajectory")

    def test_seed3_existing_anchor_is_hash_bound_and_protocol_complete(self):
        payload = self.executable_logical_manifest()
        rows = RUNNER.validate_manifest(payload)
        anchor = RUNNER.read_existing_seed3(rows[3], payload["canonical"])
        self.assertFalse(anchor["layer_a"]["executed"])
        self.assertFalse(anchor["layer_b"]["executed"])
        self.assertEqual(anchor["state_kimg"], 256)
        self.assertEqual(anchor["training_state"]["sha256"], rows[3]["training_state"]["sha256"])

    def test_rejects_mutable_latest_alias(self):
        payload = self.executable_logical_manifest()
        seed4 = next(row for row in payload["seed_rows"] if row["training_seed"] == 4)
        seed4["training_state"]["path"] = "/data/archive/seed4/training-state-latest.pt"
        with self.assertRaisesRegex(SystemExit, "latest alias"):
            RUNNER.validate_manifest(payload)

    def test_summary_writes_three_seed_table_and_preserves_q_accounting(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "operation"
            root.mkdir()
            seed_results = []
            for seed, schedule_q in ((3, 128), (4, 256), (5, 256)):
                layer_a_path = root / f"seed{seed}" / "layer_a.json"
                layer_b_path = root / f"seed{seed}" / "layer_b.json"
                layer_a_path.parent.mkdir()
                layer_a = {
                    "whole_model": {
                        "gauge_defined": True, "a_K_star": 0.75, "R_grad": 0.04,
                        "s_K_star": 0.95, "c_K_star": 1.05, "R_opt": 0.08,
                        "on_support_gauge_dispersion_energy": 0.0063,
                        "off_support_candidate_energy_exact": 0.0001, "H_K": 0.08,
                    }
                }
                layer_b = {
                    "T_steps": 20, "eval_step": 19, "a_star_mean": 0.77,
                    "a_star_std": 0.02, "weighted_R2_scalar_vs_actual": 0.70,
                    "corr_scalar_vs_actual": 0.84, "weighted_RMSE_scalar_vs_actual": 0.03,
                }
                layer_a_path.write_text(json.dumps(layer_a), encoding="utf-8")
                layer_b_path.write_text(json.dumps(layer_b), encoding="utf-8")
                seed_results.append({
                    "training_seed": seed, "row_kind": "existing_artifact" if seed == 3 else "new_independent_training_trajectory",
                    "training_trajectory_id": f"trajectory-{seed}", "state_kimg": 256, "schedule_q": schedule_q,
                    "layer_a": {"storage": "operation", "receipt_path": str(layer_a_path.relative_to(root)), "receipt_sha256": sha256(layer_a_path)},
                    "layer_b": {"storage": "operation", "receipt_path": str(layer_b_path.relative_to(root)), "receipt_sha256": sha256(layer_b_path)},
                })
            manifest = {
                "protocol": SUMMARY.PROTOCOL_ID, "status": "passed", "seed_results": seed_results,
                "replication_accounting": {
                    "all_kimg_equal": True, "all_training_trajectory_ids_distinct": True,
                    "all_schedule_q_equal": False, "schedule_q_by_seed": {"3": 128, "4": 256, "5": 256},
                    "batch_replication_is_not_training_seed_replication": True,
                },
            }
            (root / "audit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            out = root / "summary"
            rows = SUMMARY.summarize(root, out)
            self.assertEqual([row["seed"] for row in rows], [3, 4, 5])
            self.assertTrue((out / "optimizer_geometry_table.csv").is_file())
            markdown = (out / "OPTIMIZER_GEOMETRY_TABLE.md").read_text(encoding="utf-8")
            self.assertIn("all schedules equal: `False`", markdown)
            self.assertIn("not a repeated-minibatch estimate", markdown)


if __name__ == "__main__":
    unittest.main()
