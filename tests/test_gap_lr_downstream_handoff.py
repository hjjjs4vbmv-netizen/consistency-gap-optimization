"""CPU-only tests for the formal downstream artifact handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/prepare_gap_lr_downstream.py"
SPEC = importlib.util.spec_from_file_location("prepare_gap_lr_downstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class GapLRDownstreamHandoffTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        data = root / "cifar10.zip"
        transfer = root / "edm.pkl"
        data.write_bytes(b"dataset")
        transfer.write_bytes(b"transfer")
        arms = {}
        for arm, run_id, gap, lr in (
            ("A", "arm_a_g1_0_lr_fixed_s3", 1.0, 1e-4),
            ("B", "arm_b_g1_3_lr_fixed_s3", 1.3, 1e-4),
            ("C", "arm_c_g1_3_lr_matched_s3", 1.3, 1.2963523762588692e-4),
        ):
            run_dir = root / "experiment" / run_id
            run_dir.mkdir(parents=True)
            states = {}
            for index, state_id in enumerate(MODULE.STATE_IDS):
                state_payload = f"{arm}-state-{state_id}".encode()
                snapshot_payload = f"{arm}-snapshot-{state_id}".encode()
                (run_dir / f"training-state-{state_id}.pt").write_bytes(state_payload)
                (run_dir / f"network-snapshot-{state_id}.pkl").write_bytes(snapshot_payload)
                actual_kimg = (32.128, 64.128, 128.128, 256.0)[index]
                states[state_id] = {
                    "actual_kimg": actual_kimg,
                    "successful_optimizer_steps": (243, 493, 993, 1991)[index],
                    "sha256": digest(state_payload),
                }
            arms[arm] = {
                "gap_scale": gap,
                "learning_rate": lr,
                "run_id": run_id,
                "states": states,
            }
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({
            "experiment_id": MODULE.EXPERIMENT_ID,
            "status": "passed",
            "source": {
                "dataset_sha256": digest(b"dataset"),
                "transfer_sha256": digest(b"transfer"),
            },
            "arms": arms,
        }))
        return data, transfer, receipt

    def args(self, root: Path, scope: str, data: Path, transfer: Path, receipt: Path):
        return Namespace(
            experiment_root=root / "experiment",
            data=data,
            transfer=transfer,
            receipt=receipt,
            scope=scope,
            deserialize=False,
            out=root / "manifest.json",
        )

    def test_role_d_scope_uses_only_arm_a_four_point_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, transfer, receipt = self.make_fixture(root)
            manifest, errors = MODULE.build_manifest(
                self.args(root, "role-d", data, transfer, receipt)
            )
        self.assertEqual(errors, [])
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual({row["arm"] for row in manifest["artifacts"]}, {"A"})
        self.assertEqual(
            [row["state_id"] for row in manifest["artifacts"]],
            list(MODULE.STATE_IDS),
        )
        self.assertFalse(manifest["role_d_contract"]["mix_arms_across_k"])

    def test_role_e_scope_freezes_three_final_numbered_ema_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, transfer, receipt = self.make_fixture(root)
            manifest, errors = MODULE.build_manifest(
                self.args(root, "role-e", data, transfer, receipt)
            )
        self.assertEqual(errors, [])
        self.assertEqual(len(manifest["artifacts"]), 3)
        self.assertEqual({row["arm"] for row in manifest["artifacts"]}, {"A", "B", "C"})
        self.assertEqual({row["state_id"] for row in manifest["artifacts"]}, {"000008"})
        self.assertEqual(manifest["role_e_contract"]["sample_seeds"], "0-4999")
        self.assertEqual(manifest["role_e_contract"]["precision"], "fp32")
        self.assertTrue(all(row["network_snapshot_sha256"] for row in manifest["artifacts"]))

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, transfer, receipt = self.make_fixture(root)
            broken = (
                root / "experiment/arm_a_g1_0_lr_fixed_s3/training-state-000008.pt"
            )
            broken.write_bytes(b"corrupted")
            manifest, errors = MODULE.build_manifest(
                self.args(root, "role-e", data, transfer, receipt)
            )
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(any("SHA256 mismatch" in error for error in errors))

    def test_launchers_freeze_the_registered_inputs(self):
        role_d = (REPO_ROOT / "scripts/run_gap_lr_longitudinal_audit.sh").read_text()
        role_e = (REPO_ROOT / "scripts/run_gap_lr_nfe1_quality.sh").read_text()
        for state_id in MODULE.STATE_IDS:
            self.assertIn(state_id, role_d)
        self.assertIn("arm_a_g1_0_lr_fixed_s3", role_d)
        self.assertNotIn("arm_b_g1_3_lr_fixed_s3", role_d)
        self.assertIn("--sample-seeds=\"$SAMPLE_SEEDS\"", role_e)
        self.assertIn("SAMPLE_SEEDS=0-4999", role_e)
        self.assertIn("--nfe=1", role_e)
        self.assertIn("--fp16=False", role_e)
        self.assertIn("network-snapshot-000008.pkl", role_e)


if __name__ == "__main__":
    unittest.main()
