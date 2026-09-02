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
    @staticmethod
    def saved_state(*, optimizer_step=243, scaler_scale=256.0):
        return {
            "attempted_iteration": 251,
            "cur_nimg": 32128,
            "cur_tick": 1,
            "elapsed_sec": 1.0,
            "gradscaler_state": {"scale": scaler_scale},
            "loss_fn_state": {},
            "net": {},
            "optimizer_state": {
                "state": {
                    index: {
                        "step": optimizer_step,
                        "exp_avg": 0,
                        "exp_avg_sq": 0,
                    }
                    for index in range(416)
                }
            },
            "successful_optimizer_steps": 243,
            "tick_start_nimg": 0,
        }

    @staticmethod
    def expected_state():
        return {
            "actual_kimg": 32.128,
            "successful_optimizer_steps": 243,
            "gradscaler_scale": 256.0,
        }

    def make_fixture(self, root: Path):
        data = root / "cifar10.zip"
        transfer = root / "edm.pkl"
        data.write_bytes(b"dataset")
        transfer.write_bytes(b"transfer")
        experiment = root / "experiment"
        (experiment / "logs").mkdir(parents=True)
        (experiment / "logs/A.log").write_text("Training complete\nExiting...\n")
        (experiment / "launch_provenance.txt").write_text(
            "experiment_id=gap_lr_matched_q128_s3_v1\n"
        )
        launcher_log = root / "experiment.launcher.log"
        launcher_log.write_text("ALL FORMAL ARMS COMPLETE\n")
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
            (run_dir / "training_options.json").write_text(json.dumps({
                "run_dir": str(run_dir.resolve()),
                "loss_kwargs": {
                    "adj": "global_sigmoid",
                    "global_gap_scale": gap,
                },
                "optimizer_kwargs": {"lr": lr},
                "seed": 3,
                "total_kimg": 256,
                "batch_size": 128,
                "batch_gpu": 16,
                "enable_amp": True,
            }))
            summary_lines = [
                "attempted_iteration,successful_optimizer_steps,processed_kimg,schedule"
            ]
            for index, state_id in enumerate(MODULE.STATE_IDS):
                summary_lines.append(
                    f"{(251, 501, 1001, 2000)[index]},"
                    f"{states[state_id]['successful_optimizer_steps']},"
                    f"{states[state_id]['actual_kimg']:.3f},global_sigmoid"
                )
            (run_dir / "train_summary.csv").write_text(
                "\n".join(summary_lines) + "\n"
            )
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
        return data, transfer, receipt, launcher_log

    def args(
        self,
        root: Path,
        scope: str,
        data: Path,
        transfer: Path,
        receipt: Path,
        launcher_log: Path,
    ):
        return Namespace(
            experiment_root=root / "experiment",
            data=data,
            transfer=transfer,
            launcher_log=launcher_log,
            receipt=receipt,
            scope=scope,
            deserialize=False,
            out=root / "manifest.json",
        )

    def test_role_d_scope_uses_only_arm_a_four_point_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, transfer, receipt, launcher_log = self.make_fixture(root)
            manifest, errors = MODULE.build_manifest(
                self.args(root, "role-d", data, transfer, receipt, launcher_log)
            )
        self.assertEqual(errors, [])
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual({row["arm"] for row in manifest["artifacts"]}, {"A"})
        self.assertEqual(
            [row["state_id"] for row in manifest["artifacts"]],
            list(MODULE.STATE_IDS),
        )
        self.assertFalse(manifest["role_d_contract"]["mix_arms_across_k"])
        provenance = manifest["role_d_provenance"]
        self.assertEqual(provenance["status"], "passed")
        self.assertTrue(provenance["single_uninterrupted_run"])
        self.assertEqual(provenance["trajectory_id"], "arm_a_g1_0_lr_fixed_s3")
        self.assertFalse(provenance["role_d_may_substitute_or_mix_runs"])
        self.assertEqual(
            set(provenance["files"]),
            {
                "train_summary",
                "training_options",
                "run_log",
                "launch_provenance",
                "launcher_log",
            },
        )
        self.assertTrue(
            all(record["sha256"] for record in provenance["files"].values())
        )
        self.assertEqual(
            set(provenance["train_summary_inspection"]["checkpoint_rows"]),
            set(MODULE.STATE_IDS),
        )

    def test_role_e_scope_freezes_three_final_numbered_ema_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, transfer, receipt, launcher_log = self.make_fixture(root)
            manifest, errors = MODULE.build_manifest(
                self.args(root, "role-e", data, transfer, receipt, launcher_log)
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
            data, transfer, receipt, launcher_log = self.make_fixture(root)
            broken = (
                root / "experiment/arm_a_g1_0_lr_fixed_s3/training-state-000008.pt"
            )
            broken.write_bytes(b"corrupted")
            manifest, errors = MODULE.build_manifest(
                self.args(root, "role-e", data, transfer, receipt, launcher_log)
            )
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(any("SHA256 mismatch" in error for error in errors))

    def test_role_d_rejects_training_options_from_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, transfer, receipt, launcher_log = self.make_fixture(root)
            options_path = (
                root
                / "experiment/arm_a_g1_0_lr_fixed_s3/training_options.json"
            )
            options = json.loads(options_path.read_text())
            options["run_dir"] = "/data/raw/ECT/ect_runs/g_screen/g1_0"
            options_path.write_text(json.dumps(options))
            manifest, errors = MODULE.build_manifest(
                self.args(root, "role-d", data, transfer, receipt, launcher_log)
            )
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["role_d_provenance"]["status"], "failed")
        self.assertTrue(any("does not identify" in error for error in errors))

    def test_role_d_requires_detached_launcher_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, transfer, receipt, launcher_log = self.make_fixture(root)
            args = self.args(
                root, "role-d", data, transfer, receipt, launcher_log
            )
            args.launcher_log = None
            manifest, errors = MODULE.build_manifest(args)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["role_d_provenance"]["status"], "failed")
        self.assertTrue(any("--launcher-log is required" in error for error in errors))

    def test_training_state_binds_optimizer_steps_to_successful_updates(self):
        original = MODULE._torch_load
        try:
            MODULE._torch_load = lambda _path: self.saved_state(optimizer_step=242)
            with self.assertRaisesRegex(RuntimeError, "optimizer parameter steps"):
                MODULE.inspect_training_state(Path("unused.pt"), self.expected_state())
        finally:
            MODULE._torch_load = original

    def test_training_state_binds_gradscaler_to_receipt(self):
        original = MODULE._torch_load
        try:
            MODULE._torch_load = lambda _path: self.saved_state(scaler_scale=128.0)
            with self.assertRaisesRegex(RuntimeError, "GradScaler scale"):
                MODULE.inspect_training_state(Path("unused.pt"), self.expected_state())
        finally:
            MODULE._torch_load = original

    def test_committed_role_d_receipt_is_passed_and_path_sanitized(self):
        path = (
            REPO_ROOT
            / "results/gap_lr_matched/role_d_formal_arm_a_handoff_receipt.json"
        )
        text = path.read_text(encoding="utf-8")
        receipt = json.loads(text)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["trajectory"]["run_id"], "arm_a_g1_0_lr_fixed_s3")
        self.assertTrue(receipt["trajectory"]["single_uninterrupted_run"])
        self.assertFalse(receipt["trajectory"]["role_d_may_substitute_or_mix_runs"])
        self.assertEqual(
            [row["actual_kimg"] for row in receipt["artifacts"]],
            [32.128, 64.128, 128.128, 256.0],
        )
        self.assertTrue(
            all(
                row["state_restoration"]["optimizer_param_state_count"] == 416
                and len(row["state_restoration"]["optimizer_parameter_steps"]) == 1
                for row in receipt["artifacts"]
            )
        )
        for forbidden in ("/data/", "172.16.", "ECT001@", "/Users/"):
            self.assertNotIn(forbidden, text)

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
