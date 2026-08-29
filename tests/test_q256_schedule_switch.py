import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from training import reproducibility, schedule_switch


class ScheduleSwitchTests(unittest.TestCase):
    def make_state(self):
        net = torch.nn.Linear(2, 2)
        ema = copy.deepcopy(net)
        trajectory = {
            "schema": reproducibility.TRAJECTORY_CONFIG_SCHEMA,
            "seed": 14,
            "total_kimg": 1024,
            "batch_size": 128,
            "loss_kwargs": {
                "factorial_protocol": "q256_target_weight_v1",
                "target_gap_scale": 1.0,
                "denominator_gap_scale": 1.0,
            },
        }
        return {
            "net": net,
            "ema": ema,
            "optimizer_state": {
                "state": {0: {"step": torch.tensor(4000),
                              "exp_avg": torch.ones(2),
                              "exp_avg_sq": torch.ones(2)}},
                "param_groups": [{"params": [0]}],
            },
            "gradscaler_state": {"scale": 65536.0},
            "attempted_iteration": 4000,
            "successful_optimizer_steps": 3994,
            "cur_nimg": 512000,
            "rank_states": [{
                "rng_state": {"schema": "test", "torch": torch.arange(3)},
                "sampler_state": {"consumed_samples": 512000},
            }],
            "factorial": {
                "enabled": True,
                "protocol": "q256_target_weight_v1",
                "arm": "A",
                "target_gap_scale": 1.0,
                "denominator_gap_scale": 1.0,
            },
            "trajectory_config": trajectory,
            "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
        }

    def make_manifest(self, source_path, state):
        return {
            "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
            "experiment_protocol": schedule_switch.PROTOCOL,
            "run_kind": "formal",
            "branch": "A_to_B",
            "seed": 14,
            "origin_arm": "A",
            "continuation_arm": "B",
            "switch_kimg": 512,
            "final_kimg": 1024,
            "protocol_sha256": "1" * 64,
            "implementation_commit": "2" * 40,
            "source_checkpoint_manifest_sha256": "3" * 64,
            "source_state": {
                "path": str(source_path.resolve()),
                "bytes": source_path.stat().st_size,
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "internal_state_sha256": schedule_switch.internal_state_hashes(state),
            },
        }

    def test_manifest_source_and_metadata_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "training-state.pt"
            source_path.write_bytes(b"immutable-source")
            state = self.make_state()
            manifest = self.make_manifest(source_path, state)
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = schedule_switch.load_run_manifest(manifest_path)
            schedule_switch.verify_resume_state_file(source_path, loaded)
            before = copy.deepcopy(state["factorial"])
            hashes = schedule_switch.verify_source_state(state, loaded)
            self.assertEqual(hashes, manifest["source_state"]["internal_state_sha256"])
            self.assertEqual(state["factorial"], before)
            metadata = schedule_switch.state_metadata(loaded)
            self.assertEqual(metadata["origin_arm"], "A")
            self.assertEqual(metadata["continuation_arm"], "B")
            self.assertEqual(metadata["source_state_sha256"],
                             manifest["source_state"]["sha256"])

    def test_continuation_scales_and_trajectory_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "training-state.pt"
            source_path.write_bytes(b"immutable-source")
            state = self.make_state()
            manifest = self.make_manifest(source_path, state)
            self.assertEqual(
                schedule_switch.continuation_factorial(manifest),
                {
                    "enabled": True,
                    "protocol": "q256_target_weight_v1",
                    "arm": "B",
                    "target_gap_scale": 1.1,
                    "denominator_gap_scale": 1.1,
                },
            )
            current = copy.deepcopy(state["trajectory_config"])
            current["loss_kwargs"]["target_gap_scale"] = 1.1
            current["loss_kwargs"]["denominator_gap_scale"] = 1.1
            state["trajectory_config"]["dataset_kwargs"] = {
                "path": "/data/raw/canonical.zip", "resolution": 32
            }
            current["dataset_kwargs"] = {
                "path": "/mnt/canonical.zip", "resolution": 32
            }
            self.assertTrue(schedule_switch.trajectory_configs_compatible(
                state["trajectory_config"], current, manifest
            ))
            current["batch_size"] = 64
            self.assertFalse(schedule_switch.trajectory_configs_compatible(
                state["trajectory_config"], current, manifest
            ))

    def test_switched_state_keeps_origin_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "training-state.pt"
            source_path.write_bytes(b"immutable-source")
            state = self.make_state()
            manifest = self.make_manifest(source_path, state)
            state["schedule_switch"] = schedule_switch.state_metadata(manifest)
            schedule_switch.verify_switched_state(state, manifest)
            state["factorial"]["arm"] = "B"
            with self.assertRaisesRegex(RuntimeError, "origin factorial"):
                schedule_switch.verify_switched_state(state, manifest)

    def test_manifest_rejects_branch_selection_after_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "training-state.pt"
            source_path.write_bytes(b"immutable-source")
            state = self.make_state()
            manifest = self.make_manifest(source_path, state)
            manifest["branch"] = "A_to_A"
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid schedule-switch branch"):
                schedule_switch.load_run_manifest(manifest_path)

    def test_seed3_7_protocol_uses_its_frozen_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "training-state.pt"
            source_path.write_bytes(b"immutable-source")
            state = self.make_state()
            state["trajectory_config"]["seed"] = 3
            state["trajectory_config_sha256"] = reproducibility.state_sha256(
                state["trajectory_config"]
            )
            manifest = self.make_manifest(source_path, state)
            manifest_path = Path(directory) / "manifest.json"
            for protocol in (
                schedule_switch.SEED3_7_PROTOCOL,
                schedule_switch.SEED3_7_PROTOCOL_V2,
                schedule_switch.SEED3_7_PROTOCOL_V3,
            ):
                manifest["experiment_protocol"] = protocol
                manifest["seed"] = 3
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                loaded = schedule_switch.load_run_manifest(manifest_path)
                self.assertEqual(loaded["seed"], 3)
                schedule_switch.verify_source_state(state, loaded)
            manifest["seed"] = 8
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outside the frozen"):
                schedule_switch.load_run_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
