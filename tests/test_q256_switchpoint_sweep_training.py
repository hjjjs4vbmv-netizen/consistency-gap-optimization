import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from training import ct_training_loop, reproducibility, schedule_switch


class SwitchpointSweepTrainingTests(unittest.TestCase):
    def test_sweep_prefix_pause_is_authorized_only_at_512(self):
        attempts = ct_training_loop.validate_planned_pause(
            stop_after_attempts=4000,
            planned_pause_protocol=schedule_switch.SWITCHPOINT_SWEEP_PROTOCOL,
            strict_reproducibility=True,
            seed=81,
            total_kimg=1024,
            resume_state_dump=None,
            schedule_switch_manifest=None,
        )
        self.assertEqual(attempts, 4000)
        with self.assertRaises(ValueError):
            ct_training_loop.validate_planned_pause(
                stop_after_attempts=3000,
                planned_pause_protocol=schedule_switch.SWITCHPOINT_SWEEP_PROTOCOL,
                strict_reproducibility=True,
                seed=81,
                total_kimg=1024,
                resume_state_dump=None,
                schedule_switch_manifest=None,
            )

    def make_state(self, switch_kimg):
        net = torch.nn.Linear(2, 2)
        trajectory = {
            "schema": reproducibility.TRAJECTORY_CONFIG_SCHEMA,
            "seed": 81,
            "total_kimg": 1024,
            "batch_size": 128,
            "loss_kwargs": {
                "factorial_protocol": "q256_target_weight_v1",
                "arm": "B",
                "target_gap_scale": 1.1,
                "denominator_gap_scale": 1.1,
            },
        }
        state = {
            "net": net,
            "ema": copy.deepcopy(net),
            "optimizer_state": {"state": {}, "param_groups": []},
            "gradscaler_state": {"scale": 65536.0},
            "attempted_iteration": switch_kimg * 1000 // 128,
            "successful_optimizer_steps": switch_kimg * 1000 // 128,
            "cur_nimg": switch_kimg * 1000,
            "rank_states": [{
                "rng_state": {"schema": "test", "torch": torch.arange(3)},
                "sampler_state": {"consumed_samples": switch_kimg * 1000},
            }],
            "factorial": {
                "enabled": True,
                "protocol": "q256_target_weight_v1",
                "arm": "B",
                "target_gap_scale": 1.1,
                "denominator_gap_scale": 1.1,
            },
            "trajectory_config": trajectory,
            "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
        }
        return state

    def make_manifest(self, source_path, state, switch_kimg):
        return {
            "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
            "experiment_protocol": schedule_switch.SWITCHPOINT_SWEEP_PROTOCOL,
            "run_kind": "formal",
            "branch": "BA",
            "seed": 81,
            "origin_arm": "B",
            "continuation_arm": "A",
            "switch_kimg": switch_kimg,
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

    def test_all_frozen_switchpoints_validate_against_their_source_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "training-state.pt"
            source_path.write_bytes(b"source")
            for switch_kimg in schedule_switch.SWEEP_SWITCH_KIMG:
                with self.subTest(switch_kimg=switch_kimg):
                    state = self.make_state(switch_kimg)
                    manifest = self.make_manifest(source_path, state, switch_kimg)
                    path = Path(directory) / f"manifest-{switch_kimg}.json"
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    loaded = schedule_switch.load_run_manifest(path)
                    schedule_switch.verify_source_state(state, loaded)
                    self.assertEqual(
                        schedule_switch.state_metadata(loaded)["switch_kimg"],
                        switch_kimg,
                    )

    def test_sweep_rejects_unplanned_switchpoint_and_ctrl_before_512(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "training-state.pt"
            source_path.write_bytes(b"source")
            state = self.make_state(128)
            manifest = self.make_manifest(source_path, state, 128)
            path = Path(directory) / "manifest.json"
            for mutation in ("switch", "ctrl"):
                changed = copy.deepcopy(manifest)
                if mutation == "switch":
                    changed["switch_kimg"] = 192
                else:
                    changed.update(
                        branch="CTRL",
                        origin_arm="A",
                        continuation_arm="A",
                    )
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                    schedule_switch.load_run_manifest(path)


if __name__ == "__main__":
    unittest.main()
