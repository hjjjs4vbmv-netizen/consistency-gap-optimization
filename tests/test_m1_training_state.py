import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import run_m1_training_slot as launcher
from training import m1, reproducibility, schedule_switch


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("running", torch.tensor([3.0]))


class M1TrainingStateTests(unittest.TestCase):
    def make_manifest(self, directory, branch="R_A"):
        source = Path(directory) / "source.pt"
        source.touch()
        origin = "B" if branch.endswith("_B") else "A"
        return {
            "schema": schedule_switch.RUN_MANIFEST_SCHEMA,
            "experiment_protocol": m1.PROTOCOL_ID,
            "run_kind": "formal",
            "branch": branch,
            "seed": 50,
            "origin_arm": origin,
            "continuation_arm": "A",
            "switch_kimg": 512,
            "final_kimg": 1024,
            "source_state": {"path": str(source.resolve())},
            "immutable_output_root": str(Path(directory).resolve()),
            "m1_shadow_update": True,
        }

    def source_state(self, arm="A"):
        trajectory = {"seed": 50, "total_kimg": 1024}
        return {
            "net": TinyModule(),
            "ema": TinyModule(),
            "optimizer_state": {
                "state": {0: {"step": torch.tensor(3.0)}},
                "param_groups": [{"lr": 1e-4, "params": [0]}],
            },
            "gradscaler_state": {"scale": 65536.0},
            "rank_states": [{
                "rng_state": {"torch_cpu": torch.get_rng_state()},
                "sampler_state": {"consumed_samples": 512000},
            }],
            "loss_fn_state": {"stage": 0},
            "attempted_iteration": 4000,
            "successful_optimizer_steps": 3999,
            "cur_nimg": 512000,
            "cur_tick": 1,
            "tick_start_nimg": 510000,
            "snapshot_grid_z": [],
            "snapshot_grid_c": [],
            "snapshot_grid_size": (0, 0),
            "factorial": {
                "protocol": "q256_target_weight_v1",
                "arm": arm,
                "target_gap_scale": 1.1 if arm == "B" else 1.0,
                "denominator_gap_scale": 1.1 if arm == "B" else 1.0,
            },
            "trajectory_config": trajectory,
            "trajectory_config_sha256": reproducibility.state_sha256(trajectory),
            "reproducibility_schema": reproducibility.TRAINING_STATE_SCHEMA,
        }

    def test_compact_manifest_needs_no_hash_or_gate_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory)
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(schedule_switch.load_run_manifest(path), manifest)
            self.assertNotIn("sha256", json.dumps(manifest).lower())

    def test_optimizer_keep_and_reset(self):
        model = TinyModule()
        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        before = copy.deepcopy(optimizer.state_dict())
        self.assertEqual(m1.apply_optimizer_intervention(optimizer, "K_A"), 0)
        self.assertEqual(optimizer.state_dict()["param_groups"], before["param_groups"])
        self.assertTrue(optimizer.state)
        self.assertEqual(m1.apply_optimizer_intervention(optimizer, "R_A"), 1)
        self.assertFalse(optimizer.state)
        self.assertEqual(optimizer.state_dict()["param_groups"], before["param_groups"])

    def test_e512_copy_and_update_do_not_consume_rng(self):
        online = TinyModule()
        rng_before = torch.get_rng_state()
        ema_512 = m1.initialize_ema_512(online)
        with torch.no_grad():
            online.weight.add_(2)
            online.running.add_(5)
        m1.update_ema_512(ema_512, online, 0.5)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(ema_512.weight, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(ema_512.running, torch.tensor([3.0])))

    def test_source_and_branch_init_semantics_are_checked_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source_state()
            for branch, reset in (("K_A", False), ("R_A", True)):
                manifest = self.make_manifest(directory, branch)
                schedule_switch.verify_source_state(source, manifest)
                branch_state = copy.deepcopy(source)
                if reset:
                    branch_state["optimizer_state"]["state"] = {}
                branch_state["ema_512"] = copy.deepcopy(source["net"])
                branch_state["m1"] = m1.initial_metadata(
                    manifest, int(reset), source["successful_optimizer_steps"]
                )
                self.assertTrue(
                    m1.validate_branch_init_against_source(
                        branch_state, source, manifest
                    )
                )
                branch_state["rank_states"][0]["sampler_state"][
                    "consumed_samples"
                ] += 1
                with self.assertRaisesRegex(RuntimeError, "rank_states"):
                    m1.validate_branch_init_against_source(
                        branch_state, source, manifest
                    )

    def test_resume_metadata_prevents_second_reset_or_e512_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory, "R_B")
            metadata = m1.initial_metadata(manifest, 1, 3990)
            state = {
                "m1": m1.checkpoint_metadata(metadata, 7),
                "ema_512": TinyModule(),
                "successful_optimizer_steps": 3997,
            }
            restored = m1.validate_resumed_state(state, manifest)
            self.assertEqual(restored["reset_count"], 1)
            self.assertEqual(restored["successful_steps_since_init"], 7)

    def test_slot_rotation_is_fixed_but_seed_is_explicit(self):
        self.assertEqual(launcher.parse_slot("S01")[1], launcher.ORDERS[0])
        self.assertEqual(launcher.parse_slot("S05")[1], launcher.ORDERS[0])
        self.assertEqual(launcher.parse_slot("S16")[1], launcher.ORDERS[3])
        with self.assertRaises(ValueError):
            launcher.parse_slot("S17")


if __name__ == "__main__":
    unittest.main()
