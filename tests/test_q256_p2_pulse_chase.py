from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from analysis.q256_p2_b384_pulse_chase_v1.analyze_results import summarize
from ct_train import main as train_cli
from training import pulse_chase
from training.ct_training_loop import (
    _FACTORIAL_TELEMETRY_FIELDS,
    _P2_TAPE_TELEMETRY_FIELDS,
)


HEX64 = "a" * 64


def manifest(path: Path, *, seed: int = 19, run_kind: str = "formal") -> dict:
    internal = {
        "online_parameters": HEX64,
        "ema": HEX64,
        "radam": HEX64,
        "gradscaler": HEX64,
        "rank_rng": [HEX64],
        "sampler": [HEX64],
        "loss_control": HEX64,
        "trajectory_config": HEX64,
        "data_cursor": [384000],
        "attempted_iteration": 3000,
        "successful_optimizer_steps": 3000,
        "cur_nimg": 384000,
    }
    return {
        "schema": pulse_chase.RUN_MANIFEST_SCHEMA,
        "experiment_protocol": pulse_chase.PROTOCOL,
        "run_kind": run_kind,
        "seed": seed,
        "branch": "Early-switch",
        "pulse_arm": "A",
        "chase_arm": "A",
        "source_kimg": 384,
        "pulse_end_kimg": 512,
        "chase_end_kimg": 640,
        "protocol_sha256": HEX64,
        "implementation_commit": "b" * 40,
        "asset_sha256": dict(pulse_chase.ASSET_SHA256),
        "source_inventory_sha256": HEX64,
        "source_state": {
            "path": str(path.resolve()), "bytes": 1, "sha256": HEX64,
            "internal_state_sha256": internal,
        },
        "gpu_index": 0,
        "gpu_uuid": "GPU-test",
        "immutable_output_root": str(path.parent.resolve()),
        "matched_randomness_audit": False,
    }


class PulseChaseContractTest(unittest.TestCase):
    def test_frozen_design_constants(self):
        self.assertEqual(pulse_chase.SEEDS, tuple(range(19, 29)))
        self.assertEqual(pulse_chase.SOURCE_ATTEMPT, 3000)
        self.assertEqual(pulse_chase.PULSE_END_ATTEMPT, 4000)
        self.assertEqual(pulse_chase.CHASE_END_ATTEMPT, 5000)
        self.assertEqual(
            pulse_chase.BRANCHES,
            {
                "Early-switch": {"pulse_arm": "A", "chase_arm": "A"},
                "Late-switch": {"pulse_arm": "B", "chase_arm": "A"},
            },
        )

    def test_manifest_formal_and_smoke_seed_domains_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            path.write_bytes(b"x")
            formal_path = Path(directory) / "formal.json"
            formal_path.write_text(json.dumps(manifest(path)))
            self.assertEqual(pulse_chase.load_run_manifest(formal_path)["seed"], 19)
            bad = manifest(path, seed=18, run_kind="formal")
            bad_path = Path(directory) / "bad.json"
            bad_path.write_text(json.dumps(bad))
            with self.assertRaises(RuntimeError):
                pulse_chase.load_run_manifest(bad_path)
            smoke = manifest(path, seed=18, run_kind="smoke")
            smoke_path = Path(directory) / "smoke.json"
            smoke_path.write_text(json.dumps(smoke))
            self.assertEqual(pulse_chase.load_run_manifest(smoke_path)["seed"], 18)

    def test_trajectory_compatibility_changes_only_budget_and_arm(self):
        base = {
            "total_kimg": 384,
            "matched_randomness_audit": False,
            "loss_kwargs": {"arm": "B", "target_gap_scale": 1.1,
                            "denominator_gap_scale": 1.1, "q": 256},
            "batch_size": 128,
        }
        current = {
            "total_kimg": 512,
            "matched_randomness_audit": True,
            "loss_kwargs": {"arm": "A", "target_gap_scale": 1.0,
                            "denominator_gap_scale": 1.0, "q": 256},
            "batch_size": 128,
        }
        phase = {"name": "pulse", "start_kimg": 384,
                 "end_kimg": 512, "arm": "A"}
        self.assertTrue(pulse_chase.trajectory_configs_compatible(
            base, current, {}, phase
        ))
        current["batch_size"] = 64
        self.assertFalse(pulse_chase.trajectory_configs_compatible(
            base, current, {}, phase
        ))

    def test_smoke_tape_schema_is_optional_superset(self):
        self.assertEqual(
            len(_P2_TAPE_TELEMETRY_FIELDS),
            len(_FACTORIAL_TELEMETRY_FIELDS) + 2,
        )
        for name in _FACTORIAL_TELEMETRY_FIELDS:
            self.assertIn(name, _P2_TAPE_TELEMETRY_FIELDS)

    def test_cli_exposes_only_scoped_p2_controls(self):
        result = CliRunner().invoke(train_cli, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--p2-pulse-chase-manifest", result.output)
        self.assertIn("--p2-matched-randomness-audit", result.output)

    def test_material_equivalent_and_unresolved_rules(self):
        material = summarize("x", [0.08] * 10)
        self.assertTrue(material["material"])
        equivalent = summarize("x", [0.0] * 10)
        self.assertTrue(equivalent["equivalent_tost_alpha_0p05"])
        unresolved = summarize("x", [-0.08, 0.08] * 5)
        self.assertEqual(unresolved["classification"], "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
