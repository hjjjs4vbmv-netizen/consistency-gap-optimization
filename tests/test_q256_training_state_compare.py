"""Contracts for canonical q256 replay-artifact comparison."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import pickle
import tempfile
from collections import OrderedDict
from pathlib import Path
import unittest

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_q256_training_states.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_q256_training_states", SCRIPT_PATH
)
COMPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPARE)


def _training_state(*, reverse: bool = False) -> dict:
    optimizer_state = {
        "state": {
            0: {
                "step": torch.tensor(32.0),
                "exp_avg": torch.tensor([0.25, -0.5], dtype=torch.float32),
                "exp_avg_sq": torch.tensor([0.125, 0.75], dtype=torch.float32),
            }
        },
        "param_groups": [
            {
                "params": [0],
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ],
    }
    fields = [
        ("net", {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}),
        ("optimizer_state", optimizer_state),
        ("loss_fn_state", {"stage": 2, "ratio": 0.75}),
        ("gradscaler_state", {"scale": 65536.0, "growth_tracker": 12}),
        ("cur_nimg", 260_096),
        ("cur_tick", 29),
        ("tick_start_nimg", 260_096),
        ("batch_idx", 2032),
        ("attempted_iteration", 2032),
        ("successful_optimizer_steps", 2022),
        ("adaptive_signal_window_state", {"sum": np.array([1.0, 2.0])}),
        ("elapsed_sec", 123.5),
        ("run_dir", "/tmp/replay-left"),
        (
            "_q256_radam_moment_transport",
            {"source_sha256": "a" * 64, "coefficient": 1.0},
        ),
    ]
    if reverse:
        fields.reverse()
    return dict(fields)


def _snapshot(*, path: str = "/data/raw/ECT/cifar10.zip") -> dict:
    return {
        "ema": {"weight": torch.tensor([3.0, -1.0], dtype=torch.float32)},
        "loss_fn": {"q": 256.0, "gap": 1.1},
        "augment_pipe": None,
        "dataset_kwargs": {"path": path, "resolution": 32, "xflip": False},
    }


def _write_pair(
    root: Path,
    left_state: dict,
    right_state: dict,
    left_snapshot: dict,
    right_snapshot: dict,
):
    paths = {
        "left_state": root / "left-state.pt",
        "right_state": root / "right-state.pt",
        "left_snapshot": root / "left-snapshot.pkl",
        "right_snapshot": root / "right-snapshot.pkl",
    }
    torch.save(left_state, paths["left_state"])
    torch.save(right_state, paths["right_state"])
    with paths["left_snapshot"].open("wb") as handle:
        pickle.dump(left_snapshot, handle)
    with paths["right_snapshot"].open("wb") as handle:
        pickle.dump(right_snapshot, handle)
    return paths


class Q256TrainingStateCompareTests(unittest.TestCase):
    def test_mapping_order_absolute_paths_and_explicit_metadata_are_nonsemantic(self):
        left = _training_state(reverse=False)
        right = _training_state(reverse=True)
        right["elapsed_sec"] = 9999.0
        right["run_dir"] = "/different/absolute/run"
        del right["_q256_radam_moment_transport"]
        left["metadata"] = {"semantic": False, "host": "worker-a"}
        left_summary = COMPARE.summarize_payload(left, kind="training_state")
        right_summary = COMPARE.summarize_payload(right, kind="training_state")
        comparison = COMPARE.compare_summaries(left_summary, right_summary)
        self.assertTrue(comparison["equal"])
        self.assertEqual(
            left_summary["canonical_sha256"], right_summary["canonical_sha256"]
        )

        left_snapshot = COMPARE.summarize_payload(
            _snapshot(path="/data/a/cifar10.zip"), kind="snapshot"
        )
        right_snapshot = COMPARE.summarize_payload(
            _snapshot(path="/mnt/b/cifar10.zip"), kind="snapshot"
        )
        self.assertTrue(
            COMPARE.compare_summaries(left_snapshot, right_snapshot)["equal"]
        )
        self.assertIn("dataset_kwargs", left_snapshot["nested_exclusions"])

    def test_tensor_hash_includes_dtype_shape_and_value_bytes(self):
        base, _ = COMPARE.canonical_hash(torch.tensor([1.0, 2.0], dtype=torch.float32))
        reordered, _ = COMPARE.canonical_hash(
            torch.tensor([2.0, 1.0], dtype=torch.float32)
        )
        different_dtype, _ = COMPARE.canonical_hash(
            torch.tensor([1.0, 2.0], dtype=torch.float64)
        )
        different_shape, _ = COMPARE.canonical_hash(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        )
        self.assertEqual(len({base, reordered, different_dtype, different_shape}), 4)

    def test_mapping_order_is_normalized_but_sequence_order_is_semantic(self):
        left = OrderedDict((("b", 2), ("a", 1)))
        right = OrderedDict((("a", 1), ("b", 2)))
        self.assertEqual(
            COMPARE.canonical_hash(left)[0], COMPARE.canonical_hash(right)[0]
        )
        self.assertNotEqual(
            COMPARE.canonical_hash([1, 2])[0], COMPARE.canonical_hash([2, 1])[0]
        )

    def test_persistent_class_hash_does_not_depend_on_random_import_module_name(self):
        attributes = {
            "_orig_module_src": "class FrozenLoss:\n    pass\n",
            "_orig_class_name": "FrozenLoss",
        }
        left_type = type("FrozenLoss", (), attributes)
        right_type = type("FrozenLoss", (), attributes)
        left_type.__module__ = "_imported_module_" + "1" * 32
        right_type.__module__ = "_imported_module_" + "2" * 32
        self.assertEqual(
            COMPARE.canonical_hash(left_type())[0],
            COMPARE.canonical_hash(right_type())[0],
        )

    def test_each_trajectory_field_difference_is_reported(self):
        mutations = {
            "net": lambda state: state["net"]["weight"].add_(1),
            "optimizer_state": lambda state: state["optimizer_state"]["state"][0][
                "exp_avg"
            ].mul_(2),
            "loss_fn_state": lambda state: state["loss_fn_state"].update(stage=3),
            "gradscaler_state": lambda state: state["gradscaler_state"].update(
                scale=32768.0
            ),
            "cur_nimg": lambda state: state.update(cur_nimg=260_224),
            "cur_tick": lambda state: state.update(cur_tick=30),
            "batch_idx": lambda state: state.update(batch_idx=2033),
        }
        for field, mutation in mutations.items():
            with self.subTest(field=field):
                left = _training_state()
                right = copy.deepcopy(left)
                mutation(right)
                comparison = COMPARE.compare_summaries(
                    COMPARE.summarize_payload(left, kind="training_state"),
                    COMPARE.summarize_payload(right, kind="training_state"),
                )
                self.assertFalse(comparison["equal"])
                self.assertEqual(
                    [item["field"] for item in comparison["differences"]], [field]
                )

    def test_module_hash_covers_state_topology_and_requires_grad(self):
        torch.manual_seed(10)
        left = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(0.25))
        right = copy.deepcopy(left)
        self.assertEqual(
            COMPARE.canonical_hash(left)[0], COMPARE.canonical_hash(right)[0]
        )
        with torch.no_grad():
            right[0].weight[0, 0].add_(1)
        self.assertNotEqual(
            COMPARE.canonical_hash(left)[0], COMPARE.canonical_hash(right)[0]
        )
        right = copy.deepcopy(left)
        right[0].weight.requires_grad_(False)
        self.assertNotEqual(
            COMPARE.canonical_hash(left)[0], COMPARE.canonical_hash(right)[0]
        )

    def test_snapshot_semantic_fields_are_compared_individually(self):
        mutations = {
            "ema": lambda snapshot: snapshot["ema"]["weight"].add_(1),
            "loss_fn": lambda snapshot: snapshot["loss_fn"].update(q=128.0),
            "augment_pipe": lambda snapshot: snapshot.update(
                augment_pipe={"probability": 0.1}
            ),
            "dataset_kwargs": lambda snapshot: snapshot["dataset_kwargs"].update(
                resolution=64
            ),
        }
        for field, mutation in mutations.items():
            with self.subTest(field=field):
                left = _snapshot()
                right = copy.deepcopy(left)
                mutation(right)
                comparison = COMPARE.compare_summaries(
                    COMPARE.summarize_payload(left, kind="snapshot"),
                    COMPARE.summarize_payload(right, kind="snapshot"),
                )
                self.assertFalse(comparison["equal"])
                self.assertEqual(
                    [item["field"] for item in comparison["differences"]],
                    [field],
                )

    def test_unknown_top_level_fields_fail_closed(self):
        payload = _training_state()
        payload["new_unreviewed_field"] = 1
        with self.assertRaisesRegex(COMPARE.ComparisonError, "unknown top-level"):
            COMPARE.summarize_payload(payload, kind="training_state")
        snapshot = _snapshot()
        snapshot["mystery"] = None
        with self.assertRaisesRegex(COMPARE.ComparisonError, "unknown top-level"):
            COMPARE.summarize_payload(snapshot, kind="snapshot")
        semantic_metadata = _training_state()
        semantic_metadata["metadata"] = {"semantic": True, "value": 1}
        with self.assertRaisesRegex(COMPARE.ComparisonError, "unknown top-level"):
            COMPARE.summarize_payload(semantic_metadata, kind="training_state")

    def test_cli_equal_and_not_equal_exit_codes_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_pair(
                root,
                _training_state(),
                _training_state(reverse=True),
                _snapshot(path="/left/data.zip"),
                _snapshot(path="/right/data.zip"),
            )
            argv = [
                "--left-state",
                str(paths["left_state"]),
                "--right-state",
                str(paths["right_state"]),
                "--left-snapshot",
                str(paths["left_snapshot"]),
                "--right-snapshot",
                str(paths["right_snapshot"]),
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = COMPARE.main(argv)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "EQUAL")

            right = _training_state()
            right["cur_nimg"] += 128
            torch.save(right, paths["right_state"])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = COMPARE.main(argv)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(payload["status"], "NOT_EQUAL")
            self.assertEqual(
                payload["comparison"]["training_state"]["differences"][0]["field"],
                "cur_nimg",
            )

    def test_corrupt_or_unknown_artifact_exits_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = _training_state()
            right = _training_state()
            right["unknown"] = 1
            paths = _write_pair(root, left, right, _snapshot(), _snapshot())
            argv = [
                "--left-state",
                str(paths["left_state"]),
                "--right-state",
                str(paths["right_state"]),
                "--left-snapshot",
                str(paths["left_snapshot"]),
                "--right-snapshot",
                str(paths["right_snapshot"]),
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = COMPARE.main(argv)
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "CORRUPT")

            paths["right_state"].write_bytes(b"not a torch checkpoint")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = COMPARE.main(argv)
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "CORRUPT")


if __name__ == "__main__":
    unittest.main()
