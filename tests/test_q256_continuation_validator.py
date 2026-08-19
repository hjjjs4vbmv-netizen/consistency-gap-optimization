"""Contracts for the fail-closed q256 continuation validator."""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "validate_q256_continuation.py"
SPEC = importlib.util.spec_from_file_location(
    "q256_continuation_validator", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Q256ContinuationValidatorTests(unittest.TestCase):
    expected_nimg = 260096
    final_nimg = 260096
    final_attempted = 2032
    seed = 3

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.dataset = self.root / "canonical.zip"
        self.dataset.write_bytes(b"synthetic-canonical-cifar10\n")
        self.identity_path = self.root / "data_identity.json"
        self.identity_path.write_text(
            json.dumps(
                {
                    "schema": "ect.q256.dataset-identity/v1",
                    "canonical_training_archive": {
                        "path": str(self.dataset),
                        "sha256": _sha256(self.dataset),
                        "size_bytes": self.dataset.stat().st_size,
                    },
                }
            ),
            encoding="utf-8",
        )
        self.state_path = self.run_dir / "training-state-latest.pt"
        self.snapshot_path = self.run_dir / "network-snapshot-latest.pkl"
        self.receipt_path = self.root / "receipt.json"
        self._write_valid_fixture()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @property
    def dataset_kwargs(self) -> dict[str, object]:
        return {
            "class_name": "training.dataset.ImageFolderDataset",
            "path": str(self.dataset),
            "use_labels": False,
            "xflip": False,
            "cache": True,
            "resolution": 32,
            "max_size": 50000,
        }

    def _options(self, schedule: str = "global_sigmoid") -> dict[str, object]:
        gap_scale = 1.1 if schedule == "global_sigmoid" else 1.0
        source_state = self.root / "source" / "training-state-latest.pt"
        source_snapshot = self.root / "source" / "network-snapshot-latest.pkl"
        return {
            "dataset_kwargs": self.dataset_kwargs,
            "data_loader_kwargs": {
                "pin_memory": True,
                "num_workers": 1,
                "prefetch_factor": 2,
            },
            "network_kwargs": {
                "class_name": "training.networks.ECMPrecond",
                "model_type": "SongUNet",
                "embedding_type": "positional",
                "encoder_type": "standard",
                "decoder_type": "standard",
                "channel_mult_noise": 1,
                "resample_filter": [1, 1],
                "model_channels": 128,
                "channel_mult": [2, 2, 2],
                "dropout": 0.2,
                "use_fp16": True,
            },
            "loss_kwargs": {
                "class_name": "training.loss.ECMLoss",
                "P_mean": -1.1,
                "P_std": 2.0,
                "q": 256.0,
                "c": 0.0,
                "k": 8.0,
                "b": 1.0,
                "adj": schedule,
                "adaptive_loss_ema_beta": 0.9,
                "adaptive_warmup_updates": 2,
                "adaptive_max_adjust": 0.05,
                "adaptive_min_gap": 0.001,
                "local_tbin_num_bins": 4,
                "local_tbin_short_beta": 0.9,
                "local_tbin_long_beta": 0.99,
                "local_tbin_warmup_updates": 32,
                "local_tbin_gain": 0.5,
                "local_tbin_min_scale": 0.75,
                "local_tbin_max_scale": 1.5,
                "local_tbin_deadband": 0.02,
                "local_tbin_min_gap": 0.001,
                "global_gap_scale": gap_scale,
            },
            "optimizer_kwargs": {
                "class_name": "torch.optim.RAdam",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "total_kimg": 260,
            "ema_halflife_kimg": None,
            "ema_beta": 0.9993,
            "batch_size": 128,
            "batch_gpu": 16,
            "loss_scaling": 1.0,
            "cudnn_benchmark": True,
            "enable_tf32": False,
            "enable_amp": True,
            "kimg_per_tick": 10.0,
            "snapshot_ticks": None,
            "state_dump_ticks": None,
            "ckpt_ticks": 10,
            "double_ticks": 10000,
            "adaptive_update_kimg": 0.5,
            "mid_t": [0.821],
            "metrics": [],
            "sample_ticks": 26,
            "eval_ticks": 50,
            "seed": self.seed,
            "resume_pkl": str(source_snapshot),
            "resume_tick": 0,
            "resume_state_dump": str(source_state),
            "run_dir": str(self.run_dir.resolve()),
        }

    @staticmethod
    def _network() -> torch.nn.Module:
        net = torch.nn.Linear(2, 2)
        net.use_fp16 = True
        net.img_resolution = 32
        net.label_dim = 0
        return net

    def _state(self, net: torch.nn.Module) -> dict[str, object]:
        optimizer = torch.optim.RAdam(
            net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8
        )
        for parameter in net.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer_state = optimizer.state_dict()
        for item in optimizer_state["state"].values():
            item["step"].fill_(self.final_attempted)
        ratio = 1.0 - 1.0 / 256.0
        return {
            "net": net,
            "optimizer_state": optimizer_state,
            "attempted_iteration": self.final_attempted,
            "successful_optimizer_steps": self.final_attempted,
            "cur_nimg": self.final_nimg,
            "cur_tick": 2,
            "tick_start_nimg": self.final_nimg,
            "elapsed_sec": 12.5,
            "loss_fn_state": {
                "schedule_name": "global_sigmoid",
                "stage": 0,
                "ratio": ratio,
                "schedule": {},
            },
            "gradscaler_state": {
                "scale": 65536.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 32,
            },
        }

    def _snapshot(self, net: torch.nn.Module) -> dict[str, object]:
        ratio = 1.0 - 1.0 / 256.0
        schedule = types.SimpleNamespace(
            name="global_sigmoid",
            q=256.0,
            k=8.0,
            b=1.0,
            stage=0,
            global_gap_scale=1.1,
        )
        loss_fn = types.SimpleNamespace(
            q=256.0,
            k=8.0,
            b=1.0,
            c=0.0,
            stage=0,
            ratio=ratio,
            schedule=schedule,
        )
        return {
            "ema": copy.deepcopy(net).eval(),
            "loss_fn": loss_fn,
            "augment_pipe": None,
            "dataset_kwargs": self.dataset_kwargs,
        }

    def _summary_rows(self) -> list[dict[str, object]]:
        rows = []
        for index, attempted in enumerate((2031, 2032)):
            processed = attempted * 128
            rows.append(
                {
                    "attempted_iteration": attempted,
                    "successful_optimizer_steps": attempted,
                    "processed_nimg": processed,
                    "processed_kimg": f"{processed / 1000:.6f}",
                    "loss": f"{1.25 - index * 0.01:.8f}",
                    "grad_scale": "65536",
                    "step_skipped": 0,
                    "schedule": "global_sigmoid",
                    "stage": 0,
                    "next_loop_cur_tick": 1 + index,
                    "loss_ema": "",
                    "loss_reference": "",
                    "correction": "0.1",
                    "signal_updates": 0,
                    "adaptive_active": 0,
                    "r_over_t_mean": "0.99",
                    "gap_mean": "0.01",
                    "gap_over_sigmoid_gap_mean": "1.1",
                    "lower_gap_clip_rate": "0",
                    "upper_gap_clip_rate": "0",
                    "elapsed_sec": f"{11 + index:.6f}",
                    "peak_vram_gb": "1.0",
                }
            )
        return rows

    def _write_valid_fixture(self) -> None:
        net = self._network()
        torch.save(self._state(net), self.state_path)
        with self.snapshot_path.open("wb") as handle:
            pickle.dump(self._snapshot(net), handle)
        (self.run_dir / "training_options.json").write_text(
            json.dumps(self._options()), encoding="utf-8"
        )
        with (self.run_dir / "train_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=VALIDATOR.TRAIN_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(self._summary_rows())
        (self.run_dir / "log.txt").write_text(
            "Loading training state from /trusted/source.pt\nExiting...\n",
            encoding="utf-8",
        )

    def _arguments(self) -> list[str]:
        return [
            "--run-dir",
            str(self.run_dir),
            "--state",
            str(self.state_path),
            "--snapshot",
            str(self.snapshot_path),
            "--expected-nimg",
            str(self.expected_nimg),
            "--expected-seed",
            str(self.seed),
            "--expected-arm",
            "G",
            "--result-receipt",
            str(self.receipt_path),
        ]

    def _run(self) -> tuple[int, dict[str, object]]:
        output = io.StringIO()
        with (
            mock.patch.multiple(
                VALIDATOR,
                CANONICAL_DATA_PATH=str(self.dataset),
                CANONICAL_DATA_SHA256=_sha256(self.dataset),
                CANONICAL_DATA_SIZE_BYTES=self.dataset.stat().st_size,
                DATA_IDENTITY_DECLARATION=self.identity_path,
            ),
            contextlib.redirect_stdout(output),
        ):
            code = VALIDATOR.main(self._arguments())
        return code, json.loads(output.getvalue())

    def test_synthetic_run_is_go(self) -> None:
        code, output = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(output["verdict"], "GO")
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["verdict"], "GO")
        self.assertEqual(receipt["observed"]["state"]["cur_nimg"], self.final_nimg)
        self.assertTrue(receipt["observed"]["net_ema"]["strict_keys_equal"])

    def test_legacy_radam_group_may_omit_safe_runtime_flags(self) -> None:
        state = torch.load(self.state_path, map_location="cpu", weights_only=False)
        for group in state["optimizer_state"]["param_groups"]:
            group.pop("maximize", None)
            group.pop("capturable", None)
        torch.save(state, self.state_path)
        code, output = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(output["verdict"], "GO")
        self.assertEqual(
            output["observed"]["state"]["optimizer"]["parameter_states"], 2
        )

    def test_nonfinite_tensor_is_no_go(self) -> None:
        state = torch.load(self.state_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            state["net"].weight[0, 0] = float("nan")
        torch.save(state, self.state_path)
        code, output = self._run()
        self.assertEqual(code, 2)
        self.assertEqual(output["verdict"], "NO_GO")
        self.assertIn("non-finite tensor", output["error"])

    def test_wrong_schedule_is_no_go(self) -> None:
        options_path = self.run_dir / "training_options.json"
        options = json.loads(options_path.read_text(encoding="utf-8"))
        options["loss_kwargs"]["adj"] = "sigmoid"
        options_path.write_text(json.dumps(options), encoding="utf-8")
        code, output = self._run()
        self.assertEqual(code, 2)
        self.assertEqual(output["verdict"], "NO_GO")
        self.assertIn("schedule", output["error"])

    def test_strict_net_ema_key_mismatch_is_no_go(self) -> None:
        with self.snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        ema = torch.nn.Sequential(copy.deepcopy(snapshot["ema"]))
        ema.use_fp16 = True
        ema.img_resolution = 32
        ema.label_dim = 0
        snapshot["ema"] = ema
        with self.snapshot_path.open("wb") as handle:
            pickle.dump(snapshot, handle)
        code, output = self._run()
        self.assertEqual(code, 2)
        self.assertEqual(output["verdict"], "NO_GO")
        self.assertIn("strict net/EMA", output["error"])

    def test_existing_receipt_is_never_overwritten(self) -> None:
        sentinel = b"immutable-existing-receipt\n"
        self.receipt_path.write_bytes(sentinel)
        code, output = self._run()
        self.assertEqual(code, 2)
        self.assertEqual(output["verdict"], "NO_GO")
        self.assertEqual(self.receipt_path.read_bytes(), sentinel)


if __name__ == "__main__":
    unittest.main()
