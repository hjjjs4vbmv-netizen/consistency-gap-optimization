import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from analysis import q256_control_compatibility as compatibility


class Q256ControlCompatibilityTest(unittest.TestCase):
    def make_source(
        self, root: Path, name: str, value: float, *, serialize_rng: bool = True
    ) -> Path:
        run = root / name
        run.mkdir(parents=True)
        optimizer_state = {
            "state": {
                0: {
                    "step": torch.tensor(1990.0),
                    "exp_avg": torch.tensor([value / 10]),
                    "exp_avg_sq": torch.tensor([value / 100]),
                }
            },
            "param_groups": [{"params": [0], "lr": 1e-4}],
        }
        state = {
            "net": {"weight": torch.tensor([value])},
            "optimizer_state": optimizer_state,
            "gradscaler_state": {"scale": torch.tensor(65536.0), "growth_tracker": 12},
            "cur_nimg": 256000,
        }
        if serialize_rng:
            state["rng_state"] = {
                "python": (3, (1, 2, 3), None),
                "numpy": ("MT19937", torch.arange(4).numpy(), 4, 0, 0.0),
                "torch_cpu": torch.arange(8, dtype=torch.uint8),
                "torch_cuda": [torch.arange(8, dtype=torch.uint8) + 1],
            }
            state["sampler_state"] = {"cursor": 8192, "epoch": 0}
        torch.save(state, run / "training-state-latest.pt")
        with (run / "network-snapshot-latest.pkl").open("wb") as handle:
            pickle.dump({"ema": {"weight": torch.tensor([value + 0.5])}}, handle)
        options = {
            "loss_kwargs": {"q": 256, "adj": "sigmoid", "global_gap_scale": 1.0},
            "batch_size": 512,
            "batch_gpu": 64,
            "augment_kwargs": {"class_name": "training.augment.AugmentPipe", "p": 0.12},
            "dataset_kwargs": {"xflip": False},
            "network_kwargs": {"use_fp16": False},
            "enable_amp": True,
            "enable_tf32": False,
            "loss_scaling": 1,
            "kimg_per_tick": 8,
            "snapshot_ticks": 32,
            "state_dump_ticks": 32,
            "ckpt_ticks": 4,
            "sample_ticks": 8,
            "eval_ticks": 32,
        }
        (run / "training_options.json").write_text(
            json.dumps(options, sort_keys=True), encoding="utf-8"
        )
        return run

    @staticmethod
    def run_spec(schedule: str, gap_scale: float) -> dict:
        return {
            "protocol": {
                "q": 256,
                "schedule": schedule,
                "gap_scale": gap_scale,
                "batch": {"batch_size": 512, "batch_gpu": 64},
                "augmentation": {
                    "augment_kwargs": {
                        "class_name": "training.augment.AugmentPipe",
                        "p": 0.12,
                    },
                    "dataset_xflip": False,
                },
                "precision": {
                    "use_fp16": False,
                    "enable_amp": True,
                    "enable_tf32": False,
                    "loss_scaling": 1,
                },
                "checkpoint_cadence": {
                    "kimg_per_tick": 8,
                    "snapshot_ticks": 32,
                    "state_dump_ticks": 32,
                    "ckpt_ticks": 4,
                    "sample_ticks": 8,
                    "eval_ticks": 32,
                },
            },
            "data": {
                "byte_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            },
            "execution_commit": "0123456789abcdef",
            "execution_core_sha256": "c" * 64,
            "start_kimg": 256,
            "endpoints_kimg": [512, 768, 1024],
            "config_sha256": "d" * 64,
        }

    def make_manifest(
        self, root: Path, *, legacy_g: bool = False, serialize_rng: bool = True
    ) -> dict:
        manifest = {
            "schema_version": 1,
            "reference": {},
            "controls": {"F": {}, "G": {}},
        }
        for seed in compatibility.SEEDS:
            common = self.make_source(
                root, f"seed{seed}-common", float(seed), serialize_rng=serialize_rng
            )
            g_source = common
            if legacy_g:
                g_source = self.make_source(
                    root,
                    f"seed{seed}-legacy-g",
                    float(seed) + 10,
                    serialize_rng=serialize_rng,
                )
            manifest["reference"][seed] = {
                "source": {"run_dir": str(common)},
                "run": self.run_spec("global_sigmoid", 1.10),
            }
            manifest["controls"]["F"][seed] = {
                "source": {"run_dir": str(common)},
                "run": self.run_spec("sigmoid", 1.0),
            }
            manifest["controls"]["G"][seed] = {
                "source": {"run_dir": str(g_source)},
                "run": self.run_spec("global_sigmoid", 1.10),
            }
        return manifest

    def test_compatible_controls_are_independently_reusable(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = compatibility.build_report(self.make_manifest(root), root)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["reusable_controls"])
        for seed in compatibility.SEEDS:
            self.assertEqual(
                report["seeds"][seed]["controls"]["F"]["reuse_decision"], "reusable"
            )
            self.assertEqual(
                report["seeds"][seed]["controls"]["G"]["reuse_decision"], "reusable"
            )
            self.assertTrue(report["seeds"][seed]["legacy_F_G_same_256k_source"])

    def test_legacy_g_source_requires_fresh_g_without_disqualifying_f(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self.make_manifest(root, legacy_g=True)
            manifest_path = root / "manifest.json"
            report_path = root / "compatibility.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            exit_code = compatibility.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(report_path),
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 2)
        self.assertEqual(report["status"], "FAIL")
        for seed in compatibility.SEEDS:
            self.assertEqual(
                report["seeds"][seed]["controls"]["F"]["reuse_decision"], "reusable"
            )
            self.assertEqual(
                report["seeds"][seed]["controls"]["G"]["reuse_decision"],
                "fresh_required",
            )
            self.assertEqual(
                report["seeds"][seed]["controls"]["G"]["action"],
                "launch_fresh_paired_G_control",
            )
        legacy_rows = [
            row for row in report["rows"] if row["field"] == "legacy_shared_256k_source"
        ]
        self.assertEqual(len(legacy_rows), 3)
        self.assertTrue(
            all(
                row["reason"] == "legacy_F_G_controls_use_different_256k_source"
                for row in legacy_rows
            )
        )
        self.assertTrue(
            any(row["field"] == "model_sha256" for row in report["blockers"])
        )

    def test_missing_rng_and_sampler_are_explicit_fail_closed_fields(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = compatibility.build_report(
                self.make_manifest(root, serialize_rng=False), root
            )
        self.assertEqual(report["status"], "FAIL")
        missing = {
            (item["arm"], item["field"])
            for item in report["missing_nonserialized_fields"]
        }
        self.assertIn(("T", "rng.python.sha256"), missing)
        self.assertIn(("F", "rng.torch_cuda.sha256"), missing)
        self.assertIn(("G", "sampler.sha256"), missing)
        for seed in compatibility.SEEDS:
            self.assertEqual(
                report["seeds"][seed]["controls"]["F"]["reuse_decision"],
                "fresh_required",
            )
            self.assertEqual(
                report["seeds"][seed]["controls"]["G"]["reuse_decision"],
                "fresh_required",
            )
        self.assertTrue(
            any(
                row["status"] == "missing"
                and row["reason"] == "required_field_missing_or_not_serialized"
                for row in report["rows"]
            )
        )

    def test_malformed_manifest_writes_error_report_and_exits_three(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "bad.json"
            report_path = root / "report.json"
            manifest_path.write_text("{}", encoding="utf-8")
            exit_code = compatibility.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--report",
                    str(report_path),
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 3)
        self.assertEqual(report["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
