from __future__ import annotations

import copy
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

from scripts import transport_radam_moments as transport


def assert_tensor_equal(
    testcase: unittest.TestCase, left: torch.Tensor, right: torch.Tensor
) -> None:
    testcase.assertEqual(left.dtype, right.dtype)
    testcase.assertEqual(left.device, right.device)
    testcase.assertEqual(left.shape, right.shape)
    testcase.assertEqual(left.layout, right.layout)
    testcase.assertEqual(left.requires_grad, right.requires_grad)
    testcase.assertEqual(transport.tensor_sha256(left), transport.tensor_sha256(right))


def assert_canonical_equal(testcase: unittest.TestCase, left: Any, right: Any) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        testcase.assertTrue(torch.is_tensor(left) and torch.is_tensor(right))
        assert_tensor_equal(testcase, left, right)
        return
    testcase.assertEqual(type(left), type(right))
    if isinstance(left, dict):
        testcase.assertEqual(list(left.keys()), list(right.keys()))
        for key in left:
            assert_canonical_equal(testcase, left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            assert_canonical_equal(testcase, left_item, right_item)
        return
    testcase.assertEqual(left, right)


def assert_numpy_rng_equal(testcase: unittest.TestCase, left: Any, right: Any) -> None:
    testcase.assertEqual(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    testcase.assertEqual(left[2:], right[2:])


class Q256MomentTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_optimizer_state(self, *, amsgrad: bool = True) -> dict:
        first_0 = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
        second_0 = torch.tensor([4.0, 9.0, 0.25], dtype=torch.float32)
        first_1 = torch.arange(1.0, 7.0, dtype=torch.float64).reshape(2, 3).t()
        second_1 = torch.arange(2.0, 8.0, dtype=torch.float64).reshape(2, 3).t()
        state = {
            0: {
                "step": torch.tensor(17.0),
                "exp_avg": first_0,
                "exp_avg_sq": second_0,
                "radam_buffer": {"rho": torch.tensor(4.5), "step_size": 0.01},
                "scalar_cache": 3.0,
            },
            1: {
                "step": 17,
                "exp_avg": first_1,
                "exp_avg_sq": second_1,
                "rectification_buffer": [torch.tensor(0.75), None],
            },
        }
        if amsgrad:
            state[0]["max_exp_avg_sq"] = second_0 + 1.0
            state[1]["max_exp_avg_sq"] = second_1 + 1.0
        return {
            "state": state,
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": torch.tensor(0.001),
                    "betas": (0.9, 0.999),
                    "eps": 1.0e-8,
                    "weight_decay": 0.0,
                }
            ],
        }

    def make_checkpoint(self, *, amsgrad: bool = True) -> dict:
        return {
            "net_state": {
                "weight": torch.arange(6.0, dtype=torch.float32).reshape(2, 3),
                "counter": 9,
            },
            "ema_state": {"weight": torch.tensor([2.0, 3.0])},
            "optimizer_state": self.make_optimizer_state(amsgrad=amsgrad),
            "gradscaler_state": {"scale": 65536.0, "growth_tracker": 22},
            "cpu_rng_state": torch.arange(16, dtype=torch.uint8),
            "sampler_state": {"epoch": 31, "offset": 7},
            "cur_nimg": 256_000,
            "cur_tick": 257,
        }

    def write_source(self, *, checkpoint: Any = None, name: str = "source.pt") -> Path:
        source = self.root / name
        torch.save(self.make_checkpoint() if checkpoint is None else checkpoint, source)
        return source

    def load(self, path: Path) -> Any:
        return torch.load(path, weights_only=False)

    def test_a_one_is_an_exact_noop_for_supported_moments(self) -> None:
        optimizer = self.make_optimizer_state()
        before = {
            (param_id, field): transport.tensor_sha256(value)
            for param_id, state in optimizer["state"].items()
            for field, value in state.items()
            if field in transport.MOMENT_FACTORS
        }

        report = transport.transform_optimizer_state(optimizer, 1.0)

        self.assertEqual(report["transformed_state_count"], 2)
        for param_id, state in optimizer["state"].items():
            for field, value in state.items():
                if field in transport.MOMENT_FACTORS:
                    self.assertEqual(
                        transport.tensor_sha256(value), before[(param_id, field)]
                    )
        self.assertTrue(
            all(item["norm_ratio_verified"] for item in report["moment_tensors"])
        )

    def test_inverse_transport_reconstructs_supported_moments(self) -> None:
        optimizer = self.make_optimizer_state()
        original = copy.deepcopy(optimizer)

        transport.transform_optimizer_state(optimizer, 1.25)
        transport.transform_optimizer_state(optimizer, 1.0 / 1.25)

        for param_id, state in optimizer["state"].items():
            for field in transport.MOMENT_FACTORS:
                if field in state:
                    torch.testing.assert_close(
                        state[field],
                        original["state"][param_id][field],
                        rtol=1e-6,
                        atol=1e-8,
                    )

    def test_first_moments_scale_by_a(self) -> None:
        optimizer = self.make_optimizer_state()
        original = copy.deepcopy(optimizer)

        report = transport.transform_optimizer_state(optimizer, 1.5)

        for param_id in (0, 1):
            torch.testing.assert_close(
                optimizer["state"][param_id]["exp_avg"],
                original["state"][param_id]["exp_avg"] * 1.5,
            )
        first_records = [
            item for item in report["moment_tensors"] if item["field"] == "exp_avg"
        ]
        self.assertEqual(len(first_records), 2)
        self.assertTrue(all(item["expected_factor"] == 1.5 for item in first_records))

    def test_second_moments_scale_by_a_squared(self) -> None:
        optimizer = self.make_optimizer_state()
        original = copy.deepcopy(optimizer)

        report = transport.transform_optimizer_state(optimizer, 1.5)

        for param_id in (0, 1):
            torch.testing.assert_close(
                optimizer["state"][param_id]["exp_avg_sq"],
                original["state"][param_id]["exp_avg_sq"] * 2.25,
            )
        records = [
            item for item in report["moment_tensors"] if item["field"] == "exp_avg_sq"
        ]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(item["expected_factor"] == 2.25 for item in records))

    def test_amsgrad_max_second_moment_scales_by_a_squared(self) -> None:
        optimizer = self.make_optimizer_state(amsgrad=True)
        original = copy.deepcopy(optimizer)

        transport.transform_optimizer_state(optimizer, 0.8)

        for param_id in (0, 1):
            torch.testing.assert_close(
                optimizer["state"][param_id]["max_exp_avg_sq"],
                original["state"][param_id]["max_exp_avg_sq"] * 0.64,
            )

    def test_non_moment_fields_remain_canonically_identical(self) -> None:
        checkpoint = self.make_checkpoint()
        source = self.write_source(checkpoint=checkpoint)
        output = self.root / "transported.pt"
        source_hash = transport.sha256_file(source)

        transport.transform_checkpoint(source, output, 1.1)
        transformed = self.load(output)

        self.assertEqual(transport.sha256_file(source), source_hash)
        for key in checkpoint:
            if key != "optimizer_state":
                assert_canonical_equal(self, transformed[key], checkpoint[key])
        before_optimizer = checkpoint["optimizer_state"]
        after_optimizer = transformed["optimizer_state"]
        assert_canonical_equal(
            self, after_optimizer["param_groups"], before_optimizer["param_groups"]
        )
        self.assertEqual(
            list(after_optimizer["state"]), list(before_optimizer["state"])
        )
        for param_id in before_optimizer["state"]:
            for field, before_value in before_optimizer["state"][param_id].items():
                if field not in transport.MOMENT_FACTORS:
                    assert_canonical_equal(
                        self, after_optimizer["state"][param_id][field], before_value
                    )

    def test_tensor_dtype_device_shape_sparsity_and_association_are_preserved(
        self,
    ) -> None:
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
        sparse_first = torch.sparse_coo_tensor(
            indices, torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32), (2, 3)
        )
        sparse_second = torch.sparse_coo_tensor(
            indices, torch.tensor([2.0, 5.0, 10.0], dtype=torch.float32), (2, 3)
        )
        optimizer = {
            "state": {
                12: {
                    "step": torch.tensor(3.0),
                    "exp_avg": sparse_first,
                    "exp_avg_sq": sparse_second,
                }
            },
            "param_groups": [{"params": [12], "lr": 0.01}],
        }
        metadata = {
            field: {
                "dtype": tensor.dtype,
                "device": tensor.device,
                "shape": tensor.shape,
                "layout": tensor.layout,
            }
            for field, tensor in optimizer["state"][12].items()
            if field in transport.MOMENT_FACTORS
        }

        report = transport.transform_optimizer_state(optimizer, 1.2)

        self.assertEqual(optimizer["param_groups"][0]["params"], [12])
        self.assertEqual(list(optimizer["state"]), [12])
        self.assertEqual(report["missing_state_parameter_ids"], [])
        for field, expected in metadata.items():
            actual = optimizer["state"][12][field]
            self.assertEqual(actual.dtype, expected["dtype"])
            self.assertEqual(actual.device, expected["device"])
            self.assertEqual(actual.shape, expected["shape"])
            self.assertEqual(actual.layout, expected["layout"])

    def test_missing_optimizer_states_are_rejected_or_explicitly_reported(self) -> None:
        optimizer = self.make_optimizer_state()
        del optimizer["state"][1]

        with self.assertRaisesRegex(
            transport.TransportError, "missing.*parameter ids: 1"
        ):
            transport.transform_optimizer_state(copy.deepcopy(optimizer), 1.1)

        report = transport.transform_optimizer_state(
            optimizer, 1.1, allow_missing_state=True
        )
        self.assertEqual(report["missing_state_parameter_ids"], ["1"])
        self.assertEqual(report["transformed_state_count"], 1)

    def test_unknown_tensor_valued_optimizer_field_fails_closed(self) -> None:
        optimizer = self.make_optimizer_state()
        optimizer["state"][0]["mystery"] = {"nested": [torch.tensor(4.0)]}

        with self.assertRaisesRegex(
            transport.TransportError, "unsupported tensor-valued.*mystery"
        ):
            transport.transform_optimizer_state(optimizer, 1.1)

    def test_source_and_existing_outputs_cannot_be_overwritten(self) -> None:
        source = self.write_source()
        source_hash = transport.sha256_file(source)

        with self.assertRaisesRegex(transport.TransportError, "paths must differ"):
            transport.transform_checkpoint(source, source, 1.1)
        self.assertEqual(transport.sha256_file(source), source_hash)

        output = self.root / "existing.pt"
        output.write_bytes(b"sentinel-output")
        with self.assertRaisesRegex(transport.TransportError, "existing output"):
            transport.transform_checkpoint(source, output, 1.1)
        self.assertEqual(output.read_bytes(), b"sentinel-output")
        self.assertEqual(transport.sha256_file(source), source_hash)

    def test_repeated_checkpoint_transform_is_blocked(self) -> None:
        source = self.write_source()
        first = self.root / "first.pt"
        second = self.root / "second.pt"

        transport.transform_checkpoint(source, first, 1.1)

        with self.assertRaisesRegex(
            transport.TransportError,
            "(matching moment-transport sidecar|already contains.*marker)",
        ):
            transport.transform_checkpoint(first, second, 1.1)
        self.assertFalse(second.exists())

    def test_save_load_preserves_transport_and_provenance(self) -> None:
        source = self.write_source()
        output = self.root / "transported.pt"
        expected_source_hash = transport.sha256_file(source)

        manifest = transport.transform_checkpoint(
            source,
            output,
            1.1,
            expected_source_sha256=expected_source_hash,
            command_line=[
                "python",
                "scripts/transport_radam_moments.py",
                "--scale",
                "1.1",
            ],
        )
        sidecar_path = output.with_name(output.name + ".manifest.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        loaded = self.load(output)

        self.assertEqual(manifest, sidecar)
        self.assertEqual(sidecar["source"]["sha256"], expected_source_hash)
        self.assertTrue(sidecar["source"]["unchanged"])
        self.assertEqual(sidecar["output"]["sha256"], transport.sha256_file(output))
        self.assertEqual(sidecar["marker"], loaded[transport.MARKER_KEY])
        self.assertEqual(sidecar["optimizer"]["transformed_state_count"], 2)
        self.assertTrue(
            all(
                item["norm_ratio_verified"]
                for item in sidecar["optimizer"]["moment_tensors"]
            )
        )

    def test_publication_failure_rolls_back_new_output(self) -> None:
        source = self.write_source()
        output = self.root / "rolled-back.pt"
        source_hash = transport.sha256_file(source)
        original_link = transport._link_noreplace
        calls = 0

        def fail_second_link(temp_path: Path, target: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise transport.TransportError("injected sidecar publication failure")
            original_link(temp_path, target)

        with mock.patch.object(
            transport, "_link_noreplace", side_effect=fail_second_link
        ):
            with self.assertRaisesRegex(transport.TransportError, "injected sidecar"):
                transport.transform_checkpoint(source, output, 1.1)

        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".manifest.json").exists())
        self.assertEqual(transport.sha256_file(source), source_hash)
        self.assertEqual(list(self.root.glob(".*-tmp-*")), [])

    def test_transform_consumes_no_python_numpy_or_torch_rng(self) -> None:
        source = self.write_source()
        output = self.root / "transported.pt"
        random.seed(20260819)
        np.random.seed(20260819)
        torch.manual_seed(20260819)
        python_before = copy.deepcopy(random.getstate())
        numpy_before = copy.deepcopy(np.random.get_state())
        torch_before = torch.random.get_rng_state().clone()

        manifest = transport.transform_checkpoint(source, output, 1.1)

        self.assertEqual(random.getstate(), python_before)
        assert_numpy_rng_equal(self, np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        self.assertTrue(all(manifest["preservation"]["rng_unchanged"].values()))

    def test_normal_optimizer_resume_does_not_reapply_transport(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.RAdam([parameter], lr=0.01)
        parameter.grad = torch.tensor([0.5, -0.25])
        optimizer.step()
        original_state = copy.deepcopy(optimizer.state_dict())
        source = self.write_source(
            checkpoint={"optimizer_state": original_state, "cur_nimg": 256_000},
            name="radam-source.pt",
        )
        output = self.root / "radam-transported.pt"
        transport.transform_checkpoint(source, output, 1.5)
        loaded = self.load(output)

        resumed_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        resumed_optimizer = torch.optim.RAdam([resumed_parameter], lr=0.01)
        resumed_optimizer.load_state_dict(loaded["optimizer_state"])
        resumed_state = resumed_optimizer.state_dict()
        torch.testing.assert_close(
            resumed_state["state"][0]["exp_avg"],
            original_state["state"][0]["exp_avg"] * 1.5,
        )
        torch.testing.assert_close(
            resumed_state["state"][0]["exp_avg_sq"],
            original_state["state"][0]["exp_avg_sq"] * 2.25,
        )

        roundtrip = self.root / "ordinary-resume-save.pt"
        torch.save({"optimizer_state": resumed_state}, roundtrip)
        roundtrip_state = self.load(roundtrip)["optimizer_state"]
        assert_tensor_equal(
            self,
            roundtrip_state["state"][0]["exp_avg"],
            resumed_state["state"][0]["exp_avg"],
        )

    def test_cli_creates_checkpoint_and_hash_sidecar(self) -> None:
        source = self.write_source()
        output = self.root / "cli-output.pt"
        script = Path(transport.__file__).resolve()
        arguments = [
            "--source",
            str(source),
            "--output",
            str(output),
            "--coefficient",
            "1.2",
            "--expected-source-sha256",
            transport.sha256_file(source),
        ]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            return_code = transport.main(arguments)

        self.assertEqual(return_code, 0)
        stdout_manifest = json.loads(stdout.getvalue())
        self.assertTrue(output.is_file())
        self.assertEqual(
            stdout_manifest["output"]["sha256"], transport.sha256_file(output)
        )
        self.assertEqual(stdout_manifest["tool"]["command_line"][1], str(script))


if __name__ == "__main__":
    unittest.main()
