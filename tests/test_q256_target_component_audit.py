import contextlib
import copy
import io
import json
import math
import pickle
from types import SimpleNamespace

import pytest
import torch

from analysis.gap_gradient_hook import module_state_hashes, sha256_file
from analysis.q256_target_component_audit import (
    CANONICAL_CIFAR10_SHA256,
    _scaled_residual,
    _loss_identity_error,
    factorial_loss_with_fixed_randomness,
    layerwise_target_geometry,
    load_common_state,
    measurement_labels,
    parse_args,
    publish_measurement_artifacts,
    realized_scale_summary,
    run_cell,
    run_probe,
    summarize_layer_geometry,
    target_geometry,
    validate_asset_receipt,
)
from training.loss import (
    ECMLoss,
    TARGET_WEIGHT_FACTORIAL_PROTOCOL,
    compute_target_weight_times,
)
from training import reproducibility


class TinyECTNet(torch.nn.Module):
    img_resolution = 2

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.7))
        self.time_gain = torch.nn.Parameter(torch.tensor(-0.2))

    def forward(
        self,
        x,
        sigma,
        labels=None,
        augment_labels=None,
        force_fp32=False,
    ):
        del labels, augment_labels, force_fp32
        return self.gain * x + self.time_gain * sigma.square()


class ZeroResidualNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x,
        sigma,
        labels=None,
        augment_labels=None,
        force_fp32=False,
    ):
        del sigma, labels, augment_labels, force_fp32
        return torch.zeros_like(x) + 0.0 * self.dummy


def make_loss(arm="A"):
    factors = {
        "A": (1.0, 1.0),
        "B": (1.1, 1.1),
        "C": (1.1, 1.0),
        "D": (1.0, 1.1),
    }
    target, denominator = factors[arm]
    with contextlib.redirect_stdout(io.StringIO()):
        return ECMLoss(
            q=256,
            c=0.0,
            adj="sigmoid",
            factorial_protocol=TARGET_WEIGHT_FACTORIAL_PROTOCOL,
            target_gap_scale=target,
            denominator_gap_scale=denominator,
        )


def minimal_cli(tmp_path, *, run_kind="primary"):
    args = [
        "--run-kind", run_kind,
        "--training-state", str(tmp_path / "state.pt"),
        "--checkpoint", str(tmp_path / "snapshot.pkl"),
        "--checkpoint-receipt", str(tmp_path / "receipt.json"),
        "--data", str(tmp_path / "data.zip"),
        "--expected-data-sha256", CANONICAL_CIFAR10_SHA256,
        "--state-arm", "A",
        "--state-kimg", "512",
        "--training-seed", "3",
        "--out", str(tmp_path / "out"),
    ]
    if run_kind == "smoke":
        args.extend(["--device", "cpu"])
    return args


def test_primary_and_smoke_are_machine_distinct_and_primary_is_frozen(tmp_path):
    primary = parse_args(minimal_cli(tmp_path))
    assert primary.run_kind == "primary"
    assert measurement_labels("primary", True) == (
        "ect.q256.target-component-audit-primary/v2",
        "PASS_PRIMARY_COMMON_STATE_GRADIENT_AUDIT",
    )
    assert measurement_labels("smoke", True) == (
        "ect.q256.target-component-audit-smoke/v1",
        "PASS_SMOKE_NOT_PRIMARY",
    )

    smoke = parse_args(minimal_cli(tmp_path, run_kind="smoke") + ["--batches", "1"])
    assert smoke.batches == 1
    with pytest.raises(SystemExit):
        parse_args(minimal_cli(tmp_path) + ["--batches", "1"])
    with pytest.raises(SystemExit):
        parse_args(minimal_cli(tmp_path) + ["--device", "cpu"])
    cpu_preflight = parse_args(
        minimal_cli(tmp_path)
        + ["--device", "cpu", "--preflight-only"]
    )
    assert cpu_preflight.preflight_only is True
    with pytest.raises(SystemExit):
        parse_args(
            minimal_cli(tmp_path, run_kind="smoke")
            + ["--identity-relative-tolerance", "0.01"]
        )


def test_measurement_artifacts_are_atomic_and_hash_bound(tmp_path):
    out = tmp_path / "published"
    manifest = {"schema": "test", "status": "PASS_SMOKE_NOT_PRIMARY"}
    hashes = publish_measurement_artifacts(
        out,
        [{"layer": "x", "r_tar": 0.5}],
        [{"batch_index": 0, "r_tar": 0.5}],
        manifest,
    )
    loaded = json.loads((out / "target_component_manifest.json").read_text())
    assert loaded["artifact_sha256"] == hashes
    for filename, digest in hashes.items():
        assert sha256_file(out / filename) == digest
    assert not list(tmp_path.glob(".published.staging-*"))
    with pytest.raises(FileExistsError):
        publish_measurement_artifacts(out, [{"layer": "x"}], [{"batch": 0}], {})


def test_direct_residual_and_target_geometry_are_numerically_exact():
    scale = 0.5
    gradient_a = {"weight": torch.tensor([2.0, 0.0])}
    gradient_b = {"weight": torch.tensor([1.0, 1.0])}

    geometry = target_geometry(gradient_a, gradient_b, scale)
    assert geometry["tau_tar_l2"] == pytest.approx(1.0)
    assert geometry["r_tar"] == pytest.approx(0.5)
    assert geometry["cos_tau_g_a"] == pytest.approx(0.0)
    assert geometry["a_star"] == pytest.approx(scale)

    exact = _scaled_residual(
        {"weight": scale * gradient_a["weight"]}, gradient_a, scale
    )
    assert exact["absolute_l2"] == 0.0
    assert exact["relative_l2"] == 0.0

    with pytest.raises(FloatingPointError, match="non-finite per-sample loss"):
        _loss_identity_error(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_layerwise_geometry_marks_zero_reference_instead_of_aborting():
    gradient_a = {
        "live.weight": torch.tensor([1.0]),
        "dead.weight": torch.tensor([0.0]),
    }
    gradient_b = {
        "live.weight": torch.tensor([2.0]),
        "dead.weight": torch.tensor([0.25]),
    }
    rows = {row["layer"]: row for row in layerwise_target_geometry(
        gradient_a, gradient_b, 0.5
    )}
    assert rows["live"]["reference_gradient_nonzero"] is True
    assert rows["dead"]["reference_gradient_nonzero"] is False
    assert math.isnan(rows["dead"]["r_tar"])
    assert rows["dead"]["tau_tar_l2"] == pytest.approx(0.25)
    assert rows["live"]["fit_scope"] == "layer_specific_scalar"

    aggregate = target_geometry(gradient_a, gradient_b, 0.5)
    summary = summarize_layer_geometry(list(rows.values()), aggregate)
    assert summary["nonzero_reference_layer_count"] == 1
    assert summary["zero_reference_layer_count"] == 1
    assert summary["energy_reconstruction_gate_passed"] is True


def test_realized_scale_summary_detects_constant_and_clipped_ratios():
    baseline = torch.tensor([1.0, 2.0], dtype=torch.float32)
    enlarged = baseline * 1.1
    summary = realized_scale_summary(
        baseline, enlarged, expected_scale=1.0 / 1.1
    )
    assert summary["constant_scale_gate_passed"] is True

    nonconstant = realized_scale_summary(
        baseline,
        torch.tensor([1.1, 2.0], dtype=torch.float32),
        expected_scale=1.0 / 1.1,
    )
    assert nonconstant["constant_scale_gate_passed"] is False


def test_q256_ratio_gate_allows_expected_fp32_subtractive_cancellation():
    loss = make_loss("A")
    generator = torch.Generator(device="cpu").manual_seed(20260823)
    normal = torch.randn((25_600, 1, 1, 1), generator=generator)
    t = (normal * loss.P_std + loss.P_mean).exp()
    base_r = loss.schedule.compute_r(t=t, stage=0)
    _, enlarged_r, baseline_delta, enlarged_delta = compute_target_weight_times(
        t,
        base_r,
        target_gap_scale=1.0,
        denominator_gap_scale=1.1,
    )
    summary = realized_scale_summary(
        baseline_delta,
        enlarged_delta,
        expected_scale=1.0 / 1.1,
        base_r=base_r,
        enlarged_r=enlarged_r,
    )
    assert summary["unclipped_support_gate_passed"] is True
    assert summary["ratio_numeric_gate_passed"] is True
    assert summary["constant_scale_gate_passed"] is True
    # The cancellation error is real and exceeds the earlier eps-only gate.
    assert summary["max_abs_error_to_expected"] > 64 * torch.finfo(torch.float32).eps


def test_virtual_cells_satisfy_exact_denominator_identities():
    net = TinyECTNet()
    loss = make_loss("A")
    images = torch.tensor(
        [
            [[[0.2, -0.4], [0.7, 0.1]]],
            [[[-0.3, 0.5], [0.9, -0.8]]],
        ],
        dtype=torch.float32,
    )
    labels = torch.empty((2, 0), dtype=torch.float32)
    t = torch.tensor([0.4, 1.3], dtype=torch.float32).reshape(2, 1, 1, 1)
    eps = torch.tensor(
        [
            [[[0.1, -0.2], [0.3, -0.4]]],
            [[[0.5, -0.6], [0.7, -0.8]]],
        ],
        dtype=torch.float32,
    )
    dropout_state = torch.get_rng_state()

    cells = {
        arm: run_cell(net, loss, images, labels, t, eps, dropout_state, arm)
        for arm in ("A", "D", "C", "B")
    }
    scale = 1.0 / 1.1
    assert _scaled_residual(cells["D"][0], cells["A"][0], scale)[
        "relative_l2"
    ] < 1e-5
    assert _scaled_residual(cells["B"][0], cells["C"][0], scale)[
        "relative_l2"
    ] < 1e-5
    assert torch.allclose(
        cells["D"][1], scale * cells["A"][1], rtol=1e-4, atol=0
    )
    assert torch.allclose(
        cells["B"][1], scale * cells["C"][1], rtol=1e-4, atol=0
    )

    per_sample, times = factorial_loss_with_fixed_randomness(
        net,
        loss,
        images,
        labels,
        t,
        eps,
        dropout_state,
        target_gap_scale=1.1,
        denominator_gap_scale=1.1,
    )
    assert per_sample.shape == (2,)
    assert torch.all(times["r_target"] < t)


def test_exact_zero_pair_residual_fails_closed_for_c_zero():
    images = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    labels = torch.empty((1, 0), dtype=torch.float32)
    t = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    eps = torch.zeros_like(images)
    with pytest.raises(FloatingPointError, match="zero pair residuals"):
        run_cell(
            ZeroResidualNet(),
            make_loss("A"),
            images,
            labels,
            t,
            eps,
            torch.get_rng_state(),
            "A",
        )


@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
def test_fixed_randomness_helper_matches_production_factorial_loss(arm):
    images = torch.tensor(
        [
            [[[0.2, -0.4], [0.7, 0.1]]],
            [[[-0.3, 0.5], [0.9, -0.8]]],
        ],
        dtype=torch.float32,
    )
    labels = torch.empty((2, 0), dtype=torch.float32)
    production_net = TinyECTNet()
    fixed_net = copy.deepcopy(production_net)
    production_loss = make_loss(arm)
    fixed_loss = make_loss(arm)
    target_scale, denominator_scale = {
        "A": (1.0, 1.0),
        "B": (1.1, 1.1),
        "C": (1.1, 1.0),
        "D": (1.0, 1.1),
    }[arm]

    seed = 9102
    torch.manual_seed(seed)
    observed = production_loss(production_net, images, labels)
    observed.mean().backward()

    torch.manual_seed(seed)
    normal = torch.randn((images.shape[0], 1, 1, 1))
    t = (normal * fixed_loss.P_std + fixed_loss.P_mean).exp()
    eps = torch.randn_like(images)
    dropout_state = torch.get_rng_state()
    reconstructed, _ = factorial_loss_with_fixed_randomness(
        fixed_net,
        fixed_loss,
        images,
        labels,
        t,
        eps,
        dropout_state,
        target_gap_scale=target_scale,
        denominator_gap_scale=denominator_scale,
    )
    reconstructed.mean().backward()

    assert torch.equal(observed, reconstructed)
    for observed_parameter, reconstructed_parameter in zip(
        production_net.parameters(), fixed_net.parameters(), strict=True
    ):
        assert torch.equal(observed_parameter.grad, reconstructed_parameter.grad)


def test_probe_is_common_state_deterministic_and_noncommitting():
    images = torch.tensor(
        [
            [[[0, 64], [128, 255]]],
            [[[255, 128], [64, 0]]],
        ],
        dtype=torch.uint8,
    )
    labels = torch.empty((2, 0), dtype=torch.float32)
    loss = make_loss("A")

    first_net = TinyECTNet()
    before = module_state_hashes(first_net)
    rng_before = torch.get_rng_state().clone()
    first = run_probe(
        first_net,
        loss,
        iter([(images, labels)]),
        batches=1,
        device=torch.device("cpu"),
        random_seed=123,
    )
    assert module_state_hashes(first_net) == before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(parameter.grad is None for parameter in first_net.parameters())
    assert first[0]["max_identity_d_equals_s_a_relative_l2"] < 1e-5
    assert first[0]["max_identity_b_equals_s_c_relative_l2"] < 1e-5

    second = run_probe(
        TinyECTNet(),
        make_loss("A"),
        iter([(images, labels)]),
        batches=1,
        device=torch.device("cpu"),
        random_seed=123,
    )
    assert first[0] == second[0]
    assert first[2] == second[2]


def test_probe_restores_rng_and_gradients_after_an_exception():
    net = TinyECTNet()
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(StopIteration):
        run_probe(
            net,
            make_loss("A"),
            iter([]),
            batches=1,
            device=torch.device("cpu"),
            random_seed=123,
        )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(parameter.grad is None for parameter in net.parameters())


def test_loss_reconstructs_from_strict_training_state_when_snapshot_omits_it(
    tmp_path,
):
    loss = make_loss("A")
    loss_kwargs = {
        "class_name": "training.loss.ECMLoss",
        "q": 256,
        "c": 0.0,
        "adj": "sigmoid",
        "factorial_protocol": TARGET_WEIGHT_FACTORIAL_PROTOCOL,
        "target_gap_scale": 1.0,
        "denominator_gap_scale": 1.0,
    }
    trajectory_config = {
        "schema": reproducibility.TRAJECTORY_CONFIG_SCHEMA,
        "seed": 3,
        "augment_kwargs": None,
        "dataset_kwargs": {
            "path": "/immutable/cifar10.zip",
            "use_labels": False,
            "xflip": False,
            "resolution": 32,
        },
        "network_kwargs": {"use_fp16": True},
        "enable_amp": True,
        "batch_gpu": 16,
        "total_kimg": 1024,
        "loss_kwargs": loss_kwargs,
    }
    state_path = tmp_path / "training-state.pt"
    checkpoint_path = tmp_path / "network-snapshot.pkl"
    online = TinyECTNet()
    ema = TinyECTNet()
    torch.save(
        {
            "net": online,
            "ema": ema,
            "cur_nimg": 512_000,
            "cur_tick": 1,
            "successful_optimizer_steps": 2,
            "attempted_iteration": 2,
            "loss_fn_state": loss.schedule_state_dict(),
            "factorial": loss.factorial,
            "reproducibility_schema": reproducibility.TRAINING_STATE_SCHEMA,
            "trajectory_config": trajectory_config,
            "trajectory_config_sha256": reproducibility.state_sha256(
                trajectory_config
            ),
        },
        state_path,
    )
    with checkpoint_path.open("wb") as handle:
        pickle.dump({"loss_fn": None, "ema": ema, "augment_pipe": None}, handle)

    _, loaded_loss, meta = load_common_state(
        state_path,
        checkpoint_path,
        torch.device("cpu"),
        expected_arm="A",
        expected_kimg=512,
        expected_training_seed=3,
    )
    assert loaded_loss.factorial["arm"] == "A"
    assert meta["loss_source"] == "training_state.trajectory_config.loss_kwargs"
    assert meta["trajectory_total_kimg"] == 1024
    assert len(meta["trajectory_dynamics_sha256"]) == 64
    with pytest.raises(SystemExit, match="trajectory seed"):
        load_common_state(
            state_path,
            checkpoint_path,
            torch.device("cpu"),
            expected_arm="A",
            expected_kimg=512,
            expected_training_seed=4,
        )

    altered = torch.load(state_path, map_location="cpu", weights_only=False)
    altered["trajectory_config"]["loss_kwargs"]["P_mean"] = -1.0
    altered["trajectory_config_sha256"] = reproducibility.state_sha256(
        altered["trajectory_config"]
    )
    torch.save(altered, state_path)
    with pytest.raises(SystemExit, match="loss contract mismatch"):
        load_common_state(
            state_path,
            checkpoint_path,
            torch.device("cpu"),
            expected_arm="A",
            expected_kimg=512,
            expected_training_seed=3,
        )


def test_asset_receipt_is_a_fail_closed_preflight(tmp_path):
    state_path = tmp_path / "state.pt"
    checkpoint_path = tmp_path / "snapshot.pkl"
    data_path = tmp_path / "data.zip"
    receipt_path = tmp_path / "snapshot.receipt.json"
    state_path.write_bytes(b"state")
    checkpoint_path.write_bytes(b"snapshot")
    data_path.write_bytes(b"dataset")
    receipt = {
        "schema": "ect.q256.replay-ema-export/v1",
        "status": "PASS",
        "seed": 3,
        "arm": "A",
        "budget_kimg": 512,
        "source_state_sha256": sha256_file(state_path),
        "snapshot_sha256": sha256_file(checkpoint_path),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    args = SimpleNamespace(
        checkpoint_receipt=receipt_path,
        training_seed=3,
        state_arm="A",
        state_kimg=512,
        expected_data_sha256=sha256_file(data_path),
        training_state=state_path,
        checkpoint=checkpoint_path,
        data=data_path,
    )
    loaded, actual = validate_asset_receipt(args)
    assert loaded == receipt
    assert actual["data"] == args.expected_data_sha256

    data_path.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="asset SHA256 mismatch"):
        validate_asset_receipt(args)
