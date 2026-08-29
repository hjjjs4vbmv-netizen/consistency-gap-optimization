#!/usr/bin/env python3
"""Sparse exact three-point forcing/feedback probes on production fork states."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dnnlib
from analysis.nonlinear_dynamics_gate.decompose_forcing_feedback import (
    CSV_FIELDS,
    _batch_receipt,
    _clone_tensor_map,
    _mechanism_summary,
    _row,
    _state_tensor_blocks,
    _transition,
    carryover_only_map,
    observable_vectors,
)
from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate.core import (
    ARM_SPECS,
    AlgorithmicState,
    AuditBatch,
    AuditBatchGroup,
    _net_forward,
    _schedule,
    get_device_rng_state,
    set_device_rng_state,
    state_sha256,
)
from torch_utils import misc
from training import reproducibility


HORIZONS = (0, 1, 4, 16, 64, 128, 256, 500)
ARMS = ("A", "B", "C", "D")
SCALES = {
    "A": (1.0, 1.0), "B": (1.1, 1.1),
    "C": (1.1, 1.0), "D": (1.0, 1.1),
}
EXOGENOUS_FIELDS = (
    "batch_sha256", "t_sha256", "base_r_sha256", "input_noise_sha256",
    "dropout_rng_sha256", "augmentation_rng_sha256",
)


def state_path(output_root: Path, source_root: Path, seed: int, arm: str,
               horizon: int) -> Path:
    if horizon == 0:
        return source_root / f"seed{seed}" / "armB" / "training-state-kimg000384.pt"
    run_dir = output_root / "runs" / f"seed{seed}" / f"B384_to_{arm}"
    if horizon == 500:
        return run_dir / "training-state-kimg000448.pt"
    return run_dir / f"training-state-attempt{3000 + horizon:06d}.pt"


def load_state(path: Path, arm: str, device: torch.device) -> tuple[AlgorithmicState, dict]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    trajectory = copy.deepcopy(raw["trajectory_config"])
    loss_kwargs = dict(trajectory["loss_kwargs"])
    target, denominator = SCALES[arm]
    loss_kwargs.update(
        factorial_protocol="q256_target_weight_v1",
        target_gap_scale=target,
        denominator_gap_scale=denominator,
    )
    net = copy.deepcopy(raw["net"]).to(device).train().requires_grad_(True)
    ema = copy.deepcopy(raw["ema"]).to(device).eval().requires_grad_(False)
    optimizer = cli_common._optimizer_from_state(net, raw["optimizer_state"])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    if not loss_fn.load_schedule_state_dict(copy.deepcopy(raw["loss_fn_state"])):
        raise RuntimeError(f"loss state is incompatible: {path}")
    scaler = torch.amp.GradScaler("cuda")
    scaler.load_state_dict(copy.deepcopy(raw["gradscaler_state"]))
    return AlgorithmicState(
        net=net, optimizer=optimizer, ema=ema, loss_fn=loss_fn,
        scaler=scaler, ema_beta=float(trajectory["ema_beta"]),
    ), raw


def next_production_batch(state: AlgorithmicState, raw: dict, *, seed: int,
                          horizon: int, device: torch.device) -> tuple[AuditBatchGroup, dict]:
    trajectory = raw["trajectory_config"]
    dataset_kwargs = dict(trajectory["dataset_kwargs"])
    loader_kwargs = dict(trajectory["data_loader_kwargs"])
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=seed)
    rank_state = raw["rank_states"][0]
    sampler.load_state_dict(rank_state["sampler_state"])
    iterator = iter(torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler,
        batch_size=int(trajectory["batch_gpu"]), **loader_kwargs,
    ))
    process_rng = reproducibility.capture_rng_state()
    source_hash = state.sha256()
    raw_batch_hashes = []
    micros = []
    audit_id = 202608290000 + seed * 1000 + horizon
    try:
        reproducibility.restore_rng_state(rank_state["rng_state"])
        for _ in range(int(trajectory["num_accumulation_rounds"])):
            raw_images, raw_labels = next(iterator)
            raw_batch_hashes.append(reproducibility.state_sha256({
                "images": raw_images, "labels": raw_labels,
            }))
            images = raw_images.to(device).to(torch.float32) / 127.5 - 1.0
            labels = raw_labels.to(device)
            t = (
                torch.randn([images.shape[0], 1, 1, 1], device=device)
                * float(state.loss_fn.P_std) + float(state.loss_fn.P_mean)
            ).exp()
            noise = torch.randn_like(images)
            dropout_rng = get_device_rng_state(device)
            micro = AuditBatch(
                images=images.detach().clone(), labels=labels.detach().clone(),
                t=t.detach().clone(), noise=noise.detach().clone(),
                dropout_rng_state=dropout_rng.detach().clone(), audit_id=audit_id,
            )
            micros.append(micro)
            # Advance exactly the schedule-independent dropout stream that one
            # production loss call consumes. The target resets to the same mask,
            # so the ending RNG equals one network forward from dropout_rng.
            r_target = _schedule(
                state.loss_fn, ARM_SPECS["A"]["target_scale"]
            ).compute_r(t, stage=int(state.loss_fn.stage))
            with torch.no_grad():
                _net_forward(state.net, images + noise * t, t, labels)
                set_device_rng_state(dropout_rng, device)
                _net_forward(state.net, images + noise * r_target, r_target, labels)
    finally:
        reproducibility.restore_rng_state(process_rng)
    if state.sha256() != source_hash:
        raise RuntimeError("production exogenous reconstruction mutated source state")
    batch = AuditBatchGroup(tuple(micros), audit_id)
    receipt = {
        "batch_sha256": reproducibility.state_sha256(raw_batch_hashes),
        "t_sha256": reproducibility.state_sha256(
            [reproducibility.state_sha256(item.t) for item in micros]
        ),
        "base_r_sha256": reproducibility.state_sha256([
            reproducibility.state_sha256(
                state.loss_fn.schedule.compute_r(item.t, stage=int(state.loss_fn.stage))
            ) for item in micros
        ]),
        "input_noise_sha256": reproducibility.state_sha256(
            [reproducibility.state_sha256(item.noise) for item in micros]
        ),
        "dropout_rng_sha256": reproducibility.state_sha256(
            [reproducibility.state_sha256(item.dropout_rng_state) for item in micros]
        ),
        "augmentation_rng_sha256": reproducibility.state_sha256(
            [reproducibility.state_sha256(None) for _ in micros]
        ),
        "audit_batch": _batch_receipt(batch),
    }
    return batch, receipt


def telemetry_row(output_root: Path, seed: int, arm: str, horizon: int) -> dict[str, str] | None:
    if horizon >= 500:
        return None
    path = output_root / "runs" / f"seed{seed}" / f"B384_to_{arm}" / "matched_training_telemetry_v1.csv"
    attempted = 3000 + horizon + 1
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["attempted_iteration"]) == attempted:
                return row
    raise RuntimeError(f"missing production telemetry attempt {attempted}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-horizons-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    probe_receipts = []
    all_streams_match = True
    all_source_states_preserved = True
    for horizon in HORIZONS:
        a_path = state_path(args.output_root, args.source_root, args.seed, "A", horizon)
        a_state, a_raw = load_state(a_path, "A", device)
        batch, exogenous = next_production_batch(
            a_state, a_raw, seed=args.seed, horizon=horizon, device=device,
        )
        expected_row = telemetry_row(args.output_root, args.seed, "A", horizon)
        stream_matches = (
            expected_row is None
            or all(exogenous[field] == expected_row[field] for field in EXOGENOUS_FIELDS)
        )
        all_streams_match &= stream_matches
        for arm in ("B", "C", "D"):
            x_path = state_path(args.output_root, args.source_root, args.seed, arm, horizon)
            x_state, _ = load_state(x_path, arm, device)
            a_before = a_state.sha256()
            x_before = x_state.sha256()
            pre_a = {
                block: _clone_tensor_map(values)
                for block, values in _state_tensor_blocks(a_state).items()
            }
            pre_x = {
                block: _clone_tensor_map(values)
                for block, values in _state_tensor_blocks(x_state).items()
            }
            pre_a_obs = observable_vectors(a_state, batch, None)
            pre_x_obs = observable_vectors(x_state, batch, None)
            if arm == "B":
                for block, values in pre_a.items():
                    horizon_rows.append({
                        "seed": args.seed, "horizon": horizon, "arm": "A",
                        "branch_label": "B384_to_A", "space": "state",
                        "block": block,
                        "value_norm": math.sqrt(sum(
                            float(value.detach().double().square().sum())
                            for value in values.values()
                        )),
                    })
                for block, values in pre_a_obs.items():
                    horizon_rows.append({
                        "seed": args.seed, "horizon": horizon, "arm": "A",
                        "branch_label": "B384_to_A", "space": "observable",
                        "block": block,
                        "value_norm": math.sqrt(sum(
                            float(value.detach().double().square().sum())
                            for value in values.values()
                        )),
                    })
            for block, values in pre_x.items():
                horizon_rows.append({
                    "seed": args.seed, "horizon": horizon, "arm": arm,
                    "branch_label": f"B384_to_{arm}", "space": "state",
                    "block": block,
                    "value_norm": math.sqrt(sum(
                        float(value.detach().double().square().sum())
                        for value in values.values()
                    )),
                })
            for block, values in pre_x_obs.items():
                horizon_rows.append({
                    "seed": args.seed, "horizon": horizon, "arm": arm,
                    "branch_label": f"B384_to_{arm}", "space": "observable",
                    "block": block,
                    "value_norm": math.sqrt(sum(
                        float(value.detach().double().square().sum())
                        for value in values.values()
                    )),
                })
            pairing_seed = int(batch.audit_id)
            a_after, a_telemetry = _transition(
                a_state, batch, "A", pairing_seed, clone_input=True,
            )
            counterfactual, cf_telemetry = _transition(
                a_state, batch, arm, pairing_seed, clone_input=True,
            )
            actual, actual_telemetry = _transition(
                x_state, batch, arm, pairing_seed, clone_input=True,
            )
            post_a = _state_tensor_blocks(a_after)
            post_cf = _state_tensor_blocks(counterfactual)
            post_x = _state_tensor_blocks(actual)
            for block in post_a:
                carryover, metadata = carryover_only_map(
                    block, pre_a[block], pre_x[block], actual,
                    optimizer_step_skipped=bool(actual_telemetry["step_skipped"]),
                    optimizer_skip_regime_paired=(
                        bool(actual_telemetry["step_skipped"])
                        == bool(cf_telemetry["step_skipped"])
                    ),
                )
                row = _row(
                    arm, horizon, "state", block,
                    post_a[block], post_cf[block], post_x[block],
                    pre_baseline=pre_a[block], pre_actual=pre_x[block],
                    carryover=carryover, carryover_metadata=metadata,
                )
                row.update(seed=args.seed, horizon=horizon)
                rows.append(row)
            obs_a = observable_vectors(a_after, batch, None)
            obs_cf = observable_vectors(counterfactual, batch, None)
            obs_x = observable_vectors(actual, batch, None)
            for block in ("residual", "feature"):
                row = _row(
                    arm, horizon, "observable", block,
                    obs_a[block], obs_cf[block], obs_x[block],
                    pre_baseline=pre_a_obs[block], pre_actual=pre_x_obs[block],
                    carryover_metadata={
                        "rule": "not_declared_for_this_readout",
                        "retention_source": None, "retention_values": [],
                    },
                )
                row.update(seed=args.seed, horizon=horizon)
                rows.append(row)
            preserved = a_state.sha256() == a_before and x_state.sha256() == x_before
            all_source_states_preserved &= preserved
            probe_receipts.append({
                "seed": args.seed, "horizon": horizon, "arm": arm,
                "stream_matches_production": stream_matches,
                "source_states_preserved": preserved,
                "exogenous": exogenous,
                "transition_receipts": {
                    "phi_A_zA": a_telemetry,
                    "phi_X_zA": cf_telemetry,
                    "phi_X_zX": actual_telemetry,
                },
            })
            del x_state, a_after, counterfactual, actual
            torch.cuda.empty_cache()
        del a_state, a_raw, batch
        torch.cuda.empty_cache()
    closure_pass = all(row["closure_pass"] for row in rows)
    summary = _mechanism_summary(rows)
    payload = {
        "schema": "ect.q256.b384-same-state-sparse-forcing-feedback/v1",
        "status": (
            "PASS" if closure_pass and all_streams_match
            and all_source_states_preserved else "FAIL_CLOSED"
        ),
        "seed": args.seed, "horizons": list(HORIZONS),
        "probe_count": len(probe_receipts),
        "all_exact_closures_pass": closure_pass,
        "max_closure_l2": max(row["closure_l2"] for row in rows),
        "max_closure_relative": max(row["closure_relative"] for row in rows),
        "all_available_streams_match_production": all_streams_match,
        "all_source_states_preserved": all_source_states_preserved,
        "mechanism_by_arm_and_block": summary,
        "probe_receipts": probe_receipts,
        "claim_guard": (
            "Norm ratios are scale diagnostics, not causal contribution percentages."
        ),
    }
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ("seed", "horizon", *CSV_FIELDS)
    with args.out_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with args.out_horizons_csv.open("x", newline="", encoding="utf-8") as handle:
        fields = (
            "seed", "horizon", "arm", "branch_label", "space", "block",
            "value_norm",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(horizon_rows)
    with args.out_json.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
