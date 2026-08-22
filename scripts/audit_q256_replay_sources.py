#!/usr/bin/env python3
"""Fail-closed audit of the 12 frozen 256 kimg replay sources."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch


ARMS = {
    "A": (1.0, 1.0),
    "B": (1.1, 1.1),
    "C": (1.1, 1.0),
    "D": (1.0, 1.1),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_rank_state_complete(rank_state: dict) -> None:
    assert isinstance(rank_state, dict)
    rng = rank_state["rng_state"]
    sampler = rank_state["sampler_state"]
    assert all(
        key in rng
        for key in (
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda_all",
            "torch_cuda_device_count",
        )
    )
    assert "consumed_samples" in sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for seed in (3, 4, 5):
        for arm, (target_scale, denominator_scale) in ARMS.items():
            path = (
                args.source_root
                / f"seed{seed}"
                / f"arm{arm}"
                / "training-state-latest.pt"
            )
            assert path.is_file() and path.stat().st_size > 0, path
            file_sha256 = sha256_file(path)
            state = torch.load(
                path, map_location="cpu", weights_only=False
            )
            assert int(state["cur_nimg"]) == 256_000
            assert int(state["attempted_iteration"]) == 2_000
            assert 0 < int(state["successful_optimizer_steps"]) <= 2_000
            assert state.get("net") is not None
            assert state.get("ema") is not None
            assert isinstance(state.get("optimizer_state"), dict)
            assert state["optimizer_state"].get("state")
            assert isinstance(state.get("gradscaler_state"), dict)
            assert state["gradscaler_state"]
            assert isinstance(state.get("rank_states"), list)
            assert len(state["rank_states"]) == 1
            assert_rank_state_complete(state["rank_states"][0])
            factorial = state["factorial"]
            assert factorial["arm"] == arm
            assert float(factorial["target_gap_scale"]) == target_scale
            assert (
                float(factorial["denominator_gap_scale"])
                == denominator_scale
            )
            trajectory = state["trajectory_config"]
            assert int(trajectory["seed"]) == seed
            assert int(trajectory["batch_size"]) == 128
            assert int(trajectory["batch_gpu"]) == 16
            assert trajectory["enable_amp"] is True
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "target_gap_scale": target_scale,
                    "denominator_gap_scale": denominator_scale,
                    "cur_nimg": int(state["cur_nimg"]),
                    "attempted_iteration": int(
                        state["attempted_iteration"]
                    ),
                    "successful_optimizer_steps": int(
                        state["successful_optimizer_steps"]
                    ),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256,
                    "trajectory_config_sha256": state[
                        "trajectory_config_sha256"
                    ],
                    "status": "PASS",
                }
            )
            del state
            gc.collect()

    assert len(rows) == 12
    payload = {
        "schema": "ect.q256.target-weight-replay-sources/v1",
        "source_root": str(args.source_root),
        "source_count": len(rows),
        "all_pass": True,
        "sources": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("SOURCE_AUDIT_PASS count=12")


if __name__ == "__main__":
    main()
