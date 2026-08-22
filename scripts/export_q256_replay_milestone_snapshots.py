#!/usr/bin/env python3
"""Deterministically export immutable EMA snapshots from replay full-states."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import reproducibility


ARMS = ("A", "B", "C", "D")
BUDGETS = (256, 384, 512, 640, 768, 896, 1024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rng_receipt() -> dict:
    return {
        "python": repr(random.getstate()),
        "numpy": repr(np.random.get_state()),
        "torch_cpu": reproducibility.state_sha256(torch.get_rng_state()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(3, 4, 5), required=True)
    args = parser.parse_args()

    for arm in ARMS:
        run_dir = args.run_root / f"seed{args.seed}" / f"arm{arm}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for budget in BUDGETS:
            if budget == 256:
                state_path = (
                    args.source_root
                    / f"seed{args.seed}"
                    / f"arm{arm}"
                    / "training-state-latest.pt"
                )
            else:
                state_path = (
                    run_dir / f"training-state-kimg{budget:06d}.pt"
                )
            output_path = (
                run_dir / f"network-snapshot-kimg{budget:06d}.pkl"
            )
            receipt_path = output_path.with_suffix(".receipt.json")
            assert state_path.is_file(), state_path
            if output_path.exists() or receipt_path.exists():
                assert output_path.is_file() and receipt_path.is_file()
                receipt = json.loads(receipt_path.read_text())
                assert receipt["snapshot_sha256"] == sha256_file(output_path)
                print(f"SNAPSHOT_SKIP seed={args.seed} arm={arm} budget={budget}")
                continue

            before_rng = rng_receipt()
            state = torch.load(
                state_path, map_location="cpu", weights_only=False
            )
            assert int(state["cur_nimg"]) == budget * 1000
            assert int(state["trajectory_config"]["seed"]) == args.seed
            assert state["factorial"]["arm"] == arm
            ema = copy.deepcopy(state["ema"]).eval().requires_grad_(False)
            ema_sha256 = reproducibility.module_state_sha256(ema)
            payload = {
                "ema": ema,
                "loss_fn": None,
                "augment_pipe": None,
                "dataset_kwargs": dict(
                    state["trajectory_config"]["dataset_kwargs"]
                ),
            }
            reproducibility.atomic_pickle_dump(
                payload, output_path, overwrite=False
            )
            after_rng = rng_receipt()
            assert before_rng == after_rng, "snapshot export changed RNG state"
            receipt = {
                "schema": "ect.q256.replay-ema-export/v1",
                "seed": args.seed,
                "arm": arm,
                "budget_kimg": budget,
                "source_state_path": str(state_path),
                "source_state_sha256": sha256_file(state_path),
                "ema_canonical_sha256": ema_sha256,
                "snapshot_path": str(output_path),
                "snapshot_sha256": sha256_file(output_path),
                "rng_unchanged": True,
                "status": "PASS",
            }
            reproducibility.atomic_json_dump(
                receipt, receipt_path, overwrite=False
            )
            print(f"SNAPSHOT_PASS seed={args.seed} arm={arm} budget={budget}")


if __name__ == "__main__":
    main()
