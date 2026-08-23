#!/usr/bin/env python3
"""Export frozen second-q EMA snapshots from immutable full states."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import reproducibility


DEFAULT_BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
DEFAULT_CELLS = (
    (3, "A"),
    (3, "B"),
    (4, "A"),
    (4, "B"),
    (5, "A"),
    (5, "B"),
)


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


def parse_cells(value: str) -> tuple[tuple[int, str], ...]:
    cells = []
    for token in value.split(","):
        seed_text, separator, arm = token.strip().partition(":")
        if separator != ":" or not seed_text.isdigit():
            raise argparse.ArgumentTypeError(f"invalid cell: {token!r}")
        cell = (int(seed_text), arm)
        if cell not in DEFAULT_CELLS:
            raise argparse.ArgumentTypeError(f"unsupported cell: {token!r}")
        cells.append(cell)
    if not cells or len(set(cells)) != len(cells):
        raise argparse.ArgumentTypeError("cells must be non-empty and unique")
    return tuple(cells)


def wait_for_state(
    path: Path, *, wait: bool, deadline: float, poll_seconds: float
) -> None:
    while not path.is_file():
        if not wait:
            raise FileNotFoundError(path)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        print(f"WAIT state={path}", flush=True)
        time.sleep(poll_seconds)


def export_one(run_root: Path, seed: int, arm: str, budget: int) -> dict:
    run_dir = run_root / f"seed{seed}" / f"arm{arm}"
    state_path = run_dir / f"training-state-kimg{budget:06d}.pt"
    output_path = run_dir / f"network-snapshot-kimg{budget:06d}.pkl"
    receipt_path = output_path.with_suffix(".receipt.json")

    if output_path.exists() or receipt_path.exists():
        if not output_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"partial snapshot export exists: {output_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("snapshot_sha256") != sha256_file(output_path):
            raise RuntimeError(f"snapshot receipt mismatch: {output_path}")
        print(f"SNAPSHOT_SKIP seed={seed} arm={arm} budget={budget}", flush=True)
        return receipt

    before_rng = rng_receipt()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state.get("cur_nimg", -1)) != budget * 1000:
        raise RuntimeError(f"state budget mismatch: {state_path}")
    trajectory = state.get("trajectory_config", {})
    if int(trajectory.get("seed", -1)) != seed:
        raise RuntimeError(f"state seed mismatch: {state_path}")
    if state.get("factorial", {}).get("arm") != arm:
        raise RuntimeError(f"state arm mismatch: {state_path}")
    if float(trajectory.get("loss_kwargs", {}).get("q", -1)) != 128.0:
        raise RuntimeError(f"state q mismatch: {state_path}")

    ema = copy.deepcopy(state["ema"]).eval().requires_grad_(False)
    ema_sha256 = reproducibility.module_state_sha256(ema)
    payload = {
        "ema": ema,
        "loss_fn": None,
        "augment_pipe": None,
        "dataset_kwargs": dict(trajectory["dataset_kwargs"]),
    }
    reproducibility.atomic_pickle_dump(payload, output_path, overwrite=False)
    after_rng = rng_receipt()
    if before_rng != after_rng:
        raise RuntimeError("snapshot export changed process RNG state")

    receipt = {
        "schema": "ect.second-q.ema-export/v1",
        "seed": seed,
        "arm": arm,
        "budget_kimg": budget,
        "schedule_q": 128,
        "source_state_path": str(state_path),
        "source_state_sha256": sha256_file(state_path),
        "ema_canonical_sha256": ema_sha256,
        "snapshot_path": str(output_path),
        "snapshot_sha256": sha256_file(output_path),
        "rng_unchanged": True,
        "status": "PASS",
    }
    reproducibility.atomic_json_dump(receipt, receipt_path, overwrite=False)
    print(f"SNAPSHOT_PASS seed={seed} arm={arm} budget={budget}", flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--cells",
        type=parse_cells,
        default=DEFAULT_CELLS,
        help="Comma-separated SEED:ARM cells",
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=5 * 60 * 60)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise SystemExit("timeouts must be positive")
    deadline = time.monotonic() + args.timeout_seconds
    receipts = []
    for budget in DEFAULT_BUDGETS:
        for seed, arm in args.cells:
            state_path = (
                args.run_root
                / f"seed{seed}"
                / f"arm{arm}"
                / f"training-state-kimg{budget:06d}.pt"
            )
            wait_for_state(
                state_path,
                wait=args.wait,
                deadline=deadline,
                poll_seconds=args.poll_seconds,
            )
            receipts.append(export_one(args.run_root, seed, arm, budget))

    summary = {
        "schema": "ect.second-q.ema-export-matrix/v1",
        "status": "PASS",
        "run_root": str(args.run_root),
        "cell_count": len(args.cells),
        "budget_count": len(DEFAULT_BUDGETS),
        "snapshot_count": len(receipts),
        "receipts": receipts,
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        reproducibility.atomic_json_dump(
            summary, args.summary_out, overwrite=False
        )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
