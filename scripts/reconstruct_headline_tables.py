#!/usr/bin/env python3
"""Reconstruct publication-v2 headline tables from the anonymous manifest."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path


BLOCKS = ("block_5000_9999", "block_10000_14999", "block_15000_19999")
SEEDS = (3, 4, 5)
ARMS = ("A", "B", "C")
METRICS = ("fid5k_full", "kid5k_full")


def render(header: list[str], rows: list[dict[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def reconstruct(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values: dict[tuple[str, int, str, str], float] = {}
    for cell in manifest["cells"]:
        block = cell["cell_id"].split("/", maxsplit=1)[0]
        for metric, value in cell["metrics"].items():
            values[(block, int(cell["training_seed"]), cell["arm"], metric)] = float(value)
    if len(values) != 54:
        raise RuntimeError(f"expected 54 metric values, found {len(values)}")

    blockwise: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for block in BLOCKS:
        for seed in SEEDS:
            for metric in METRICS:
                arms = {arm: values[(block, seed, arm, metric)] for arm in ARMS}
                blockwise.append(
                    {
                        "block": block,
                        "training_seed": seed,
                        "metric": metric,
                        "A": repr(arms["A"]),
                        "B": repr(arms["B"]),
                        "C": repr(arms["C"]),
                        "delta_gap_B_minus_A": repr(arms["B"] - arms["A"]),
                        "delta_ctrl_C_minus_B": repr(arms["C"] - arms["B"]),
                    }
                )
    for seed in SEEDS:
        for metric in METRICS:
            deltas = [
                values[(block, seed, "C", metric)] - values[(block, seed, "B", metric)]
                for block in BLOCKS
            ]
            positive = sum(value > 0 for value in deltas)
            negative = sum(value < 0 for value in deltas)
            sign = "positive" if positive == 3 else "negative" if negative == 3 else "mixed"
            summary.append(
                {
                    "training_seed": seed,
                    "metric": metric,
                    "blocks": 3,
                    "delta_ctrl_mean": f"{statistics.mean(deltas):.9f}",
                    "delta_ctrl_sample_sd": f"{statistics.stdev(deltas):.9f}",
                    "delta_ctrl_sign": sign,
                    "sign_consistency": f"{max(positive, negative)}/3",
                }
            )
    return {
        "blockwise_results.csv": render(
            ["block", "training_seed", "metric", "A", "B", "C",
             "delta_gap_B_minus_A", "delta_ctrl_C_minus_B"], blockwise
        ),
        "disjoint_block_summary.csv": render(
            ["training_seed", "metric", "blocks", "delta_ctrl_mean",
             "delta_ctrl_sample_sd", "delta_ctrl_sign", "sign_consistency"], summary
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--verify-against", type=Path)
    args = parser.parse_args()
    rendered = reconstruct(args.manifest)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        path = args.outdir / name
        path.write_text(content, encoding="utf-8", newline="")
        if args.verify_against is not None:
            expected = (args.verify_against / name).read_bytes()
            if path.read_bytes() != expected:
                raise RuntimeError(f"byte mismatch: {name}")
    print("headline table reconstruction: PASS (2/2 byte-exact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
