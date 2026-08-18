#!/usr/bin/env python3
"""Rebuild PR #53's disjoint-block CSVs from the SHA-bound metric JSONLs.

The source may be a Git commit, so this remains usable before the Role E branch
is merged into the current checkout.  It deliberately consumes only the 54
versioned metric JSONLs; it does not access checkpoints or generated samples.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE_REF = "fbdcb13ed7dab8a5fb179382e4453df9c1a0f7d2"
ROOT = "results/gap_lr_matched/disjoint_5k_0813"
BLOCKWISE_HEADER = [
    "block",
    "training_seed",
    "metric",
    "A",
    "B",
    "C",
    "delta_gap_B_minus_A",
    "delta_ctrl_C_minus_B",
]
SUMMARY_HEADER = [
    "training_seed",
    "metric",
    "blocks",
    "delta_ctrl_mean",
    "delta_ctrl_sample_sd",
    "delta_ctrl_sign",
    "sign_consistency",
]
METRIC_RE = re.compile(
    rf"^{re.escape(ROOT)}/blocks/"
    r"(?P<block>block_(?P<start>\d+)_(?P<end>\d+))/"
    r"seed(?P<seed>[345])/arm_(?P<arm>[abc])/"
    r"metric-(?P<metric>fid5k_full|kid5k_full)\.jsonl$"
)
EXPECTED_BLOCKS = (
    "block_5000_9999",
    "block_10000_14999",
    "block_15000_19999",
)


def git_bytes(repo: Path, ref: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def git_paths(repo: Path, ref: str, root: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, root],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line]


@dataclass(frozen=True)
class MetricRecord:
    block: str
    training_seed: int
    arm: str
    metric: str
    value: float
    path: str


def load_metric_records(repo: Path, ref: str) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for path in git_paths(repo, ref, f"{ROOT}/blocks"):
        match = METRIC_RE.fullmatch(path)
        if match is None:
            continue
        raw = json.loads(git_bytes(repo, ref, path))
        metric = match.group("metric")
        if raw.get("metric") != metric:
            raise ValueError(f"metric field/path mismatch: {path}")
        value = raw.get("results", {}).get(metric)
        if not isinstance(value, (int, float)):
            raise ValueError(f"missing numeric metric value: {path}")
        records.append(
            MetricRecord(
                block=match.group("block"),
                training_seed=int(match.group("seed")),
                arm=match.group("arm").upper(),
                metric=metric,
                value=float(value),
                path=path,
            )
        )
    if len(records) != 54:
        raise ValueError(f"expected 54 metric files, found {len(records)}")
    return records


def build_tables(records: list[MetricRecord]) -> tuple[list[dict], list[dict]]:
    indexed = {
        (item.block, item.training_seed, item.metric, item.arm): item.value
        for item in records
    }
    if len(indexed) != 54:
        raise ValueError("duplicate block/seed/metric/arm cell")

    blockwise: list[dict] = []
    for block in EXPECTED_BLOCKS:
        for seed in (3, 4, 5):
            for metric in ("fid5k_full", "kid5k_full"):
                values = {
                    arm: indexed[(block, seed, metric, arm)]
                    for arm in ("A", "B", "C")
                }
                blockwise.append(
                    {
                        "block": block,
                        "training_seed": seed,
                        "metric": metric,
                        **values,
                        "delta_gap_B_minus_A": values["B"] - values["A"],
                        "delta_ctrl_C_minus_B": values["C"] - values["B"],
                    }
                )

    summary: list[dict] = []
    for seed in (3, 4, 5):
        for metric in ("fid5k_full", "kid5k_full"):
            deltas = [
                row["delta_ctrl_C_minus_B"]
                for row in blockwise
                if row["training_seed"] == seed and row["metric"] == metric
            ]
            positive = sum(value > 0 for value in deltas)
            negative = sum(value < 0 for value in deltas)
            if positive == len(deltas):
                sign = "positive"
                consistent = positive
            elif negative == len(deltas):
                sign = "negative"
                consistent = negative
            else:
                sign = "mixed"
                consistent = max(positive, negative)
            summary.append(
                {
                    "training_seed": seed,
                    "metric": metric,
                    "blocks": len(deltas),
                    "delta_ctrl_mean": statistics.mean(deltas),
                    "delta_ctrl_sample_sd": statistics.stdev(deltas),
                    "delta_ctrl_sign": sign,
                    "sign_consistency": f"{consistent}/{len(deltas)}",
                }
            )
    return blockwise, summary


def render_csv(header: list[str], rows: list[dict], summary: bool = False) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rendered = dict(row)
        if summary:
            rendered["delta_ctrl_mean"] = f"{row['delta_ctrl_mean']:.9f}"
            rendered["delta_ctrl_sample_sd"] = f"{row['delta_ctrl_sample_sd']:.9f}"
        writer.writerow(rendered)
    return buffer.getvalue()


def verify_committed_tables(repo: Path, ref: str) -> None:
    records = load_metric_records(repo, ref)
    blockwise, summary = build_tables(records)
    expected_blockwise = git_bytes(repo, ref, f"{ROOT}/blockwise_results.csv").decode()
    expected_summary = git_bytes(repo, ref, f"{ROOT}/disjoint_block_summary.csv").decode()
    actual_blockwise = render_csv(BLOCKWISE_HEADER, blockwise)
    actual_summary = render_csv(SUMMARY_HEADER, summary, summary=True)
    if actual_blockwise != expected_blockwise:
        raise ValueError("rebuilt blockwise_results.csv differs from PR #53")
    if actual_summary != expected_summary:
        raise ValueError("rebuilt disjoint_block_summary.csv differs from PR #53")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--outdir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_metric_records(args.repo, args.source_ref)
    blockwise, summary = build_tables(records)
    if args.verify:
        verify_committed_tables(args.repo, args.source_ref)
        print("PASS: PR #53 CSVs reproduce exactly from 54 metric JSONLs")
    if args.outdir is not None:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "blockwise_results.csv").write_text(
            render_csv(BLOCKWISE_HEADER, blockwise), encoding="utf-8"
        )
        (args.outdir / "disjoint_block_summary.csv").write_text(
            render_csv(SUMMARY_HEADER, summary, summary=True), encoding="utf-8"
        )
    elif not args.verify:
        sys.stdout.write(render_csv(SUMMARY_HEADER, summary, summary=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
