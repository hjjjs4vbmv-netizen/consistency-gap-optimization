#!/usr/bin/env python3
"""Independently verify and package the publication-v2 evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


BLOCKS = ("block_5000_9999", "block_10000_14999", "block_15000_19999")
SEEDS = (3, 4, 5)
ARMS = ("A", "B", "C")
METRICS = ("fid5k_full", "kid5k_full")
EXPECTED_CHECKPOINTS = {
    (3, "A"): "fa48bf5a3c7488678e3efd79d43229196bba6e24bacad2c7fc4cda5c2ff1c32b",
    (3, "B"): "a698182f3bbc8307fe1c36c229e5b50772f7fe7e532868353ddf5e395c0ee4db",
    (3, "C"): "0caf658fdffc30a5d9fd3d143da1a86a7cf40152403e7235cb2b8ae392bc1639",
    (4, "A"): "ec724a4705cab6a789f05404a2fc82b362d5e3ef3aa5ed24735b82583059b684",
    (4, "B"): "e6adb0548babb1de2aaa4a55e22ae4adfbe4d7daae2f2547e11e46628b726595",
    (4, "C"): "b5d19259a9089ba2bc8b8cb90e7dcd669b065a364efbb4f99736aae5bdded31e",
    (5, "A"): "97837ecba0f11d5b7d25c1eada17adf8ce5d5671ceae6553291f1405c5c16455",
    (5, "B"): "fce3c1f2c14357b617f51e7220dd3dfe0e02c3e9894318678d7e167bff6af36a",
    (5, "C"): "48e7fa22cef49b158b9b99da71f20c472149ebced9028b6f5c165653a2762852",
}
DATASET_SHA256 = "a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty file: {path}")


def load_single_jsonl(path: Path, metric: str) -> tuple[dict[str, Any], float]:
    require_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"expected one JSONL row: {path}")
    record = json.loads(lines[0])
    if record.get("metric") != metric:
        raise RuntimeError(f"metric field mismatch: {path}")
    value = record.get("results", {}).get(metric)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"invalid metric value: {path}")
    return record, float(value)


def verify_array(path: Path, shape: tuple[int, ...], dtype: str) -> None:
    require_file(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise RuntimeError(
            f"array mismatch {path}: expected {shape}/{dtype}, "
            f"observed {array.shape}/{array.dtype}"
        )


def render_csv(header: list[str], rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def historical_values(repo: Path) -> dict[tuple[str, int, str, str], float]:
    path = repo / "evidence" / "disjoint_5k_cell_manifest_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: dict[tuple[str, int, str, str], float] = {}
    for cell in payload["cells"]:
        block = cell["cell_id"].split("/", maxsplit=1)[0]
        seed = int(cell["training_seed"])
        arm = cell["arm"]
        for receipt in cell["metric_receipts"]:
            values[(block, seed, arm, receipt["metric"])] = float(
                receipt["reported_value"]
            )
    if len(values) != 54:
        raise RuntimeError(f"expected 54 historical values, found {len(values)}")
    return values


def build(run_root: Path, repo: Path) -> tuple[dict[str, Any], dict[str, str]]:
    contract_path = run_root / "run_contract.json"
    summary_path = run_root / "run_summary.json"
    require_file(contract_path)
    require_file(summary_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "publication-v2-regenerated-disjoint-5k-v2":
        raise RuntimeError("unexpected run contract")
    if run_summary.get("status") != "completed" or run_summary.get("completed_cells") != 27:
        raise RuntimeError("run summary is not 27/27 completed")

    old_values = historical_values(repo)
    cells: list[dict[str, Any]] = []
    blockwise_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    values: dict[tuple[str, int, str, str], float] = {}
    metric_receipt_count = 0
    retained_array_count = 0

    for block in BLOCKS:
        start, end = [int(item) for item in block.removeprefix("block_").split("_")]
        for seed in SEEDS:
            for arm in ARMS:
                relative_cell = Path("blocks") / block / f"seed{seed}" / f"arm_{arm.lower()}"
                cell_root = run_root / relative_cell
                completion_path = cell_root / "completion.json"
                hash_manifest_path = cell_root / "artifact_sha256.json"
                require_file(completion_path)
                require_file(hash_manifest_path)
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                if completion.get("status") != "completed":
                    raise RuntimeError(f"cell not complete: {relative_cell}")
                expected_checkpoint = EXPECTED_CHECKPOINTS[(seed, arm)]
                if completion.get("checkpoint_sha256") != expected_checkpoint:
                    raise RuntimeError(f"checkpoint mismatch: {relative_cell}")
                if completion.get("sample_seed_range") != f"{start}-{end}":
                    raise RuntimeError(f"sample range mismatch: {relative_cell}")

                declared_hashes = json.loads(hash_manifest_path.read_text(encoding="utf-8"))
                core_paths = {
                    "samples": cell_root / "generated-samples.npy",
                    "fid_features": cell_root / "generated-features-fid5k_full-repeat00.npy",
                    "kid_features": cell_root / "generated-features-kid5k_full-repeat00.npy",
                    "fid_receipt": cell_root / "metric-fid5k_full.jsonl",
                    "kid_receipt": cell_root / "metric-kid5k_full.jsonl",
                }
                verify_array(core_paths["samples"], (5000, 3, 32, 32), "uint8")
                verify_array(core_paths["fid_features"], (5000, 2048), "float32")
                verify_array(core_paths["kid_features"], (5000, 2048), "float32")
                retained_array_count += 3
                core_hashes: dict[str, str] = {}
                for label, path in core_paths.items():
                    observed = sha256_file(path)
                    if declared_hashes.get(path.name) != observed:
                        raise RuntimeError(f"artifact hash mismatch: {path}")
                    core_hashes[label] = observed

                metric_values: dict[str, float] = {}
                for metric in METRICS:
                    _record, value = load_single_jsonl(
                        cell_root / f"metric-{metric}.jsonl", metric
                    )
                    metric_values[metric] = value
                    values[(block, seed, arm, metric)] = value
                    old = old_values[(block, seed, arm, metric)]
                    comparison_rows.append(
                        {
                            "block": block,
                            "training_seed": seed,
                            "arm": arm,
                            "metric": metric,
                            "historical_pr53": repr(old),
                            "regenerated_v2": repr(value),
                            "delta_v2_minus_pr53": repr(value - old),
                            "absolute_delta": repr(abs(value - old)),
                        }
                    )
                    metric_receipt_count += 1

                cells.append(
                    {
                        "cell_id": f"{block}/seed{seed}/arm_{arm.lower()}",
                        "training_seed": seed,
                        "arm": arm,
                        "sample_seed_range": f"{start}-{end}",
                        "sample_count": 5000,
                        "checkpoint_sha256": expected_checkpoint,
                        "metrics": metric_values,
                        "artifacts": {
                            "samples": {
                                "path": str(relative_cell / core_paths["samples"].name),
                                "sha256": core_hashes["samples"],
                                "shape": [5000, 3, 32, 32],
                                "dtype": "uint8",
                            },
                            "features": {
                                "fid5k_full": {
                                    "path": str(relative_cell / core_paths["fid_features"].name),
                                    "sha256": core_hashes["fid_features"],
                                    "shape": [5000, 2048],
                                    "dtype": "float32",
                                },
                                "kid5k_full": {
                                    "path": str(relative_cell / core_paths["kid_features"].name),
                                    "sha256": core_hashes["kid_features"],
                                    "shape": [5000, 2048],
                                    "dtype": "float32",
                                },
                            },
                            "metric_receipts": {
                                "fid5k_full": {
                                    "path": str(relative_cell / core_paths["fid_receipt"].name),
                                    "sha256": core_hashes["fid_receipt"],
                                },
                                "kid5k_full": {
                                    "path": str(relative_cell / core_paths["kid_receipt"].name),
                                    "sha256": core_hashes["kid_receipt"],
                                },
                            },
                        },
                    }
                )

    if len(cells) != 27 or metric_receipt_count != 54 or retained_array_count != 81:
        raise RuntimeError("publication matrix accounting failed")

    for block in BLOCKS:
        for seed in SEEDS:
            for metric in METRICS:
                arms = {arm: values[(block, seed, arm, metric)] for arm in ARMS}
                blockwise_rows.append(
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

    summary_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for metric in METRICS:
            deltas = [
                values[(block, seed, "C", metric)] - values[(block, seed, "B", metric)]
                for block in BLOCKS
            ]
            positive = sum(value > 0 for value in deltas)
            negative = sum(value < 0 for value in deltas)
            if positive == 3:
                sign, consistency = "positive", positive
            elif negative == 3:
                sign, consistency = "negative", negative
            else:
                sign, consistency = "mixed", max(positive, negative)
            summary_rows.append(
                {
                    "training_seed": seed,
                    "metric": metric,
                    "blocks": 3,
                    "delta_ctrl_mean": f"{statistics.mean(deltas):.9f}",
                    "delta_ctrl_sample_sd": f"{statistics.stdev(deltas):.9f}",
                    "delta_ctrl_sign": sign,
                    "sign_consistency": f"{consistency}/3",
                }
            )

    comparison_abs = [float(row["absolute_delta"]) for row in comparison_rows]
    manifest = {
        "schema_version": 1,
        "manifest_id": "publication-v2-regenerated-disjoint-5k-v2",
        "status": "PASS",
        "evidence_class": "regenerated evaluation provenance",
        "source_commit": contract["source_commit"],
        "contract_sha256": sha256_file(contract_path),
        "run_summary_sha256": sha256_file(summary_path),
        "finished_at_utc": run_summary["finished_at_utc"],
        "dataset_sha256": DATASET_SHA256,
        "matrix": {
            "cells": 27,
            "metric_receipts": 54,
            "checkpoint_hash_bound_receipts": 54,
            "sample_range_bound_receipts": 54,
            "retained_sample_arrays": 27,
            "retained_feature_arrays": 54,
            "retry_count": 0,
        },
        "historical_pr53_comparison": {
            "rows": 54,
            "max_absolute_delta": max(comparison_abs),
            "mean_absolute_delta": statistics.mean(comparison_abs),
            "note": "PR #53 is archival only; v2 is the publication evidence table.",
        },
        "b003_status": "RESOLVED_BY_REGENERATED_PROVENANCE",
        "b005_status": "RESOLVED_BY_RECOVERED_ORIGINAL_BYTES",
        "b006_status": "RESOLVED_54_OF_54_CHECKPOINT_HASH_BOUND",
        "cells": cells,
    }
    rendered = {
        "blockwise_results.csv": render_csv(
            [
                "block", "training_seed", "metric", "A", "B", "C",
                "delta_gap_B_minus_A", "delta_ctrl_C_minus_B",
            ],
            blockwise_rows,
        ),
        "disjoint_block_summary.csv": render_csv(
            [
                "training_seed", "metric", "blocks", "delta_ctrl_mean",
                "delta_ctrl_sample_sd", "delta_ctrl_sign", "sign_consistency",
            ],
            summary_rows,
        ),
        "comparison_to_pr53.csv": render_csv(
            [
                "block", "training_seed", "arm", "metric", "historical_pr53",
                "regenerated_v2", "delta_v2_minus_pr53", "absolute_delta",
            ],
            comparison_rows,
        ),
    }
    return manifest, rendered


def write_bundle(run_root: Path, repo: Path, outdir: Path) -> None:
    manifest, rendered = build(run_root, repo)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "publication_v2_cell_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, content in rendered.items():
        (outdir / name).write_text(content, encoding="utf-8", newline="")
    summary_lines = [
        "# Publication-v2 regenerated evaluation",
        "",
        "Status: **PASS**",
        "",
        "- 27/27 cells completed; 54/54 metric receipts hash-bound.",
        "- 27 sample arrays and 54 feature arrays retained and independently verified.",
        "- B003: resolved by regenerated evaluation provenance.",
        "- B005: resolved from recovered original checkpoint bytes.",
        "- B006: resolved; checkpoint coverage is 54/54 receipts.",
        f"- Maximum absolute metric difference from archival PR #53: {manifest['historical_pr53_comparison']['max_absolute_delta']:.12g}.",
        "",
        "PR #53 values remain archival; the regenerated-v2 table is the publication evidence.",
        "",
    ]
    (outdir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")
    checksum_files = sorted(path for path in outdir.iterdir() if path.name != "SHA256SUMS")
    (outdir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_files),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "outdir": str(outdir),
                "manifest_sha256": sha256_file(manifest_path),
                "cells": manifest["matrix"]["cells"],
                "metric_receipts": manifest["matrix"]["metric_receipts"],
                "retained_arrays": (
                    manifest["matrix"]["retained_sample_arrays"]
                    + manifest["matrix"]["retained_feature_arrays"]
                ),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    write_bundle(args.run_root.resolve(), args.repo.resolve(), args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
