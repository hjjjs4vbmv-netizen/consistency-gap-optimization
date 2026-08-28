#!/usr/bin/env python3
"""Materialize the four preregistered audit minibatches from canonical data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from analysis.operator_clock_gate import cli_common
from analysis.operator_clock_gate.core import write_json
from training.dataset import ImageFolderDataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def run(args) -> int:
    actual_sha = cli_common.sha256_file(args.data)
    if actual_sha != args.expected_data_sha256:
        raise RuntimeError(
            f"dataset SHA256 mismatch: {actual_sha} != {args.expected_data_sha256}")
    frozen = cli_common.protocol()
    construction = frozen["batch_construction"]
    dataset = ImageFolderDataset(
        path=str(args.data), use_labels=False, xflip=False,
        cache=False, resolution=32)
    batches = []
    index_batches = []
    for audit_id in frozen["audit_minibatch_ids"]:
        generator = torch.Generator(device="cpu").manual_seed(int(audit_id))
        permutation = torch.randperm(len(dataset), generator=generator)
        indices = permutation[:construction["batch_size"]].tolist()
        samples = [dataset[index] for index in indices]
        images = torch.stack([torch.as_tensor(item[0]) for item in samples])
        labels = torch.stack([torch.as_tensor(item[1]) for item in samples])
        batches.append({"images": images, "labels": labels})
        index_batches.append({"audit_id": audit_id, "indices": indices})
    payload = {
        "schema": "ect.operator-clock.frozen-batches/v1",
        "dataset_path_at_creation": str(args.data.resolve()),
        "dataset_sha256": actual_sha,
        "construction": construction,
        "index_batches": index_batches,
        "batches": batches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    receipt = {
        key: value for key, value in payload.items() if key != "batches"
    }
    receipt["batch_file"] = str(args.out.resolve())
    receipt["batch_file_sha256"] = cli_common.sha256_file(args.out)
    write_json(args.out.with_suffix(args.out.suffix + ".json"), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
