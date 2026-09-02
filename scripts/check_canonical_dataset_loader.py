#!/usr/bin/env python3
"""Exhaustive loader smoke for the canonical CIFAR-10 training archive."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SmokeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SmokeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def run_smoke(args: argparse.Namespace) -> dict:
    repo = args.repo.resolve()
    dataset_path = args.dataset.resolve()
    require(repo.is_dir(), f"missing repository: {repo}")
    require(dataset_path.is_file(), f"missing dataset: {dataset_path}")
    require(sha256_file(dataset_path) == args.expected_dataset_sha256, "canonical dataset SHA256 mismatch")

    loader_path = repo / "training/dataset.py"
    require(loader_path.is_file(), f"missing loader: {loader_path}")
    loader_sha256 = sha256_file(loader_path)
    require(loader_sha256 == args.expected_loader_sha256, "dataset loader SHA256 mismatch")
    git_commit = git_output(repo, "rev-parse", "HEAD")

    sys.path.insert(0, str(repo))
    from training.dataset import ImageFolderDataset

    dataset = ImageFolderDataset(
        path=str(dataset_path),
        resolution=32,
        use_pyspng=True,
        use_labels=True,
        xflip=False,
        cache=False,
    )
    try:
        require(len(dataset) == 50000, f"expected 50000 samples, found {len(dataset)}")
        require(dataset.image_shape == [3, 32, 32], f"unexpected image shape: {dataset.image_shape}")
        require(dataset.label_shape == [10], f"unexpected label shape: {dataset.label_shape}")
        require(dataset.has_onehot_labels, "dataset.json does not provide integer class labels")

        filename_digest = hashlib.sha256()
        image_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        normalized_digest = hashlib.sha256()
        histogram: Counter[int] = Counter()
        normalized_min = float("inf")
        normalized_max = float("-inf")

        for index, filename in enumerate(dataset._image_fnames):
            image, onehot = dataset[index]
            details = dataset.get_details(index)
            require(image.shape == (3, 32, 32), f"sample {index} is not RGB 32x32")
            require(image.dtype == np.uint8, f"sample {index} is not uint8")
            require(details.xflip is False, f"sample {index} unexpectedly uses xflip")
            require(onehot.shape == (10,) and onehot.dtype == np.float32, f"sample {index} has invalid one-hot label")
            require(float(onehot.sum()) == 1.0, f"sample {index} label is not one-hot")
            label = int(details.raw_label)
            require(0 <= label < 10 and onehot[label] == 1.0, f"sample {index} label mapping mismatch")

            normalized = image.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
            normalized_min = min(normalized_min, float(normalized.min()))
            normalized_max = max(normalized_max, float(normalized.max()))
            filename_digest.update(filename.encode("utf-8") + b"\0")
            image_digest.update(image.tobytes(order="C"))
            label_digest.update(label.to_bytes(8, byteorder="little", signed=False))
            normalized_digest.update(normalized.tobytes(order="C"))
            histogram[label] += 1

        require(dict(sorted(histogram.items())) == {index: 5000 for index in range(10)}, f"unexpected class histogram: {dict(histogram)}")
        require(normalized_min == -1.0 and normalized_max == 1.0, "preprocessing range is not exactly [-1, 1]")
        return {
            "schema": "ect.canonical-dataset-loader-smoke/v1",
            "status": "PASS",
            "sample_count": len(dataset),
            "class_count": 10,
            "class_histogram": {str(key): value for key, value in sorted(histogram.items())},
            "image_layout": "CHW",
            "image_shape": [3, 32, 32],
            "image_dtype": "uint8",
            "color_space": "RGB",
            "xflip": False,
            "label_source": "dataset.json",
            "label_mapping": "sorted image filename -> integer class -> one-hot[10]",
            "preprocessing": "float32(uint8_image) / 127.5 - 1.0",
            "preprocessed_dtype": "float32",
            "preprocessed_range": [normalized_min, normalized_max],
            "dataset_path": str(dataset_path),
            "dataset_sha256": args.expected_dataset_sha256,
            "loader_path": str(loader_path),
            "loader_sha256": loader_sha256,
            "loader_git_commit": git_commit,
            "sorted_filename_stream_sha256": filename_digest.hexdigest(),
            "decoded_chw_image_stream_sha256": image_digest.hexdigest(),
            "integer_label_stream_sha256": label_digest.hexdigest(),
            "preprocessed_float32_stream_sha256": normalized_digest.hexdigest(),
            "q256_detector_sha256": args.detector_sha256,
            "q256_fid_reference_sha256": args.fid_reference_sha256,
            "q256_kid_reference_sha256": args.kid_reference_sha256,
        }
    finally:
        dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-loader-sha256", required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--fid-reference-sha256", required=True)
    parser.add_argument("--kid-reference-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        print(json.dumps(run_smoke(parse_args()), indent=2, sort_keys=True))
        return 0
    except SmokeError as exc:
        print(f"[canonical-dataset-smoke] NO-GO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
