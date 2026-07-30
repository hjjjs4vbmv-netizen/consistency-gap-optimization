#!/usr/bin/env python3
"""Validate the fixed-seed sampling acceptance artifact from a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SEEDS = list(range(64))
EXPECTED_MODES = {
    "nfe1": {"nfe": 1, "mid_t": []},
    "nfe2": {"nfe": 2, "mid_t": [0.821]},
}


def fail(message: str) -> None:
    raise SystemExit(f"[verify_fixed_seed_determinism] ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(result_dir: Path) -> dict:
    path = result_dir / "metadata.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read metadata {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"metadata must be a JSON object: {path}")
    return payload


def verify_manifest(result_dir: Path) -> int:
    manifest_path = result_dir / "sha256_manifest.txt"
    try:
        lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    except OSError as exc:
        fail(f"cannot read manifest {manifest_path}: {exc}")
    if not lines:
        fail("manifest is empty")
    names = set()
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError:
            fail(f"invalid manifest line: {line!r}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            fail(f"invalid manifest SHA256: {digest!r}")
        if name in names:
            fail(f"duplicate manifest entry: {name}")
        names.add(name)
        path = result_dir / name
        if not path.is_file():
            fail(f"manifest file is missing: {path}")
        if sha256_file(path) != digest:
            fail(f"manifest SHA256 mismatch: {path}")
    return len(names)


def verify_mode_files(result_dir: Path, mode_name: str) -> None:
    image_dir = result_dir / mode_name / "images"
    actual = {path.name for path in image_dir.glob("seed*.png")}
    expected = {f"seed{seed:06d}.png" for seed in EXPECTED_SEEDS}
    if actual != expected:
        fail(
            f"{mode_name} image set is not exactly seeds 0-63; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if not (result_dir / mode_name / "grid_8x8.png").is_file():
        fail(f"{mode_name} grid is missing")


def verify(result_dir: Path) -> dict:
    metadata = load_metadata(result_dir)
    if metadata.get("seed_list") != EXPECTED_SEEDS or metadata.get("seed_count") != 64:
        fail("fixed seed list must be exactly 0-63")
    if metadata.get("nfe_modes") != [1, 2]:
        fail("fixed seed evaluation must include NFE=1 and NFE=2")
    if metadata.get("mid_t_by_mode") != {
        name: config["mid_t"] for name, config in EXPECTED_MODES.items()
    }:
        fail("NFE/mid_t contract does not match the frozen protocol")
    if metadata.get("precision") != "fp32":
        fail("fixed seed acceptance requires FP32")
    if metadata.get("model_forward_batch_size") != 1:
        fail("model forward batch size must be one")
    if metadata.get("work_group_sizes_verified") != [8, 16]:
        fail("work-group verification must be exactly 8 and 16")
    if metadata.get("repeat_runs_verified", 0) < 2 or metadata.get("determinism_passed") is not True:
        fail("repeated-run determinism did not pass")
    if metadata.get("image_count_by_mode") != {"nfe1": 64, "nfe2": 64}:
        fail("each NFE mode must contain 64 images")
    if metadata.get("image_count_total") != 128:
        fail("fixed seed evaluation must contain 128 images total")
    for mode_name in EXPECTED_MODES:
        verify_mode_files(result_dir, mode_name)
    manifest_entry_count = verify_manifest(result_dir)
    return {
        "schema_version": 1,
        "result_directory": str(result_dir),
        "checkpoint_id": metadata.get("checkpoint_id"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "nfe_modes": metadata["nfe_modes"],
        "mid_t_by_mode": metadata["mid_t_by_mode"],
        "seed_list": metadata["seed_list"],
        "work_group_sizes_verified": metadata["work_group_sizes_verified"],
        "repeat_runs_verified": metadata["repeat_runs_verified"],
        "manifest_entry_count": manifest_entry_count,
        "status": "passed",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result_dir = args.result_dir.expanduser().resolve()
    report = verify(result_dir)
    report_path = args.report.expanduser().resolve() if args.report else result_dir / "determinism_verification.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Fixed-seed determinism passed: {report_path}")


if __name__ == "__main__":
    main()
