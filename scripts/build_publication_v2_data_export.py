#!/usr/bin/env python3
"""Build a history-free publication-v2 data payload using same-filesystem links."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    copied: set[str] = set()
    for cell in manifest["cells"]:
        records = [cell["artifacts"]["samples"]]
        records.extend(cell["artifacts"]["features"].values())
        records.extend(cell["artifacts"]["metric_receipts"].values())
        for record in records:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe manifest path: {relative}")
            if relative.as_posix() in copied:
                continue
            src = source / relative
            if sha256_file(src) != record["sha256"]:
                raise RuntimeError(f"source hash mismatch: {relative}")
            dst = output / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
            copied.add(relative.as_posix())
    bundle = output / "results" / "publication_v2_regenerated"
    bundle.mkdir(parents=True)
    for name in ("README.md", "SHA256SUMS", "blockwise_results.csv",
                 "comparison_to_pr53.csv", "disjoint_block_summary.csv",
                 "publication_v2_cell_manifest.json"):
        shutil.copy2(args.tables / name, bundle / name)
    (output / "README.md").write_text(
        "# Anonymous publication-v2 data payload\n\n"
        "This history-free payload contains 27 exact generated-sample arrays, "
        "54 exact feature arrays, and 54 metric receipts. Paths are relative; "
        "the companion code artifact verifies every hash, shape, dtype, sample "
        "range, and checkpoint binding.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksum_path = output / "RELEASE_SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "files": len(files), "linked_core_files": len(copied)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
