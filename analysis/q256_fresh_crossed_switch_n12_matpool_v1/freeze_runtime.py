#!/usr/bin/env python3
"""Bind the rebuilt Conda runtime to immutable lock and archive artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--explicit-lock", type=Path, required=True)
    parser.add_argument("--pip-freeze", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--pip-index-url", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    if probe.get("status") != "PASS" or probe.get("cuda_device_count") != 6:
        raise RuntimeError("runtime probe must PASS on all six GPUs")
    prefix = args.prefix.resolve(strict=True)
    if not (prefix / "bin" / "python").is_file():
        raise RuntimeError("runtime prefix has no Python executable")
    payload = {
        "schema": "ect.q256.rebuilt-runtime/v1", "status": "PASS",
        "environment_kind": "conda-pack-archive-plus-frozen-prefix",
        "environment_prefix": str(prefix),
        "environment_archive": artifact(args.archive),
        "explicit_lock": artifact(args.explicit_lock),
        "pip_freeze": artifact(args.pip_freeze),
        "requirements": artifact(args.requirements),
        "pip_index_url": args.pip_index_url,
        "probe_receipt": artifact(args.probe),
        "probe": probe,
        "old_sif_reused": False,
        "requires_runtime_specific_exact_parity": True,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "manifest": str(args.output),
                      "sha256": hashlib.sha256(encoded).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
