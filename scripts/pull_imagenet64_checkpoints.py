#!/usr/bin/env python3
"""Pull completed ImageNet-64 full-state milestones into private storage."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time


MILESTONES_KIMG = tuple(range(1280, 12801, 1280))


def parse_cells(value):
    cells = tuple(PurePosixPath(item.strip()) for item in value.split(","))
    if not cells or any(
        len(cell.parts) != 2
        or cell.parts[0] not in {"seed101", "seed102", "seed103"}
        or cell.parts[1] not in {"IA", "IB"}
        for cell in cells
    ):
        raise argparse.ArgumentTypeError(
            "--cells must be comma-separated seedNNN/IA or seedNNN/IB paths"
        )
    if len(set(cells)) != len(cells):
        raise argparse.ArgumentTypeError("--cells must not contain duplicates")
    return cells


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssh_base(args):
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-i", str(args.identity),
        "-p", str(args.port),
        args.source,
    ]


def expected_files(
    source_root, cells
):
    return tuple(
        source_root / cell / f"training-state-kimg{kimg:06d}.pt"
        for cell in cells
        for kimg in MILESTONES_KIMG
    )


def remote_sha256(args, path):
    result = subprocess.run(
        [*ssh_base(args), "sha256sum", str(path)],
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )
    digest, reported_path = result.stdout.rstrip("\n").split(maxsplit=1)
    if reported_path.strip() != str(path) or len(digest) != 64:
        raise RuntimeError(f"invalid remote checksum response for {path}")
    return digest


def remote_files(args):
    command = "find {} -type f -name {} -print".format(
        shlex.quote(str(args.source_root)),
        shlex.quote("training-state-kimg*.pt"),
    )
    result = subprocess.run(
        [*ssh_base(args), command],
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
    )
    return {
        PurePosixPath(line)
        for line in result.stdout.splitlines()
        if line
    }


def pull_one(
    args,
    source_path,
    destination_path,
):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination_path.with_name(destination_path.name + ".partial")
    if destination_path.exists():
        local_digest = sha256(destination_path)
    else:
        ssh_transport = " ".join(shlex.quote(item) for item in [
            "ssh", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-i", str(args.identity), "-p", str(args.port),
        ])
        subprocess.run([
            "rsync", "--archive", "--partial", "--append-verify",
            "--protect-args", "-e", ssh_transport,
            f"{args.source}:{source_path}", str(partial_path),
        ], check=True)
        local_digest = sha256(partial_path)
    source_digest = remote_sha256(args, source_path)
    if local_digest != source_digest:
        raise RuntimeError(f"checksum mismatch for {source_path}")
    if not destination_path.exists():
        os.replace(partial_path, destination_path)
    return {
        "source": f"{args.source}:{source_path}",
        "destination": str(destination_path),
        "bytes": destination_path.stat().st_size,
        "sha256": local_digest,
        "verified_unix_time": time.time(),
    }


def append_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="SSH user@host")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--source-root", type=PurePosixPath, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--cells", type=parse_cells, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    expected = expected_files(args.source_root, args.cells)
    receipt_path = args.archive_root / "checkpoint-transfer-receipts.jsonl"
    if args.dry_run:
        print(f"source={args.source} port={args.port} files={len(expected)}")
        for path in expected:
            relative = path.relative_to(args.source_root)
            print(f"{path} -> {args.archive_root / relative}")
        return
    if not args.identity.is_file():
        parser.error(f"missing SSH identity: {args.identity}")

    completed = set()
    while len(completed) < len(expected):
        progress = False
        available = remote_files(args)
        for source_path in expected:
            if source_path in completed or source_path not in available:
                continue
            relative = source_path.relative_to(args.source_root)
            receipt = pull_one(args, source_path, args.archive_root / relative)
            append_receipt(receipt_path, receipt)
            completed.add(source_path)
            progress = True
            print(
                f"verified {len(completed)}/{len(expected)} "
                f"{receipt['sha256']} {relative}",
                flush=True,
            )
        if args.once:
            return
        if len(completed) < len(expected) and not progress:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
