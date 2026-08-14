#!/usr/bin/env python3
"""Run the one-point stateful RAdam audit for seeds 3/4/5 at 256 kimg.

The runner is read-only with respect to training artifacts: each call forks
disposable g=1.0/g=1.3 branches from a restored Arm-A endpoint and writes a
new receipt directory. Use --dry-run locally to inspect the command matrix
without external checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "analysis" / "radam_stateful_update_audit.py"
EXPECTED_SEEDS = (3, 4, 5)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise SystemExit("manifest schema_version must be 1")
    design = manifest.get("design")
    cells = manifest.get("endpoint_cells")
    if not isinstance(design, dict) or not isinstance(cells, list):
        raise SystemExit("manifest requires object design and list endpoint_cells")
    if tuple(sorted(cell.get("training_seed") for cell in cells)) != EXPECTED_SEEDS:
        raise SystemExit("manifest must contain exactly one endpoint for seeds 3, 4, and 5")
    if design.get("reference_arm") != "A" or design.get("reference_gap_scale") != 1.0:
        raise SystemExit("the cross-seed reference must be Arm A at g=1.0")
    if design.get("candidate_gap_scale") != 1.3 or float(design.get("state_kimg", -1)) != 256.0:
        raise SystemExit("the frozen design is a g=1.0/g=1.3 audit at 256 kimg")
    if len({cell.get("run_id") for cell in cells}) != len(cells):
        raise SystemExit("endpoint run_id values must be unique")
    return manifest


def is_placeholder(value: str) -> bool:
    return not value or value.startswith("REPLACE_WITH_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_for_cell(manifest: dict[str, Any], cell: dict[str, Any], *, out: Path, python: str) -> list[str]:
    design = manifest["design"]
    paths = {
        "data": str(manifest.get("data", "")),
        "training_state": str(cell.get("training_state", "")),
        "checkpoint": str(cell.get("checkpoint", "")),
    }
    missing = [name for name, value in paths.items() if is_placeholder(value)]
    if missing:
        raise SystemExit(f"seed {cell['training_seed']} has placeholder path(s): {', '.join(missing)}")
    return [
        python, str(AUDIT),
        "--training-state", paths["training_state"],
        "--checkpoint", paths["checkpoint"],
        "--data", paths["data"],
        "--batch-size", str(design["batch_size"]),
        "--batch-gpu", str(design["batch_gpu"]),
        "--seed", str(design["audit_random_seed"]),
        "--state-kimg", "256",
        "--device", "cuda",
        "--amp" if design.get("amp", True) else "--no-amp",
        "--lr", str(design["learning_rate"]),
        "--betas", ",".join(str(value) for value in design["betas"]),
        "--eps", str(design["eps"]),
        "--support-atol", str(design.get("support_atol", 0.0)),
        "--out", str(out),
    ]


def validate_real_paths(manifest: dict[str, Any], cell: dict[str, Any]) -> None:
    paths = {
        "data": Path(str(manifest.get("data", ""))),
        "training_state": Path(str(cell.get("training_state", ""))),
        "checkpoint": Path(str(cell.get("checkpoint", ""))),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"seed {cell['training_seed']} missing input path(s): {', '.join(missing)}")
    for name in ("training_state", "checkpoint"):
        expected = str(cell.get(f"expected_{name}_sha256", ""))
        if is_placeholder(expected) or len(expected) != 64 or any(letter not in "0123456789abcdef" for letter in expected):
            raise SystemExit(f"seed {cell['training_seed']} has no valid expected {name} SHA256")
        observed = file_sha256(paths[name])
        if observed != expected:
            raise SystemExit(
                f"seed {cell['training_seed']} {name} SHA256 mismatch: expected {expected}, observed {observed}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="new external output root; refuses an existing path")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter with the audited Torch environment")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the endpoint matrix; permits placeholder paths and writes nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.dry_run:
        for cell in sorted(manifest["endpoint_cells"], key=lambda item: item["training_seed"]):
            print(f"seed {cell['training_seed']}: {cell['run_id']}")
            print("  training-state:", cell["training_state"])
            print("  checkpoint:", cell["checkpoint"])
            print("  output:", args.out / f"seed{cell['training_seed']}")
        return 0
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {args.out}")
    planned = []
    for cell in sorted(manifest["endpoint_cells"], key=lambda item: item["training_seed"]):
        point_out = args.out / f"seed{cell['training_seed']}"
        command = command_for_cell(manifest, cell, out=point_out, python=args.python)
        validate_real_paths(manifest, cell)
        planned.append((cell, command))
    args.out.mkdir(parents=True)
    for cell, command in planned:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    (args.out / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("CROSS_SEED_OPTIMIZER_GEOMETRY_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
