#!/usr/bin/env python3
"""Validate three stateful-audit receipts and write a 3-seed geometry table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (3, 4, 5)
PAIRING_FIELDS = ("same_minibatch", "same_t", "same_noise", "same_dropout_rng_state")
METRICS = ("R_grad", "R_opt", "c_K_star", "s_K_star", "h_dispersion_rms")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(name: str, value: Any, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{name} is missing or non-numeric") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise SystemExit(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SystemExit("manifest schema_version must be 1")
    if float(manifest.get("design", {}).get("state_kimg", -1)) != 256.0:
        raise SystemExit("summary requires the frozen 256-kimg endpoint design")
    cells = manifest.get("endpoint_cells", [])
    by_seed = {cell.get("training_seed"): cell for cell in cells}
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS or len(by_seed) != len(cells):
        raise SystemExit("manifest must contain exactly one seed 3, seed 4, and seed 5 cell")
    for seed, cell in by_seed.items():
        quality = cell.get("quality_control", {})
        finite(f"seed {seed} quality FID delta", quality.get("fid5k_delta_C_minus_B"))
        finite(f"seed {seed} quality KID delta", quality.get("kid5k_delta_C_minus_B"))
        for name in ("training_state", "checkpoint"):
            expected = str(cell.get(f"expected_{name}_sha256", ""))
            if len(expected) != 64 or any(letter not in "0123456789abcdef" for letter in expected):
                raise SystemExit(f"seed {seed} has no valid expected {name} SHA256")
    return by_seed


def read_receipt(path: Path, *, seed: int, cell: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing seed {seed} receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("gains") != [1.0, 1.3]:
        raise SystemExit(f"seed {seed} receipt lacks the required g=1.0/g=1.3 pair")
    stateful = receipt.get("stateful_radam", {})
    if not stateful.get("moments_nontrivial") or not stateful.get("gradscaler_restored"):
        raise SystemExit(f"seed {seed} receipt lacks nontrivial RAdam moments or restored GradScaler")
    if int(stateful.get("n_K", 0)) < 6:
        raise SystemExit(f"seed {seed} receipt does not pass the nonzero-state step gate")
    randomness = receipt.get("randomness_contract", {})
    if not all(randomness.get(field) is True for field in PAIRING_FIELDS):
        raise SystemExit(f"seed {seed} receipt violates the paired-randomness contract")
    if receipt.get("source_state_non_committing", {}).get("preserved") is not True:
        raise SystemExit(f"seed {seed} receipt did not preserve its source state")
    branches = receipt.get("branches", [])
    if len(branches) != 2 or any(branch.get("step_skipped") for branch in branches):
        raise SystemExit(f"seed {seed} receipt has a skipped or incomplete virtual optimizer step")
    whole = receipt.get("whole_model", {})
    if whole.get("gauge_defined") is not True:
        raise SystemExit(f"seed {seed} optimizer geometry is undefined: {whole.get('gauge_error')}")
    if whole.get("H_K_equals_R_opt_identity") is not True:
        raise SystemExit(f"seed {seed} receipt fails its H_K = R_opt identity check")
    if float(receipt.get("provenance", {}).get("state_kimg", -1)) != 256.0:
        raise SystemExit(f"seed {seed} receipt is not the 256-kimg endpoint")
    provenance = receipt["provenance"]
    for name in ("training_state", "checkpoint"):
        if provenance.get(f"{name}_sha256") != cell[f"expected_{name}_sha256"]:
            raise SystemExit(f"seed {seed} receipt {name} SHA256 does not match the frozen manifest")
    quality = cell["quality_control"]
    dispersion_energy = finite("on-support h_i dispersion energy", whole.get("on_support_gauge_dispersion_energy"))
    return {
        "training_seed": seed,
        "run_id": cell["run_id"],
        "quality_delta_ctrl_fid5k": float(quality["fid5k_delta_C_minus_B"]),
        "quality_delta_ctrl_kid5k": float(quality["kid5k_delta_C_minus_B"]),
        "R_grad": finite("R_grad", whole.get("R_grad")),
        "R_opt": finite("R_opt", whole.get("R_opt")),
        "c_K_star": finite("c_K_star", whole.get("c_K_star"), positive=True),
        "s_K_star": finite("s_K_star", whole.get("s_K_star"), positive=True),
        "h_dispersion_rms": math.sqrt(dispersion_energy),
        "off_support_candidate_energy_exact": finite("off-support candidate energy", whole.get("off_support_candidate_energy_exact")),
        "optimizer_steps": int(stateful["n_K"]),
        "receipt_sha256": sha256(path),
        "receipt_path": str(path),
    }


def with_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    reducers = (("mean", statistics.fmean), ("sample_sd", statistics.stdev),
                ("range", lambda values: max(values) - min(values)))
    for name, reducer in reducers:
        aggregate: dict[str, Any] = {"training_seed": name, "run_id": "descriptive"}
        for metric in METRICS:
            aggregate[metric] = reducer([float(row[metric]) for row in rows])
        aggregate.update({
            "quality_delta_ctrl_fid5k": None,
            "quality_delta_ctrl_kid5k": None,
            "off_support_candidate_energy_exact": None,
            "optimizer_steps": None,
            "receipt_sha256": None,
            "receipt_path": None,
        })
        output.append(aggregate)
    return output


def write_outputs(rows: list[dict[str, Any]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    columns = [
        "training_seed", "run_id", "quality_delta_ctrl_fid5k", "quality_delta_ctrl_kid5k",
        "R_grad", "R_opt", "c_K_star", "s_K_star", "h_dispersion_rms",
        "off_support_candidate_energy_exact", "optimizer_steps", "receipt_sha256", "receipt_path",
    ]
    aggregates = with_aggregates(rows)
    with (out / "optimizer_geometry_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggregates)

    lines = [
        "# Cross-seed optimizer geometry",
        "",
        "Each row is one non-committing same-state current-gap RAdam audit at the 256-kimg Arm-A endpoint.",
        "The paired branches share minibatch, time/noise, and dropout RNG state and differ only by `g=1.0` versus `g=1.3`.",
        "",
        "`quality delta ctrl` is pre-existing endpoint-quality context from the three disjoint 5k blocks: `C−B` for FID; KID.",
        "It is not an outcome of this optimizer audit. Negative FID/KID is better. `h_i dispersion RMS` is the square root of the receipt's on-support energy-weighted dispersion.",
        "`H_K = R_opt` remains an identity check, not independent evidence.",
        "",
        "| seed | quality delta ctrl (FID; KID) | R_grad | R_opt | c_K* | s_K* | h_i dispersion RMS | off-support energy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| seed {seed} | {fid:+.2f}; {kid:+.6f} | {rgrad:.6f} | {ropt:.6f} | {cstar:.6f} | {sstar:.6f} | {hdisp:.6f} | {off:.2e} |".format(
                seed=row["training_seed"], fid=row["quality_delta_ctrl_fid5k"], kid=row["quality_delta_ctrl_kid5k"],
                rgrad=row["R_grad"], ropt=row["R_opt"], cstar=row["c_K_star"], sstar=row["s_K_star"],
                hdisp=row["h_dispersion_rms"], off=row["off_support_candidate_energy_exact"],
            )
        )
    for row in aggregates[3:]:
        lines.append(
            "| {label} | — | {rgrad:.6f} | {ropt:.6f} | {cstar:.6f} | {sstar:.6f} | {hdisp:.6f} | — |".format(
                label=row["training_seed"], rgrad=row["R_grad"], ropt=row["R_opt"],
                cstar=row["c_K_star"], sstar=row["s_K_star"], hdisp=row["h_dispersion_rms"],
            )
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This table answers whether measured local optimizer geometry varies across the three endpoints while their quality-control signs differ.",
        "It does not establish that any geometry difference causes endpoint quality. A claim of numerical similarity must report the displayed range and a prospectively stated tolerance.",
    ])
    (out / "OPTIMIZER_GEOMETRY_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cells = load_manifest(args.manifest)
    rows = [read_receipt(args.audit_root / f"seed{seed}" / "radam_update_audit_stateful.json", seed=seed, cell=cells[seed]) for seed in EXPECTED_SEEDS]
    write_outputs(rows, args.out)
    print(f"wrote {args.out / 'optimizer_geometry_table.csv'}")
    print(f"wrote {args.out / 'OPTIMIZER_GEOMETRY_TABLE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
