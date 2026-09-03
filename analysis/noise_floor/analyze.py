#!/usr/bin/env python3
"""Summarize checkpoint, exact-contrast, and rotation block variation."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .run_matrix import build_jobs, load_manifest
from .validation import validate_receipt


METRICS = ("fid", "kid")
ROTATION_FIELDS = (
    "rotation_id", "early", "late", "nfe", "metric", "anchor_B0_rotation",
    "B1", "B2", "B3", "B4", "B5", "rotation_k", "status",
)


def transformed(metric, value):
    if value is None:
        return None
    if metric == "fid":
        if value <= 0:
            raise RuntimeError("FID must be positive before log transform")
        return math.log(value)
    return value


def summarize(values):
    complete = all(value is not None for value in values)
    if not complete:
        return {"n_valid": sum(value is not None for value in values),
                "mean": None, "sd": None, "two_sd": None, "status": "INCOMPLETE"}
    numeric = [float(value) for value in values]
    sd = statistics.stdev(numeric)
    return {"n_valid": 5, "mean": statistics.mean(numeric), "sd": sd,
            "two_sd": 2 * sd, "status": "COMPLETE"}


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        columns = list(rows[0]) if rows else list(fieldnames or ())
        if not columns:
            raise RuntimeError("empty CSV requires field names")
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_values(manifest):
    root = Path(manifest["output_root"])
    values = {}
    statuses = {}
    for job in build_jobs(manifest):
        receipt_path = root / "receipts" / f"{job['job_id']}.json"
        if not receipt_path.is_file():
            status, receipt = "MISSING", None
        else:
            receipt = json.loads(receipt_path.read_text())
            status = receipt["status"]
            if status == "PASS":
                validate_receipt(manifest, job, receipt)
        for metric in METRICS:
            key = (job["checkpoint"]["id"], job["nfe"], metric, job["block"]["id"])
            statuses[key] = status
            if status == "PASS":
                raw_name = "fid50k_full" if metric == "fid" else "kid50k_full"
                values[key] = transformed(metric, float(receipt["metrics"][raw_name]))
            else:
                values[key] = None
    return values, statuses


def checkpoint_rows(manifest, values):
    rows = []
    for checkpoint in manifest["checkpoints"]:
        for nfe in (1, 2):
            for metric in METRICS:
                blocks = [values[(checkpoint["id"], nfe, metric, f"B{index}")]
                          for index in range(1, 6)]
                row = {
                    "checkpoint_id": checkpoint["id"], "identity": checkpoint["identity"],
                    "cohort": checkpoint["cohort"], "nfe": nfe, "metric": metric,
                    "scale": "log" if metric == "fid" else "raw",
                    "anchor_B0": transformed(metric, checkpoint["b0"][str(nfe)][metric]),
                    **{f"B{index}": blocks[index - 1] for index in range(1, 6)},
                    **summarize(blocks),
                }
                rows.append(row)
    expected = manifest["expected_checkpoints"] * 4
    if len(rows) != expected:
        raise RuntimeError("unexpected checkpoint output row count")
    return rows


def contrast_rows(manifest, values):
    checkpoints = {item["id"]: item for item in manifest["checkpoints"]}
    rows, block_values = [], {}
    for contrast in manifest["contrasts"]:
        lhs, rhs = checkpoints[contrast["lhs"]], checkpoints[contrast["rhs"]]
        for nfe in (1, 2):
            for metric in METRICS:
                lhs_anchor = transformed(metric, lhs["b0"][str(nfe)][metric])
                rhs_anchor = transformed(metric, rhs["b0"][str(nfe)][metric])
                anchor = (None if lhs_anchor is None or rhs_anchor is None
                          else lhs_anchor - rhs_anchor)
                blocks = []
                for index in range(1, 6):
                    block = f"B{index}"
                    left = values[(lhs["id"], nfe, metric, block)]
                    right = values[(rhs["id"], nfe, metric, block)]
                    delta = None if left is None or right is None else left - right
                    block_values[(contrast["id"], nfe, metric, block)] = delta
                    blocks.append(delta)
                summary = summarize(blocks)
                same_sign = None
                if (summary["status"] == "COMPLETE" and anchor is not None
                        and sign(anchor) != 0):
                    same_sign = sum(sign(value) == sign(anchor) for value in blocks)
                ratio = None
                if (summary["status"] == "COMPLETE" and anchor is not None
                        and summary["two_sd"] > 0):
                    ratio = abs(anchor) / summary["two_sd"]
                rows.append({
                    "contrast_id": contrast["id"], "lhs": lhs["id"], "rhs": rhs["id"],
                    "nfe": nfe, "metric": metric,
                    "scale": "log" if metric == "fid" else "raw", "anchor_B0": anchor,
                    **{f"B{index}": blocks[index - 1] for index in range(1, 6)},
                    **summary, "same_anchor_sign_k": same_sign,
                    "anchor_abs_over_two_sd": ratio,
                })
    expected = manifest["expected_contrasts"] * 4
    if len(rows) != expected:
        raise RuntimeError("unexpected contrast output row count")
    return rows, block_values


def rotation_rows(manifest, contrast_rows_, block_values):
    contrast_index = {(row["contrast_id"], row["nfe"], row["metric"]): row
                      for row in contrast_rows_}
    rows = []
    for rotation in manifest["rotations"]:
        nfes = (rotation["nfe"],) if "nfe" in rotation else (1, 2)
        metrics = (rotation["metric"],) if "metric" in rotation else METRICS
        for nfe in nfes:
            for metric in metrics:
                early = contrast_index[(rotation["early"], nfe, metric)]
                late = contrast_index[(rotation["late"], nfe, metric)]
                if early["anchor_B0"] is None or late["anchor_B0"] is None:
                    anchor_rotation = None
                else:
                    anchor_rotation = int(early["anchor_B0"] * late["anchor_B0"] < 0)
                block_results = []
                for index in range(1, 6):
                    block = f"B{index}"
                    first = block_values[(rotation["early"], nfe, metric, block)]
                    second = block_values[(rotation["late"], nfe, metric, block)]
                    if first is None or second is None:
                        block_results.append(None)
                    elif first == 0 or second == 0:
                        block_results.append("ZERO")
                    else:
                        block_results.append(int(first * second < 0))
                complete = all(value is not None for value in block_results)
                rotation_k = None if not complete else sum(value == 1 for value in block_results)
                rows.append({
                    "rotation_id": rotation["id"], "early": rotation["early"],
                    "late": rotation["late"], "nfe": nfe, "metric": metric,
                    "anchor_B0_rotation": anchor_rotation,
                    **{f"B{index}": block_results[index - 1] for index in range(1, 6)},
                    "rotation_k": rotation_k, "status": "COMPLETE" if complete else "INCOMPLETE",
                })
    expected = manifest["expected_rotation_rows"]
    if len(rows) != expected:
        raise RuntimeError("unexpected rotation output row count")
    return rows


def report(manifest, checkpoints, contrasts, rotations):
    completed = sum(row["status"] == "COMPLETE" for row in checkpoints)
    lines = [
        "# Generation-block sensitivity", "",
        f"Checkpoint summaries complete: {completed}/{len(checkpoints)}.", "",
        "B0 is a historical anchor and is excluded from all means and SDs. "
        "FID is on the natural-log scale; KID is on the raw scale. The 2SD value "
        "is descriptive and is not a TIE or confidence rule.", "",
        "## Exact contrasts", "",
        "| Contrast | NFE | Metric | Status | Same B0 sign | 2SD |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in contrasts:
        same = "" if row["same_anchor_sign_k"] is None else f"{row['same_anchor_sign_k']}/5"
        two_sd = "" if row["two_sd"] is None else f"{row['two_sd']:.9g}"
        lines.append(f"| {row['contrast_id']} | {row['nfe']} | {row['metric']} | "
                     f"{row['status']} | {same} | {two_sd} |")
    lines.extend(["", "## Rotation", "",
                  "| Rotation | NFE | Metric | B0 | New blocks | Status |",
                  "|---|---:|---|---:|---:|---|"])
    for row in rotations:
        count = "" if row["rotation_k"] is None else f"{row['rotation_k']}/5"
        lines.append(f"| {row['rotation_id']} | {row['nfe']} | {row['metric']} | "
                     f"{row['anchor_B0_rotation']} | {count} | {row['status']} |")
    lines.extend(["", "## Not evaluated", ""])
    if manifest["not_evaluated"]:
        lines.extend(f"- `{item['id']}`: {item['reason']}" for item in manifest["not_evaluated"])
    else:
        lines.append("None.")
    lines.extend(["", "All summaries are contrast-specific, post-seal descriptive "
                  "checks and do not modify frozen inference.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest.resolve())
    values, _ = load_values(manifest)
    checkpoints = checkpoint_rows(manifest, values)
    contrasts, block_values = contrast_rows(manifest, values)
    rotations = rotation_rows(manifest, contrasts, block_values)
    write_csv(args.outdir / "checkpoint_variation.csv", checkpoints)
    write_csv(args.outdir / "contrast_variation.csv", contrasts)
    write_csv(args.outdir / "rotation_variation.csv", rotations,
              fieldnames=ROTATION_FIELDS)
    write_csv(args.outdir / "not_evaluated.csv", manifest["not_evaluated"],
              fieldnames=("id", "reason"))
    (args.outdir / "NOISE_FLOOR.md").write_text(
        report(manifest, checkpoints, contrasts, rotations))
    print(json.dumps({"checkpoint_rows": len(checkpoints),
                      "contrast_rows": len(contrasts),
                      "rotation_rows": len(rotations),
                      "not_evaluated_rows": len(manifest["not_evaluated"])}))


if __name__ == "__main__":
    main()
