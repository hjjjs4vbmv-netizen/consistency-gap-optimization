#!/usr/bin/env python3
"""Normalize the q256 four-arm FID-50k endpoint records for paper assets.

The normalizer accepts one receipt-backed raw CSV per frozen budget.  It never
infers missing cells and it only emits a two-budget source when every required
seed, arm, and NFE record is present and marked PASS.  This makes it impossible
to accidentally combine the historical FID-5k checkpoints with FID-50k
endpoints in the compute-to-quality pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PREFIX = "normalize_q256_two_budget_fid50k"
RAW_REQUIRED = {"seed", "arm", "nfe", "fid50k_full", "status"}


def fail(message: str) -> None:
    raise SystemExit("[{}] ERROR: {}".format(PREFIX, message))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail("cannot read config {}: {}".format(path, exc))
        raise exc
    required = {
        "schema_version", "kind", "budgets_kimg", "training_seeds", "nfes", "arms",
        "metric_name", "sample_count", "generation_seed_range", "metric_seed",
        "evidence_class", "evaluation_contract", "analysis_track",
    }
    missing = required - set(config)
    if missing:
        fail("config is missing {}".format(sorted(missing)))
    if config["schema_version"] != 1 or config["kind"] != "q256_two_budget_fid50k_source":
        fail("config must declare schema_version=1 and kind='q256_two_budget_fid50k_source'")
    if config["metric_name"] != "fid50k_full":
        fail("this normalizer only accepts metric_name='fid50k_full'")
    if config["analysis_track"] != "two_budget_endpoint":
        fail("analysis_track must be 'two_budget_endpoint'")
    if not isinstance(config["budgets_kimg"], list) or len(config["budgets_kimg"]) != 2:
        fail("budgets_kimg must contain exactly two endpoints")
    try:
        config["budgets_kimg"] = [int(value) for value in config["budgets_kimg"]]
        config["training_seeds"] = [int(value) for value in config["training_seeds"]]
        config["nfes"] = [int(value) for value in config["nfes"]]
        config["sample_count"] = int(config["sample_count"])
        config["metric_seed"] = int(config["metric_seed"])
    except (TypeError, ValueError) as exc:
        fail("budgets, seeds, NFEs, sample_count, and metric_seed must be integers")
        raise exc
    if (config["budgets_kimg"] != sorted(config["budgets_kimg"])
            or len(set(config["budgets_kimg"])) != 2 or any(value <= 0 for value in config["budgets_kimg"])):
        fail("budgets_kimg must contain two strictly increasing positive values")
    if (not config["training_seeds"] or len(set(config["training_seeds"])) != len(config["training_seeds"])
            or any(value < 0 for value in config["training_seeds"])):
        fail("training_seeds must be a non-empty unique list of non-negative integers")
    if (not config["nfes"] or len(set(config["nfes"])) != len(config["nfes"])
            or any(value < 1 for value in config["nfes"])):
        fail("nfes must be a non-empty unique list of positive integers")
    if not isinstance(config["arms"], dict) or set(config["arms"]) != {"A", "B", "C", "D"}:
        fail("arms must map exactly A, B, C, and D to output method names")
    if any(not isinstance(method, str) or not method.strip() for method in config["arms"].values()):
        fail("each arm must map to a non-empty output method name")
    if len(set(config["arms"].values())) != 4:
        fail("the four output method names must be distinct")
    required_text = ("generation_seed_range", "evidence_class", "evaluation_contract")
    if any(not isinstance(config[field], str) or not config[field].strip() for field in required_text):
        fail("generation_seed_range, evidence_class, and evaluation_contract must be non-empty strings")
    if config["sample_count"] != 50000 or config["metric_seed"] < 0:
        fail("the source must identify the formal 50,000-sample FID protocol and a non-negative metric seed")
    return config


def parse_budget_inputs(raw_values: list[str], expected_budgets: list[int]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for raw in raw_values:
        if "=" not in raw:
            fail("--budget-input must use BUDGET=PATH")
        budget_raw, path_raw = raw.split("=", 1)
        try:
            budget = int(budget_raw)
        except ValueError as exc:
            fail("budget in --budget-input must be an integer: {}".format(budget_raw))
            raise exc
        if budget in parsed or not path_raw.strip():
            fail("each --budget-input requires one unique non-empty path")
        parsed[budget] = Path(path_raw).expanduser().resolve()
    if set(parsed) != set(expected_budgets):
        fail("budget inputs must match frozen budgets {}; got {}".format(expected_budgets, sorted(parsed)))
    return parsed


def read_raw(path: Path, budget: int, config: dict) -> list[dict]:
    if not path.is_file():
        fail("raw CSV for {} kimg does not exist: {}".format(budget, path))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = RAW_REQUIRED - fields
        if missing:
            fail("raw CSV {} is missing {}".format(path, sorted(missing)))
        raw_rows = list(reader)
    index: dict[tuple[int, str, int], dict] = {}
    for row_number, row in enumerate(raw_rows, start=2):
        try:
            seed = int(row["seed"])
            nfe = int(row["nfe"])
            value = float(row["fid50k_full"])
        except (TypeError, ValueError) as exc:
            fail("{} row {} has an invalid seed, NFE, or FID-50k value".format(path, row_number))
            raise exc
        arm = row["arm"].strip()
        if row["status"].strip().upper() != "PASS":
            fail("{} row {} is not a PASS receipt".format(path, row_number))
        if not math.isfinite(value):
            fail("{} row {} has a non-finite FID-50k value".format(path, row_number))
        if seed not in config["training_seeds"] or arm not in config["arms"] or nfe not in config["nfes"]:
            fail("{} row {} is outside the frozen seed/arm/NFE matrix".format(path, row_number))
        key = (seed, arm, nfe)
        if key in index:
            fail("{} has a duplicate endpoint record {}".format(path, key))
        index[key] = {
            "seed": seed,
            "arm": arm,
            "nfe": nfe,
            "metric_value": value,
            "receipt_sha256": row.get("receipt_sha256", "").strip(),
            "artifacts_tree_sha256": row.get("artifacts_tree_sha256", "").strip(),
            "generated_features_sha256": row.get("generated_features_sha256", "").strip(),
        }
    expected = {
        (seed, arm, nfe)
        for seed in config["training_seeds"] for arm in ("A", "B", "C", "D") for nfe in config["nfes"]
    }
    if set(index) != expected:
        fail(
            "{} is not a complete frozen {}-kimg matrix; missing={}, extra={}".format(
                path, budget, sorted(expected - set(index)), sorted(set(index) - expected),
            ),
        )
    return [index[key] for key in sorted(index)]


def write_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "method", "training_seed", "budget_kimg", "nfe", "metric_name", "metric_value",
        "checkpoint_sha256", "source_receipt_sha256", "source_artifacts_tree_sha256",
        "source_generated_features_sha256", "sample_count", "generation_seed_range", "metric_seed",
        "evidence_class", "evaluation_contract", "analysis_track",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget-input", action="append", required=True, metavar="BUDGET=PATH")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = read_config(config_path)
    inputs = parse_budget_inputs(args.budget_input, config["budgets_kimg"])
    records = []
    for budget in config["budgets_kimg"]:
        for raw in read_raw(inputs[budget], budget, config):
            records.append({
                "method": config["arms"][raw["arm"]],
                "training_seed": raw["seed"],
                "budget_kimg": budget,
                "nfe": raw["nfe"],
                "metric_name": config["metric_name"],
                "metric_value": "{:.12g}".format(raw["metric_value"]),
                "checkpoint_sha256": "",
                "source_receipt_sha256": raw["receipt_sha256"],
                "source_artifacts_tree_sha256": raw["artifacts_tree_sha256"],
                "source_generated_features_sha256": raw["generated_features_sha256"],
                "sample_count": config["sample_count"],
                "generation_seed_range": config["generation_seed_range"],
                "metric_seed": config["metric_seed"],
                "evidence_class": config["evidence_class"],
                "evaluation_contract": config["evaluation_contract"],
                "analysis_track": config["analysis_track"],
            })
    output = args.out_csv.resolve()
    write_csv(output, records)
    manifest = {
        "kind": "q256_two_budget_fid50k_normalized_source",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "raw_inputs": {
            str(budget): {"path": str(inputs[budget]), "sha256": sha256(inputs[budget])}
            for budget in config["budgets_kimg"]
        },
        "output_csv": str(output),
        "output_csv_sha256": sha256(output),
        "record_count": len(records),
        "frozen_protocol": {
            field: config[field]
            for field in ("metric_name", "sample_count", "generation_seed_range", "metric_seed",
                          "evaluation_contract", "analysis_track")
        },
        "prohibition": "No FID-5k record is accepted or emitted by this normalizer.",
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Normalized {} FID-50k records to {}".format(len(records), output))


if __name__ == "__main__":
    main()
