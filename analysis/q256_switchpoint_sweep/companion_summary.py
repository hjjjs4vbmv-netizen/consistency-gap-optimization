from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def summarize(decoded: dict, receipts: list[dict]) -> dict:
    blocks: dict[int, dict[str, float]] = {index: {} for index in range(5)}
    for row in decoded["results"]:
        if (row["seed"], row["kimg"], row["trajectory"]) in {
            (81, 1024, "CTRL"), (81, 1024, "BA512")
        } and row["status"] == "PASS":
            blocks[0][row["trajectory"]] = float(row["fid50k_full"])
    failures = []
    for receipt in receipts:
        job = receipt.get("job", {})
        if receipt.get("status") != "PASS":
            failures.append(job or receipt.get("job"))
            continue
        block = int(job["block"])
        blocks[block][job["trajectory"]] = float(receipt["values"]["fid50k_full"])
    values = []
    rows = []
    for block in range(5):
        pair = blocks[block]
        value = None
        if set(pair) == {"CTRL", "BA512"} and all(math.isfinite(v) and v > 0 for v in pair.values()):
            value = math.log(pair["BA512"]) - math.log(pair["CTRL"])
            values.append(value)
        rows.append({"block": block, "status": "PASS" if value is not None else "MISSING", "paired_logfid": value})
    return {
        "status": "PASS" if len(values) == 5 else "COMPANION_INCOMPLETE",
        "independent_unit": "NONE_REPEATED_GENERATION_BLOCKS",
        "inferential_role": "NONE", "blocks": rows, "n_blocks": len(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "range": [min(values), max(values)] if values else None,
        "failed_receipts": failures,
    }


def write(decoded_path: Path, receipt_root: Path, output: Path) -> None:
    decoded = json.loads(decoded_path.read_text())
    receipts = [json.loads(path.read_text()) for path in sorted(receipt_root.glob("*.json"))]
    result = summarize(decoded, receipts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
