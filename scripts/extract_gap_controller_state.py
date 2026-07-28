#!/usr/bin/env python3
"""Extract auditable final schedule/controller state from a training snapshot."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite controller value: {value}")
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return jsonable(value.item())
    raise TypeError(f"unsupported controller value: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    loss_fn = payload.get("loss_fn")
    schedule = getattr(loss_fn, "schedule", None)
    if schedule is None:
        raise SystemExit("snapshot does not contain loss_fn.schedule")

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "schedule_class": type(schedule).__name__,
        "metadata": jsonable(schedule.metadata()),
        "state_dict": jsonable(schedule.state_dict()),
        "runtime_metrics": jsonable(schedule.runtime_metrics()),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
