"""Shared validation helpers for seed-resolved paper assets.

The helpers deliberately accept only one metric/NFE/protocol trajectory at a
time.  This prevents a paper plot from accidentally joining values produced
under different sample counts or evaluation contracts.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

try:  # Supports both ``python scripts/...`` and package-level tests.
    from .collect_multibudget_results import read_rows
except ImportError:
    from collect_multibudget_results import read_rows


DEFAULT_BUDGETS = (256, 512, 768, 1024)


def fail(prefix: str, message: str) -> None:
    raise SystemExit("[{}] ERROR: {}".format(prefix, message))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_one_protocol(rows: list[dict], analysis_track: str, prefix: str) -> dict:
    tracks = {row["analysis_track"] for row in rows}
    if tracks != {analysis_track}:
        fail(prefix, "selected rows must all have analysis_track={!r}; got {}".format(analysis_track, sorted(tracks)))
    contracts = {row["evaluation_contract"] for row in rows}
    protocol = {
        (row["sample_count"], row["generation_seed_range"], row["metric_seed"])
        for row in rows
    }
    if (len(contracts) != 1 or "" in contracts or len(protocol) != 1
            or None in next(iter(protocol)) or "" in next(iter(protocol))):
        fail(
            prefix,
            "one explicit evaluation_contract and one complete "
            "(sample_count, generation_seed_range, metric_seed) protocol are required",
        )
    sample_count, generation_seed_range, metric_seed = next(iter(protocol))
    return {
        "evaluation_contract": next(iter(contracts)),
        "sample_count": sample_count,
        "generation_seed_range": generation_seed_range,
        "metric_seed": metric_seed,
        "evidence_classes": sorted({row["evidence_class"] for row in rows}),
    }


def load_trajectory(
    source: Path, metric_name: str, nfe: int, budgets: tuple[int, ...], analysis_track: str, prefix: str,
) -> tuple[list[dict], dict]:
    """Load a complete, protocol-matched metric/NFE trajectory from long-form CSV."""
    all_rows = read_rows(source.resolve())
    selected = [
        row for row in all_rows
        if row["metric_name"] == metric_name and row["nfe"] == nfe
    ]
    if not selected:
        available = sorted({(row["metric_name"], row["nfe"]) for row in all_rows})
        fail(prefix, "no rows for metric_name={!r}, nfe={}; available={}".format(metric_name, nfe, available))
    observed_budgets = {row["budget_kimg"] for row in selected}
    expected_budgets = {float(budget) for budget in budgets}
    if observed_budgets != expected_budgets:
        fail(
            prefix,
            "frozen budgets {} do not match observed budgets {}. "
            "Do not silently truncate a paper trajectory.".format(list(budgets), sorted(observed_budgets)),
        )
    return selected, require_one_protocol(selected, analysis_track, prefix)


def complete_matrix(
    rows: list[dict], methods: tuple[str, ...], budgets: tuple[int, ...], prefix: str,
) -> tuple[list[int], dict[tuple[str, int, float], dict]]:
    """Return a closed method × seed × budget matrix or fail with exact cells."""
    selected = [row for row in rows if row["method"] in methods]
    observed_methods = {row["method"] for row in selected}
    if observed_methods != set(methods):
        fail(prefix, "requested methods {} do not match observed methods {}".format(list(methods), sorted(observed_methods)))
    seeds = sorted({row["training_seed"] for row in selected})
    if not seeds:
        fail(prefix, "no training seeds found for requested methods")
    index: dict[tuple[str, int, float], dict] = {}
    for row in selected:
        key = (row["method"], row["training_seed"], row["budget_kimg"])
        if key in index:
            fail(prefix, "duplicate metric point: {}".format(key))
        if not math.isfinite(row["metric_value"]):
            fail(prefix, "non-finite metric value at {}".format(key))
        index[key] = row
    expected = {
        (method, seed, float(budget))
        for method in methods for seed in seeds for budget in budgets
    }
    missing = expected - set(index)
    extra = set(index) - expected
    if missing or extra:
        fail(prefix, "matrix incomplete; missing={}, extra={}".format(sorted(missing), sorted(extra)))
    return seeds, index

