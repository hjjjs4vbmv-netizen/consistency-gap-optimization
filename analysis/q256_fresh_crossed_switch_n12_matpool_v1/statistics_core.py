"""Dependency-free descriptive statistics used by the frozen analysis."""

from __future__ import annotations

from collections.abc import Iterable


def median(values: Iterable[float]) -> float:
    """Return the conventional sample median for a non-empty iterable."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median requires at least one value")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
