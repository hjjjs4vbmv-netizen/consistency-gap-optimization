"""Compatibility helpers for stateful RAdam audit whole-model metrics.

PR #65 generalized several whole-model field names from the historical
K-specific spelling (``a_K_star``, ``s_K_star``, ``c_K_star``) to generic
reference/probe names (``a_star``, ``s_star``, ``c_star``).  Existing analysis
consumers may still read the historical spelling.  This module provides one
fail-closed adapter so those readers can migrate without silently changing
metric semantics.

The aliases are exact name aliases only.  No value transformation, residual
renormalization, or convention change is performed.
"""
from __future__ import annotations

from typing import Any, Mapping


LEGACY_TO_GENERIC = {
    "a_K_star": "a_star",
    "s_K_star": "s_star",
    "c_K_star": "c_star",
}


def metric(whole_model: Mapping[str, Any], name: str) -> Any:
    """Read a metric using either the generic or historical spelling.

    If both spellings are present they must agree exactly; otherwise the
    adapter fails closed rather than choosing one silently.
    """
    generic = LEGACY_TO_GENERIC.get(name, name)
    legacy = next((old for old, new in LEGACY_TO_GENERIC.items() if new == generic), None)

    generic_present = generic in whole_model
    legacy_present = legacy is not None and legacy in whole_model
    if not generic_present and not legacy_present:
        raise KeyError(name)

    if generic_present and legacy_present and whole_model[generic] != whole_model[legacy]:
        raise ValueError(
            f"conflicting stateful-audit aliases: {generic}={whole_model[generic]!r}, "
            f"{legacy}={whole_model[legacy]!r}"
        )
    return whole_model[generic] if generic_present else whole_model[legacy]


def with_legacy_aliases(whole_model: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy containing exact historical aliases for generic metrics."""
    result = dict(whole_model)
    for legacy, generic in LEGACY_TO_GENERIC.items():
        if generic not in result:
            raise KeyError(generic)
        if legacy in result and result[legacy] != result[generic]:
            raise ValueError(
                f"conflicting stateful-audit aliases: {generic}={result[generic]!r}, "
                f"{legacy}={result[legacy]!r}"
            )
        result[legacy] = result[generic]
    return result
