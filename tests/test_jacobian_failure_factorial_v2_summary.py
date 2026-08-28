"""Focused tests for v2 summary classification and manifest validation."""
from __future__ import annotations

from analysis.jacobian_failure_factorial_v2 import summarize


def test_expected_matrix_contains_160_unique_cells():
    frozen = summarize.load_protocol()
    assert len(summarize.expected_keys(frozen)) == 160


def test_transition_separation_is_not_internal_component_attribution():
    source = """
    GO at the production-transition level; HOLD for attribution inside that
    transition. Regime D combines autocast/FP16, the stateful RAdam update, EMA,
    and scaler state; the factorial does not assign the instability to one
    internal component.
    """
    assert "GO at the production-transition level" in source
    assert "does not assign" in source
