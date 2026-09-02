"""Non-committing operator-clock audits for ECT training dynamics."""

from .core import (  # noqa: F401
    ARM_SPECS,
    AlgorithmicState,
    AuditBatch,
    algorithmic_jvp,
    field_jvp,
    matched_micro_rollout,
    squared_gn_operator_jvp,
)
