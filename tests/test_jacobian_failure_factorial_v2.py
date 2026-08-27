"""Protocol-boundary tests for the calibrated v2 factorial."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.jacobian_failure_factorial_v2 import run_factorial


def test_v2_grid_is_the_frozen_interior_calibration_triple():
    frozen = run_factorial.protocol()
    assert frozen["epsilon_grid"] == [
        0.00390625, 0.001953125, 0.0009765625]
    assert frozen["calibration"]["admitted_plateau"][:3] == frozen["epsilon_grid"]
    assert frozen["replacement_after_results"] == "PROHIBITED"


def test_v2_matrix_has_160_cells():
    assert len(run_factorial.tasks(run_factorial.protocol(), "formal")) == 160


def test_v2_rejects_correctness_receipt_from_another_protocol(tmp_path: Path):
    receipt = tmp_path / "gate.json"
    receipt.write_text(json.dumps({
        "formal_admissible": True,
        "protocol_sha256": "0" * 64,
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="another protocol"):
        run_factorial.load_correctness(receipt)

