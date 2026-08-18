"""Regression tests for the stateful-audit generic/legacy metric adapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "analysis" / "radam_stateful_schema_compat.py"
SPEC = importlib.util.spec_from_file_location("radam_stateful_schema_compat", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StatefulSchemaCompatTests(unittest.TestCase):
    def test_legacy_metric_names_read_generic_receipt(self):
        whole = {"a_star": 0.8, "s_star": 1.1, "c_star": 0.9}
        self.assertEqual(MODULE.metric(whole, "a_K_star"), 0.8)
        self.assertEqual(MODULE.metric(whole, "s_K_star"), 1.1)
        self.assertEqual(MODULE.metric(whole, "c_K_star"), 0.9)

    def test_adapter_adds_exact_aliases_without_transforming_values(self):
        whole = {"a_star": 0.8, "s_star": 1.1, "c_star": 0.9, "R_opt": 0.07}
        adapted = MODULE.with_legacy_aliases(whole)
        self.assertEqual(adapted["a_K_star"], adapted["a_star"])
        self.assertEqual(adapted["s_K_star"], adapted["s_star"])
        self.assertEqual(adapted["c_K_star"], adapted["c_star"])
        self.assertEqual(adapted["R_opt"], whole["R_opt"])
        self.assertNotIn("a_K_star", whole)

    def test_conflicting_aliases_fail_closed(self):
        with self.assertRaises(ValueError):
            MODULE.metric({"a_star": 0.8, "a_K_star": 0.7}, "a_K_star")
        with self.assertRaises(ValueError):
            MODULE.with_legacy_aliases({
                "a_star": 0.8,
                "a_K_star": 0.7,
                "s_star": 1.1,
                "c_star": 0.9,
            })

    def test_missing_metric_fails_closed(self):
        with self.assertRaises(KeyError):
            MODULE.metric({}, "a_K_star")


if __name__ == "__main__":
    unittest.main()
