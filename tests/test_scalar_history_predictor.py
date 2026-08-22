"""Regression test: predictor time-index must align with the actual-update index.

This guards the bug where `--eval-step 0` switched the predictor to step 0 but
still compared against the FINAL-step actual update (u1.npy / ug.npy are the
step-(T-1) updates). With a full per-step history (u1_history / ug_history,
shape (T,d)), the actual update at eval step t must be row t — not the last row.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "scalar_history_predictor.py"
SPEC = importlib.util.spec_from_file_location("scalar_history_predictor", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _save_history(tmp_path, T, d):
    # distinct, unambiguous values per step so a wrong index is impossible to miss
    u1 = (np.arange(T * d, dtype=np.float64).reshape(T, d) + 1.0)
    ug = (np.arange(T * d, dtype=np.float64).reshape(T, d) + 1000.0)
    np.save(tmp_path / "u1_history.npy", u1)
    np.save(tmp_path / "ug_history.npy", ug)
    # single final-step files (what the buggy code used)
    np.save(tmp_path / "u1.npy", u1[-1])
    np.save(tmp_path / "ug.npy", ug[-1])
    return u1, ug


def test_eval_step_selects_correct_update(tmp_path):
    T, d = 3, 7
    u1, ug = _save_history(tmp_path, T, d)
    hp = tmp_path / "u1_history.npy"
    gp = tmp_path / "ug_history.npy"

    # eval_step=0 must use row 0 (not the last row)
    u1_0, ug_0, _, _ = M.select_eval_update(hp, gp, tmp_path / "u1.npy", tmp_path / "ug.npy", t=0, T=T)
    assert np.array_equal(u1_0, u1[0]), "eval_step=0 must select update row 0"
    assert np.array_equal(ug_0, ug[0]), "eval_step=0 must select update row 0"

    # eval_step=T-1 must use the last row
    u1_l, ug_l, _, _ = M.select_eval_update(hp, gp, tmp_path / "u1.npy", tmp_path / "ug.npy", t=T - 1, T=T)
    assert np.array_equal(u1_l, u1[T - 1]), "eval_step=T-1 must select the last row"
    assert np.array_equal(ug_l, ug[T - 1]), "eval_step=T-1 must select the last row"

    # middle step must use the middle row (not accidentally the last row)
    u1_m, ug_m, _, _ = M.select_eval_update(hp, gp, tmp_path / "u1.npy", tmp_path / "ug.npy", t=1, T=T)
    assert np.array_equal(u1_m, u1[1]), "eval_step=1 must select update row 1"
    assert np.array_equal(ug_m, ug[1]), "eval_step=1 must select update row 1"


def test_no_history_falls_back_to_final_step(tmp_path):
    # without --u1-history/--ug-history, t=0 must fall back to the FINAL-step
    # update (u1.npy/ug.npy) — this is the legacy path; the point is that it is
    # only correct when t == T-1, which the warning surfaces.
    T, d = 3, 5
    u1, ug = _save_history(tmp_path, T, d)
    u1_0, ug_0, _, _ = M.select_eval_update(None, None, tmp_path / "u1.npy", tmp_path / "ug.npy", t=0, T=T)
    assert np.array_equal(u1_0, u1[T - 1]), "legacy fallback uses the final-step update"
    assert np.array_equal(ug_0, ug[T - 1]), "legacy fallback uses the final-step update"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_eval_step_selects_correct_update(tmp)
        test_no_history_falls_back_to_final_step(tmp)
    print("ALL SCALAR-HISTORY PREDICTOR INDEX CHECKS PASSED")
