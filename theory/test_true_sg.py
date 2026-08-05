"""Unit tests for the true stop-gradient operator (PR #34 review).

Covers the review's requested tests:
  1. single-step analytic recursion vs large-sample MC;
  2. matrix-power (T^K) vs iterative recursion;
  3. A-matched control makes error-vs-g approximately flat;
plus a regression test for the exact_E_K cumulative-step bug.

Run: python theory/test_true_sg.py
"""
import os
import sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from toy_core import sample_t, base_gap_sigmoid, hessian_symmetric
from true_sg_operator import (build_A, build_T_matrix, build_vg_Jt,
                              exact_E_K, mc_sgd, eta_lr_match_A)


def _base(n=50000, seed=0):
    rng = np.random.default_rng(seed)
    t = sample_t(n, rng=rng)
    d0 = base_gap_sigmoid(t)
    return t, d0


def test_single_step_recursion_vs_mc():
    """M1 from T_g must match large-sample MC within MC tolerance."""
    sigma_d, g, eta = 0.5, 1.0, 1e-4
    t, d0 = _base(n=50000)
    beta0 = np.array([0.01, 0.02])
    T, _, _, _ = build_T_matrix(sigma_d, t, d0, g, eta)
    m0 = np.array([beta0[0] ** 2, beta0[0] * beta0[1], beta0[1] ** 2])
    m1 = T @ m0
    vg, Jt = build_vg_Jt(t, d0, g)
    n = len(t)
    rng = np.random.default_rng(0)
    ntr = 200000
    M1 = np.zeros((2, 2))
    for _ in range(ntr):
        idx = rng.integers(0, n)
        z = rng.normal(0, sigma_d)
        b1 = beta0 - eta * (z ** 2) * (vg[idx] @ beta0) * Jt[idx]
        M1 += np.outer(b1, b1)
    M1 /= ntr
    m1_mc = np.array([M1[0, 0], M1[0, 1], M1[1, 1]])
    rel = np.linalg.norm(m1_mc - m1) / max(np.linalg.norm(m1_mc), 1e-30)
    # note: heavy-tail J_t = [t, t^2] makes the MC estimate high-variance even
    # at 200k draws (large t^4 draws dominate); a 30% band is the honest
    # tolerance for a sanity check under this parameterization.
    assert rel < 0.30, f"single-step recursion vs MC rel_err={rel:.3f} > 0.30"


def test_matrix_power_equals_iterative():
    """T^K m0 must equal iterating T K times (guard against cumulative bug)."""
    sigma_d, g, eta = 0.5, 1.0, 1e-4
    t, d0 = _base()
    beta0 = np.ones(2) * 1e-2
    T, _, _, _ = build_T_matrix(sigma_d, t, d0, g, eta)
    m0 = np.array([beta0[0] ** 2, beta0[0] * beta0[1], beta0[1] ** 2])
    K = 30
    m_pow = np.linalg.matrix_power(T, K) @ m0
    m_it = m0.copy()
    for _ in range(K):
        m_it = T @ m_it
    np.testing.assert_allclose(m_pow, m_it, rtol=1e-10)


def test_exact_E_K_no_cumulative_bug():
    """exact_E_K([20, 50]) must give T^20 and T^50, NOT T^20 and T^70."""
    sigma_d, g, eta = 0.5, 1.0, 1e-4
    t, d0 = _base()
    beta0 = np.ones(2) * 1e-2
    out, _ = exact_E_K(sigma_d, t, d0, g, eta, beta0, [20, 50])
    T, _, _, _ = build_T_matrix(sigma_d, t, d0, g, eta)
    m0 = np.array([beta0[0] ** 2, beta0[0] * beta0[1], beta0[1] ** 2])
    # independent T^50 -> trace
    M50 = np.linalg.matrix_power(T, 50) @ m0
    e50 = M50[0] + M50[2]
    assert abs(out[50] - e50) / abs(e50) < 1e-9, "cumulative-step bug present"


def test_A_matched_curve_flat():
    """Under A-matching, error-vs-g must be approximately flat (spread << 1)."""
    sigma_d = 0.5
    t, d0 = _base()
    H1 = hessian_symmetric(sigma_d, t, 1.0, d0)
    A1, _, _ = build_A(sigma_d, t, d0, 1.0)
    rhoA1 = max(abs(np.linalg.eigvals(A1)))
    eta1 = 0.005 / rhoA1
    etas = eta_lr_match_A(sigma_d, t, d0, eta1, None)
    beta0 = np.ones(2) * 1e-2
    K = 200
    vals = []
    for g in [0.5, 0.8, 1.0, 1.2, 1.5]:
        E, _ = exact_E_K(sigma_d, t, d0, g, etas[g], beta0, [K])
        vals.append(E[K])
    vals = np.array(vals)
    spread = (vals.max() - vals.min()) / max(vals.min(), 1e-30)
    assert spread < 1e-2, f"A-matched spread={spread} > 1e-2 (should be ~1e-6)"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
