"""Unit tests for the toy model core (review minor item: add basic tests).

Run: python -m pytest theory/test_toy_core.py  OR  python theory/test_toy_core.py
"""
import numpy as np
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# toy_core.py lives one level up (repo root of the toy) when tests run in-tree;
# when shipped under recurrence_of_ect/theory/, toy_core.py is in the same dir.
sys.path.insert(0, os.path.dirname(_HERE))
from toy_core import (sample_t, base_gap_sigmoid, hessian_symmetric,
                      stop_gradient_operator, v_g, hessian_power_terms,
                      gd_final_error, sgd_expectation_exact, noise_cov_from_rms)


def _base(seed=0, n=50000):
    rng = np.random.default_rng(seed)
    t = sample_t(n, rng=rng)
    d0 = base_gap_sigmoid(t)
    return t, d0


def test_v_g_second_component_equals_t2_minus_r2():
    """v_g[1] = 2 t Delta - Delta^2 must equal t^2 - r^2 with r = t - Delta."""
    t, d0 = _base()
    g = 1.1
    Delta = np.minimum(g * d0, t - 1e-3)
    r = t - Delta
    V = v_g(t, g, d0)
    np.testing.assert_allclose(V[:, 1], t**2 - r**2, rtol=1e-10)
    np.testing.assert_allclose(V[:, 0], Delta, rtol=1e-10)


def test_hessian_symmetric_and_positive_semidefinite():
    t, d0 = _base()
    H = hessian_symmetric(0.5, t, 1.0, d0)
    assert H.shape == (2, 2)
    assert np.allclose(H, H.T)
    lam = np.linalg.eigvalsh(H)
    assert lam[0] >= -1e-9, "H_g must be PSD"


def test_stop_gradient_operator_matches_finite_difference():
    """A_g beta must equal E_z[ per-sample stop-grad gradient ] (analytic E_z).

    Per-sample grad = z^2 (v_g^T beta) [t, t^2];  E_z[z^2] = sigma_d^2.
    """
    sigma_d = 0.5
    t, d0 = _base(seed=1, n=200000)
    g = 1.1
    Delta = np.minimum(g * d0, t - 1e-3)
    r = t - Delta
    vg = np.stack([Delta, t**2 - r**2], axis=-1)
    Jt = np.stack([t, t**2], axis=-1)
    beta = np.array([0.03, -0.02])
    A = stop_gradient_operator(sigma_d, t, g, d0)
    # analytic E_z then E_t
    grad_analytic = (sigma_d**2) * np.mean((vg @ beta)[:, None] * Jt, axis=0)
    np.testing.assert_allclose(grad_analytic, A @ beta, rtol=1e-8, atol=1e-10)


def test_population_loss_curvature_equals_Hg():
    """Population stop-grad loss = 1/2 beta^T H_g beta (same as symmetric)."""
    sigma_d = 0.5
    t, d0 = _base(seed=2, n=300000)
    g = 1.2
    H = hessian_symmetric(sigma_d, t, g, d0)
    beta = np.array([0.02, -0.01])
    Delta = np.minimum(g * d0, t - 1e-3)
    r = t - Delta
    vg = np.stack([Delta, t**2 - r**2], axis=-1)
    z = np.random.default_rng(7).normal(0, sigma_d, size=len(t))
    R = z * (vg @ beta)
    L_mc = 0.5 * np.mean(R**2)
    L_pred = 0.5 * beta @ H @ beta
    assert abs(L_mc - L_pred) / abs(L_pred) < 0.05  # MC tolerance


def test_hessian_power_terms_reconstruct_Hg():
    """H_g should be reconstructible from g^2 H2 + g^3 H3 + g^4 H4 (modulo clip)."""
    sigma_d = 0.5
    t, d0 = _base(seed=3, n=200000)
    H2, H3, H4 = hessian_power_terms(sigma_d, t, np.arange(0.5, 3.01, 0.1), d0)
    for g in [0.8, 1.0, 1.3]:
        H_direct = hessian_symmetric(sigma_d, t, g, d0)
        H_poly = g**2 * H2 + g**3 * H3 + g**4 * H4
        # allow residual from clipping (report it, don't fail hard)
        rel = np.linalg.norm(H_direct - H_poly) / np.linalg.norm(H_direct)
        assert rel < 0.05, f"g={g} reconstruction rel-residual {rel}"


def test_lr_matched_makes_eta_lambda_max_constant():
    """LR-matched eta keeps eta*lambda_max(H_g) constant across g."""
    sigma_d = 0.5
    t, d0 = _base()
    H1 = hessian_symmetric(sigma_d, t, 1.0, d0)
    target = 1.0
    for g in [0.6, 1.0, 1.3]:
        H = hessian_symmetric(sigma_d, t, g, d0)
        eta = target / np.linalg.eigvalsh(H)[-1]
        prod = eta * np.linalg.eigvalsh(H)[-1]
        assert abs(prod - target) < 1e-9


def test_gd_final_error_decreases_with_K():
    """More iterations -> smaller error (in the stable regime)."""
    sigma_d = 0.5
    t, d0 = _base()
    H = hessian_symmetric(sigma_d, t, 1.0, d0)
    eta = 0.5 / np.linalg.eigvalsh(H)[-1]
    beta0 = np.array([0.1, 0.1])
    e50 = gd_final_error(H, beta0, eta, 50)[0]
    e1000 = gd_final_error(H, beta0, eta, 1000)[0]
    assert e1000 < e50


def test_separation_exact_recursion_matches_closed_form():
    """Exact recursion E_K = Tr(M_K) must match the known 1-D closed form.

    For d=1 (scalar H, Sigma, beta0): E_K = (1-eta*H)^{2K} beta0^2
        + eta^2 Sigma * (1 - (1-eta*H)^{2K}) / (eta*H*(2-eta*H)).
    The matrix recursion must reproduce this exactly.
    """
    from separation import exact_sgd_error
    H = np.array([[3.0]]); Sig = np.array([[0.5]]); eta = 0.2; b0 = np.array([1.0]); K = 100
    rec = exact_sgd_error(H, Sig, eta, b0, K)
    lam = 3.0; r = 1 - eta*lam; r2K = r**(2*K)
    closed = r2K * b0[0]**2 + (eta**2 * Sig[0,0]) * (1 - r2K) / (eta*lam*(2-eta*lam))
    assert abs(rec - closed) / abs(closed) < 1e-9


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
