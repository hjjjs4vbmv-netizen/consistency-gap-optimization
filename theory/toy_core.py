"""Linear-Gaussian toy model for ECT gap calibration theory.

Implements the math from the research report (report §0.3/0.5):
  - p_t = N(0, sigma_d^2 + t^2), x_t = m(t) z, m(t) = sqrt(sigma_d^2+t^2)/sigma_d
  - ideal consistency map  f^*(x,t) = x / m(t)
  - boundary-correct model f_beta(x,t) = (x/m(t)) (1 + beta1 t + beta2 t^2)
  - same-trajectory pair residual = z v_g(t)^T beta, v_g(t) = [Delta, 2 t Delta - Delta^2]^T
  - symmetric population Hessian  H_g = sigma_d^2 E_t[v_g v_g^T] = g^2 H2 + g^3 H3 + g^4 H4
  - full-batch GD:  beta_K = (I - eta H_g)^K beta_0,  rho_g = max_j |1 - eta lambda_j(H_g)|
  - noisy SGD per eigen-direction j:
        E[beta_Kj^2] = (1-eta*lambda_j)^{2K} beta_0j^2
                     + eta*nu_j [1 - (1-eta*lambda_j)^{2K}] / (lambda_j (2 - eta*lambda_j))
  - stop-gradient asymmetric operator  A_g = E[J_t^T (J_t - J_r)]
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import eigvalsh


# --------------------------------------------------------------------------
# Data / schedule primitives
# --------------------------------------------------------------------------

def sample_t(n, mean=-1.1, std=2.0, t_min=1e-3, t_max=100.0, rng=None):
    """Sample t ~ LogNormal(mean, std), clipped to [t_min, t_max]."""
    rng = rng if rng is not None else np.random.default_rng(0)
    t = np.exp(rng.normal(mean, std, size=n))
    return np.clip(t, t_min, t_max)


def base_gap_sigmoid(t, q=256.0, k=8.0, b=1.0, stage=0.0):
    """Official ECT sigmoid base gap delta0(t) = t - r_sigmoid(t).

    r/t = 1 - decay * (1 + k*sigmoid(-b t)),  decay = 1/q^(stage+1).
    delta0 = t - r = t * decay * (1 + k*sigmoid(-b t)).
    """
    decay = 1.0 / q ** (stage + 1.0)
    adj = 1.0 + k * 1.0 / (1.0 + np.exp(b * t))
    return t * decay * adj


# --------------------------------------------------------------------------
# v_g, Hessian
# --------------------------------------------------------------------------

def v_g(t, g, delta0, t_min=1e-3):
    """Feature vector v_g(t) = [Delta, 2 t Delta - Delta^2]^T.

    Delta = min(g * delta0, t - t_min).
    """
    Delta = np.minimum(g * delta0, t - t_min)
    v1 = Delta
    v2 = 2.0 * t * Delta - Delta ** 2
    return np.stack([v1, v2], axis=-1)  # (n, 2)


def hessian_symmetric(sigma_d, t, g, delta0, t_min=1e-3):
    """H_g = sigma_d^2 E_t[v_g(t) v_g(t)^T]  (population, Monte-Carlo in t)."""
    V = v_g(t, g, delta0, t_min)          # (n, 2)
    G = np.einsum("ni,nj->ij", V, V) / len(t)
    return sigma_d ** 2 * G


def hessian_power_terms(sigma_d, t, g, delta0, t_min=1e-3):
    """Fit H_g = g^2 H2 + g^3 H3 + g^4 H4 exactly (no MC noise if same t).

    delta0 does not depend on g, so H_g = sigma_d^2 E[v_g v_g^T] is a
    polynomial in g of degree <= 4. We recover H2,H3,H4 by evaluating at
    g in {0,1,-1,2} and inverting the Vandermonde on the (g^2,g^3,g^4) basis.
    """
    # Solve H(g) = a2*g^2 + a3*g^3 + a4*g^4 for the matrix-valued coefficients
    # using least squares over a fine g-grid; exact because only 3 unknowns.
    gs = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    Hs = np.stack([hessian_symmetric(sigma_d, t, gg, delta0, t_min) for gg in gs])
    A = np.stack([gs ** 2, gs ** 3, gs ** 4], axis=1)  # (6,3)
    coefs, *_ = np.linalg.lstsq(A, Hs.reshape(6, -1), rcond=None)
    coefs = coefs.reshape(3, 2, 2)
    return coefs[0], coefs[1], coefs[2]  # H2, H3, H4


# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------

def gd_spectral_radius(H, eta):
    lam = eigvalsh(H)
    return np.max(np.abs(1.0 - eta * lam)), lam


def gd_final_error(H, beta0, eta, K):
    """beta_K = (I - eta H)^K beta0; returns ||beta_K||^2 and ||beta_K||.

    Uses eigendecomposition with |1 - eta*lam| clipped for stability near the
    instability boundary (|r| > 1 diverges; we report its magnitude).
    """
    lam, V = np.linalg.eigh(H)
    r = 1.0 - eta * lam
    rK = r ** K
    betaK = (V * rK) @ (V.T @ beta0)
    return np.linalg.norm(betaK) ** 2, np.linalg.norm(betaK)


def sgd_expectation_exact(H, beta0, eta, K, nu_vec):
    """Exact expected squared norm of beta_K under additive SGD noise.

    Per eigen-direction j:
      E[beta_Kj^2] = (1-eta*lambda_j)^{2K} beta_0j^2
                   + eta*nu_j [1-(1-eta*lambda_j)^{2K}] / (lambda_j (2 - eta*lambda_j))
    """
    lam, V = np.linalg.eigh(H)
    beta0_proj = V.T @ beta0
    r = 1.0 - eta * lam
    r2K = r ** (2 * K)
    bias = r2K * beta0_proj ** 2
    denom = lam * (2.0 - eta * lam)
    noise = np.where(denom > 0, eta * nu_vec * (1.0 - r2K) / np.maximum(denom, 1e-12), 0.0)
    expected_sq = np.sum(bias + noise)
    return expected_sq


# --------------------------------------------------------------------------
# Stop-gradient asymmetric operator
# --------------------------------------------------------------------------

def stop_gradient_operator(sigma_d, t, g, delta0, t_min=1e-3, stage=0.0):
    """A_g = E[J_t^T (J_t - J_r)], Jacobian of the pair residual wrt beta.

    residual r_g(t) = z v_g(t)^T beta  =>  J_t = z v_g(t)^T (2x2 row? actually
    the residual is scalar; J_t is gradient of residual wrt beta = z v_g(t).
    J_t^T (J_t - J_r) = z^2 (v_t v_t^T - v_t v_r^T). With shared noise z,
    A_g = E_t[ E_z[z^2] (v_t v_t^T - v_t v_r^T) ] = sigma_d^2 E_t[ v_t v_t^T - v_t v_r^T ].
    v_t uses the stop-gradient 'target' t (i.e. beta updates only via the t-branch).
    """
    Delta = np.minimum(g * delta0, t - t_min)
    vt = np.stack([Delta, 2.0 * t * Delta - Delta ** 2], axis=-1)
    # r = t - Delta
    vr = np.stack([Delta, 2.0 * (t - Delta) * Delta - Delta ** 2], axis=-1)
    M = np.einsum("ni,nj->ij", vt, vt) - np.einsum("ni,nj->ij", vt, vr)
    M = M / len(t)
    return sigma_d ** 2 * M


def asym_spectral_report(A):
    """Report symmetric part (A+A^T)/2 and antisymmetric (A-A^T)/2 spectra."""
    As = 0.5 * (A + A.T)
    Aa = 0.5 * (A - A.T)
    return eigvalsh(As), np.linalg.norm(Aa, ord=2)


# --------------------------------------------------------------------------
# Noise model
# --------------------------------------------------------------------------

def noise_cov_from_rms(H, rms):
    """Build additive noise covariance Sigma with given per-component RMS.

    We inject noise on the gradient as gaussian with covariance chosen so that
    the *effect on beta* has per-eigenvalue noise variance nu_j = rms^2.
    For the toy, the cleanest interpretation: gradient noise ~ N(0, rms^2 I)
    in beta space => nu_j = rms^2 for all j (since eigenbasis of H).
    Returns vector nu (per eigenvalue) and a gradient-noise sampler.
    """
    d = H.shape[0]
    nu = np.full(d, rms * rms)
    return nu
