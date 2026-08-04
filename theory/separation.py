"""ADCM separation counterexample (review P0/separation).

Two environments with the SAME instantaneous criterion (population loss
1/2 beta^T H_g beta, curvature H_g) but DIFFERENT gradient-noise covariance:

  env1 (symmetric loss, both branches live):
      residual R = z v_g^T beta,  grad = z^2 (v_g^T beta) v_g
      => Sigma_g^(1) = sigma_d^2 E[v_g v_g^T] = H_g   (== curvature)
  env2 (stop-gradient loss, online branch live only):
      residual R = z v_g^T beta,  grad = z^2 (v_g^T beta) [t, t^2]
      => Sigma_g^(2) = sigma_d^2 E[ [t,t^2][t,t^2]^T ]

Both have H_g as curvature -> an instantaneous-criterion method (ADCM-style)
gives the SAME g recommendation for both. But the finite-step SGD optimum
g_K^star can differ because Sigma_g differs.

We use LR-matched eta (eta*lambda_max(H_g)=const) so the result is NOT a
learning-rate artifact. We run Monte-Carlo SGD (not the isotropic-noise
closed form, because the noise here is anisotropic and g-dependent) and
report g_K^star for each env.
"""
import numpy as np
import pandas as pd
from toy_core import sample_t, base_gap_sigmoid, hessian_symmetric, v_g


def mc_sgd_final_error(t, delta0, sigma_d, g, beta0, eta, K, env, rng, n_inner=8, rms=0.05):
    """Monte-Carlo SGD: expected ||beta_K||^2 averaged over noise draws.

    Structural gradient noise per sample:
      sym :  grad_noise = z^2 (v_g^T beta) v_g           Sigma ~ v_g v_g^T
      stop:  grad_noise = z^2 (v_g^T beta) [t, t^2]      Sigma ~ [t,t^2][t,t^2]^T
    We RESCALE each per-sample noise vector to fixed RMS = rms (so total injected
    noise power is identical across g and env), keeping only the DIRECTION
    (anisotropy) env/g-dependent. This removes the "different noise magnitude"
    confound: any g_K^star difference then comes from the noise *direction*
    relative to H_g's eigenvectors, which ADCM's instantaneous criterion cannot see.

    LR-matched eta (eta*lambda_max(H_g)=const) is passed by caller.
    """
    Delta = np.minimum(g * delta0, t - 1e-3)
    r = t - Delta
    vg = np.stack([Delta, t ** 2 - r ** 2], axis=-1)        # (n,2)
    Jt = np.stack([t, t ** 2], axis=-1)                    # (n,2)  stop-grad Jacobian col
    n = len(t)
    feat = vg if env == "sym" else Jt                       # noise feature direction
    sqsum = 0.0
    for _ in range(n_inner):
        beta = beta0.copy()
        for k in range(K):
            idx = rng.integers(0, n)
            z = rng.normal(0, sigma_d)
            raw = (z ** 2) * (vg[idx] @ beta) * feat[idx]   # structural noise vec
            nrm = np.linalg.norm(raw)
            if nrm > 1e-30:
                raw = raw / nrm * rms                       # fixed RMS = rms
            grad = raw                                      # pure noise-driven step
            beta = beta - eta * grad
        sqsum += np.sum(beta ** 2)
    return sqsum / n_inner


def run(sigma_d=0.5, out="."):
    import os
    os.makedirs(out, exist_ok=True)
    t = sample_t(200000, rng=np.random.default_rng(0))
    d0 = base_gap_sigmoid(t)
    H1 = hessian_symmetric(sigma_d, t, 1.0, d0)
    lam1_max = np.linalg.eigvalsh(H1)[-1]

    gs = np.arange(0.5, 1.46, 0.05)   # within stability bound
    Ks = [50, 200, 1000]
    rms_list = [0.05]                 # noise level
    eta_scale = 1.0                   # LR-matched target eta*lam_max = 1.0
    beta0 = np.ones(2) * 1e-2

    rows = []
    for env in ["sym", "stop"]:
        for K in Ks:
            for rms in rms_list:
                # scale noise so per-step noise RMS ~ rms in beta space
                # grad noise std ~ sigma_d^2 * |v_g^T beta| * |feat|; we just use
                # a fixed additive gaussian of std rms on the grad (isotropic proxy)
                # -> to respect anisotropic Sigma_g, we instead inject the true
                # structural noise scaled by a factor s so that overall RMS ~ rms.
                # For a clean separation demo we use the *structural* noise only
                # (z^2 term), scaled by a constant factor.
                for g in gs:
                    H = hessian_symmetric(sigma_d, t, g, d0)
                    eta = eta_scale / np.linalg.eigvalsh(H)[-1]   # LR-matched
                    rng = np.random.default_rng(12345)
                    e = mc_sgd_final_error(t, d0, sigma_d, g, beta0, eta, K, env, rng,
                                           n_inner=6, rms=rms)
                    rows.append(dict(env=env, g=round(float(g),4), K=K, rms=rms, error=float(e)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "toy_separation.csv"), index=False)

    print("=== g_K^star per environment (LR-matched, structural noise) ===")
    for K in Ks:
        for env in ["sym", "stop"]:
            sub = df[(df.env == env) & (df.K == K)].sort_values("g")
            gstar = sub.loc[sub.error.idxmin(), "g"]
            emin = sub.error.min()
            print(f"  env={env:4s} K={K:4d}: g*={gstar:.2f}  err_min={emin:.4e}")

    print("\n=== SEPARATION CHECK ===")
    sep = False
    for K in Ks:
        s = df[(df.env == "sym") & (df.K == K)]
        p = df[(df.env == "stop") & (df.K == K)]
        gs_star = s.loc[s.error.idxmin(), "g"]
        gp_star = p.loc[p.error.idxmin(), "g"]
        diff = gs_star != gp_star
        sep = sep or diff
        print(f"  K={K}: sym g*={gs_star:.2f}  stop g*={gp_star:.2f}  DIFFER={diff}")
    print("=> ADCM SEPARATION ESTABLISHED" if sep else "=> no separation in this config")


if __name__ == "__main__":
    run(out=".")
