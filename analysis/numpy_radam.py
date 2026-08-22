"""Pure-numpy RAdam replay (float32), matching torch.optim.RAdam.

Why: the matpool node the sweep data lives on runs glibc 2.17, too old for the
conda/venv python (needs glibc >= 2.28) that has torch. The RAdam update rule is
deterministic and torch-independent:

    m <- beta1*m + (1-beta1)*g                 (torch: exp_avg.lerp_(g, 1-beta1))
    v <- beta2*v + (1-beta2)*g*g               (torch: exp_avg_sq.mul_(b2).addcmul_(g,g,value=1-b2))
    bc1 = 1 - beta1^step ; bc2 = 1 - beta2^step
    m_hat = m / bc1
    rho_inf = 2/(1-beta2) - 1
    rho_t  = rho_inf - 2*step*(beta2^step)/bc2
    update = sqrt((rho_t-4)(rho_t-2)rho_inf/((rho_inf-4)(rho_inf-2)rho_t)) * sqrt(bc2)/(sqrt(v)+eps)   if rho_t > 5
           = 1.0                                                                                       else
    delta = -lr * m_hat * update

All moment/update arithmetic is float32 (matching torch's float32 tensors). The
per-step parameter change is independent of the parameter VALUE (depends only on
m, v, step, grad), so we never track params — delta is returned directly.

Validation: replay(G1) reproduces the stored torch-generated u1_history.npy to
float32 precision (max|d| ~ 2^-23), because the backfill was itself torch-CPU
float32 and numpy float32 IEEE arithmetic agrees with it to ~1 ulp.
"""
import numpy as np

BETA1, BETA2 = 0.9, 0.999


def radam_step(m, v, step, g, lr=1e-4, beta1=BETA1, beta2=BETA2, eps=1e-8):
    """One RAdam step in float32. m,v,g are float32 (d,) arrays; returns (m,v,delta).

    delta is float32 and equals p_after - p_before (the sweep's u).
    """
    m = np.ascontiguousarray(m, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    step = int(step) + 1

    # exp_avg.lerp_(g, 1-beta1)  ==  m + (g - m)*(1-beta1)
    m = m + (g - m) * (1.0 - beta1)
    # exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1-beta2)  ==  beta2*v + (1-beta2)*g*g
    v = beta2 * v + (1.0 - beta2) * (g * g)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = m / bc1

    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    rho_t = rho_inf - 2.0 * step * (beta2 ** step) / bc2

    if rho_t > 5.0:
        rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)) ** 0.5
        adaptive = bc2 ** 0.5 / (np.sqrt(v) + eps)
        update = rect * adaptive
    else:
        update = 1.0

    delta = -lr * m_hat * update
    return m, v, delta


def replay_numpy(grad_hist_f32, m0, v0, step0, lr=1e-4,
                 beta1=BETA1, beta2=BETA2, eps=1e-8):
    """Replay RAdam from a real state over a float32 gradient history.

    grad_hist_f32: (T, d) float32 array (or iterable of (d,) float32).
    Returns list of float32 per-step deltas (the parameter updates), length T.
    """
    m = np.array(m0, dtype=np.float32).copy()
    v = np.array(v0, dtype=np.float32).copy()
    step = int(step0)
    deltas = []
    for j in range(grad_hist_f32.shape[0]):
        g = grad_hist_f32[j]
        m, v, delta = radam_step(m, v, step, g, lr, beta1, beta2, eps)
        step += 1
        deltas.append(delta)
    return deltas
