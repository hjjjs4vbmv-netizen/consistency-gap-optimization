"""Planning calculations for the q256 fresh confirmatory v2 protocol draft.

Deterministic (no RNG): noncentral-t exact power, exact MDE by root finding,
and assurance by Gauss-Hermite quadrature over the posterior of the effect.

All inputs are the published fresh n=11 per-seed values from PR #97
(final_11seed/H_C_I_Q_G_per_seed.csv, SHA256
4d8bc83f7e9254878294a38fc3ad2ac40c84445d497dd166cec4f37b2e197461),
used for design planning only; no new inference is performed on them.

Run:  python planning_calculations.py
Out:  planning_calculations.json (committed alongside this script)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import optimize, special, stats
import numpy as np

# ---------------------------------------------------------------------------
# Source data: fresh n=11 per-seed values (PR #97, unblinded, published).
# H_A = log FID(BA@1024) - log FID(AA@1024) per seed.
# ---------------------------------------------------------------------------
H_A_FRESH = [-0.3313880604349322, 0.017103218093632133, -0.039304810308030014,
             -0.07956901234322977, 0.056797872619032574, 0.006212055871322342,
             -0.04627475067322706, -0.013274311247696069,
             -0.2236898119883306, -0.10509247158221235, -0.09515634304906806]
# Pooled-history endpoint contrast H per seed (for the four-arm comparison arm).
H_FRESH = [-0.33245675999668856, 0.02811969551058069, -0.02175261734920464,
           -0.08680594933146635, 0.03479244615671906, 0.008889246365893166,
           -0.027237725223582254, -0.0003208962443752572,
           -0.25875243518235447, -0.0898781152917556, -0.08582730276093464]
# Continuation contrasts C1 = Y_BA - Y_AA, C2 = Y_BB - Y_AB (for the redundancy
# correlation that motivates the two-arm design).
C1_FRESH = [-0.3313880604349322, 0.017103218093632133, -0.039304810308030014,
            -0.07956901234322977, 0.056797872619032574, 0.006212055871322342,
            -0.04627475067322706, -0.013274311247696069,
            -0.2236898119883306, -0.10509247158221235, -0.09515634304906806]
C2_FRESH = [2.5020355094112494 - 2.8355609689696943,
            2.3849862839098283 - 2.345850310982299,
            2.0258276392185137 - 2.030028063608893,
            2.2008328162518213 - 2.294875202571524,
            2.115449074704447 - 2.1026624550100414,
            2.151239724074637 - 2.139673287614173,
            2.0288537119897595 - 2.0370540117636964,
            2.1495695008092017 - 2.136936982050256,
            2.4389811563143704 - 2.7327962146907487,
            2.2421370866607924 - 2.316800845662091,
            2.342233375576492 - 2.418731638149293]

SOURCE_CSV_SHA256 = ("4d8bc83f7e9254878294a38fc3ad2ac40c84445d"
                     "497dd166cec4f37b2e197461")
# Transcribed from the published CSV; the script re-derives every statistic
# from the embedded values and records them, so the JSON is self-verifying.


def one_sided_t_power(delta: float, n: int, sd: float, alpha: float = 0.05) -> float:
    """Exact power of a one-sided paired t-test via the noncentral t.

    H1 direction: mean < 0.  Returns Pr(reject | true mean = delta).
    """
    if n < 2:
        raise ValueError
    df = n - 1
    t_crit = stats.t.ppf(1.0 - alpha, df)
    if delta == 0.0:
        return alpha
    ncp = delta * math.sqrt(n) / sd
    # scipy's nct.cdf returns NaN for large |ncp| (observed for ncp >= 11 on
    # the upper side with df=23); saturate analytically where the true power is
    # outside double-precision resolution of {0, 1}.
    if ncp < -30.0:
        return 1.0
    if ncp > 10.0:
        return 0.0
    # Reject when t < -t_crit; t ~ nct(df, ncp).
    power = float(stats.nct.cdf(-t_crit, df, ncp))
    if not math.isfinite(power):
        # Defensive normal-approximation fallback for any residual NaN region.
        z = -t_crit - ncp
        power = float(stats.norm.cdf(z))
    return power


def two_sided_t_power(delta: float, n: int, sd: float, alpha: float = 0.05) -> float:
    df = n - 1
    if delta == 0.0:
        return alpha
    ncp = abs(delta) * math.sqrt(n) / sd
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.cdf(t_crit, df, ncp, loc=-ncp) -
                 stats.nct.cdf(-t_crit, df, ncp, loc=-ncp))


def mde_one_sided(n: int, sd: float, target_power: float = 0.80,
                  alpha: float = 0.05) -> float:
    """Smallest |effect| reaching target_power (exact, by bisection)."""
    def power_gap(effect: float) -> float:
        return one_sided_t_power(-effect, n, sd, alpha) - target_power
    lo, hi = 0.0, 1.0
    while power_gap(hi) < 0:
        hi *= 2.0
    root = optimize.brentq(power_gap, lo, hi, xtol=1e-10)
    return float(root)


def assurance(planning_mean: float, planning_sd: float, evidence_n: int,
              design_n: int, design_sd: float, alpha: float = 0.05,
              quad_order: int = 200) -> float:
    """Bayesian predictive power.

    Prior: flat on the true mean effect delta.
    Posterior: delta | fresh evidence ~ N(planning_mean, planning_sd/sqrt(evidence_n))
    (normal approximation to the t posterior; the evidence n is large enough
    relative to the intended use that the flat-prior normal posterior is the
    standard planning choice, and the t-vs-normal difference is immaterial
    next to the model risk).
    Predictive power: integral of the exact noncentral-t power over this
    posterior, by deterministic Gauss-Hermite quadrature.

    This is an assurance (predictive power), not a frequentist guarantee;
    it assumes the fresh cohort is exchangeable with the new cohort.
    """
    se = planning_sd / math.sqrt(evidence_n)
    # Probabilists' Hermite (He): nodes x_i, weights w_i integrate
    # int f(x) exp(-x^2/2) dx = sqrt(2*pi) * sum w_i f(x_i).
    # scipy's roots_hermitenorm is numerically stable at order 200; numpy's
    # hermegauss produces NaN weights at this order and must not be used.
    nodes, weights = special.roots_hermitenorm(quad_order)
    deltas = planning_mean + nodes * se
    powers = [one_sided_t_power(float(d), design_n, design_sd, alpha)
              for d in deltas]
    return float(np.sum(np.asarray(powers) * weights) / math.sqrt(2.0 * math.pi))


def main() -> None:
    h_a = np.asarray(H_A_FRESH, dtype=float)
    h_pool = np.asarray(H_FRESH, dtype=float)
    c1 = np.asarray(C1_FRESH, dtype=float)
    c2 = np.asarray(C2_FRESH, dtype=float)

    mean_h_a = float(h_a.mean())
    sd_h_a = float(h_a.std(ddof=1))
    mean_h_pool = float(h_pool.mean())
    sd_h_pool = float(h_pool.std(ddof=1))
    corr_c1c2 = float(np.corrcoef(c1, c2)[0, 1])

    # --- Two-arm n=24 design (primary candidate) ---------------------------
    n_two = 24
    power_two = one_sided_t_power(mean_h_a, n_two, sd_h_a)
    # Concordance variant recomputed during external review: noncentral-t power
    # at the pooled-H point estimate (-0.0756) with the H_A SD. The two-arm
    # design's primary is H_A, so power_two above is the planning number; this
    # variant is recorded so the review's 0.938 is traceable.
    power_two_pooled_variant = one_sided_t_power(mean_h_pool, n_two, sd_h_a)
    mde_two = mde_one_sided(n_two, sd_h_a, 0.80)
    assurance_two = assurance(mean_h_a, sd_h_a, len(h_a), n_two, sd_h_a)

    # --- Four-arm n=16 comparison (rejected alternative) -------------------
    n_four = 16
    power_four = one_sided_t_power(mean_h_pool, n_four, sd_h_pool)
    mde_four = mde_one_sided(n_four, sd_h_pool, 0.80)
    assurance_four = assurance(mean_h_pool, sd_h_pool, len(h_pool), n_four,
                               sd_h_pool)

    # --- Dose within-seed linear contrast, n=8 -----------------------------
    # Contrast on terminal log FID at g in {1.0, 1.1, 1.2}, equally spaced
    # (g - 1.0) in {0, 0.1, 0.2}:  L_s = (-0.5) * Y_{s,1.0} + 0 * Y_{s,1.1}
    #                            + (+0.5) * Y_{s,1.2}.
    # Planning effect: linear dose response whose per-0.1-in-g step equals the
    # fresh H_A estimate (E[L] = mean_h_a under linearity).
    # Planning SD: Var(L) = 0.25*Var(Y_1.0) + 0.25*Var(Y_1.2); approximating
    # per-arm log-FID endpoint noise as equal and independent across arms,
    # Var(L) = 0.5 * Var(two-arm difference)  =>  SD(L) = sd_h_a / sqrt(2).
    n_dose = 8
    sd_contrast = sd_h_a / math.sqrt(2.0)
    power_dose = two_sided_t_power(mean_h_a, n_dose, sd_contrast)
    mde_dose = mde_one_sided(n_dose, sd_contrast, 0.80)  # magnitude only

    out = {
        "schema": "ect.q256.fresh-confirmatory-v2-planning/v1",
        "source": {
            "csv": "results/q256_fresh_crossed_switch_n12_matpool_v1/final_11seed/H_C_I_Q_G_per_seed.csv",
            "sha256": SOURCE_CSV_SHA256,
            "use": "design planning on unblinded published data; no new inference",
        },
        "fresh_evidence": {
            "n": len(h_a),
            "mean_H_A": mean_h_a,
            "sd_H_A": sd_h_a,
            "neg_count_H_A": int((h_a < 0).sum()),
            "mean_H_pooled": mean_h_pool,
            "sd_H_pooled": sd_h_pool,
            "corr_C1_C2": corr_c1c2,
        },
        "two_arm_n24": {
            "test": "one-sided paired t, H1 mean(H_A) < 0, alpha 0.05",
            "power_at_fresh_point_estimate": power_two,
            "power_at_pooled_effect_review_concordance": power_two_pooled_variant,
            "mde_80pct_power": mde_two,
            "assurance": assurance_two,
            "assurance_method": (
                "flat prior on delta; posterior N(mean_H_A, sd_H_A/sqrt(11)) "
                "from the fresh cohort; predictive power = Gauss-Hermite "
                "quadrature (order 200) of the exact noncentral-t power over "
                "this posterior; assumes cohort exchangeability"
            ),
        },
        "four_arm_n16_comparison": {
            "test": "one-sided paired t on pooled H, alpha 0.05",
            "power_at_fresh_point_estimate": power_four,
            "mde_80pct_power": mde_four,
            "assurance": assurance_four,
        },
        "dose_within_seed_contrast_n8": {
            "weights": {"g1.0": -0.5, "g1.1": 0.0, "g1.2": 0.5},
            "endpoint": "terminal log FID at 1024 kimg",
            "planning_effect": "E[L] = fresh mean(H_A) under a linear dose response",
            "planning_sd_contrast": sd_contrast,
            "planning_sd_rationale": (
                "Var(L) = 0.25*Var(Y_g1.0) + 0.25*Var(Y_g1.2); with equal "
                "independent per-arm endpoint noise this is half the variance "
                "of a two-arm difference, so SD(L) = sd(H_A)/sqrt(2). "
                "To be replaced by the Z2-upgraded noise-floor value at freeze."
            ),
            "test": "two-sided paired t over the 8 per-seed contrasts, alpha 0.05",
            "power_at_linear_projection": power_dose,
            "mde_80pct_power": mde_dose,
        },
        "tooling": {
            "scipy": stats.__dict__.get("__version__", None) or __import__("scipy").__version__,
            "numpy": np.__version__,
        },
    }
    # scipy version reporting made robust:
    import scipy
    out["tooling"]["scipy"] = scipy.__version__

    out_path = Path(__file__).with_name("planning_calculations.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()