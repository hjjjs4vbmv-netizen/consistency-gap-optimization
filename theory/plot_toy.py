"""Plot toy-model results.

Figures:
  figures/toy_condition_number.pdf   - g vs Hessian condition number & eta*lambda_max
  figures/toy_error_vs_g.pdf         - error vs g, one line per (K, noise)
  figures/toy_gstar_vs_budget.pdf    - optimal g* vs budget K (per noise)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    os.makedirs("figures", exist_ok=True)
    h = pd.read_csv("theory/toy_hessian.csv")
    b = pd.read_csv("theory/toy_finite_budget.csv")

    # ---- Figure 1: condition number + stability boundary ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(h.g, h.cond, "-o", ms=3, label="condition number")
    ax.set_xlabel("g")
    ax.set_ylabel("condition number")
    ax.set_title("Hessian condition number vs gap scale g")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(h.g, h.eta_lambda_max, "-o", ms=3, color="tab:red",
            label=r"$\eta\,\lambda_{\max}(H_g)$")
    ax.axhline(2.0, color="black", ls="--", lw=1, label="stability boundary")
    ax.fill_between(h.g, 2.0, h.eta_lambda_max.max() * 1.05,
                    color="red", alpha=0.1)
    ax.set_xlabel("g")
    ax.set_ylabel(r"$\eta\,\lambda_{\max}$")
    ax.set_title("Stability margin: gap over-drives GD")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/toy_condition_number.pdf")
    plt.close(fig)

    # ---- Figure 2: error vs g (one line per K, noise) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    colors = plt.cm.viridis(np.linspace(0, 1, 3))
    for j, K in enumerate([50, 200, 1000]):
        ax = axes[j]
        for i, noise in enumerate([0, 0.01, 0.05]):
            sub = b[(b.K == K) & (b.noise == noise)]
            sub = sub.sort_values("g")
            ax.plot(sub.g, sub.error, "-o", ms=3, color=colors[i],
                    label=f"noise={noise}")
            gmin = sub.loc[sub.error.idxmin(), "g"]
            emin = sub.error.min()
            ax.plot(gmin, emin, "x", ms=8, color="red", zorder=5)
            ax.annotate(f"g*={gmin:.2f}", (gmin, emin),
                        textcoords="offset points", xytext=(6, -4), fontsize=8)
        ax.set_xlabel("g")
        ax.set_ylabel("final error")
        ax.set_title(f"K={K}")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Toy-model final error vs gap scale g")
    fig.tight_layout()
    fig.savefig("figures/toy_error_vs_g.pdf")
    plt.close(fig)

    # ---- Figure 3: g* vs budget K (per noise) ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, noise in enumerate([0, 0.01, 0.05]):
        gstars = []
        Ks = []
        for K in [50, 200, 1000]:
            sub = b[(b.K == K) & (b.noise == noise)]
            gmin = sub.loc[sub.error.idxmin(), "g"]
            gstars.append(gmin)
            Ks.append(K)
        ax.plot(Ks, gstars, "-o", ms=6, color=colors[i], label=f"noise={noise}")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="g=1 (official)")
    ax.set_xlabel("optimization budget K (iterations)")
    ax.set_ylabel(r"optimal $g_K^*$")
    ax.set_title(r"Internal optimal gap $g_K^*$ vs budget")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/toy_gstar_vs_budget.pdf")
    plt.close(fig)

    print("saved figures:")
    for f in ["toy_condition_number", "toy_error_vs_g", "toy_gstar_vs_budget"]:
        print("  figures/%s.pdf" % f)


if __name__ == "__main__":
    main()
