"""Plot R_opt(K) and R²_scalar(K) across K = {32, 64, 128, 256} kimg.

Reads analysis/real_history/k{K}/scalar_prediction.json for each K.
Shows whether the scalar-history mechanism's predictive power (R²) is stable
across training horizon — the leader's requested K→{R_opt(K), R²_scalar(K)}.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [32, 64, 128, 256]


def main(out_dir: Path = Path("figures")):
    out_dir.mkdir(parents=True, exist_ok=True)
    r2, ropt, a_star = [], [], []
    for k in KS:
        p = Path(f"analysis/real_history/k{k}/scalar_prediction.json")
        if not p.exists():
            print(f"K={k}: MISSING {p}")
            continue
        d = json.load(open(p))
        r2.append((k, d.get("weighted_R2_scalar_vs_actual")))
        ropt.append((k, d.get("R_opt")))
        a_star.append((k, d.get("a_star_mean")))
        print(f"K={k}: R²={d.get('weighted_R2_scalar_vs_actual',0):.4f}, "
              f"R_opt={d.get('R_opt',0):.4f}, a*={d.get('a_star_mean',0):.4f}")

    if not r2:
        print("no data"); return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot([k for k, _ in r2], [v for _, v in r2], "-o", label="Weighted R²")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="R²=0.5")
    ax.set_xlabel("K (kimg)"); ax.set_ylabel("R²_scalar")
    ax.set_title("Scalar-history predictor: explained variance vs horizon")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([k for k, _ in ropt], [v for _, v in ropt], "-s", label="R_opt")
    ax.plot([k for k, _ in ropt], [v for _, v in ropt], "-o", alpha=0, label="_")
    ax.set_xlabel("K (kimg)"); ax.set_ylabel("R_opt")
    ax.set_title("Optimizer update residual vs horizon")
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("K → {R_opt(K), R²_scalar(K)} — prospective scalar-history mechanism")
    fig.tight_layout()
    fig.savefig(out_dir / "k_horizon_R2_Ropt.pdf")
    print(f"saved {out_dir}/k_horizon_R2_Ropt.pdf")


if __name__ == "__main__":
    main()
