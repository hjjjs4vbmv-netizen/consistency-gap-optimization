#!/usr/bin/env python3
"""Render the publication figure for the frozen Arm A longitudinal audit.

The script reads the versioned cross-state summary and only changes its
presentation.  It does not rerun the audit, recalculate measurements, or add
inferential statistics.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ect-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


EXPECTED_K_KIMG = (32.128, 64.128, 128.128, 256.000)
REQUIRED_COLUMNS = {
    "K_kimg",
    "stateful_n_K",
    "R_grad",
    "R_opt",
    "c_K_star",
}

# Okabe-Ito colors, paired with distinct markers and line styles for grayscale.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
INK = "#202124"
MID_GRAY = "#666666"
GRID = "#D9D9D9"


@dataclass(frozen=True)
class StateMeasurement:
    k_kimg: float
    n_k: int
    r_grad: float
    r_opt: float
    c_k_star: float


def read_summary(path: Path) -> list[StateMeasurement]:
    """Read and validate the frozen four-state, figure-ready summary."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise SystemExit(
                f"longitudinal summary is missing required fields: {sorted(missing)}"
            )
        raw_rows = list(reader)

    rows: list[StateMeasurement] = []
    for row_number, row in enumerate(raw_rows, start=2):
        try:
            measurement = StateMeasurement(
                k_kimg=float(row["K_kimg"]),
                n_k=int(row["stateful_n_K"]),
                r_grad=float(row["R_grad"]),
                r_opt=float(row["R_opt"]),
                c_k_star=float(row["c_K_star"]),
            )
        except (TypeError, ValueError) as error:
            raise SystemExit(f"invalid numeric value on CSV row {row_number}: {error}")
        values = (
            measurement.k_kimg,
            measurement.r_grad,
            measurement.r_opt,
            measurement.c_k_star,
        )
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"non-finite figure value on CSV row {row_number}")
        if measurement.n_k <= 0:
            raise SystemExit(f"non-positive optimizer step on CSV row {row_number}")
        if not (0.0 <= measurement.r_grad <= 1.0):
            raise SystemExit(f"R_grad is outside [0, 1] on CSV row {row_number}")
        if not (0.0 <= measurement.r_opt <= 1.0):
            raise SystemExit(f"R_opt is outside [0, 1] on CSV row {row_number}")
        if measurement.c_k_star <= 0.0:
            raise SystemExit(f"c_K_star is non-positive on CSV row {row_number}")
        rows.append(measurement)

    rows.sort(key=lambda item: item.k_kimg)
    observed_k = tuple(row.k_kimg for row in rows)
    if observed_k != EXPECTED_K_KIMG:
        raise SystemExit(
            "expected the frozen Arm A states at "
            f"{EXPECTED_K_KIMG}; observed {observed_k}"
        )
    return rows


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MID_GRAY)
    axis.spines["bottom"].set_color(MID_GRAY)
    axis.tick_params(width=0.7, length=3, color=MID_GRAY, pad=2)
    axis.set_axisbelow(True)


def render_main_figure(
    rows: list[StateMeasurement],
    outdir: Path,
    stem: str = "same_trajectory_residuals",
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
    png_dpi: int = 600,
) -> list[Path]:
    """Render vector masters plus a high-resolution raster preview."""
    if tuple(row.k_kimg for row in rows) != EXPECTED_K_KIMG:
        raise ValueError("render_main_figure requires the four frozen Arm A states")

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.7,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "ect-arm-a-longitudinal-v1",
        "axes.unicode_minus": False,
    }
    with plt.rc_context(rc):
        figure, (residual_axis, scale_axis) = plt.subplots(
            1,
            2,
            figsize=(7.0, 2.8),
            gridspec_kw={"width_ratios": (1.42, 1.0)},
        )
        figure.subplots_adjust(
            left=0.083, right=0.988, bottom=0.205, top=0.91, wspace=0.34
        )

        x = [row.k_kimg for row in rows]
        x_labels = [f"{row.k_kimg:.3f}" for row in rows]
        r_grad = [100.0 * row.r_grad for row in rows]
        r_opt = [100.0 * row.r_opt for row in rows]
        c_k_star = [row.c_k_star for row in rows]

        residual_axis.plot(
            x,
            r_grad,
            color=BLUE,
            marker="o",
            markersize=4.5,
            markeredgewidth=0.7,
            label=r"$R_{\mathrm{grad}}$",
            zorder=3,
        )
        residual_axis.plot(
            x,
            r_opt,
            color=VERMILLION,
            linestyle=(0, (4.0, 2.0)),
            marker="s",
            markersize=4.3,
            markeredgewidth=0.7,
            label=r"$R_{\mathrm{opt}}$",
            zorder=3,
        )
        for index, (k_kimg, value) in enumerate(zip(x, r_opt)):
            offset = 7 if index in (0, 2) else -12
            residual_axis.annotate(
                f"{value:.2f}%",
                (k_kimg, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                color=INK,
                fontsize=7.0,
                zorder=4,
            )
        residual_axis.set_xscale("log", base=2)
        residual_axis.set_xlim(x[0] / 1.12, x[-1] * 1.12)
        residual_axis.set_ylim(0.0, 12.6)
        residual_axis.yaxis.set_major_locator(MultipleLocator(2.0))
        residual_axis.set_xticks(x, x_labels)
        residual_axis.set_xlabel(r"Restored state $K$ (kimg)")
        residual_axis.set_ylabel("Reference-normalized residual (%)")
        residual_axis.legend(
            loc="lower left",
            ncols=2,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.0,
            borderaxespad=0.15,
        )
        residual_axis.text(
            0.0,
            1.035,
            "(a)",
            transform=residual_axis.transAxes,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        style_axis(residual_axis)

        scale_axis.axhline(
            1.0,
            color=MID_GRAY,
            linewidth=0.85,
            linestyle=(0, (3.0, 2.0)),
            zorder=1,
        )
        scale_axis.plot(
            x,
            c_k_star,
            color=INK,
            marker="D",
            markersize=4.2,
            markeredgewidth=0.7,
            zorder=3,
        )
        for index, (k_kimg, value) in enumerate(zip(x, c_k_star)):
            offset = 7 if index in (0, 2) else -12
            scale_axis.annotate(
                f"{value:.3f}",
                (k_kimg, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                color=INK,
                fontsize=7.0,
                zorder=4,
            )
        scale_axis.set_xscale("log", base=2)
        scale_axis.set_xlim(x[0] / 1.12, x[-1] * 1.12)
        scale_axis.set_ylim(0.998, 1.041)
        scale_axis.yaxis.set_major_locator(MultipleLocator(0.01))
        scale_axis.set_xticks(x, x_labels)
        scale_axis.set_xlabel(r"Restored state $K$ (kimg)")
        scale_axis.set_ylabel(r"Candidate LR multiplier $c_K^*$")
        scale_axis.text(
            x[-1] / 1.02,
            1.0010,
            r"$c_K^*=1$",
            ha="right",
            va="bottom",
            color=MID_GRAY,
            fontsize=7.0,
        )
        scale_axis.text(
            0.0,
            1.035,
            "(b)",
            transform=scale_axis.transAxes,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        style_axis(scale_axis)

        outdir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for extension in formats:
            output = outdir / f"{stem}.{extension}"
            if extension == "pdf":
                metadata: dict[str, object] | None = {
                    "Title": "State-conditioned residuals along one frozen Arm A trajectory",
                    "Creator": "scripts/plot_same_trajectory_longitudinal.py",
                    "CreationDate": None,
                    "ModDate": None,
                }
            elif extension == "svg":
                metadata = {
                    "Title": "State-conditioned residuals along one frozen Arm A trajectory",
                    "Creator": "scripts/plot_same_trajectory_longitudinal.py",
                    "Date": None,
                }
            else:
                # Omit PNG timestamps and software chunks so repeated renders
                # are byte-for-byte stable as well as visually identical.
                metadata = None
            figure.savefig(
                output,
                format=extension,
                dpi=png_dpi if extension == "png" else 300,
                facecolor="white",
                metadata=metadata,
            )
            if extension == "svg":
                # Matplotlib's path serializer leaves spaces at line ends;
                # normalize them so the checked-in vector passes diff checks.
                lines = output.read_text(encoding="utf-8").splitlines()
                output.write_text(
                    "\n".join(line.rstrip() for line in lines) + "\n",
                    encoding="utf-8",
                )
            outputs.append(output)
        plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/same_trajectory_longitudinal/longitudinal_summary.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/same_trajectory_longitudinal"),
    )
    parser.add_argument("--stem", default="same_trajectory_residuals")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=("pdf", "svg", "png"),
    )
    parser.add_argument("--png-dpi", type=int, default=600)
    args = parser.parse_args()

    if args.png_dpi <= 0:
        raise SystemExit("--png-dpi must be positive")
    rows = read_summary(args.summary.resolve())
    outputs = render_main_figure(
        rows,
        args.outdir.resolve(),
        stem=args.stem,
        formats=tuple(args.formats),
        png_dpi=args.png_dpi,
    )
    print("Wrote " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
