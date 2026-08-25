#!/usr/bin/env python3
"""Quality-blind calibration for the frozen q128 matched-spacing protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = "ect.q128-matched-spacing-calibration/v1"
DEFAULT_SEED = 20260824
DEFAULT_SAMPLE_COUNT = 1_000_000
DEFAULT_SEARCH_INTERVAL = (0.25, 1.50)
DEFAULT_TOLERANCE = 1e-13
P_MEAN = -1.1
P_STD = 2.0
K = 8.0
B = 1.0
STAGE = 0
REFERENCE_Q = 256.0
REFERENCE_G = 1.10
TARGET_Q = 128.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid_adjustment(t: np.ndarray) -> np.ndarray:
    # sigmoid(-b*t) written in the stable negative-domain form.
    exp_neg = np.exp(-B * t)
    return 1.0 + K * exp_neg / (1.0 + exp_neg)


def realized_gap(t: np.ndarray, q: float, g: float) -> tuple[np.ndarray, np.ndarray]:
    adjustment = sigmoid_adjustment(t)
    base_gap = np.minimum(t, t * adjustment / (q ** (STAGE + 1)))
    intended = base_gap * float(g)
    gap = np.minimum(t, intended)
    clipped = intended > t
    return gap, clipped


def objective(t: np.ndarray, reference_log_gap: np.ndarray, g: float) -> float:
    candidate_gap, _ = realized_gap(t, TARGET_Q, g)
    residual = np.log(candidate_gap) - reference_log_gap
    return float(np.mean(np.square(residual, dtype=np.float64), dtype=np.float64))


def golden_section_search(t: np.ndarray, reference_log_gap: np.ndarray,
                          lower: float, upper: float, tolerance: float) -> tuple[float, int]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = float(lower)
    right = float(upper)
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = objective(t, reference_log_gap, c)
    fd = objective(t, reference_log_gap, d)
    iterations = 0
    while right - left > tolerance:
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = objective(t, reference_log_gap, c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = objective(t, reference_log_gap, d)
        iterations += 1
        if iterations > 256:
            raise RuntimeError("golden-section search did not converge")
    return (left + right) / 2.0, iterations


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    measured = np.quantile(values, probabilities)
    return {
        f"p{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, measured)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--search-lower", type=float, default=DEFAULT_SEARCH_INTERVAL[0])
    parser.add_argument("--search-upper", type=float, default=DEFAULT_SEARCH_INTERVAL[1])
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise SystemExit("sample-count must be positive")
    if not 0 < args.search_lower < args.search_upper:
        raise SystemExit("invalid search interval")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    standard_normal = rng.standard_normal(args.sample_count, dtype=np.float64)
    t = np.exp(standard_normal * P_STD + P_MEAN).astype("<f8", copy=False)
    sample_path = output_dir / "t_samples_f64.npy"
    np.save(sample_path, t, allow_pickle=False)

    reference_gap, reference_clipped = realized_gap(t, REFERENCE_Q, REFERENCE_G)
    reference_log_gap = np.log(reference_gap)
    golden_g, iterations = golden_section_search(
        t,
        reference_log_gap,
        args.search_lower,
        args.search_upper,
        args.tolerance,
    )
    # The unclipped stage-0 mapping has an exact cross-q candidate. Evaluate it
    # explicitly so the frozen value is stable across optimizer tolerances.
    exact_ratio_candidate = REFERENCE_G * TARGET_Q / REFERENCE_Q
    candidates = [golden_g, exact_ratio_candidate]
    selected_g = min(candidates, key=lambda value: objective(t, reference_log_gap, value))
    selected_g = float(round(selected_g, 15))
    selected_objective = objective(t, reference_log_gap, selected_g)
    candidate_gap, candidate_clipped = realized_gap(t, TARGET_Q, selected_g)

    manifest = {
        "schema": SCHEMA,
        "status": "FROZEN_PASS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "quality_blind": True,
        "quality_metrics_read": [],
        "calibration_rng": {
            "seed": args.seed,
            "algorithm": "NumPy PCG64",
            "numpy_version": np.__version__,
        },
        "sample_count": args.sample_count,
        "t_distribution": {
            "family": "lognormal",
            "construction": "exp(N(0,1) * P_std + P_mean)",
            "P_mean": P_MEAN,
            "P_std": P_STD,
            "dtype": "little-endian float64",
        },
        "t_sample_raw_sha256": sha256_bytes(t.tobytes(order="C")),
        "t_sample_file": sample_path.name,
        "t_sample_file_sha256": sha256_file(sample_path),
        "spacing_definition": {
            "schedule": "official sigmoid",
            "stage": STAGE,
            "k": K,
            "b": B,
            "base_r": "clamp(t * (1 - (1 + k*sigmoid(-b*t))/q^(stage+1)), min=0)",
            "realized_gap": "min(t, g * (t - base_r))",
            "clipping_rule": "r_g = t - min(t, g*(t-base_r)); 0 <= r_g <= t",
        },
        "reference": {"q": REFERENCE_Q, "g": REFERENCE_G},
        "target_q": TARGET_Q,
        "objective": "mean((log(delta_128_g(t)) - log(delta_256_1.10(t)))**2)",
        "search": {
            "interval": [args.search_lower, args.search_upper],
            "algorithm": "bounded golden-section plus exact-ratio candidate",
            "tolerance": args.tolerance,
            "iterations": iterations,
            "golden_section_solution": golden_g,
            "exact_ratio_candidate": exact_ratio_candidate,
            "optimum_on_boundary": selected_g in (args.search_lower, args.search_upper),
        },
        "selected_g128_star": selected_g,
        "selected_objective": selected_objective,
        "clipping_statistics": {
            "reference_fraction": float(np.mean(reference_clipped)),
            "candidate_fraction": float(np.mean(candidate_clipped)),
        },
        "spacing_quantiles": {
            "reference_delta": quantiles(reference_gap),
            "candidate_delta": quantiles(candidate_gap),
            "reference_delta_over_t": quantiles(reference_gap / t),
            "candidate_delta_over_t": quantiles(candidate_gap / t),
            "log_gap_residual": quantiles(np.log(candidate_gap) - reference_log_gap),
        },
        "calibration_source": str(Path(__file__).resolve()),
        "calibration_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "calibration_manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    (output_dir / "calibration_manifest.json.sha256").write_text(
        f"{sha256_bytes(payload.encode('utf-8'))}  calibration_manifest.json\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "selected_g128_star": selected_g,
        "selected_objective": selected_objective,
        "manifest": str(manifest_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
