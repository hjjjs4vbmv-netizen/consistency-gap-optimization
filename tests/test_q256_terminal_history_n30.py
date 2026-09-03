import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "analysis" / "q256_terminal_history_n30_matpool_v1"
RESULTS = EXPERIMENT / "final_results"
EXPECTED_PROTOCOL_SHA256 = "317d3ef93102050276c1366d9633e322d60fbc9000cd56c8fc8a24c1d4eef544"
EXPECTED_MISSING = {
    (58, "AA"),
    (58, "BA"),
    (65, "AA"),
    (67, "AA"),
    (68, "AA"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_protocol_hash_is_frozen() -> None:
    assert sha256(EXPERIMENT / "protocol.json") == EXPECTED_PROTOCOL_SHA256
    assert (EXPERIMENT / "protocol.sha256").read_text().split()[0] == EXPECTED_PROTOCOL_SHA256


def test_secondwave_assignments_are_disjoint_and_complete() -> None:
    endpoints = []
    expected_gpu_counts = {"eval5": 5, "eval6": 6, "single1": 1, "single2": 1}
    for worker, gpu_count in expected_gpu_counts.items():
        assignments = json.loads((EXPERIMENT / f"eval_assignments_{worker}.json").read_text())
        assert set(assignments) == {str(index) for index in range(gpu_count)}
        endpoints.extend((int(seed), str(cell)) for jobs in assignments.values() for seed, cell in jobs)
    expected = {
        (seed, cell)
        for seed in [*range(58, 66), *range(73, 80)]
        for cell in ("AA", "BA")
    }
    assert len(endpoints) == 30
    assert len(set(endpoints)) == 30
    assert set(endpoints) == expected


def test_final_result_checksums_and_coverage() -> None:
    for line in (RESULTS / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert sha256(RESULTS / name.strip()) == expected

    rows = read_csv(RESULTS / "combined_results.csv")
    keys = {(int(row["seed"]), row["cell"]) for row in rows}
    planned = {(seed, cell) for seed in range(50, 80) for cell in ("AA", "BA")}
    assert len(rows) == 55
    assert len(keys) == 55
    assert planned - keys == EXPECTED_MISSING

    failures = read_csv(RESULTS / "scientific_failures.csv")
    assert {(int(row["seed"]), row["cell"]) for row in failures} == EXPECTED_MISSING


def test_recorded_paired_statistics_match_pair_table() -> None:
    pairs = read_csv(RESULTS / "paired_results.csv")
    stats = json.loads((RESULTS / "statistics.json").read_text())
    contrasts = [float(row["log_fid_contrast_ba_minus_aa"]) for row in pairs]
    mean = sum(contrasts) / len(contrasts)

    assert len(pairs) == 26
    assert sum(value < 0 for value in contrasts) == 22
    assert sum(value > 0 for value in contrasts) == 4
    assert math.isclose(mean, stats["mean_log_fid_contrast"], rel_tol=0, abs_tol=1e-15)
    assert stats["complete_pairs_n"] == 26
    assert stats["classification"] == "DIRECTIONAL_NEGATIVE"
    assert stats["ci95"][1] < 0
    assert stats["ci90"][1] < -math.log(1.03)
    assert stats["tost"]["equivalent_at_0.05"] is False
    assert stats["arms"]["AA"]["available_n"] == 26
    assert stats["arms"]["BA"]["available_n"] == 29
