#!/usr/bin/env python3
"""Check manuscript assets and archived data without rerunning training.

Run after scripts/build.sh. Optional final-pass TeX logs and AUX files provide
additional compiler checks; this is an artifact check, not a training audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import fitz


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def csv_count(path: str) -> int:
    with (ROOT / path).open(newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-log", type=Path)
    parser.add_argument("--tex-aux", type=Path)
    args = parser.parse_args()

    manifest = read_json("data/source_manifest.json")
    for entry in manifest:
        path = ROOT / entry["file"]
        require(path.is_file(), f"missing source: {entry['file']}")
        require(sha256(path) == entry["sha256"], f"source hash: {entry['file']}")

    tex_files = [ROOT / "main.tex", *sorted((ROOT / "sections/en").glob("*.tex"))]
    tex = "\n".join(path.read_text() for path in tex_files)
    tex = re.sub(r"(?<!\\)%[^\n]*", "", tex)
    labels = re.findall(r"\\label\{([^{}]+)\}", tex)
    duplicate_labels = [key for key, n in Counter(labels).items() if n > 1]
    require(not duplicate_labels, f"duplicate labels: {duplicate_labels}")
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^{}]+)\}", tex))
    require(refs <= set(labels), f"unknown references: {sorted(refs - set(labels))}")
    bib = (ROOT / "references.bib").read_text()
    bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib)
    require(len(bib_keys) == len(set(bib_keys)), "duplicate bibliography keys")
    cited = set()
    for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^{}]+)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    require(cited <= set(bib_keys), f"missing citations: {sorted(cited - set(bib_keys))}")

    inputs = re.findall(r"\\(?:input|inputrows)\{([^{}]+)\}", tex)
    for name in inputs:
        path = ROOT / name
        if not path.suffix:
            path = path.with_suffix(".tex")
        require(path.is_file(), f"missing TeX input: {name}")

    figure_names = re.findall(r"\\figfile\{([^{}]+)\}", tex)
    require(len(figure_names) == len(set(figure_names)), "repeated figure asset")
    for name in figure_names:
        path = ROOT / "figures/en" / f"{name}.pdf"
        require(path.is_file(), f"missing figure: {name}")
        with fitz.open(path) as figure:
            require(len(figure) == 1, f"figure is not a single-page PDF: {name}")

    pdf_path = ROOT / "compiled/ECT_Draft_EN.pdf"
    with fitz.open(pdf_path) as pdf:
        page_texts = [page.get_text() for page in pdf]
        require(all(len(text.strip()) > 120 for text in page_texts),
                "near-empty page; inspect the compiled PDF")
        require("\ufffd" not in "\n".join(page_texts), "replacement glyph in PDF text")
        reference_page = next((i + 1 for i, text in enumerate(page_texts)
                               if re.search(r"\nreferences\s*\n", text, re.IGNORECASE)), None)
        require(reference_page is not None, "bibliography heading missing from PDF")
        pdf_pages = len(pdf)

    diagnostics = "Not supplied; PDF and source structure checked."
    if args.tex_log:
        log = args.tex_log.read_text(errors="replace")
        bad = re.findall(
            r"^.*(?:Overfull|Undefined control sequence|Missing character|"
            r"LaTeX Error|(?:Citation|Reference).*undefined|"
            r"There were undefined|multiply defined).*?$", log, re.MULTILINE)
        require(not bad, "TeX diagnostics: " + " | ".join(bad))
        diagnostics = "Final-pass log checked: no undefined citations/references, missing glyphs, overfull boxes, or LaTeX errors."
    counters = {}
    if args.tex_aux:
        aux = args.tex_aux.read_text()
        for kind in ("figure", "table"):
            counters[kind] = len(re.findall(
                r"\\contentsline\s*\{" + kind + r"\}", aux))

    summary = read_json("data/startup_summary.json")
    window = read_json("data/window_recomputation.json")
    require(window["status"] == "PASS" and window["n"] == 8, "window reconstruction missing")
    require(window["block_count"] == 72 and window["reused_blocks"] == 48, "window block count")
    require(csv_count("data/raw/q128_complete_per_seed.csv") == 8, "q128 seed rows")
    require(csv_count("data/raw/q128_effects.csv") == 4, "q128 preset contrasts")
    report = {
        "status": "PASS",
        "scope": "Included source hashes, TeX dependencies, figure PDFs, and compiled-document structure; no training rerun or private checkpoint rehash.",
        "archived_source_hashes_verified": len(manifest),
        "unique_labels": len(labels),
        "bibliography_entries": len(bib_keys),
        "cited_entries": len(cited),
        "figure_assets": len(figure_names),
        "generated_table_row_files": len(list((ROOT / "data").glob("*_rows.tex"))),
        "component_outcome_rows": csv_count("data/raw/component_outcomes.csv"),
        "startup_seeds": summary["n"],
        "startup_snapshot": summary["source_commit"],
        "window_snapshot": window["source_commit"],
        "window_seeds": window["n"],
        "q128_seeds": 8,
        "q128_contrasts": 4,
        "window_blocks": window["block_count"],
        "window_reused_control_values": window["control_values_exactly_reconciled_with_PR111"],
        "window_telemetry_attempts": window["telemetry"]["attempts"],
        "pdf": {"file": str(pdf_path.relative_to(ROOT)), "pages": pdf_pages,
                "reference_start_page": reference_page, "sha256": sha256(pdf_path)},
        "tex_counters": counters,
        "compiler_diagnostics": diagnostics,
        "visual_review": "Requires human inspection; not inferred from these checks.",
    }
    (ROOT / "data/artifact_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
