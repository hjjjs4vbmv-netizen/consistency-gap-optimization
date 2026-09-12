# Beyond Pair Spacing: Loss Scale and Startup Updates in Consistency Tuning

Anonymous English research draft, updated 12 September 2026 from the existing PR112 Overleaf project. This edition adds the complete PR113 fresh q128 cohort and a bounded audit of existing records. **New training and generation evaluation: 0 GPUh.**

Upload the ZIP to Overleaf, select `main.tex`, and use pdfLaTeX. All figure PDFs and table rows are included; compilation requires no experimental assets or Python. The local preview is `compiled/ECT_Draft_EN.pdf`. `bash scripts/build.sh` supports latexmk/pdfLaTeX or Tectonic. This preview was compiled with Tectonic; the style and anonymous structure are retained.

The narrative connects coupled target/denominator changes, the q256 denominator-history evidence, startup/window interventions in the responder-enriched cohort, and the fresh q128 cohort's limits on generalization. The q128 primary contrast is not supported as a stable benefit. Local update changes are visible, but existing evidence does not explain their difference from endpoint response.

- `sections/en/paper.tex`: revised abstract, contributions, q128 results, interpretation and conclusion.
- `sections/en/appendix.tex`: all eight q128 seed outcomes, preset effects, local exposure readings and audit boundaries.
- `figures/en/q128_*.{pdf,png,svg}`: seed/effect intervals and descriptive local response.
- `data/q128_*.csv`, `data/q128_*_rows.tex`: generated complete tables.
- `sources/pr113/`: unchanged result records and original verifier; older source snapshots remain historical.
- `data/source_manifest.json`: hashes and identities of preserved source files.
- `BUILD_INFO.json`, `data/artifact_validation.json`: actual final build checks.

The companion repository directory `analysis/research_closeout_v1` contains the public redacted evidence, original/extracted/public hash bindings, CPU analysis scripts and Chinese closeout report. Rebuild q128 diagnostics and figures there with `scripts/audit_existing.py` and `scripts/make_tables_figures.py`; these are not needed for Overleaf compilation. Original PR112 plotting scripts remain available for their original figures.

Reviewed PR113 results: `42871769be317f6baa1bd6e030bec120c22d9371`; PR112: `3d1330f5d794d05fe63b67aa40e2dc6213f1387f`. Scientific deployment: `fcc94d26b1085011ac935084f7c1cf4c149e2947`; evaluator: `d6aba02fb88e9db0993623895eb2228ed717d810`. Result commits are not substituted for execution identities.

All 32 q128 paths have matching initial receipts and continuous telemetry through successful update 16. Endpoint checks are supported by original worker receipts and bound checkpoint seals; this round does not reload checkpoints. An independent full boundary-state comparison is available for seed302/DA only. Old q256 early net displacement remains missing. Public evidence is a post-hoc supplement, not a newly invented prospective freeze.

The paper remains a research draft. Anonymous placeholder authors are retained. `\iclrfinalcopy` removes preview rulers and does not signify acceptance. Venue-specific page limits and submission settings remain an author decision.
