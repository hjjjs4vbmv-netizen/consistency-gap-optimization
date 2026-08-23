# ICLR Figure Contract and QA Record

Audited on 2026-08-23. PDF files are the manuscript assets, SVG files are editable masters, and PNG files are raster previews. All four figures were inspected at rendered manuscript size in the compiled 11-page PDF.

## Figure contracts

| Figure | Core conclusion | Evidence/transform | Replicate and display rule | Export |
|---|---|---|---|---|
| Figure 1 — `figure1_composite_intervention` | Pair spacing changes the detached target endpoint and explicit denominator; matched-state factorization precedes separate training trajectories. | Conceptual rendering of exact theory; no numerical data. | Labels $g=1.10$ as a controlled probe, not a universal method. | PDF 442.7×165.4 pt (PDF 1.4); SVG; 1847×689 PNG at 300 dpi. |
| Figure 2 — `figure2_seed_resolved_learning_curves` | Formal arm separations contract and cross with seed, budget, and NFE. | 168 rows from the formal deterministic replay source. | Every training seed and all four arms shown; budget/NFE are repeated readouts. | PDF 396×249 pt; SVG; 3300×2078 PNG at 600 dpi. |
| Figure 3 — `figure3_time_to_quality_seed_resolved` | The complete intervention is earlier for several trajectories but not all. | First observed NFE1 FID-50k $\leq10$ from the 456-row unified source; no interpolation. | Ten seed-level A/B connectors separated by evidence group; open markers show censoring; seeds 17/18 retained. | PDF 452.2×257.4 pt (PDF 1.4); SVG; 1885×1073 PNG at 300 dpi. |
| Figure 4 — `figure4_budget_dependent_factorial_contrasts` | Complete, target, denominator, and interaction FID contrasts change with horizon. | Formal NFE1 contrasts from the exact-budget replay. | All three seed paths shown; thick line is a descriptive arithmetic mean, not a replicate. | PDF 396×260 pt; SVG; 3300×2173 PNG at 600 dpi. |

## Data-integrity gates

- Formal replay source: 3 seeds × 4 arms × 7 budgets × 2 NFE = 168 rows; no missing or duplicate design cells; all jobs PASS.
- Unified curve source: 168 formal replay + 48 secondary A/B + 240 secondary factorial = 456 rows.
- Time-to-quality output: 10 training seeds; classification exactly `{B earlier: 5, tie: 2, B later: 2, censored: 1}`.
- Evidence-group, status, PR, commit, and source-file provenance columns remain in the unified CSV.
- Figure 2/4 source SHA-256 is retained in the frozen upstream audit; unified source SHA-256 is `35cb722bab101cb85feb4e15e4e5a80bccd86bf9a585dd49dcb9f5963a77f872`.
- Time-to-quality source SHA-256 is `451a357e83e79bd3f35159647d13ef1b3d63ca390da59b0e772d32328f6e358d`.

## Visual QA

- No clipped labels, overlapping panels, missing glyphs, or unreadable legends were observed.
- Figure 1 preserves the two intervention branches and visibly separates matched-state factorization from finite training.
- Figure 2 remains legible on a log scale; A/B emphasis does not hide C/D.
- Figure 3 visibly separates evidence groups, displays both later seeds, and uses an open marker for censoring.
- Figure 4 preserves zero, sign, individual seed trajectories, and the descriptive mean on the pseudo-log axis.
- Captions identify the training-seed unit, repeated readouts, transforms, censoring, and descriptive status where applicable.

## Builder

`docs/scripts/build_manuscript_assets.py` rebuilds Figures 1 and 3 plus the unified/time-to-quality source tables and asserts the frozen classification counts. Figures 2 and 4 retain their audited upstream generation path.
