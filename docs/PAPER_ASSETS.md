# Paper assets

## Asset A — FID versus training budget

Use `scripts/render_paper_asset_a.py` to turn one protocol-matched learning
curve matrix into the paper asset.  The figure has training budget in kimg on
the horizontal axis and FID on the vertical axis.  It requires the four
checkpoints `256, 512, 768, 1024` for every method and seed.

The source CSV follows the long-form schema used by
`scripts/collect_multibudget_results.py`.  For a plotted curve it must have
one exact `metric_name` (normally `fid5k_full`), NFE, `analysis_track`,
`evaluation_contract`, sample count, generation-seed range, and metric seed.
The renderer refuses partial trajectories and refuses to join FID-5k and
FID-50k in one line.

The first two selected seeds are the main A/B curves.  C/D are contextual
curves: use `--secondary-mode faded` (the default) to make them pale in the
main panel, or `--secondary-mode panel` for a second panel.

```bash
python scripts/render_paper_asset_a.py \
  --input-csv results/q256_learning_curve/evaluation_results.csv \
  --outdir paper_assets/asset_a \
  --metric-name fid5k_full --nfe 2 \
  --analysis-track budget_curve \
  --primary-seeds 3,4 --seed-labels 3=A,4=B,5=C,6=D
```

The command emits SVG, PDF, and PNG versions of `asset_a_fid_vs_budget`, a
normalized figure-data CSV, and a provenance manifest with hashes.  The 1024
kimg FID-5k point is an auxiliary same-protocol curve point; a 1024 kimg
FID-50k value belongs in a separate endpoint asset, never this trajectory.

## Asset B — Compute-to-quality

Asset B is driven by a version-controlled frozen threshold JSON.  It defines
the threshold, metric, NFE, two paired arms, protocol, and checkpoint budgets
before results are rendered.  `crossing_mode` is one global choice for an
asset: `first_observed` reports the first checkpoint at or below the threshold;
`linear_interpolation_descriptive` labels every crossing as a descriptive
between-checkpoint estimate.  The renderer rejects a chart that would mix
those definitions, and it rejects an interpolation that is unbracketed before
the first checkpoint.

For each seed it writes both `tau_A`, `tau_B`, and
`Δtau = tau_B − tau_A` in kimg.  The figure keeps each paired seed as a thin
connector; median lines are summaries only.  The manifest flags the special
case in which every observed paired `Δtau` is negative, without making a
population-level claim.

## Asset C — Arm dispersion contraction

Asset C uses a frozen arm configuration that maps A/B/C/D to four distinct
methods.  At every seed and checkpoint it computes
`S_s(K) = max_a FID_{s,a}(K) − min_a FID_{s,a}(K)`.  The output includes the
complete seed-by-budget values, a mean/median summary, and each seed's change
from the first to final budget.  It refuses incomplete four-arm cells and
refuses to generate a single-seed mean-only figure.

## Asset D — Seed-resolved reporting rule

Every paper plot must retain one visible mark or thin trajectory per seed.
Mean and median are permitted only as explicitly labelled overlays or summary
rows; they may not replace the seed-level data.  This is a hard renderer rule
for Assets A–C, so heterogeneous seeds cannot be hidden by an aggregate curve.
