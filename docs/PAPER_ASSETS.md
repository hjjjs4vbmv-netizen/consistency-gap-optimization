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
