# Cross-seed optimizer geometry protocol

## Question

At the 256-kimg Arm-A endpoints for training seeds 3, 4, and 5, does the
same-state paired current-gap RAdam geometry vary in parallel with the already
observed reversal of the fresh-state-control quality effect?

This is a descriptive mechanism screen, not a causal test. It does not repeat
the earlier seed-3 longitudinal points at 32/64/128/256 kimg. There is exactly
one endpoint audit per seed, all at 256 kimg.

## Frozen comparison

For every seed, restore the complete Arm-A state
`z_256 = (theta, m, v, n, GradScaler)` from `training-state-000008.pt`, then
make two disposable virtual branches. They share the minibatch, `t`, noise,
and dropout RNG state and differ only by current `global_gap_scale`: `g=1.0`
versus `g=1.3`. Neither branch may alter the restored source state.

The runner reads a host-bound copy of
`configs/cross_seed_optimizer_geometry_matrix.example.json`. Before a real
run, replace only its marked path placeholders, preserve the remaining frozen
design values, and save that exact manifest beside the external outputs.

## Required per-seed gates

- Arm A, global sigmoid, `g=1.0`, 256 kimg, and the numbered final endpoint;
- complete nontrivial RAdam state and restored GradScaler state;
- paired randomness contract satisfied;
- neither virtual optimizer step skipped;
- source parameters, optimizer state, and GradScaler state preserved;
- receipt reports finite `R_grad`, `R_opt`, `c_K_star`, `s_K_star`, and
  on-support `h_i` dispersion;
- `H_K = R_opt` identity check passes. This is a receipt-integrity check, not
  a second mechanism measurement.

## Commands

Local, no-checkpoint inspection:

```bash
python scripts/run_cross_seed_optimizer_geometry_audit.py \
  --manifest configs/cross_seed_optimizer_geometry_matrix.example.json \
  --out /tmp/cross-seed-optimizer-geometry \
  --dry-run
```

After binding external checkpoint paths, run the audit in a Torch/CUDA
environment and write to a new external directory:

```bash
python scripts/run_cross_seed_optimizer_geometry_audit.py \
  --manifest /path/to/bound-matrix.json \
  --out /path/to/new/cross-seed-optimizer-geometry \
  --python /path/to/ect-python
```

Then generate the table:

```bash
python scripts/summarize_cross_seed_optimizer_geometry.py \
  --manifest /path/to/bound-matrix.json \
  --audit-root /path/to/new/cross-seed-optimizer-geometry \
  --out /path/to/new/cross-seed-optimizer-geometry/summary
```

## Interpretation boundary

The table juxtaposes fixed, disjoint-block quality-control deltas `C−B` with
local endpoint geometry. The quality inputs are context only. The optimizer
audit neither recomputes FID/KID nor identifies a causal pathway.

If the three geometry rows differ, they are a lead for a subsequent
predeclared mechanism test. If they are numerically similar, that is negative
evidence against this local geometry as an explanation for the endpoint
reversal. In either case, report raw values, sample SD, and range. A claim of
“nearly the same” additionally requires a tolerance decided before interpreting
the completed table.
