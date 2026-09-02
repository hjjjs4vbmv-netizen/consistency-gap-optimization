# Operator-clock Jacobian audit

This directory implements three deliberately distinct predictors:

1. `squared_loss_simplified_operator`: the theoretical squared-pair baseline
   `I - eta sum w J_i^T(J_i-J_j)`. It is never reported as the true ECT map.
2. `recompute_and_detach_field_jacobian`: central finite differences of the
   complete gradient-field evaluation. Every `theta +/- epsilon*u` evaluation
   reruns both forwards and detaches the newly recomputed target inside that
   evaluation.
3. `full_algorithmic_state_transition_jacobian`: central finite differences
   of a cloned one-step transition over parameters, floating model buffers,
   RAdam moments, GradScaler scale, EMA parameters/buffers, and the associated
   discrete step/scale decisions.

`protocol.json` freezes the four A/B/C/D arms, four minibatch IDs, eight
projection seeds, four horizons, and the epsilon sweep before formal results.
It also freezes seed 3 at q256/256 kimg by the result-independent rule
"lowest eligible seed in the archived verified fixed-baseline source manifest"
and records the source/dataset SHA256 values.
JVP directions use a state-relative per-tensor RMS convention so perturbations
remain resolvable in a large FP32 model; positive second-moment/scaler
coordinates are clipped before any result is observed so both FD branches stay
inside the valid state space.
The field and algorithmic maps use separate frozen epsilon grids after a
single pre-formal correctness calibration cell. The strict 5% adjacent-change
gate remains unchanged. AMP `+/-` skip equality and a constant AMP regime
across the full epsilon sweep are both required; a discontinuity is retained
as `FAIL_CLOSED`, not coerced into a Jacobian.
The runners require trusted, matching artifacts:

```bash
python analysis/operator_clock_gate/run_field_jvp.py \
  --training-state /path/training-state-latest.pt \
  --checkpoint /path/network-snapshot-latest.pkl \
  --batch-file /path/four-frozen-batches.pt

python analysis/operator_clock_gate/run_algorithmic_jvp.py \
  --training-state /path/training-state-latest.pt \
  --checkpoint /path/network-snapshot-latest.pkl \
  --batch-file /path/four-frozen-batches.pt

python analysis/operator_clock_gate/run_matched_micro_rollout.py \
  --training-state /path/training-state-latest.pt \
  --checkpoint /path/network-snapshot-latest.pkl \
  --batch-file /path/four-frozen-batches.pt
```

The batch file is a trusted `torch.save` artifact with a `batches` list. Each
entry is either `(images, labels)` or `{"images": ..., "labels": ...}`. The
runners hash all source files and fail before compute when an explicitly
provided `--expected-*-sha256` does not match.

Create that file without inspecting model results using:

```bash
python analysis/operator_clock_gate/prepare_frozen_batches.py \
  --data /path/cifar10-32x32.zip \
  --expected-data-sha256 a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1 \
  --out /path/four-frozen-batches.pt
```

The field and algorithmic runners accept `--shard-index i --num-shards n`.
Assignment is the frozen lexicographic A/B/C/D, batch, direction order modulo
`n`; sharding changes execution placement only, not scientific selection.

Formal outputs go to `results/raw_receipts/` by default. JSON receipts contain
the state/RNG preservation checks, epsilon convergence table, AMP skip pairing,
and hashes of raw JVP tensors; the JVP tensors are stored beside them as `.pt`.

After all shards and the rollout finish, validate the exact 128+128 coverage,
JSON/PT pairing, source preservation, and matched horizons with
`summarize_formal_results.py --results ... --out formal_summary.json`.
