# Seed-Replication Blind Adjudication

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: ANALYZED (machine recommendation only; independent
  quality-blind review is required for final acceptance)
- Version Label: gap_lr_seed_replication_blind_adjudication_v1

## Purpose and temporal boundary

This is a post-run evidence and adjudication layer for the seed-4/5 extension
of `gap_lr_matched_q128_s3_v1`. It does not amend the preregistered matrix or
retroactively redefine protocol commit
`583c2fe0f914fc1191903d747737fd54b4ba1eef`.

The accept-versus-rerun decision is frozen before any seed-4/5 FID, KID, or
other generation-quality result is opened. The evidence builder reads only
training configurations and artifacts, per-run integrity receipts, the frozen
transfer checkpoint and dataset metadata, runtime provenance, and launcher
logs. Quality outputs are excluded inputs.

## Observed deviations

Four facts require adjudication rather than a claim of protocol-exact
execution:

1. The original fail-stop launcher ended after seed 4 arm A. Two manual
   recovery launchers freshly started the remaining arms; none resumed a
   trained state.
2. Seed 4 B/C and seed 5 ran concurrently on two logged GPU indices, although
   the execution policy specified one fully serial seed group at a time.
3. Seed 4 B/C used logged GPU index 0 instead of the planned index 1. Both
   devices are the same A100 model, memory capacity, and driver; the public
   evidence hashes rather than publishes their full UUIDs.
4. Seed 5 arm A's generated `model_init.png` differs from B/C by at most one
   8-bit level. The observed pattern is compatible with an FP16/cuDNN forward
   followed by PNG quantization; this preview is not a parameter hash.
5. The original seed-4-A verifier output and exit status were not preserved;
   the same artifact set passed a later strengthened re-verification.

## Initialization question

The historical processes did not save a hash immediately after transfer copy
and before their first forward pass. The package therefore must not claim an
`observed_preupdate_hash`.

The reconstruction script instead replays the frozen initialization path from
each receipt-bound `training_options.json`, the hash-bound transfer checkpoint,
the dataset-derived `(img_resolution, img_channels, label_dim)` interface, and
the frozen implementation. It uses the actual destination-iterating name-based
copy semantics and canonicalizes all parameters and buffers with
`ECT_CANONICAL_TORCH_MODULE_V1`:

- tensors ordered by UTF-8 fully qualified name and kind;
- kind, name, dtype, rank, shape, byte count, and raw bytes length-prefixed;
- raw bytes produced by detach, CPU copy, contiguous row-major layout, and
  little-endian representation;
- module mode, `requires_grad`, and non-tensor attributes excluded and bound
  separately through the configuration/code receipt.

The result is correctly named
`reconstructed_expected_initialization_hash`. It proves what the frozen inputs
and code uniquely imply, not what was independently attested inside historical
process memory.

## Decision policy

The machine adjudicator returns `machine_recommends_acceptance` only if:

- all six strengthened per-run receipts pass and every listed artifact hash is
  recomputed;
- all runs reach 256 kimg and retain finite final state/EMA artifacts;
- attempted iterations, successful updates, AMP skips, and final GradScaler
  values agree between CSV and state;
- A/B/C and cross-seed normalized configurations differ only on the frozen
  axes;
- the transfer covers every destination parameter/buffer with no shape or
  dtype mismatch, and all six reconstructed expected initialization hashes are
  equal;
- preview drift is no greater than one 8-bit level;
- concurrent intervals occur on distinct logged indices whose devices have
  the same model, memory, and driver;
- all deviations, evidence limitations, and excluded claims remain explicit.

Any failure of those gates returns `rerun_required`. A machine recommendation
does not authorize quality evaluation. Final
`accepted_with_documented_deviation` requires an independent quality-blind
review bound to the exact candidate/evidence hashes. It must preserve the
exclusions of protocol-exact execution, historically observed bitwise
initialization identity, cross-device bitwise training equivalence, and
performance comparability.

## Evidence products

The Git-tracked package contains:

- six path-sanitized public per-run integrity receipts;
- one initialization reconstruction receipt;
- one objective runtime/configuration/deviation evidence receipt;
- one final blind-adjudication receipt.

Large checkpoints, states, raw logs, datasets, transfer archives, full GPU
UUIDs, host/account/IP identifiers, and absolute server paths remain external
to Git. Public receipts bind retained artifacts by SHA256 and byte size.

## Reproduction commands

Run the strengthened per-run verifier for all six runs first, then:

```bash
PYTHONPATH=. python scripts/reconstruct_gap_lr_seed_initialization.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --integrity-receipt-dir "$INTERNAL_RECEIPTS" \
  --data "$DATA" \
  --transfer "$TRANSFER" \
  --repo "$REPO" \
  --adjudication-tooling-commit "$ADJUDICATION_TOOLING_COMMIT" \
  --output "$AUDIT_DIR/initialization_reconstruction.json"

PYTHONPATH=. python scripts/build_gap_lr_seed_replication_blind_evidence.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --internal-receipt-dir "$INTERNAL_RECEIPTS" \
  --initialization-reconstruction "$AUDIT_DIR/initialization_reconstruction.json" \
  --original-launcher-log "$ORIGINAL_LAUNCHER_LOG" \
  --adjudication-tooling-commit "$ADJUDICATION_TOOLING_COMMIT" \
  --public-receipt-dir "$AUDIT_DIR/public_receipts" \
  --output "$AUDIT_DIR/blind_evidence.json"

python scripts/adjudicate_gap_lr_seed_replication.py \
  --evidence "$AUDIT_DIR/blind_evidence.json" \
  --initialization-reconstruction "$AUDIT_DIR/initialization_reconstruction.json" \
  --public-receipt-dir "$AUDIT_DIR/public_receipts" \
  --adjudication-tooling-commit "$ADJUDICATION_TOOLING_COMMIT" \
  --output "$AUDIT_DIR/blind_adjudication.json"
```
