# q=256 target-component audit — Amendment 001

Date: 2026-08-25 (Asia/Singapore)  
Status: **frozen before v2 regeneration and before matrix-level result analysis**

## Trigger

The first complete v1 artifact set passed every per-cell CUDA, identity,
state-preservation, and artifact-publication gate. The frozen matrix validator
then rejected the set because `trajectory_config_sha256` differed between the
256-kimg source state and the 512/768/1024-kimg continuation states.

No matrix table, cross-budget summary, or learning-curve alignment was produced
before this amendment. The only inspected gradient values were the declared
two-state smoke outputs and the seed-3/256-kimg timing cell reported while the
matrix was running.

## Diagnosis

A direct read-only comparison of the embedded seed-3 trajectory configs showed
exactly one unequal field: `total_kimg` was 256 in the source run and 1024 in
the continuation. Loss, network, optimizer, data, RNG/runtime, batching, and
all other trajectory fields were byte-equivalent after serialization. The
original validator therefore conflated a declared terminal horizon with the
dynamics contract it intended to hold fixed.

## Bounded correction

Version 2 retains and validates the raw `trajectory_config_sha256`, records
`trajectory_total_kimg`, and additionally records
`trajectory_dynamics_sha256`, defined as the canonical trajectory-config hash
after removing exactly the top-level `total_kimg` field. Matrix admission now
requires:

1. `trajectory_total_kimg` is an integer at least as large as the audited
   checkpoint budget;
2. the raw config digest remains present for provenance; and
3. `trajectory_dynamics_sha256` agrees across budgets within each seed.

No other gate, estimand, seed, budget, batch, RNG, precision, tolerance, model
state, or interpretation rule changes. The runner, validator, protocol, and
this amendment are hash-bound into every v2 artifact. All v1 measurement
directories remain preserved as failed-matrix-gate provenance and are excluded
from the admitted primary matrix.
