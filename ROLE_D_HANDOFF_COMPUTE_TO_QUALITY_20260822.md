# Role D handoff — compute-to-quality and paper assets

**Version:** 2026-08-22
**Owner:** Role D / Alicia
**Priority:** P0 — paper-facing analysis
**Status:** active

## Current paper question

Can consistency-gap choice change the compute required to reach useful
generation quality?

The companion mechanism question is why an exact objective-level
factorization does not yield a seed-stable endpoint factorization. The paper
spine is:

```text
gap intervention
    -> target geometry + explicit weighting
    -> finite-budget trajectory effects
    -> compute-to-quality
    -> long-budget contraction
```

Role D turns frozen experimental outputs into seed-resolved,
protocol-matched, publication-ready evidence for that chain. It does not own
new optimizer-mechanism claims or training interventions.

## Evidence base

At q256 and 256 kimg, the four factorial arms are:

```text
A = (1.0, 1.0)    B = (1.1, 1.1)
C = (1.1, 1.0)    D = (1.0, 1.1)
```

The preregistered seeds are 3–5. Their complete 24/24 FID/KID-50k formal
matrix is recorded in
[`analysis/q256_target_weight_factorial/`](analysis/q256_target_weight_factorial/).
Seeds 6–7, 8–13, and 14–18 are secondary sensitivity analyses. They may be
shown descriptively, but they are not a pooled prospective confirmatory
sample. In particular, seed-resolved observations must accompany any
aggregate summary.

For seeds 3–5, all four arms have reportedly completed continuation to 1024
kimg. The handoff reports approximate FID ranges of 7.6–10.7 at NFE1 and
2.7–3.1 at NFE2. The raw, receipt-backed 1024-kimg cell records have not yet
been added to this workspace, so those ranges are not a renderable data
source.

## Assets

### Asset A — complete FID-versus-budget curve

Asset A is restricted to one evaluation protocol at all four budgets
256/512/768/1024. It retains every seed as a line or mark; a mean or median
can only be a labelled overlay. It rejects incomplete trajectories and any
attempt to connect FID-5k and FID-50k.

### Asset B — compute-to-quality

For threshold \(\eta\), define
\(\tau_a(\eta)=\min\{K:\mathrm{FID}_a(K)\le\eta\}\) and compare the
paired quantity \(\Delta\tau_s=\tau_{B,s}-\tau_{A,s}\). A single figure
uses exactly one frozen crossing rule: `first_observed` or
`linear_interpolation_descriptive`. Interpolation is always labelled as a
descriptive between-checkpoint estimate.

The current primary two-budget configuration freezes FID-50k, NFE1, threshold
12, and descriptive linear interpolation in
[`q256_compute_to_quality_nfe1_eta12.frozen.json`](configs/paper_assets/q256_compute_to_quality_nfe1_eta12.frozen.json).

### Asset C — four-arm dispersion contraction

At each seed and budget, report
\(S_s(K)=\max_{a\in\{A,B,C,D\}}\mathrm{FID}_{s,a}(K)-\min_{a\in\{A,B,C,D\}}\mathrm{FID}_{s,a}(K)\).
Show every seed and its first-to-final change. A pooled summary cannot replace
the seed-level values.

The presently planned 256-to-1024 output is a two-budget endpoint comparison,
not a complete learning curve. Its frozen contract is described in
[`docs/ROLE_D_PROTOCOL_CONTRACT.md`](docs/ROLE_D_PROTOCOL_CONTRACT.md).

## Critical protocol boundary

The historical 512/768 evaluations used FID/KID-5k, while the formal endpoint
measure is FID/KID-50k. These protocols are not interchangeable:

```text
FID-5k @ 512 -> FID-5k @ 768 -> FID-50k @ 1024
```

must never be plotted or analysed as one learning curve. The allowed paths
are: wait for FID-50k at all four budgets for Asset A; show a complete FID-5k
series only as an explicitly labelled auxiliary analysis; or use the matched
256/1024 FID-50k records as a two-budget endpoint comparison.

## Scope and claim boundary

The paper can report large but trajectory-dependent finite-budget effects; a
failure of objective-level target/weight factorization to produce a
seed-stable endpoint factorization; and, when generated from the final matched
records, contraction of arm-level differences from 256 to 1024 kimg.

The current evidence does not establish a universal benefit of `g=1.10`, a
dominant target or denominator mechanism, an optimizer attribution, or
asymptotic convergence to one solution. Compute-to-quality language is used
only if the frozen threshold analysis supports it.

## Required workflow

Role D may normalize frozen outputs, calculate crossings and dispersion,
write captions and provenance manifests, and QA figures. It must not launch
full training, new sweeps, balanced-beta variants, optimizer novelty analyses,
or retrospectively change thresholds, crossing rules, budgets, or seed sets.

Follow the normalisation and rendering commands in
[`docs/ROLE_D_PROTOCOL_CONTRACT.md`](docs/ROLE_D_PROTOCOL_CONTRACT.md). The
pipeline fails closed for a missing cell, a failed receipt, a mixed protocol,
or a partial trajectory. Every final asset emits SVG and PDF masters, a
600-dpi PNG, source data, SHA-256 manifest, exact command, caption,
interpretation boundary, and grayscale preview.

## Current completion state

The renderer and frozen configuration work are complete. The generated
two-budget assets are blocked only on delivery of the raw 1024-kimg FID-50k
receipt matrix; the complete primary Asset A additionally awaits
protocol-matched intermediate-budget FID-50k evaluations. See
[`ROLE_D_STATUS.md`](ROLE_D_STATUS.md) for the live checklist.

Before paper integration, restack this work onto the clean paper base so that
obsolete theory or optimizer-analysis commits are not carried into the
submission branch.
