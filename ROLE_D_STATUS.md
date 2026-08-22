# Role D status — compute-to-quality paper assets

**Updated:** 2026-08-22

## DONE

- [x] The active paper question, evidence boundary, and asset definitions are
  recorded in `ROLE_D_HANDOFF_COMPUTE_TO_QUALITY_20260822.md`.
- [x] The frozen FID-50k two-budget source contract and frozen NFE=1 threshold
  configuration are versioned under `configs/paper_assets/`.
- [x] Renderers preserve every seed, reject incomplete cells and mixed
  protocols, and emit paper-facing provenance, caption, interpretation, and
  grayscale QA sidecars.

## BLOCKED ON DATA

- [x] The checked-in formal four-arm 256-kimg FID-50k receipt matrix is
  available, but the matching raw 1024-kimg receipt matrix for seeds 3–5 is
  absent from this workspace. No two-budget source or figure will be inferred
  from endpoint ranges or FID-5k proxy values.
- [x] A complete primary learning curve also awaits protocol-matched FID-50k
  evaluations at 512 and 768 kimg.

## NEXT WHEN DATA ARRIVES

- [ ] Normalize the delivered 1024-kimg raw receipts with
  `scripts/normalize_q256_two_budget_fid50k.py`.
- [ ] Regenerate the two-budget endpoint, compute-to-quality, and dispersion
  assets with the frozen configs in `configs/paper_assets/`.
- [ ] Perform the retained grayscale preview review and move the generated
  vector masters into the paper integration branch.
