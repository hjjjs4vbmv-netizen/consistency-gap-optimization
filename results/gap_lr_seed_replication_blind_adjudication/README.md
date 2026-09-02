# Seed-Replication Blind-Adjudication Candidate

This directory is the sanitized, quality-blind evidence package for seed 4/5
of `gap_lr_matched_q128_s45_replication_v1`. No seed-4/5 FID, KID, or other
generation-quality output was read to produce it.

## Machine result

- Verdict: `machine_recommends_acceptance`
- Failed conditions: none
- Runs requiring rerun: none
- Protocol exact: no
- Scientific-use authorization: no
- Quality-evaluation authorization: no
- Required next step: independent quality-blind review bound to the hashes
  below, followed by a separate `accepted_with_documented_deviation` receipt
  if the reviewer accepts deviations D1–D5.

All six strengthened per-run integrity receipts passed. Each run reached 256
kimg and 2000 attempted iterations; successful optimizer steps were 1992 for
five runs and 1991 for seed 5 arm C, with 8–9 AMP skips inside the frozen
allowance. The six reconstructed expected transferred tensor states have one
common canonical SHA256:

`cabefcddf9ca190fe50eee05714f2cd5d92680b6974a6d8bd2be3be42a1ada19`

This is a post-run reconstruction, not a historical in-process pre-update
attestation. The candidate preserves that distinction and all other claim
exclusions.

## Review bindings

- Adjudication tooling commit:
  `d375b321c02935c8802f0bb18a8aa51f1e3abc3e`
- Execution protocol commit:
  `583c2fe0f914fc1191903d747737fd54b4ba1eef`
- Machine candidate SHA256:
  `e7a158f235e2644c244297758b293e04701714d904707ce637e841225284901f`
- Objective evidence SHA256:
  `8a6ee20300056a4353851ba919134486313629749d541c892d3a7b8fc8c82c12`
- Initialization reconstruction SHA256:
  `b2e53f2ac865f8729ac7566c8c165c4bb773262399363da53085085b44a17d18`

The six public receipt hashes are recorded inside
`blind_adjudication.json`. Internal receipts, absolute paths, raw logs, stable
device identifiers, checkpoints, and training states remain external to Git.
