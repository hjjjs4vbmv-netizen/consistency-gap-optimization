# Fresh q256 crossed-switch replication report

## Execution and integrity status

- Training matrix: PASS (22/22 prefixes; 44/44 suffixes).
- Blind evaluation: SEALED_PASS (242/242 jobs), decoded only after the full amended matrix seal.
- Manual evaluation recovery: 1; the failed attempt is preserved and the replacement cache passed a non-metric storage gate.
- Manual postseal report recovery: 2; no evaluation rerun or re-decode was performed.
- Analysis population: 11 complete seeds (31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42); seed38 excluded by explicit author amendment; the original n=12 claim is abandoned: True.
- Protocol SHA256: `df7f584c4af0017b6d655b843460033e22f8801b843859d7dbe32aa852f167b1`.
- Implementation commit: `c12c278b60808e1120c035bb68e7c866c3208df7`; final HEAD `aa1896aa4b39343282266ae45005841f87eaf999`; clean worktree: `True`.
- Host: `Po408E`; GPU UUIDs: GPU-dbf0a977-e9bf-8469-c0f3-d9eed4bf0be4, GPU-74e16478-3980-7d5d-f504-de28a8022fcd, GPU-b6d4a8cf-b094-504a-fcb2-6cc7554185d8, GPU-a30c5dfd-23c6-b2d4-0aee-5b2c87a59a13, GPU-d0834467-cb74-7998-f3fc-85a1bb472a75, GPU-f367feae-962a-c42a-98a6-2cb97a6ecfb2.
- Dataset SHA256: `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
- Transfer SHA256: `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`.
- Rebuilt runtime manifest SHA256: `feb0cf0d824550b0776bbc47ee220fc4d2e9489b468aa4c550a5c58a51a5e1ac`.

## Statistical primary verdict

- Decision: **INCONCLUSIVE**.
- H mean: -0.0755664012; median: -0.056532514; sample SD: 0.118719134.
- Two-sided 95% CI: [-0.155323, 0.00419019794].
- TOST two-sided 90% CI: [-0.140443747, -0.0106890552].
- Exact two-sided sign-flip p: 0.0419921875; negative directions: 8/11.

## Claim boundary

The primary classification is determined only by seed-level H from 1024-kimg NFE1 FID-50k under the frozen rules. NFE2, KID, intermediate milestones, AULC, single-cell BA, interaction, and checkpoint-quality diagnostics are descriptive and cannot alter or rescue the primary verdict. Execution PASS establishes protocol and data integrity; it does not itself establish the scientific hypothesis.
