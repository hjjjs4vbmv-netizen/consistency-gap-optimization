# Reproducible validation record

This experiment has two distinct validation layers.

## Frozen formal-runtime record

- Formal finalization source HEAD: `aa1896aa4b39343282266ae45005841f87eaf999`
- Frozen implementation commit: `c12c278b60808e1120c035bb68e7c866c3208df7`
- Protocol SHA256: `df7f584c4af0017b6d655b843460033e22f8801b843859d7dbe32aa852f167b1`
- Rebuilt runtime-manifest SHA256: `feb0cf0d824550b0776bbc47ee220fc4d2e9489b468aa4c550a5c58a51a5e1ac`
- Formal matrix: 242/242 `SEALED_PASS`; zero recovery-matrix failures
- Finalization: `PASS`; primary verdict `INCONCLUSIVE`

The copied final receipt and public evidence manifest preserve these bindings.
This historical record must not be represented as a test run of later PR heads.

## Current-source checks

GitHub Actions runs `.github/workflows/q256-fresh-protocol.yml` on every PR head.
That check is the commit-bound, reviewable record for current source. It executes:

```bash
cd analysis/q256_fresh_crossed_switch_n12_matpool_v1
sha256sum --check protocol.sha256

python -m py_compile analysis/q256_fresh_crossed_switch_n12_matpool_v1/*.py
bash -n analysis/q256_fresh_crossed_switch_n12_matpool_v1/*.sh
python -m unittest discover -v -s tests \
  -p 'test_q256_eleven_seed_authorization.py'
```

The dependency-free unit suite includes a valid-chain case and field/source
tampering cases for the amendment schema, quality-observation attestation,
decision threshold, numeric-recovery authorization, terminal failure receipt,
and all eleven completion-receipt bindings. It does not claim to replace the
frozen GPU-runtime tests; it specifically makes the authorization-chain fix
reviewable without Torch or a GPU.
