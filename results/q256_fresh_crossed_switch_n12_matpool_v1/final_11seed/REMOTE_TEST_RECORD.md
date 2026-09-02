# Frozen-runtime test record for PR97 remediation

- Test source commit: `31b127e42b33caff0f21b9a0fbbe9b014512f5c3`
- Frozen runtime-manifest SHA256: `feb0cf0d824550b0776bbc47ee220fc4d2e9489b468aa4c550a5c58a51a5e1ac`
- Runtime Python: `/root/q256_fresh_crossed_switch_n12_matpool_v1/runtime/env/bin/python`
- Isolated worktree: `/root/q256_fresh_crossed_switch_n12_matpool_v1/pr97-validation-31b127e`
- Result: 29/29 original q256 tests PASS; 14/14 authorization/remediation tests PASS.

Commands executed on the formal six-A100 node:

```bash
cd /root/q256_fresh_crossed_switch_n12_matpool_v1/pr97-validation-31b127e

/root/q256_fresh_crossed_switch_n12_matpool_v1/runtime/env/bin/python \
  -m unittest discover -v -s tests \
  -p 'test_q256_fresh_crossed_switch_n12.py'

/root/q256_fresh_crossed_switch_n12_matpool_v1/runtime/env/bin/python \
  -m unittest discover -v -s tests \
  -p 'test_q256_eleven_seed_authorization.py'
```

Log bindings:

- `remote_test_31b127e_main.log`:
  `2b5d2c3b9b8272932e352bfa788486afcca7ff7b4ff83112f408b42cff36f416`
- `remote_test_31b127e_authorization.log`:
  `02ed72d0e8927b51c34d4bf9ec65e331561cfb75498fa32edc81c0affed5d740`

The later PR head adds only evidence files, documentation, the public checksum
manifest, and its CI check. GitHub Actions is the exact-head validation record
for that final commit; it rechecks protocol SHA, Python compilation, shell
syntax, the 14 dependency-free tampering tests, and every public evidence hash.
