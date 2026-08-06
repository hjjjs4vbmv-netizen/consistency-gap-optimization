# q=128 1024-kimg prospective formal protocol

`configs/q128_1024k_prospective_matrix.frozen.json` is a new, pre-training
declaration for six **new** checkpoint identities.  Its repository commit is
the time-order evidence.  It cannot make any existing `q128-1024k-*` result
prospective.

For each cell, retain and hash the final checkpoint, final training state,
`training_options.json`, dataset, integrity receipt, and a transfer archive.
Persist `commit_sha.txt` in the run directory before training so the receipt
can attest the frozen training commit.  Do not resume from the historical
1024-kimg or exploratory 1000-kimg checkpoints.

After all six runs pass integrity, create a sender manifest containing the
archive SHA and six checkpoint file SHA values.  A person operating as Role D
must execute the following from a separate checkout or account, on the
received files—not from the sender's run directory:

```bash
python scripts/verify_checkpoint_handoff.py \
  --sender-manifest q128_1024k_handoff_sender.json \
  --archive q128_1024k_handoff.tar.gz \
  --verifier-identity 'Role D / <independent operator>' \
  --output receiver_verification.json
```

Only a `status: passed` receipt, returned without editing by the sender, is
Role D receiver verification.  It verifies the archive SHA and every declared
checkpoint SHA, but does not establish q128/q256 dataset-semantic equivalence.
That equivalence remains a separate formal blocker.

Before metric generation, bind the completed six cells into a portable
runtime manifest and run `scripts/preflight_formal_evaluation.sh`.  The formal
job must include all six cells and both NFE contracts; it may not use quick
metrics as a selection rule.
