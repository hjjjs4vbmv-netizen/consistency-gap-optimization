# Source Provenance

## Formal training source

- Source archive SHA256: `6af4d04198b97f469fdbb168a84848d61e43e585a4f1a7b90bb3e2cd60e1b59b`
- Extracted branch head at review: `1395daa16415a2f960a0d1fede09e0a9bf8ae70b`
- Formal runs covered: fixed/global-only x seeds 3/4/5 at 256 kimg
- Reference merged implementation: `main@3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43`

The codeload archive did not contain `.git` metadata. The branch head is
recorded separately and does not prove that a mutable branch archive was
downloaded at that exact head; the archive SHA256 is the authoritative
identity of the executed source.

## Relevant-file comparison

| File | Source SHA256 | Reference SHA256 | Result |
| --- | --- | --- | --- |
| `ct_train.py` | `edb7d0065d560e79825042b5100b8bede45d60246e2d28d7a46c98c5c12c4180` | `c46ca27588e0fe1526922b4eb2767d1e207f32eea92d56688cb8685f8fe6fdc6` | materially different |
| `training/ct_training_loop.py` | `132d6c638ebbc14b33859736a163a634a63f03c6b0bf543d77fe84c80a5348f2` | `071750bcf2990f5e8a162e52a965532b13c640625fa491e5bc8f6f968853a3aa` | materially different |
| `training/loss.py` | `f25af844199e1637a1d2c341ad5cf8f8b538a0d2bd6aa7a6a444c7e4ca9f5084` | `9a44e96e1a5c273a8cace179240e032f78c56b1172afc8ff8a4dfbf3b1c3ec34` | materially different |
| `training/schedules.py` | `d0a675c73588351b9a9891c179ea96ca244ecf54bb4ab8e151db8421bd48422b` | `0d5a7ea79dbd6168b127924b008bab47070be1463f3a4bd2dff13bea351959e0` | materially different |
| `ct_eval.py` | `0d0f7cb4790f3c089fbcd3690c8fee45dcc421a93a08c6f1f85a19efd3d85c03` | `0d0f7cb4790f3c089fbcd3690c8fee45dcc421a93a08c6f1f85a19efd3d85c03` | byte-identical |

- Executed evaluation/metric code tree SHA256: `eb8fae9f7dd78cb3e9414fabb560b19ce6e61d18ae4f6fce329f94ce288851f8`
- `ct_eval.py` is byte-identical to the reference checkout.
- Training-related files are not byte-identical and are treated as
  materially different for claim classification; no equivalence is inferred.

## Classification

These runs are **legacy retrospective exploratory evidence produced from
a pre-merge implementation**. No retraining is claimed or required by this
record, and the results are not formally comparable confirmatory evidence.

## Dataset qualification

The executed dataset archive differs bytewise from the canonical archive.
A canonical archive was unavailable on the evaluation node, so the same
content-manifest algorithm could not be applied to both assets. Semantic
equivalence and the claim that only q changed are therefore not established.
