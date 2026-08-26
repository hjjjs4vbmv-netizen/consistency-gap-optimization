# Formal operator-clock remote storage

The compact receipt tree in this directory was copied from the completed
Matpool execution. Large raw tensors remain on the execution node and are not
duplicated in Git.

- Endpoint: `root@px-cloud1.matpool.com:27391`
- Execution root: `/root/operator-clock-audit-v1`
- Formal result root: `/root/operator-clock-audit-v1/results`
- Field raw tensors: `results/field/shard{0,1}/*.pt`
- Algorithmic raw tensors: `results/algorithmic/shard{0,1}/*.pt`
- Raw tensor count: 256
- Raw tensor bytes: 342,534,144,896
- Per-file checksum manifest: `results/raw_tensor_sha256.txt`
- Checksum-manifest SHA256:
  `62270c70a06ce61c9c6cf0802d74a00c0cd4243bba167be0d915086250b13bfc`

The transferred compact archive excluded all `.pt` files.

- Compact archive SHA256:
  `0def4fd7aaff5b5a0d43b41e532be772c510974217d745a05f0d3bbff105e2fa`
- Formal summary SHA256:
  `071ca793f793f70b35fd23165f01ce7ded022cd99de205991accd4a2414d1e2d`
- Matched rollout SHA256:
  `a51dead69a3776de12d10b2a6478c331476409ebac0a9dc03295a14fcd667323`
- Execution environment SHA256:
  `16a7d055915fc93341d9a91a177ed66e72469dbdcf2076f00daaeaef5e47af3f`

Execution commits:

- Formal field/algorithmic JVP: `6e0ac3469b2e4d33591a45c908513aa42349318d`
- Matched rollout: `73e78e5564b6d4075a14ef69f17617d5990cc1c1`
- Completeness summary: `dd95ae5481f9a88d62db93ecb0841224b6f2ddf3`

No passwords or private keys are stored in this receipt tree.
