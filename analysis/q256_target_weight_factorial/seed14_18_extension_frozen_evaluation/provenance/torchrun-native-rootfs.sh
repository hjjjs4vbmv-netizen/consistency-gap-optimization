#!/usr/bin/env bash
set -euo pipefail

rootfs=${Q256_NATIVE_ROOTFS:?Q256_NATIVE_ROOTFS is required}
library_path=${Q256_NATIVE_LIBRARY_PATH:?Q256_NATIVE_LIBRARY_PATH is required}

exec "$rootfs/lib64/ld-linux-x86-64.so.2" \
  --library-path "$library_path" \
  "$rootfs/usr/bin/python" -m torch.distributed.run "$@"
