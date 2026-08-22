#!/usr/bin/env bash

set -euo pipefail

Q256_EVAL_ROOT="${Q256_EVAL_ROOT:-/root/q256_eval}"
Q256_SANDBOX_ROOT="${Q256_SANDBOX_ROOT:-${Q256_EVAL_ROOT}/runtime/sandbox}"
Q256_RUNTIME_PYTHON="${Q256_RUNTIME_PYTHON:-${Q256_SANDBOX_ROOT}/usr/bin/python}"

q256_runtime_ld_library_path="${Q256_SANDBOX_ROOT}/usr/local/lib/python3.10/dist-packages/torch/lib"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/usr/local/cuda/compat/lib"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/usr/local/nvidia/lib:${Q256_SANDBOX_ROOT}/usr/local/nvidia/lib64"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/lib:${Q256_SANDBOX_ROOT}/lib/x86_64-linux-gnu"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/opt/hpcx/clusterkit/lib:${Q256_SANDBOX_ROOT}/opt/hpcx/hcoll/lib"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/opt/hpcx/nccl_rdma_sharp_plugin/lib:${Q256_SANDBOX_ROOT}/opt/hpcx/ompi/lib"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/opt/hpcx/sharp/lib:${Q256_SANDBOX_ROOT}/opt/hpcx/ucc/lib:${Q256_SANDBOX_ROOT}/opt/hpcx/ucx/lib"
q256_runtime_ld_library_path+=":${Q256_SANDBOX_ROOT}/usr/local/cuda/targets/x86_64-linux/lib:${Q256_SANDBOX_ROOT}/usr/local/lib"

q256_runtime_path="${Q256_SANDBOX_ROOT}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin"
q256_runtime_path+=":${Q256_SANDBOX_ROOT}/usr/local/mpi/bin:${Q256_SANDBOX_ROOT}/usr/local/nvidia/bin"
q256_runtime_path+=":${Q256_SANDBOX_ROOT}/usr/local/cuda/bin:${Q256_SANDBOX_ROOT}/usr/local/sbin:${Q256_SANDBOX_ROOT}/usr/local/bin"
q256_runtime_path+=":${Q256_SANDBOX_ROOT}/usr/sbin:${Q256_SANDBOX_ROOT}/usr/bin:${Q256_SANDBOX_ROOT}/sbin:${Q256_SANDBOX_ROOT}/bin"
q256_runtime_path+=":${Q256_SANDBOX_ROOT}/usr/local/ucx/bin:${Q256_SANDBOX_ROOT}/opt/tensorrt/bin"

export Q256_EVAL_ROOT Q256_SANDBOX_ROOT Q256_RUNTIME_PYTHON
export Q256_RUNTIME_LD_LIBRARY_PATH="${q256_runtime_ld_library_path}"
export Q256_RUNTIME_PATH="${q256_runtime_path}"

test -x "${Q256_RUNTIME_PYTHON}"

