#!/usr/bin/env bash
set -euo pipefail

Q256_RUNTIME_ROOTFS="${1:?source runtime_env.sh RUNTIME_ROOTFS}"
Q256_RUNTIME_PYTHON="${Q256_RUNTIME_ROOTFS}/usr/bin/python"
Q256_RUNTIME_LD_LIBRARY_PATH="${Q256_RUNTIME_ROOTFS}/usr/local/lib/python3.10/dist-packages/torch/lib"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/local/cuda/compat/lib"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/local/nvidia/lib:${Q256_RUNTIME_ROOTFS}/usr/local/nvidia/lib64"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/lib:${Q256_RUNTIME_ROOTFS}/lib/x86_64-linux-gnu"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/opt/hpcx/clusterkit/lib:${Q256_RUNTIME_ROOTFS}/opt/hpcx/hcoll/lib"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/opt/hpcx/nccl_rdma_sharp_plugin/lib:${Q256_RUNTIME_ROOTFS}/opt/hpcx/ompi/lib"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/opt/hpcx/sharp/lib:${Q256_RUNTIME_ROOTFS}/opt/hpcx/ucc/lib:${Q256_RUNTIME_ROOTFS}/opt/hpcx/ucx/lib"
Q256_RUNTIME_LD_LIBRARY_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/local/cuda/targets/x86_64-linux/lib:${Q256_RUNTIME_ROOTFS}/usr/local/lib"
Q256_RUNTIME_PATH="${Q256_RUNTIME_ROOTFS}/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin"
Q256_RUNTIME_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/local/mpi/bin:${Q256_RUNTIME_ROOTFS}/usr/local/nvidia/bin"
Q256_RUNTIME_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/local/cuda/bin:${Q256_RUNTIME_ROOTFS}/usr/local/sbin:${Q256_RUNTIME_ROOTFS}/usr/local/bin"
Q256_RUNTIME_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/sbin:${Q256_RUNTIME_ROOTFS}/usr/bin:${Q256_RUNTIME_ROOTFS}/sbin:${Q256_RUNTIME_ROOTFS}/bin"
Q256_RUNTIME_PATH+=":${Q256_RUNTIME_ROOTFS}/usr/local/ucx/bin:${Q256_RUNTIME_ROOTFS}/opt/tensorrt/bin"
export Q256_RUNTIME_ROOTFS Q256_RUNTIME_PYTHON
export Q256_RUNTIME_LD_LIBRARY_PATH Q256_RUNTIME_PATH
[[ -x "${Q256_RUNTIME_PYTHON}" ]]
