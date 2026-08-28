#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-/root/imagenet64_gap_ab/downloads/imagenet/ILSVRC/Data/CLS-LOC/train}
DEST_DIR=${DEST_DIR:-/root/imagenet64_gap_ab/datasets}
PYTHON=${PYTHON:-/root/imagenet64_gap_ab/env/bin/python}

mkdir -p "$DEST_DIR"
"$PYTHON" dataset_tool.py convert \
    --source="$DATA_DIR" \
    --dest="$DEST_DIR/edm2-imagenet-64x64.zip" \
    --resolution=64x64 \
    --transform=center-crop-dhariwal
