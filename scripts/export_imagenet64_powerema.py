#!/usr/bin/env python3
"""Export the PowerEMA 0.050 model from one full state on CPU."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POWER_EMA_STD = 0.050
STATE_NAME = re.compile(r"training-state-kimg(?P<kimg>\d{6})\.pt$")


def default_output(state_path: Path) -> Path:
    match = STATE_NAME.fullmatch(state_path.name)
    if match is None:
        raise ValueError(
            "--output is required unless the state is named "
            "training-state-kimgXXXXXX.pt"
        )
    return state_path.with_name(
        f"network-snapshot-kimg{match.group('kimg')}-0.050.pkl"
    )


def export_snapshot(state_path: Path, output_path: Path) -> None:
    import torch

    from training import reproducibility

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    source_net = state.get("net")
    power_state = state.get("power_ema_state")
    if not isinstance(source_net, torch.nn.Module):
        raise RuntimeError("full state is missing the net module")
    if not isinstance(power_state, dict):
        raise RuntimeError("full state is missing power_ema_state")
    state_name = STATE_NAME.fullmatch(state_path.name)
    if state_name is not None and int(state.get("cur_nimg", -1)) != int(
        state_name.group("kimg")
    ) * 1000:
        raise RuntimeError("full-state cur_nimg does not match its milestone name")

    stds = tuple(float(value) for value in power_state.get("stds", ()))
    matches = [
        index for index, value in enumerate(stds)
        if math.isclose(value, POWER_EMA_STD, rel_tol=0, abs_tol=1e-12)
    ]
    emas = power_state.get("emas")
    if len(matches) != 1 or not isinstance(emas, (list, tuple)):
        raise RuntimeError("power_ema_state does not contain one 0.050 profile")
    if len(emas) != len(stds):
        raise RuntimeError("power_ema_state std/profile counts differ")

    compact = copy.deepcopy(source_net).cpu().eval().requires_grad_(False)
    compact.load_state_dict(emas[matches[0]], strict=True)
    source_buffers = dict(source_net.named_buffers())
    compact_buffers = dict(compact.named_buffers())
    if source_buffers.keys() != compact_buffers.keys():
        raise RuntimeError("live-net and PowerEMA buffer names differ")
    with torch.no_grad():
        for name, buffer in compact_buffers.items():
            buffer.copy_(source_buffers[name].detach().cpu())

    reproducibility.atomic_pickle_dump(
        {"ema": compact}, str(output_path), overwrite=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        output_path = args.output or default_output(args.state)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(
            f"state={args.state} output={output_path} "
            f"power_ema_std={POWER_EMA_STD:.3f} device=cpu"
        )
        return
    if not args.state.is_file():
        parser.error(f"missing full state: {args.state}")
    if output_path.exists():
        parser.error(f"refusing to overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_snapshot(args.state, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
