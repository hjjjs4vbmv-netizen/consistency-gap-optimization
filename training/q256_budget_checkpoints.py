"""Pure schedule helpers for q256 128-kimg durability checkpoints.

This module deliberately contains no Torch or training math.  It only decides
whether a committed image count is one of the predeclared learning-curve
budgets.
"""

from __future__ import annotations


INTERVAL_KIMG = 128
START_KIMG = 384
FINAL_KIMG = 1024
BUDGETS_KIMG = tuple(range(START_KIMG, FINAL_KIMG + 1, INTERVAL_KIMG))


def validate_contract(
    *, interval_kimg: int, start_kimg: int, total_kimg: int
) -> None:
    """Require the exact seed6/7 A/B learning-curve durability contract."""

    if isinstance(interval_kimg, bool) or interval_kimg != INTERVAL_KIMG:
        raise ValueError(f"budget checkpoint interval must be {INTERVAL_KIMG} kimg")
    if isinstance(start_kimg, bool) or start_kimg != START_KIMG:
        raise ValueError(f"budget checkpoint start must be {START_KIMG} kimg")
    if isinstance(total_kimg, bool) or total_kimg != FINAL_KIMG:
        raise ValueError(f"budget checkpoint total must be {FINAL_KIMG} kimg")


def checkpoint_budget_kimg(
    cur_nimg: int,
    *,
    interval_kimg: int,
    start_kimg: int,
    total_kimg: int,
) -> int | None:
    """Return the exact budget at a checkpoint boundary, otherwise ``None``."""

    validate_contract(
        interval_kimg=interval_kimg,
        start_kimg=start_kimg,
        total_kimg=total_kimg,
    )
    if isinstance(cur_nimg, bool) or not isinstance(cur_nimg, int):
        raise TypeError("cur_nimg must be an integer")
    if cur_nimg < 0:
        raise ValueError("cur_nimg must be nonnegative")
    if cur_nimg % 1000 != 0:
        return None
    budget_kimg = cur_nimg // 1000
    if budget_kimg < start_kimg or budget_kimg > total_kimg:
        return None
    if (budget_kimg - start_kimg) % interval_kimg != 0:
        return None
    return budget_kimg
