#!/usr/bin/env python3
"""Run the 890a85 baseline with only the M1 absolute gate pause enabled."""

import os
import sys
from pathlib import Path


M1_PROTOCOL = "m1_r1_history_persistence_q256"


def main() -> None:
    baseline = Path(os.environ["M1_BASELINE_REPO"]).resolve(strict=True)
    sys.path.insert(0, str(baseline))
    from training import ct_training_loop, schedule_switch

    original = ct_training_loop.validate_planned_pause

    def validate_planned_pause(**values):
        if (
            values.get("stop_after_attempts") == 4032
            and values.get("planned_pause_protocol") == M1_PROTOCOL
            and values.get("strict_reproducibility") is True
            and values.get("seed") in range(50, 80)
            and values.get("total_kimg") == 1024
            and values.get("resume_state_dump") is not None
            and values.get("schedule_switch_manifest") is not None
        ):
            manifest = schedule_switch.load_run_manifest(
                values["schedule_switch_manifest"]
            )
            if (
                manifest.get("experiment_protocol")
                == schedule_switch.TERMINAL_HISTORY_N30_PROTOCOL
            ):
                return 4032
        return original(**values)

    ct_training_loop.validate_planned_pause = validate_planned_pause
    import ct_train
    ct_train.main()


if __name__ == "__main__":
    main()
