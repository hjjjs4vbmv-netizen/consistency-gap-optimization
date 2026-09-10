"""Independent q128 scientific identity; shared execution, separately frozen outputs."""
from analysis.startup_window_experiments_v1.protocol import specification
from training.startup_windows import Q128 as PROTOCOL_ID


def spec(cohort):
    return specification(PROTOCOL_ID, cohort)
