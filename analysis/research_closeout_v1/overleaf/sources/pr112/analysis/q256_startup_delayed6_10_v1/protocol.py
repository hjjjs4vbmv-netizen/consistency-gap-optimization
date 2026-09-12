"""Exploratory q256 adjacent-window identity; controls remain exact PR111 records."""
from analysis.startup_window_experiments_v1.protocol import specification
from training.startup_windows import Q256 as PROTOCOL_ID


def spec():
    return specification(PROTOCOL_ID, ())
