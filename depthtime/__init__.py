"""depthtime: a computational model of how depth of thought shapes felt time."""

from .model import (ARCHETYPES, DEPTH_ANCHORS, NEUTRAL, ORDINARY, WAITING,
                    DEFAULT_PARAMS, Individual, MentalState, ModelParams,
                    TimeReadout, depth_label, depth_sweep, readout,
                    state_at_depth)
from .practices import PRACTICES, Practice, Segment, Spike
from .simulate import SessionResult, monte_carlo, simulate_session

__all__ = [
    "ARCHETYPES", "DEPTH_ANCHORS", "NEUTRAL", "ORDINARY", "WAITING",
    "DEFAULT_PARAMS", "Individual", "MentalState", "ModelParams", "TimeReadout",
    "depth_label", "depth_sweep", "readout", "state_at_depth",
    "PRACTICES", "Practice", "Segment", "Spike",
    "SessionResult", "monte_carlo", "simulate_session",
]
