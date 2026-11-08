"""Fit the model to ONE person's own time-estimation data (n-of-1 calibration).

Data format (CSV, one row per session)::

    practice,clock_min,prospective_min,retrospective_min,passage
    open_monitoring,20,24,18,-1
    sequential,30,22,40,2

* ``prospective_min``   - estimate made by stopping a hidden timer / producing an
                          interval *while* practising (or "how long so far?" asked
                          mid-session).  Leave blank if not collected.
* ``retrospective_min`` - answer to "how long did that last?" asked afterwards
                          WITHOUT forewarning.  Leave blank if not collected.
* ``passage``           - Likert from -3 (time dragged) to +3 (time flew).

The fit searches over the individual's traits (experience, interoception,
mind_wandering, arousal_reactivity, clock_rate) using random search followed by
local refinement - dependency-free and robust to the noisy objective.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import math

import numpy as np

from .model import DEFAULT_PARAMS, Individual, ModelParams
from .practices import PRACTICES, Practice, Segment
from .simulate import simulate_session

BOUNDS = {
    "experience": (0.0, 1.0),
    "interoception": (0.5, 1.5),
    "mind_wandering": (0.2, 3.0),
    "arousal_reactivity": (0.5, 1.5),
    "clock_rate": (0.8, 1.2),
}


def load_observations(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row = {"practice": r["practice"].strip(), "clock_min": float(r["clock_min"])}
            for k in ("prospective_min", "retrospective_min", "passage"):
                v = (r.get(k) or "").strip()
                row[k] = float(v) if v else None
            if row["practice"] not in PRACTICES:
                raise ValueError(f"unknown practice {row['practice']!r}; "
                                 f"choose from {sorted(PRACTICES)}")
            rows.append(row)
    return rows


def _rescaled(practice: Practice, minutes: float) -> Practice:
    k = minutes / practice.minutes
    segs = tuple(Segment(s.minutes * k, s.depth, s.context, s.adjust, s.spikes,
                         s.wander_scale, s.label) for s in practice.segments)
    return replace(practice, segments=segs)


def _predict(ind: Individual, obs: list[dict], p: ModelParams, reps: int, seed: int):
    preds = []
    for i, o in enumerate(obs):
        prac = _rescaled(PRACTICES[o["practice"]], o["clock_min"])
        sims = [simulate_session(prac, replace(ind, weber=0.0), p, seed=seed + 97 * i + j)
                for j in range(reps)]
        preds.append(dict(
            prospective_min=float(np.mean([s.felt_prospective_min for s in sims])),
            retrospective_min=float(np.mean([s.felt_retrospective_min for s in sims])),
            passage=float(np.mean([3 * math.tanh(2 * math.log(s.mean_passage)) for s in sims])),
        ))
    return preds


def loss(ind: Individual, obs: list[dict], p: ModelParams = DEFAULT_PARAMS,
         reps: int = 4, seed: int = 0) -> float:
    total, n = 0.0, 0
    for o, pr in zip(obs, _predict(ind, obs, p, reps, seed)):
        for k in ("prospective_min", "retrospective_min"):
            if o[k] is not None:
                total += (math.log(o[k] / o["clock_min"])
                          - math.log(pr[k] / o["clock_min"])) ** 2
                n += 1
        if o["passage"] is not None:
            total += ((o["passage"] - pr["passage"]) / 3.0) ** 2
            n += 1
    return total / max(n, 1)


def fit_individual(obs: list[dict], n_random: int = 40, n_refine: int = 40,
                   p: ModelParams = DEFAULT_PARAMS, seed: int = 0,
                   name: str = "fitted individual") -> tuple[Individual, float]:
    rng = np.random.default_rng(seed)

    def make(v: dict) -> Individual:
        return Individual(name=name, **v)

    def sample() -> dict:
        return {k: float(rng.uniform(*b)) for k, b in BOUNDS.items()}

    best_v = {k: getattr(Individual(), k) for k in BOUNDS}
    best_l = loss(make(best_v), obs, p, seed=seed)
    for _ in range(n_random):
        v = sample()
        l = loss(make(v), obs, p, seed=seed)
        if l < best_l:
            best_v, best_l = v, l
    step = 0.25
    for _ in range(n_refine):
        v = {k: float(np.clip(best_v[k] + rng.normal(0, step * (b[1] - b[0])), *b))
             for k, b in BOUNDS.items()}
        l = loss(make(v), obs, p, seed=seed)
        if l < best_l:
            best_v, best_l = v, l
        else:
            step = max(0.03, step * 0.93)
    return make(best_v), best_l
