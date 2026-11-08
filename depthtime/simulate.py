"""Second-by-second simulation of a practice session for one individual.

Dynamics
--------
* depth d(t) relaxes toward min(target, individual.max_depth) with time constant
  ``individual.settle_tau_s``.
* Each segment has an *entry* depth (min(target, 0.40)): the level reachable
  simply by intending the activity. Deeper stages must be settled into.
* Mind-wandering episodes start as a Poisson process whose rate falls with
  experience and with depth (a settled mind wanders less). During an episode the
  state is blended toward ORDINARY; afterwards depth drops back toward the
  entry level and must be re-settled.
* Spikes (threat, jumps) temporarily override context.
* At every second the readout is integrated:
      felt_prospective += prospective * dt
      felt_retrospective += retrospective * dt
  Final estimates receive scalar (Weber) noise, the hallmark of interval timing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .model import (DEFAULT_PARAMS, ORDINARY, Individual, ModelParams, blend,
                    clip, readout, state_at_depth)
from .practices import PRACTICES, Practice

ENTRY_DEPTH = 0.40


@dataclass
class SessionResult:
    practice: str
    individual: str
    clock_min: float
    felt_prospective_min: float
    felt_retrospective_min: float
    mean_passage: float          # >1 time flies, <1 drags (felt moments only)
    mean_timelessness: float
    peak_timelessness: float
    time_wandering_frac: float
    trace: dict | None = None

    def row(self) -> dict:
        d = dict(self.__dict__)
        d.pop("trace")
        return d


def simulate_session(practice: Practice | str, ind: Individual,
                     p: ModelParams = DEFAULT_PARAMS, seed: int | None = None,
                     dt: float = 1.0, keep_trace: bool = False) -> SessionResult:
    if isinstance(practice, str):
        practice = PRACTICES[practice]
    rng = np.random.default_rng(seed)

    depth = 0.0
    entry = 0.0
    wander_left = 0.0
    spike_left = 0.0
    spike_ctx: dict = {}
    spike_shift = 0.0

    pro = retro = 0.0
    passage_w = passage_sum = 0.0
    tl_sum = tl_peak = 0.0
    wander_t = 0.0
    total_t = 0.0
    trace = {k: [] for k in ("t", "depth", "prospective", "retrospective",
                             "passage", "timelessness", "wandering", "spike")} \
        if keep_trace else None

    tau = ind.settle_tau_s
    first = True
    for seg in practice.segments:
        target = min(seg.depth, ind.max_depth)
        # Where a person "lands" when they (re)enter this activity: you can
        # intend to watch the breath at once, but deeper stages take time.
        entry = min(target, ENTRY_DEPTH)
        if first:
            depth, first = entry, False
        n_steps = int(round(seg.minutes * 60 / dt))
        for _ in range(n_steps):
            # --- mind-wandering process ------------------------------------
            if wander_left <= 0:
                rate_per_min = (ind.mind_wandering * seg.wander_scale
                                * (1.0 - 0.75 * ind.experience)
                                * (1.0 - 0.6 * depth))
                if rng.random() < rate_per_min * dt / 60.0:
                    mean_len = 25.0 * (1.0 - 0.6 * ind.experience) + 5.0
                    wander_left = rng.exponential(mean_len)
                    # fall-out: on return you re-enter near the entry level
                    depth = max(entry, entry + (depth - entry) * 0.4)
            wandering = wander_left > 0
            if wandering:
                wander_left -= dt

            # --- spikes ----------------------------------------------------
            if spike_left <= 0 and seg.spikes:
                for sp in seg.spikes:
                    if rng.random() < sp.per_min * dt / 60.0:
                        spike_left = rng.exponential(sp.duration_s)
                        spike_ctx, spike_shift = sp.context, sp.depth_shift
                        break
            spiking = spike_left > 0
            if spiking:
                spike_left -= dt

            # --- depth dynamics -------------------------------------------
            if not wandering:
                depth += (target - depth) * (1.0 - math.exp(-dt / tau))
            d_eff = clip(depth + (spike_shift if spiking else 0.0))

            ctx = dict(seg.context)
            if spiking:
                ctx.update(spike_ctx)
            ctx["arousal"] = ind.effective_arousal(ctx.get("arousal", 0.3), p)
            state = state_at_depth(d_eff, **ctx)
            if seg.adjust:
                state = state.adjusted(**seg.adjust)
            if wandering and not spiking:
                state = blend(state, ORDINARY.with_values(
                    movement=state.movement, arousal=state.arousal,
                    novelty=state.novelty), 0.8)

            r = readout(state, ind, p)
            pro += r.prospective * dt
            retro += r.retrospective * dt
            w = 1.0 - r.timelessness       # passage only exists where time is felt
            passage_sum += r.passage * w * dt
            passage_w += w * dt
            tl_sum += r.timelessness * dt
            tl_peak = max(tl_peak, r.timelessness)
            wander_t += dt if wandering else 0.0
            total_t += dt
            if trace is not None:
                trace["t"].append(total_t / 60.0)
                trace["depth"].append(d_eff)
                trace["prospective"].append(r.prospective)
                trace["retrospective"].append(r.retrospective)
                trace["passage"].append(r.passage)
                trace["timelessness"].append(r.timelessness)
                trace["wandering"].append(wandering)
                trace["spike"].append(spiking)

    noise = lambda: float(np.exp(rng.normal(0.0, ind.weber)))  # noqa: E731
    return SessionResult(
        practice=practice.key, individual=ind.name, clock_min=total_t / 60.0,
        felt_prospective_min=pro / 60.0 * noise(),
        felt_retrospective_min=retro / 60.0 * noise(),
        mean_passage=passage_sum / max(passage_w, 1e-9),
        mean_timelessness=tl_sum / total_t,
        peak_timelessness=tl_peak,
        time_wandering_frac=wander_t / total_t,
        trace=trace,
    )


def monte_carlo(practice: str, ind: Individual, n: int = 100,
                p: ModelParams = DEFAULT_PARAMS, seed: int = 0) -> dict:
    """Mean and spread of the session readouts over n simulated sessions."""
    rows = [simulate_session(practice, ind, p, seed=seed + i).row() for i in range(n)]
    keys = ("felt_prospective_min", "felt_retrospective_min", "mean_passage",
            "mean_timelessness", "peak_timelessness", "time_wandering_frac")
    out = dict(practice=practice, individual=ind.name, clock_min=rows[0]["clock_min"])
    for k in keys:
        vals = np.array([r[k] for r in rows])
        out[k] = float(vals.mean())
        out[k + "_sd"] = float(vals.std())
    return out
