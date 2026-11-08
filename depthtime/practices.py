"""Practice profiles: how each form of 'thinking' moves a person along the depth axis.

A practice is a sequence of segments. Each segment sets a *target* depth (how far
the practice invites discursive thought to settle) plus the bodily/environmental
context (arousal, movement demand, novelty, action preparation), optional
additive adjustments to the mental state (e.g. introspection boosts
self-reference), and optional stochastic *spikes* (a BJJ scramble, a parkour
jump) that briefly override the context.

The person never sits exactly at the target. Their trajectory depends on their
traits (see ``simulate.py``): novices settle slowly, plateau shallower and keep
falling back into ordinary mind-wandering.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spike:
    per_min: float                 # expected spikes per minute
    duration_s: float              # mean duration
    context: dict = field(default_factory=dict)   # overrides during spike
    depth_shift: float = 0.0       # where the spike pushes depth (additive)


@dataclass(frozen=True)
class Segment:
    minutes: float
    depth: float
    context: dict = field(default_factory=dict)
    adjust: dict = field(default_factory=dict)
    spikes: tuple[Spike, ...] = ()
    wander_scale: float = 1.0      # how much this activity invites mind-wandering
    label: str = ""


@dataclass(frozen=True)
class Practice:
    key: str
    name: str
    family: str                    # cognitive | contemplative | movement | combat
    segments: tuple[Segment, ...]
    note: str = ""

    @property
    def minutes(self) -> float:
        return sum(s.minutes for s in self.segments)


CALM = dict(arousal=0.20, movement=0.0, novelty=0.10)


PRACTICES: dict[str, Practice] = {}


def _add(p: Practice) -> None:
    PRACTICES[p.key] = p


# ----- controls ------------------------------------------------------------
_add(Practice(
    "waiting", "Bored waiting (control)", "control",
    (Segment(30, depth=0.0, context=dict(arousal=0.25, novelty=0.10),
             adjust=dict(content=-0.65, focus=-0.65, awareness=0.10, self_ref=0.35),
             wander_scale=2.0, label="waiting room"),),
    note="Nothing to do, attention drifts to the clock: the textbook case of time dragging."))

# ----- cognitive -----------------------------------------------------------
_add(Practice(
    "sequential", "Sequential reasoning / deep problem solving", "cognitive",
    (Segment(30, depth=0.0, context=dict(arousal=0.35, novelty=0.25),
             wander_scale=0.8, label="step-by-step reasoning"),),
    note="High cognitive load: Block, Hancock & Zakay (2010) - shorter prospective, "
         "longer retrospective estimates."))

_add(Practice(
    "introspection", "Authentic introspection / reflective self-inquiry", "cognitive",
    (Segment(30, depth=0.08, context=dict(arousal=0.30, novelty=0.15),
             adjust=dict(self_ref=0.45, awareness=0.30, content=-0.25),
             wander_scale=1.2, label="journaling / self-inquiry"),),
    note="Attention turned on one's own states: self-reference and interoception up, "
         "content moderate."))

# ----- contemplative -------------------------------------------------------
_add(Practice(
    "focused_attention", "Focused-attention meditation (breath)", "contemplative",
    (Segment(5, depth=0.40, context=CALM, label="settling"),
     Segment(25, depth=0.80, context=CALM, label="one-pointed breath focus")),
    note="Samatha / concentration practice; experts approach absorption."))

_add(Practice(
    "open_monitoring", "Open-monitoring / vipassana meditation", "contemplative",
    (Segment(5, depth=0.40, context=CALM, label="settling"),
     Segment(25, depth=0.52, context=CALM, label="noting sensations as they arise")),
    note="Mindfulness of the present moment: the condition used in most "
         "time-perception mindfulness studies."))

_add(Practice(
    "nondual", "Nondual / 'no-thought' resting (e.g. self-inquiry, shikantaza)", "contemplative",
    (Segment(5, depth=0.50, context=CALM, label="settling"),
     Segment(25, depth=1.00, context=CALM, label="resting without object")),
    note="Only advanced practitioners reach the far end of the axis; the model caps "
         "depth by experience."))

_add(Practice(
    "yoga", "Yoga class (asana -> pranayama -> savasana)", "movement",
    (Segment(35, depth=0.45, context=dict(arousal=0.40, movement=0.45, novelty=0.25),
             adjust=dict(awareness=0.10), label="asana flow"),
     Segment(10, depth=0.60, context=dict(arousal=0.25, movement=0.10, novelty=0.10),
             label="pranayama"),
     Segment(15, depth=0.75, context=dict(arousal=0.12, movement=0.0, novelty=0.05),
             wander_scale=1.3, label="savasana / yoga nidra")),
    note="Breath-linked movement then stillness; savasana invites both deep rest and "
         "dozing (higher wander_scale)."))

_add(Practice(
    "tai_chi", "Tai chi form", "movement",
    (Segment(30, depth=0.50, context=dict(arousal=0.20, movement=0.35, novelty=0.15),
             adjust=dict(awareness=0.05), label="slow continuous form"),),
    note="Slow, continuous, interoceptively guided movement: 'moving meditation'."))

_add(Practice(
    "qigong", "Qigong / zhan zhuang standing", "movement",
    (Segment(10, depth=0.45, context=dict(arousal=0.20, movement=0.20, novelty=0.10),
             label="flowing qigong"),
     Segment(20, depth=0.62, context=dict(arousal=0.18, movement=0.05, novelty=0.05),
             adjust=dict(awareness=0.05), label="standing post")),
    note="Breath + posture + interoceptive 'qi' attention, mostly still."))

_add(Practice(
    "bjj", "Brazilian jiu-jitsu sparring (rolling)", "combat",
    tuple(
        seg for _ in range(5) for seg in (
            Segment(5, depth=0.25,
                    context=dict(arousal=0.65, movement=0.85, novelty=0.70),
                    spikes=(Spike(per_min=1.2, duration_s=8,
                                  context=dict(arousal=0.95, movement=1.0,
                                               novelty=0.95, action_prep=0.85),
                                  depth_shift=-0.05),),
                    wander_scale=0.15, label="roll"),
            Segment(1, depth=0.05, context=dict(arousal=0.45, movement=0.05, novelty=0.20),
                    wander_scale=1.0, label="rest / replay"),
        )
    ),
    note="Live resistance, chess-like problem solving under threat (chokes, "
         "scrambles). Spikes = threat/escape moments."))

_add(Practice(
    "parkour", "Parkour session", "movement",
    tuple(
        seg for _ in range(6) for seg in (
            Segment(3, depth=0.25,
                    context=dict(arousal=0.55, movement=0.80, novelty=0.80),
                    spikes=(Spike(per_min=2.0, duration_s=4,
                                  context=dict(arousal=0.90, movement=1.0,
                                               novelty=0.95, action_prep=1.0),
                                  depth_shift=0.0),),
                    wander_scale=0.2, label="run / line"),
            Segment(2, depth=0.05, context=dict(arousal=0.40, movement=0.10, novelty=0.40),
                    adjust=dict(content=0.05), label="scouting next line"),
        )
    ),
    note="Novel terrain, risk, and repeated pre-jump 'commitment' moments."))
