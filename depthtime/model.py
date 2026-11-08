"""Core equations of the Depth-of-Thought Time Perception model (DTP).

The model treats subjective time as the output of four interacting mechanisms,
each grounded in a separate empirical literature:

1. Pacemaker rate (r)            -- Treisman 1963; Gibbon 1977 (Scalar Expectancy
                                    Theory); arousal speeds the clock (Droit-Volet &
                                    Meck 2007); interoceptive/insular accumulation
                                    (Craig 2009; Wittmann 2013).
2. Attentional gate (g)          -- Zakay & Block 1997: only attended pulses are
                                    counted, so a mind full of non-temporal content
                                    lets fewer pulses through ("time flies").
3. Temporal self-model (R)       -- the "reader" of the clock. Time is felt by a
                                    self that persists across moments; when
                                    narrative content, self-reference and bodily
                                    awareness all fall away, duration stops being
                                    represented (timelessness reports: Berkovich-
                                    Ohana et al. 2013; Wittmann 2015).
4. Memory segmentation (M)       -- retrospective duration is reconstructed from
                                    the number of encoded changes/events (Ornstein
                                    1969; Block & Zakay 1997; Roseboom et al. 2019).

Every quantity is dimensionless and expressed relative to an "ordinary,
mind-wandering" reference state, so a prospective ratio of 1.25 means an
interval *feels* 25 % longer than it would in ordinary waking life.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# State of mind at one moment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MentalState:
    """Instantaneous phenomenological state. All fields lie in [0, 1].

    content     Density of discursive / conceptual thought (inner speech, plans,
                reasoning steps, memories). High in sequential reasoning and
                mind-wandering, ~0 in "no-thought".
    focus       Stability / single-pointedness of attention on *something*
                (a problem, the breath, a movement). Low in mind-wandering.
    awareness   Non-conceptual present-moment and interoceptive meta-awareness:
                knowing what is happening in body and mind *as it happens*.
    self_ref    Self-referential / narrative-self processing ("me", "my
                performance", "how long have I been sitting?").
    arousal     Physiological activation (calm 0 ... threat/panic 1).
    movement    Attentional demand of motor control (still 0 ... scramble 1).
    novelty     Rate of external perceptual change / unpredictability.
    action_prep Readiness for an imminent, precise action (pre-jump, pre-sweep).
    """

    content: float
    focus: float
    awareness: float
    self_ref: float
    arousal: float = 0.3
    movement: float = 0.0
    novelty: float = 0.3
    action_prep: float = 0.0

    def __post_init__(self) -> None:
        for name, val in self.__dict__.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name}={val} outside [0, 1]")

    def adjusted(self, **deltas: float) -> "MentalState":
        """Return a copy with additive deltas, clipped to [0, 1]."""
        vals = {k: clip(getattr(self, k) + v) for k, v in deltas.items()}
        return replace(self, **vals)

    def with_values(self, **vals: float) -> "MentalState":
        return replace(self, **{k: clip(v) for k, v in vals.items()})


# ---------------------------------------------------------------------------
# The depth axis: from deep (sequential) thought to no-thought
# ---------------------------------------------------------------------------
# Each anchor is (depth, content, focus, awareness, self_ref, label).
# The axis describes *progressive release of discursive thought while
# attention stays gathered*.  Ordinary mind-wandering is deliberately NOT on
# the axis: it is the "fall-out" state that every practice keeps returning to.
DEPTH_ANCHORS: tuple[tuple[float, float, float, float, float, str], ...] = (
    (0.00, 0.90, 0.80, 0.20, 0.30, "deep sequential thought"),
    (0.25, 0.50, 0.90, 0.35, 0.15, "absorbed task focus / flow"),
    (0.50, 0.15, 0.65, 0.90, 0.35, "present-moment open awareness"),
    (0.75, 0.05, 0.95, 0.30, 0.10, "concentrative absorption"),
    (1.00, 0.00, 1.00, 0.05, 0.00, "no-thought / cessation-like"),
)

ORDINARY = MentalState(content=0.70, focus=0.20, awareness=0.20, self_ref=0.70,
                       arousal=0.30, movement=0.0, novelty=0.30)
"""Reference state: ordinary waking mind-wandering. All ratios are relative to it."""

WAITING = MentalState(content=0.25, focus=0.15, awareness=0.30, self_ref=0.65,
                      arousal=0.25, movement=0.0, novelty=0.10)
"""Bored waiting: little to think about, nothing to do, attention drifts to time."""


def depth_label(depth: float) -> str:
    best = min(DEPTH_ANCHORS, key=lambda a: abs(a[0] - depth))
    return best[5]


def state_at_depth(depth: float, **context: float) -> MentalState:
    """Piecewise-linear interpolation of the mental state along the depth axis.

    ``context`` sets the non-depth variables (arousal, movement, novelty,
    action_prep).
    """
    d = clip(depth)
    for (d0, *v0, _), (d1, *v1, _) in zip(DEPTH_ANCHORS, DEPTH_ANCHORS[1:]):
        if d0 <= d <= d1:
            t = (d - d0) / (d1 - d0)
            c, f, a, s = (x0 + t * (x1 - x0) for x0, x1 in zip(v0, v1))
            break
    ctx = dict(arousal=0.3, movement=0.0, novelty=0.3, action_prep=0.0)
    ctx.update(context)
    return MentalState(content=c, focus=f, awareness=a, self_ref=s,
                       **{k: clip(v) for k, v in ctx.items()})


def blend(a: MentalState, b: MentalState, w: float) -> MentalState:
    """(1-w)*a + w*b, field by field."""
    w = clip(w)
    return MentalState(**{k: (1 - w) * getattr(a, k) + w * getattr(b, k)
                          for k in a.__dict__})


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelParams:
    # attentional gate
    w_content: float = 1.0      # how much conceptual content occupies attention
    mw_floor: float = 0.45      # even unfocused content (mind-wandering) occupies some
    w_motor: float = 0.85       # how much motor control occupies attention
    gamma: float = 0.55         # max fraction of temporal attention a task can steal
    kappa_a: float = 0.15       # present-moment awareness opens the gate
    # pacemaker
    alpha_e: float = 0.80       # arousal -> clock speed
    e_ref: float = 0.30         # arousal of the reference state
    alpha_a: float = 0.08       # interoceptive accumulation -> clock speed
    alpha_prep: float = 0.50    # action preparation -> temporal resolution (Hagura 2012)
    # memory / retrospective
    k_content: float = 0.80     # each thought shift is a context change
    k_awareness: float = 0.45   # noticed micro-events (sensations) are encoded
    k_novelty: float = 1.00     # external change
    enc0: float = 0.35          # baseline encoding strength
    enc_att: float = 0.50       # extra encoding when attention is engaged
    beta_e: float = 0.60        # arousal strengthens encoding (Stetson et al. 2007)
    ret_exp: float = 0.60       # compressive mapping memory -> remembered duration
    # temporal self-model
    r_floor: float = 0.02
    r_exp: float = 1.5          # >1: the self-model is robust until supports nearly all vanish


DEFAULT_PARAMS = ModelParams()


# ---------------------------------------------------------------------------
# Individual differences
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Individual:
    """Trait-level parameters of one person.

    experience          Contemplative / embodied practice expertise in [0, 1].
                        Raises the depth reachable, speeds settling, reduces
                        mind-wandering and threat reactivity.
    interoception       Interoceptive accuracy multiplier (~0.5-1.5).
    mind_wandering      Baseline propensity to drift (episodes per minute, novice).
    arousal_reactivity  Gain on arousal deviations (~0.5-1.5).
    clock_rate          Baseline pacemaker speed (dopaminergic tone), ~0.85-1.15.
    weber               Scalar-timing coefficient of variation (~0.10-0.25).
    """

    name: str = "average adult"
    experience: float = 0.2
    interoception: float = 1.0
    mind_wandering: float = 1.5
    arousal_reactivity: float = 1.0
    clock_rate: float = 1.0
    weber: float = 0.15

    @property
    def max_depth(self) -> float:
        return clip(0.45 + 0.55 * self.experience)

    @property
    def settle_tau_s(self) -> float:
        """Time constant (seconds) for settling toward a target depth."""
        return 240.0 * (1.25 - self.experience)

    def effective_arousal(self, arousal: float, p: ModelParams = DEFAULT_PARAMS) -> float:
        gain = self.arousal_reactivity * (1.0 - 0.45 * self.experience)
        return clip(p.e_ref + (arousal - p.e_ref) * gain)


NEUTRAL = Individual(name="neutral reference", experience=0.2)

ARCHETYPES: dict[str, Individual] = {
    "novice": Individual("novice", experience=0.05, mind_wandering=2.0),
    "intermediate": Individual("intermediate", experience=0.45, mind_wandering=1.2,
                               interoception=1.1),
    "expert": Individual("expert", experience=0.95, mind_wandering=0.4,
                         interoception=1.3, arousal_reactivity=0.8),
    "anxious novice": Individual("anxious novice", experience=0.05, mind_wandering=2.5,
                                 arousal_reactivity=1.4, weber=0.22),
}


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------
def task_demand(s: MentalState, p: ModelParams = DEFAULT_PARAMS) -> float:
    """Share of attention captured by non-temporal content (thought + movement)."""
    content = p.w_content * s.content * (p.mw_floor + (1 - p.mw_floor) * s.focus)
    motor = p.w_motor * s.movement * (0.5 + 0.5 * s.focus)
    return clip(content + motor - content * motor)  # probabilistic OR


def attentional_gate(s: MentalState, p: ModelParams = DEFAULT_PARAMS) -> float:
    return (1.0 - p.gamma * task_demand(s, p)) * (1.0 + p.kappa_a * s.awareness)


def pacemaker_rate(s: MentalState, ind: Individual = NEUTRAL,
                   p: ModelParams = DEFAULT_PARAMS) -> float:
    e = s.arousal
    r = ind.clock_rate * (1.0
                          + p.alpha_e * (e - p.e_ref)
                          + p.alpha_a * ind.interoception * s.awareness
                          + p.alpha_prep * s.action_prep)
    return max(0.05, r)


def temporal_self_model(s: MentalState, p: ModelParams = DEFAULT_PARAMS) -> float:
    """Coherence R of the self that 'reads' the clock.

    The temporal self is sustained by any of three supports: conceptual content
    (thinking *in* time), narrative self-reference, and bodily/interoceptive
    awareness. R only collapses when all three fall away together.
    """
    gone = (1 - s.content) * (1 - s.self_ref) * (1 - s.awareness)
    return clip(1.0 - gone ** p.r_exp, p.r_floor, 1.0)


def event_rate(s: MentalState, p: ModelParams = DEFAULT_PARAMS) -> float:
    """Rate of context changes / event boundaries available to memory."""
    return (p.k_novelty * s.novelty
            + p.k_content * s.content
            + p.k_awareness * s.awareness * (1 - s.content)
            + 0.5 * s.action_prep)


def encoding_strength(s: MentalState, p: ModelParams = DEFAULT_PARAMS) -> float:
    engaged = max(s.focus, s.awareness)
    return (temporal_self_model(s, p)
            * (p.enc0 + p.enc_att * engaged)
            * (1.0 + p.beta_e * s.arousal))


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeReadout:
    prospective: float      # felt duration "in the moment" relative to ordinary
    passage: float          # felt speed of time's passage (>1 = flies, <1 = drags)
    retrospective: float    # remembered duration relative to ordinary
    timelessness: float     # 1 - R: degree to which time is not represented at all
    rate: float = field(repr=False, default=0.0)
    gate: float = field(repr=False, default=0.0)
    self_model: float = field(repr=False, default=0.0)

    def as_dict(self) -> dict[str, float]:
        return dict(prospective=self.prospective, passage=self.passage,
                    retrospective=self.retrospective, timelessness=self.timelessness,
                    rate=self.rate, gate=self.gate, self_model=self.self_model)


def _raw(s: MentalState, ind: Individual, p: ModelParams) -> tuple[float, float, float, float]:
    r = pacemaker_rate(s, ind, p)
    g = attentional_gate(s, p)
    R = temporal_self_model(s, p)
    m = event_rate(s, p) * encoding_strength(s, p)
    return r, g, R, m


_REF_CACHE: dict[ModelParams, tuple[float, float]] = {}


def _reference(p: ModelParams) -> tuple[float, float]:
    if p not in _REF_CACHE:
        r, g, R, m = _raw(ORDINARY, NEUTRAL, p)
        _REF_CACHE[p] = (r * g * R, m)
    return _REF_CACHE[p]


def readout(s: MentalState, ind: Individual = NEUTRAL,
            p: ModelParams = DEFAULT_PARAMS) -> TimeReadout:
    """Instantaneous time-perception readout for one mental state."""
    ref_pro, ref_mem = _reference(p)
    r, g, R, m = _raw(s, ind, p)
    prospective = r * g * R / ref_pro
    # Passage is how fast time *seems to move* while it is still being felt.
    # It is computed from the clock and gate only: in timelessness there is no
    # felt passage at all (reported separately), not an infinitely fast one.
    ref_r, ref_g, _, _ = _raw(ORDINARY, NEUTRAL, p)
    passage = (ref_r * ref_g) / (r * g)
    return TimeReadout(prospective=prospective, passage=passage,
                       retrospective=(m / ref_mem) ** p.ret_exp, timelessness=1.0 - R,
                       rate=r, gate=g, self_model=R)


def depth_sweep(n: int = 101, ind: Individual = NEUTRAL,
                p: ModelParams = DEFAULT_PARAMS, **context: float):
    """Readouts across the depth axis (steady state, no fall-outs)."""
    out = []
    for i in range(n):
        d = i / (n - 1)
        out.append((d, readout(state_at_depth(d, **context), ind, p)))
    return out


def log2_ratio(x: float) -> float:
    return math.log2(max(x, 1e-9))
