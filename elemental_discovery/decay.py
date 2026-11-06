"""Decay-mode systematics: alpha decay, spontaneous fission, beta / EC.

Every formula here is an *empirical systematic calibrated in this module on
the reference data* (data.py), and each reports its own calibration scatter
so that predicted half-lives carry an honest error bar (in log10 seconds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np

from . import data
from .nuclear import MassModel, fissility, get_model

ALPHA_FINE = 1 / 137.035999
ME_C2 = 0.51099895  # MeV


# ---------------------------------------------------------------------------
# Alpha decay: Viola-Seaborg form, refitted here.
#   log10 T(s) = (a Z + b) / sqrt(Q) + c Z + d + h_odd
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def viola_seaborg_fit():
    rows, y = [], []
    for z, a, q, t in data.ALPHA_DECAYS:
        n = a - z
        rows.append([z / math.sqrt(q), 1 / math.sqrt(q), z, 1.0, z % 2, n % 2])
        y.append(math.log10(t))
    X, y = np.array(rows), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    return coef, rms


def log10_t_alpha(z: int, n: int, q: float) -> float:
    if q <= 0.5:
        return math.inf
    c, _ = viola_seaborg_fit()
    return float(
        c[0] * z / math.sqrt(q) + c[1] / math.sqrt(q) + c[2] * z + c[3]
        + c[4] * (z % 2) + c[5] * (n % 2)
    )


def dlogt_dq(z: int, q: float) -> float:
    c, _ = viola_seaborg_fit()
    return -0.5 * (c[0] * z + c[1]) * q ** -1.5


# ---------------------------------------------------------------------------
# Spontaneous fission: Swiatecki-type systematic.
#   log10 T_sf(s) = s0 + s1 u + s2 u^2 + k (-E_shell)   (+ odd hindrance)
#   u = Z^2/A - 37.5
# The fissility polynomial carries the liquid-drop trend and is held flat
# beyond its minimum: once the liquid-drop barrier has vanished (Z^2/A ~ 44)
# further charge no longer shortens the half-life -- only shell effects matter.
# The shell slope k (orders of magnitude per MeV of shell energy) is chosen
# on a grid.  258Fm (anomalous symmetric fission) is excluded from the fit.
# ---------------------------------------------------------------------------
ODD_SF_HINDRANCE = 3.0  # orders of magnitude per odd nucleon type (typical)
SF_EXCLUDED = {(100, 258)}


def liquid_drop_barrier(z: int, n: int) -> float:
    """Cohen-Swiatecki style liquid-drop fission barrier, MeV."""
    a = z + n
    i = (n - z) / a
    es0 = 17.9439 * (1 - 1.7826 * i * i) * a ** (2 / 3)
    x = fissility(z, n)
    if x >= 1:
        return 0.0
    if x > 2 / 3:
        return 0.83 * (1 - x) ** 3 * es0
    return 0.38 * (0.75 - x) * es0


def fission_barrier(model: MassModel, z: int, n: int) -> float:
    """Estimated fission barrier: liquid drop + |ground-state shell energy|."""
    return max(liquid_drop_barrier(z, n) - model.shell_energy(z, n), 0.0)


@lru_cache(maxsize=8)
def sf_fit(hypothesis: str):
    """Returns ((s0, s1, s2, k, u_max), rms)."""
    m = get_model(hypothesis)
    pts = [(z, a, t) for z, a, t in data.SF_HALF_LIVES if (z, a) not in SF_EXCLUDED]
    u = np.array([z * z / a - 37.5 for z, a, _ in pts])
    esh = np.array([m.shell_energy(z, a - z) for z, a, _ in pts])
    logt = np.array([math.log10(t) for *_, t in pts])
    X = np.column_stack([np.ones_like(u), u, u * u])
    best = None
    for k in np.linspace(1.0, 3.5, 26):
        y = logt + k * esh
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        rms = float(np.sqrt(np.mean((X @ c - y) ** 2)))
        if best is None or rms < best[1]:
            best = ((*c, k), rms)
    (s0, s1, s2, k), rms = best
    u_max = -s1 / (2 * s2) if s2 > 0 else math.inf
    return (float(s0), float(s1), float(s2), float(k), float(u_max)), rms


def log10_t_sf(model: MassModel, z: int, n: int) -> float:
    (s0, s1, s2, k, u_max), _ = sf_fit(model.hypothesis)
    u = min(z * z / (z + n) - 37.5, u_max)
    return float(
        s0 + s1 * u + s2 * u * u
        - k * model.shell_energy(z, n)
        + ODD_SF_HINDRANCE * ((z % 2) + (n % 2))
    )


# ---------------------------------------------------------------------------
# Weak decays: deliberately crude (+/- 1.5 decades).  Allowed transitions with
# a fixed log ft; electron-capture phase space from the K-shell density with a
# rough relativistic enhancement.
# ---------------------------------------------------------------------------
LOG_FT = 5.0
WEAK_SIGMA = 1.5


def log10_t_ec(z: int, q_ec: float) -> float:
    if q_ec <= 0:
        return math.inf
    za = min(z * ALPHA_FINE, 0.99)
    q = q_ec / ME_C2
    f = math.pi * za**3 * q * q / (1 - za * za)
    return LOG_FT - math.log10(f)


def log10_t_beta_minus(q_b: float) -> float:
    if q_b <= 0:
        return math.inf
    w0 = 1 + q_b / ME_C2
    f = max((w0**5) / 30.0, 1e-6)  # Sargent rule
    return LOG_FT - math.log10(f)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
@dataclass
class DecayPrediction:
    z: int
    n: int
    hypothesis: str
    q_alpha: float
    q_alpha_sigma: float
    e_alpha: float  # kinetic energy of the alpha particle (what detectors see)
    q_ec: float
    fission_barrier: float
    log_t: dict = field(default_factory=dict)  # partial log10 half-lives (s)
    log_t_sigma: dict = field(default_factory=dict)

    @property
    def a(self) -> int:
        return self.z + self.n

    @property
    def name(self) -> str:
        return f"{self.a}{data.symbol(self.z)}"

    @property
    def log_t_total(self) -> float:
        lam = sum(10 ** -v for v in self.log_t.values() if math.isfinite(v))
        return -math.log10(lam) if lam > 0 else math.inf

    @property
    def half_life(self) -> float:
        return 10 ** self.log_t_total

    @property
    def branches(self) -> dict:
        lam = {k: 10 ** -v for k, v in self.log_t.items() if math.isfinite(v)}
        tot = sum(lam.values())
        return {k: v / tot for k, v in lam.items()}

    @property
    def dominant_mode(self) -> str:
        return max(self.branches.items(), key=lambda kv: kv[1])[0]

    @property
    def log_t_total_sigma(self) -> float:
        b = self.branches
        return math.sqrt(sum((b[k] * self.log_t_sigma[k]) ** 2 for k in b))


def predict(z: int, n: int, hypothesis: str = "Z=114") -> DecayPrediction:
    m = get_model(hypothesis)
    q, qs = m.q_alpha_with_uncertainty(z, n)
    _, rms_a = viola_seaborg_fit()
    _, rms_sf = sf_fit(hypothesis)
    qec = m.q_ec(z, n)
    qbm = m.q_beta_minus(z, n)
    p = DecayPrediction(
        z=z, n=n, hypothesis=hypothesis, q_alpha=q, q_alpha_sigma=qs,
        e_alpha=q * (z + n - 4) / (z + n), q_ec=qec,
        fission_barrier=fission_barrier(m, z, n),
    )
    p.log_t["alpha"] = log10_t_alpha(z, n, q)
    p.log_t_sigma["alpha"] = math.hypot(rms_a, dlogt_dq(z, max(q, 1.0)) * qs)
    p.log_t["SF"] = log10_t_sf(m, z, n)
    # Shell-energy error is comparable to the local Q-alpha error.
    k_sf = sf_fit(hypothesis)[0][3]
    p.log_t_sigma["SF"] = math.hypot(rms_sf, k_sf * qs)
    if qec > 0:
        p.log_t["EC"] = log10_t_ec(z, qec)
        p.log_t_sigma["EC"] = WEAK_SIGMA
    if qbm > 0:
        p.log_t["beta-"] = log10_t_beta_minus(qbm)
        p.log_t_sigma["beta-"] = WEAK_SIGMA
    return p


def decay_chain(z: int, n: int, hypothesis: str = "Z=114", max_steps: int = 10):
    """Follow the dominant alpha/EC path until fission, a known nucleus, or Z < 100.

    Returns a list of DecayPrediction plus the reason the chain ended.
    """
    chain = []
    for _ in range(max_steps):
        p = predict(z, n, hypothesis)
        chain.append(p)
        mode = p.dominant_mode
        if len(chain) > 1 and data.is_known(z, z + n):
            return chain, f"reaches known nucleus {p.name}"
        if mode == "SF":
            return chain, f"terminates by spontaneous fission at {p.name}"
        if mode == "alpha":
            z, n = z - 2, n - 2
        elif mode == "EC":
            z, n = z - 1, n + 1
        else:
            z, n = z + 1, n - 1
        if z < 100:
            return chain, "decays below fermium"
    return chain, "chain truncated"


def calibration_summary(hypothesis: str = "Z=114") -> dict:
    _, rms_a = viola_seaborg_fit()
    _, rms_sf = sf_fit(hypothesis)
    return {
        "alpha_halflife_rms_log10": rms_a,
        "sf_halflife_rms_log10": rms_sf,
        "weak_decay_assumed_sigma_log10": WEAK_SIGMA,
    }
