"""How could a new element be made?  Fusion-evaporation reaction planning.

For every beam + actinide-target combination we compute
  * the compound nucleus, the fusion Q value and the Coulomb barrier,
  * the excitation energy at the barrier and the most likely xn channel,
  * an *empirical* evaporation-residue cross section, calibrated on every
    hot-fusion reaction that has actually produced a superheavy element and
    checked against published non-observations for elements 119 and 120,
  * the beam time needed to see a given number of decay chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from . import data
from .nuclear import get_model

# Beams that can be delivered at high intensity (stable or very long-lived).
BEAMS = [(20, 48), (22, 50), (23, 51), (24, 54), (26, 58), (28, 64)]

# Actinide targets that exist in usable (milligram) quantities.
TARGETS = [
    (92, 238, "238U"), (94, 242, "242Pu"), (94, 244, "244Pu"), (95, 243, "243Am"),
    (96, 248, "248Cm"), (97, 249, "249Bk"), (98, 249, "249Cf"), (98, 251, "251Cf"),
    (99, 254, "254Es (microgram only)"),
]

PNA = 6.241509e12  # particles per second per particle-microampere


def effective_fissility(zp, ap, zt, at):
    """Swiatecki/Blocki mean fissility x_m = 0.75 x_eff + 0.25 x_CN."""
    z, a = zp + zt, ap + at
    i = (a - 2 * z) / a
    crit = 50.883 * (1 - 1.7826 * i * i)
    z2a_eff = 4 * zp * zt / (ap ** (1 / 3) * at ** (1 / 3) * (ap ** (1 / 3) + at ** (1 / 3)))
    x_eff = z2a_eff / crit
    x_cn = (z * z / a) / crit
    return 0.75 * x_eff + 0.25 * x_cn


def coulomb_barrier(zp, ap, zt, at) -> float:
    """Bass-like interaction barrier in the centre-of-mass frame, MeV."""
    r = 1.16 * (ap ** (1 / 3) + at ** (1 / 3)) + 2.2
    return 1.44 * zp * zt / r


@lru_cache(maxsize=1)
def cross_section_fit():
    """log10 sigma(pb) = c0 + c1 * x_m, by censored (Tobit) maximum likelihood.

    Measured cross sections enter as Gaussian points in log10 sigma; published
    non-observations enter as upper limits through the normal CDF, so the
    failed searches for 119/120 genuinely constrain the extrapolation.
    Returns ((c0, c1), scatter sigma in log10).
    """
    x = np.array([effective_fissility(*r[:4]) for r in data.CROSS_SECTIONS])
    y = np.log10([r[4] for r in data.CROSS_SECTIONS])
    xl = np.array([effective_fissility(*r[:4]) for r in data.CROSS_SECTION_LIMITS])
    yl = np.log10([r[4] for r in data.CROSS_SECTION_LIMITS])

    c1 = np.linspace(-40, -5, 141)[:, None, None]
    off = np.linspace(-1.5, 1.5, 61)[None, :, None]
    sig = np.linspace(0.15, 0.8, 27)[None, None, :]
    c0 = y.mean() - c1 * x.mean() + off
    pred = c0[..., None] + c1[..., None] * x
    r = (y - pred) / sig[..., None]
    ll = -0.5 * (r * r).sum(-1) - len(y) * np.log(sig)
    pl = c0[..., None] + c1[..., None] * xl
    t = (yl - pl) / (sig[..., None] * math.sqrt(2))
    cdf = 0.5 * (1 + np.vectorize(math.erf)(t))
    ll = ll + np.log(np.maximum(cdf, 1e-300)).sum(-1)
    i, j, k = np.unravel_index(np.argmax(ll), ll.shape)
    c0b = float(np.broadcast_to(c0, ll.shape)[i, j, k])
    return (c0b, float(c1[i, 0, 0])), float(sig[0, 0, k])


def log10_sigma_pb(zp, ap, zt, at) -> tuple[float, float]:
    (c0, c1), sig = cross_section_fit()
    return c0 + c1 * effective_fissility(zp, ap, zt, at), sig


@dataclass
class Reaction:
    zp: int
    ap: int
    zt: int
    at: int
    target_label: str
    hypothesis: str = "Z=114"

    @property
    def z_cn(self):
        return self.zp + self.zt

    @property
    def a_cn(self):
        return self.ap + self.at

    @property
    def label(self):
        return f"{self.ap}{data.symbol(self.zp)} + {self.target_label.split()[0]}"

    def q_fusion(self) -> float:
        m = get_model(self.hypothesis)
        b = lambda z, a: m.binding(z, a - z)
        bp = 415.991 if (self.zp, self.ap) == (20, 48) else b(self.zp, self.ap)
        return b(self.z_cn, self.a_cn) - bp - b(self.zt, self.at)

    def barrier(self) -> float:
        return coulomb_barrier(self.zp, self.ap, self.zt, self.at)

    def e_star_at_barrier(self) -> float:
        return self.barrier() + self.q_fusion()

    def lab_energy_at_barrier(self) -> float:
        return self.barrier() * (self.ap + self.at) / self.at

    def best_xn(self) -> int:
        """Neutron evaporation channel open at ~barrier + 5 MeV."""
        m = get_model(self.hypothesis)
        e = self.e_star_at_barrier() + 5.0
        z, n = self.z_cn, self.a_cn - self.z_cn
        k = 0
        while k < 6:
            cost = m.s_n(z, n - k) + 2.0  # separation energy + mean kinetic energy
            if e - cost < 0:
                break
            e -= cost
            k += 1
        return max(k, 2)

    def residue(self) -> tuple[int, int]:
        """(Z, N) of the xn evaporation residue."""
        return self.z_cn, self.a_cn - self.z_cn - self.best_xn()

    def sigma_pb(self) -> tuple[float, float]:
        """(log10 sigma[pb], 1-sigma in log10)."""
        return log10_sigma_pb(self.zp, self.ap, self.zt, self.at)

    def events_per_day(self, beam_pua=1.0, target_mg_cm2=0.5, efficiency=0.5) -> float:
        lg, _ = self.sigma_pb()
        sigma_cm2 = 10**lg * 1e-36
        n_target = target_mg_cm2 * 1e-3 / self.at * 6.02214076e23
        return sigma_cm2 * beam_pua * PNA * n_target * efficiency * 86400


def beam_days(events_per_day: float, n_events: int = 1, confidence: float = 0.95) -> float:
    """Days to observe >= n_events with the given probability (Poisson)."""
    if events_per_day <= 0:
        return math.inf
    lo, hi = 0.0, 1e9
    for _ in range(200):
        mid = (lo + hi) / 2
        mu = events_per_day * mid
        p_less = sum(math.exp(-mu) * mu**k / math.factorial(k) for k in range(n_events))
        if 1 - p_less >= confidence:
            hi = mid
        else:
            lo = mid
    return hi


def candidate_reactions(z_new: int, hypothesis: str = "Z=114") -> list[Reaction]:
    out = []
    for zp, ap in BEAMS:
        for zt, at, label in TARGETS:
            if zp + zt == z_new:
                out.append(Reaction(zp, ap, zt, at, label, hypothesis))
    return sorted(out, key=lambda r: -r.sigma_pb()[0])


def validate_against_limits() -> list[dict]:
    """Compare predicted cross sections with published non-observations."""
    rows = []
    for zp, ap, zt, at, limit, label in data.CROSS_SECTION_LIMITS:
        lg, rms = log10_sigma_pb(zp, ap, zt, at)
        rows.append({
            "reaction": label,
            "predicted_pb": 10**lg,
            "upper_limit_pb": limit,
            "within_1_sigma": bool(10**lg <= limit * 10**rms),
        })
    return rows
