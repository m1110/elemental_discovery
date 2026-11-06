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


# ---------------------------------------------------------------------------
# Cross-section law: log10 sigma(pb) = c0 + c1 x_m + eps,  eps ~ N(0, s^2).
#
# Measured cross sections are Gaussian points in log10 sigma.  A published
# non-observation is NOT a censored Gaussian point: it is a Poisson count of
# zero at a known exposure.  A 90 % CL limit sigma_lim means an integrated
# exposure L = ln(10) / sigma_lim, and the probability of seeing nothing is
#     P(0 | c, s) = E_eps[ exp(-L 10^(c0 + c1 x + eps)) ].
# This is the exact likelihood, and it is harsher on over-predictions than the
# censored-Gaussian (Tobit) treatment: a true cross section equal to the limit
# still gives zero events 10 % of the time, twice the limit only 1 %.
#
# The posterior over (c0, c1, s) is kept on a grid (flat priors), so the
# predictive distribution of any reaction includes parameter uncertainty and
# grows away from the calibration data.
# ---------------------------------------------------------------------------
from .linear import _GH_W, _GH_X  # noqa: E402

LIMIT_CL = 0.90


def exposure_from_limit(limit_pb: float, cl: float = LIMIT_CL) -> float:
    """Integrated exposure (1/pb) implied by a zero-event upper limit."""
    return -math.log(1 - cl) / limit_pb


def _grid(limits):
    x = np.array([effective_fissility(*r[:4]) for r in data.CROSS_SECTIONS])
    y = np.log10([r[4] for r in data.CROSS_SECTIONS])
    c1 = np.linspace(-45, 0, 181)[:, None, None]
    off = np.linspace(-1.2, 1.2, 49)[None, :, None]
    s = np.linspace(0.12, 1.0, 45)[None, None, :]
    c0 = y.mean() - c1 * x.mean() + off
    pred = c0[..., None] + c1[..., None] * x
    ll = -0.5 * (((y - pred) / s[..., None]) ** 2).sum(-1) - len(y) * np.log(s)
    for zp, ap, zt, at, lim, _ in limits:
        xl = effective_fissility(zp, ap, zt, at)
        lg = (c0 + c1 * xl)[..., None] + s[..., None] * _GH_X
        p0 = (np.exp(-exposure_from_limit(lim) * 10.0 ** lg) * _GH_W).sum(-1)
        ll = ll + np.log(np.maximum(p0, 1e-300))
    w = np.exp(ll - ll.max())
    w /= w.sum()
    shape = w.shape
    return (np.broadcast_to(c0, shape).ravel(), np.broadcast_to(c1, shape).ravel(),
            np.broadcast_to(s, shape).ravel(), w.ravel())


@lru_cache(maxsize=8)
def cross_section_posterior(exclude: str | None = None):
    """Grid posterior (c0, c1, s, weight), optionally leaving one limit out."""
    lims = [r for r in data.CROSS_SECTION_LIMITS if r[5] != exclude]
    c0, c1, s, w = _grid(lims)
    # Keep the highest-weight grid points holding 99.99 % of the mass.
    order = np.argsort(w)[::-1]
    n = int(np.searchsorted(np.cumsum(w[order]), 0.9999)) + 1
    keep = order[:n]
    return c0[keep], c1[keep], s[keep], w[keep] / w[keep].sum()


@lru_cache(maxsize=1)
def cross_section_fit():
    """Posterior means ((c0, c1), s), back-compatible with the old interface."""
    c0, c1, s, w = cross_section_posterior()
    return (float(w @ c0), float(w @ c1)), float(w @ s)


def predictive_log10_sigma(zp, ap, zt, at, exclude=None):
    """Mixture-of-normals predictive distribution: (means, sds, weights)."""
    c0, c1, s, w = cross_section_posterior(exclude)
    return c0 + c1 * effective_fissility(zp, ap, zt, at), s, w


def log10_sigma_pb(zp, ap, zt, at) -> tuple[float, float]:
    """(mean, predictive sd) of log10 sigma[pb], parameter uncertainty included."""
    m, s, w = predictive_log10_sigma(zp, ap, zt, at)
    mean = float(w @ m)
    return mean, float(np.sqrt(w @ (s**2 + (m - mean) ** 2)))


def past_exposure(zp, ap, zt, at) -> float:
    """Integrated exposure (1/pb) already spent on this exact reaction without
    a single event (from the published limits)."""
    return sum(exposure_from_limit(r[4]) for r in data.CROSS_SECTION_LIMITS if r[:4] == (zp, ap, zt, at))


@lru_cache(maxsize=64)
def log10_sigma_samples(zp, ap, zt, at, condition_on_nulls: bool = True):
    """Weighted samples (log10 sigma_pb, weight) of this reaction's cross section.

    With condition_on_nulls the reaction's own idiosyncratic deviation eps is
    updated by its past zero-event exposure (exact hierarchical Bayes: the
    global parameters already saw that limit, so only eps is re-weighted).
    """
    m, s, w = predictive_log10_sigma(zp, ap, zt, at)
    # Collapse the mixture onto a fixed grid in log10 sigma.
    mean = float(w @ m)
    sd = float(np.sqrt(w @ (s**2 + (m - mean) ** 2)))
    y = np.linspace(mean - 7 * sd, mean + 7 * sd, 281)
    dens = np.zeros_like(y)
    for i in range(0, len(m), 4096):
        mm, ss, ww_ = m[i:i + 4096, None], s[i:i + 4096, None], w[i:i + 4096, None]
        dens += (ww_ / ss * np.exp(-0.5 * ((y[None, :] - mm) / ss) ** 2)).sum(0)
    L = past_exposure(zp, ap, zt, at) if condition_on_nulls else 0.0
    if L > 0:
        # Exact hierarchical update: the global parameters already saw this
        # limit, so only the reaction's own deviation is re-weighted, i.e.
        # p(y | all) ~ sum_g w_g N(y; m_g, s_g) P0(y) / P0_g.
        p0g = (np.exp(-L * 10.0 ** (m[:, None] + s[:, None] * _GH_X[None, :])) @ _GH_W)
        dens = np.zeros_like(y)
        for i in range(0, len(m), 4096):
            mm, ss = m[i:i + 4096, None], s[i:i + 4096, None]
            ww_ = (w[i:i + 4096] / np.maximum(p0g[i:i + 4096], 1e-300))[:, None]
            dens += (ww_ / ss * np.exp(-0.5 * ((y[None, :] - mm) / ss) ** 2)).sum(0)
        dens *= np.exp(-L * 10.0**y)
    return y, dens / dens.sum()


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
    """Leave-one-out predictive check of every published non-observation:
    refit without that search, then ask how probable zero events were."""
    rows = []
    for zp, ap, zt, at, limit, label in data.CROSS_SECTION_LIMITS:
        m, s, w = predictive_log10_sigma(zp, ap, zt, at, exclude=label)
        mean = float(w @ m)
        L = exposure_from_limit(limit)
        lg = m[:, None] + s[:, None] * _GH_X[None, :]
        p0 = float(w @ (np.exp(-L * 10.0**lg) @ _GH_W))
        rows.append({
            "reaction": label,
            "predicted_pb": 10**mean,
            "upper_limit_pb": limit,
            "expected_events": L * float(w @ (10.0 ** lg @ _GH_W)),
            "p_zero_events": p0,
            "within_1_sigma": bool(p0 > 0.1587),
        })
    return rows
