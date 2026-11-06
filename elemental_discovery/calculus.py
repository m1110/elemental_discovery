"""The discovery calculus: one probability for "will we discover element Z?".

The rest of the package produces point estimates with error bars.  A
discovery, however, is a yes/no event, and planning one is a question about
probability: *given everything we know and don't know, how likely is it that
k decay chains anchored to known nuclei are seen within D beam days?*

Every ingredient is a random variable in log space, and the answer is one
expectation:

    P_disc(D) = P_anchor * E_{sigma, T}[ P_Poisson(>= k ; lambda(sigma, T) D) ]

    lambda    = sigma * I * n_t * eps * S(T)               (chains per day)
    S(T)      = 2^(-t_flight / T)                          (survive the separator)
    P_anchor  = prod_{unknown links i} P(alpha_i is the fastest mode)
              = prod_i Phi((log T_SF,i - log T_alpha,i) / sqrt(sd_SF,i^2 + sd_alpha,i^2))

  * log10 sigma has the predictive distribution of the cross-section law,
    conditioned on the zero-event exposure already spent on that reaction
    (production.log10_sigma_samples);
  * log10 T of the residue is Normal with the predictive sd of the decay laws
    (decay.predict), which already includes the Q-alpha extrapolation error;
  * the expectation is evaluated by Gauss-Hermite quadrature, so the result is
    deterministic and exact to quadrature precision.

Why this is stronger than point estimates: a cross section uncertain by x4
has a log-normal mean that is several times its median, while the beam time
needed is set by the lower tail; a half-life of 1 us +/- 1 decade survives a
1 us flight with probability ~0.4, not 0.5; and a chain only counts as a
discovery under the IUPAC criteria if it is genetically anchored.  Point
estimates get all three wrong in the optimistic direction.

The model is deliberately simple: three log-normal inputs, one Bernoulli and
one Poisson.  Correlations between the residue's Q-alpha, its half-life and
the anchoring probability are neglected (they share the Q-alpha error and
move the same way, so this is mildly conservative for P_disc).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import data
from .decay import predict
from .linear import _GH_W, _GH_X, normal_cdf
from .nuclear import PROTON_HYPOTHESES, forward_calibration
from .production import PNA, Reaction, candidate_reactions, log10_sigma_samples


@dataclass
class Setup:
    beam_pua: float = 6.0
    efficiency: float = 0.5  # separator transmission x detection
    flight_time_s: float = 1.0e-6
    chains_needed: int = 3
    default_target_mg_cm2: float = 0.5


# Target areal densities limited by how much material exists.  Einsteinium-254
# is produced in microgram quantities: ~1 ug spread over ~1 cm^2 is ~1e-3
# mg/cm^2, 500x thinner than a standard actinide target.  Ranking 48Ca + 254Es
# as "3 beam days" (as a fixed 0.5 mg/cm^2 assumption does) is an artefact.
TARGET_LIMIT_MG_CM2 = {(99, 254): 1.0e-3}


def target_density(r: Reaction, setup: Setup) -> float:
    return min(setup.default_target_mg_cm2, TARGET_LIMIT_MG_CM2.get((r.zt, r.at), math.inf))


def p_alpha_fastest(p) -> float:
    """Probability that alpha decay out-runs every competing mode."""
    la, sa = p.log_t["alpha"], p.log_t_sigma["alpha"]
    if not math.isfinite(la):
        return 0.0
    prob = 1.0
    for mode, lt in p.log_t.items():
        if mode == "alpha" or not math.isfinite(lt):
            continue
        prob *= normal_cdf((lt - la) / math.hypot(sa, p.log_t_sigma[mode]))
    return prob


def anchor(z: int, n: int, hypothesis: str, max_steps: int = 8):
    """Follow the alpha path until a known nucleus.  Returns (P_anchor, path).
    P_anchor is the probability that every unknown link alpha-decays."""
    path, prob = [], 1.0
    for _ in range(max_steps):
        if path and data.is_known(z, z + n):
            path.append(f"{z + n}{data.symbol(z)}")
            return prob, path
        p = predict(z, n, hypothesis)
        path.append(p.name)
        prob *= p_alpha_fastest(p)
        z, n = z - 2, n - 2
    return 0.0, path


def flight_survival(log_t: float, sd: float, t_flight: float) -> float:
    """E[2^(-t_flight/T)] for log10 T ~ N(log_t, sd^2)."""
    lt = log_t + sd * _GH_X
    return float(_GH_W @ np.exp(-math.log(2) * t_flight / 10.0**lt))


def _poisson_at_least(k: int, mu: np.ndarray) -> np.ndarray:
    term = np.exp(-mu)
    cdf = np.zeros_like(mu)
    for i in range(k):
        cdf = cdf + term
        term = term * mu / (i + 1)
    return np.clip(1.0 - cdf, 0.0, 1.0)


@dataclass
class DiscoveryModel:
    reaction: Reaction
    hypothesis: str
    setup: Setup

    def __post_init__(self):
        r, s = self.reaction, self.setup
        zr, nr = r.residue()
        self.residue = predict(zr, nr, self.hypothesis)
        self.p_anchor, self.path = anchor(zr, nr, self.hypothesis)
        self.log_sigma, self.w_sigma = log10_sigma_samples(r.zp, r.ap, r.zt, r.at)
        lt, sd = self.residue.log_t_total, self.residue.log_t_total_sigma
        self.lt_nodes = lt + sd * _GH_X
        self.survival_nodes = np.exp(-math.log(2) * s.flight_time_s / 10.0**self.lt_nodes)
        self.survival = float(_GH_W @ self.survival_nodes)
        n_t = target_density(r, s) * 1e-3 / r.at * 6.02214076e23
        # chains/day per pb, before flight losses
        self.per_pb_day = 1e-36 * s.beam_pua * PNA * n_t * s.efficiency * 86400

    def rate_samples(self):
        """(chains/day, weight) over the joint cross-section x half-life grid."""
        lam = (10.0 ** self.log_sigma)[:, None] * self.per_pb_day * self.survival_nodes[None, :]
        return lam, self.w_sigma[:, None] * _GH_W[None, :]

    def p_chains(self, days: float, k: int | None = None) -> float:
        """P(at least k chains are recorded within `days`), anchoring aside."""
        k = self.setup.chains_needed if k is None else k
        lam, w = self.rate_samples()
        return float((w * _poisson_at_least(k, lam * days)).sum())

    def p_discovery(self, days: float, k: int | None = None) -> float:
        """P(at least k *anchored* chains within `days`)."""
        return self.p_anchor * self.p_chains(days, k)

    def days_for(self, prob: float = 0.5, k: int | None = None, anchored: bool = True) -> float:
        """Beam days for the discovery probability to reach `prob` (inf if never)."""
        f = self.p_discovery if anchored else self.p_chains
        if f(1e6, k) < prob:
            return math.inf
        lo, hi = -3.0, 6.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(10**mid, k) < prob else (lo, mid)
        return 10**hi

    def median_sigma_fb(self) -> float:
        order = np.argsort(self.log_sigma)
        c = np.cumsum(self.w_sigma[order])
        return float(10 ** self.log_sigma[order][np.searchsorted(c, 0.5)] * 1000)


def hypothesis_weights(temper: bool = True) -> dict:
    """Weights for the shell hypotheses from their *forward* predictive
    likelihood (how well each predicted heavier nuclei before seeing them).

    The forward predictions are far from independent: nested cuts reuse the
    same nuclei, and the isotopes of one element share the model's shell bias.
    The log-likelihood is therefore tempered to one effective observation per
    tested element: t = (distinct Z among test nuclei) / (test predictions)."""
    cal = {h: forward_calibration(h) for h in PROTON_HYPOTHESES}
    t = 1.0
    if temper:
        elements = {z for z, *_ in data.ALPHA_DECAYS if z >= min(_cuts())}
        t = len(elements) / next(iter(cal.values()))["n"]
    ll = np.array([cal[h]["loglik"] * t for h in cal])
    w = np.exp(ll - ll.max())
    return dict(zip(cal, w / w.sum()))


def _cuts():
    from .nuclear import FORWARD_CUTS
    return FORWARD_CUTS


def discovery_table(z_values=(119, 120, 121, 122), setup: Setup = Setup(),
                    campaign_days: float = 300.0, per_z: int = 3) -> list[dict]:
    """For every candidate reaction: discovery probability in one campaign,
    per hypothesis and model-averaged with forward-likelihood weights."""
    wts = hypothesis_weights()
    rows = []
    for z in z_values:
        for r in candidate_reactions(z)[:per_z]:
            per_h = {}
            for h in PROTON_HYPOTHESES:
                dm = DiscoveryModel(Reaction(r.zp, r.ap, r.zt, r.at, r.target_label, h), h, setup)
                per_h[h] = dm
            p_h = {h: dm.p_discovery(campaign_days) for h, dm in per_h.items()}
            ref = per_h["Z=114"]
            rows.append({
                "Z": z,
                "reaction": r.label,
                "residue": ref.residue.name,
                "target_mg_cm2": target_density(r, setup),
                "sigma_median_fb": ref.median_sigma_fb(),
                "past_null_exposure": bool(ref.log_sigma.size and _has_null(r)),
                "survival": {h: dm.survival for h, dm in per_h.items()},
                "p_anchor": {h: dm.p_anchor for h, dm in per_h.items()},
                "p_chains": ref.p_chains(campaign_days),
                "p_discovery": p_h,
                "p_discovery_avg": float(sum(wts[h] * p_h[h] for h in p_h)),
                "p_discovery_equal": float(np.mean(list(p_h.values()))),
                "days_50pc_Z114": ref.days_for(0.5),
                "path_Z114": ref.path,
            })
    return rows


def _has_null(r: Reaction) -> bool:
    return any(x[:4] == (r.zp, r.ap, r.zt, r.at) for x in data.CROSS_SECTION_LIMITS)
