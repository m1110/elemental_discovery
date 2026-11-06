"""How would we know a new element really exists?

Four kinds of evidence, each quantified where possible:

1. Genetic decay-chain correlation: implantation -> alpha -> alpha -> ... ->
   a known nucleus or fission, all in the same detector pixel within
   predicted time windows.  We compute the expected number of chains that
   background alone would fake (random-correlation analysis).
2. Counting statistics: Poisson significance of k observed chains over an
   expected background b.
3. Independent Z and A identification: K X-ray energies (atomic number) and
   A/q mass measurement (mass number), plus cross-bombardments and
   excitation functions.
4. Natural / astrophysical occurrence: would any primordial or cosmic
   inventory survive long enough to be searched for?

The discovery criteria follow the IUPAC/IUPAP Joint Working Group
(Wilkinson et al. 1991; Hofmann et al., Pure Appl. Chem. 90, 1773, 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import data
from .atomic import k_alpha1_kev, kalpha_screening
from .decay import decay_chain

IUPAC_CRITERIA = [
    "Z must be established beyond reasonable doubt; A is desirable but not required.",
    "Genetic correlation of the decay chain to previously known nuclei is the "
    "strongest evidence; chains that end in fission must be supported by "
    "cross-bombardments, excitation functions or independent Z identification.",
    "Random-correlation probability must be shown to be negligible.",
    "Results must be reproduced, preferably by an independent laboratory "
    "and/or a different reaction.",
    "Energy calibration, separator transmission and detector response must be "
    "documented so that independent reviewers can assess the claim.",
]


def poisson_sf(k: int, mu: float) -> float:
    """P(X >= k) for X ~ Poisson(mu)."""
    if k <= 0:
        return 1.0
    term, cdf = math.exp(-mu), 0.0
    for i in range(k):
        cdf += term
        term *= mu / (i + 1)
    return max(1.0 - cdf, 0.0)


def gaussian_sigma(p: float) -> float:
    """One-sided p-value -> equivalent Gaussian significance (bisection)."""
    if p <= 0:
        return math.inf
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(mid / math.sqrt(2)) > p:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class DetectorConditions:
    """Typical gas-filled-separator focal-plane conditions (beam on)."""

    evr_like_rate_per_pixel: float = 1e-4  # recoil-like signals / s / pixel
    alpha_like_rate_per_pixel: float = 2e-6  # alpha-like signals in window / s / pixel
    sf_like_rate_per_pixel: float = 1e-8  # high-energy, fission-like / s / pixel
    pixels: int = 10_000
    experiment_days: float = 100.0
    window_factor: float = 5.0  # correlation window = factor x predicted half-life


def random_chain_expectation(chain, det: DetectorConditions = DetectorConditions()) -> float:
    """Expected number of purely random chains mimicking the predicted one.

    n_random = N_evr * prod_i (rate_i * window_i), where N_evr is the total
    number of recoil-like events, and each subsequent link must fall within
    its time window in the same pixel.
    """
    t_total = det.experiment_days * 86400
    n_evr = det.evr_like_rate_per_pixel * det.pixels * t_total
    prob = 1.0
    for p in chain:
        window = det.window_factor * max(p.half_life, 1e-6)
        rate = det.sf_like_rate_per_pixel if p.dominant_mode == "SF" else det.alpha_like_rate_per_pixel
        prob *= min(rate * window, 1.0)
    return n_evr * prob


def significance(k_observed: int, background: float) -> dict:
    p = poisson_sf(k_observed, background)
    return {"k": k_observed, "background": background, "p_value": p, "sigma": gaussian_sigma(p)}


def upper_limit_pb(days: float, events_per_day_per_pb: float, cl: float = 0.90) -> float:
    """Cross-section upper limit after seeing zero events (Poisson, no background)."""
    mu = -math.log(1 - cl)  # 2.30 at 90 % CL
    return mu / (events_per_day_per_pb * days)


# ---------------------------------------------------------------------------
# Natural and astrophysical searches
# ---------------------------------------------------------------------------
ENVIRONMENTS = [
    # name, characteristic age (s), what would be searched for
    ("Kilonova ejecta (neutron-star merger), days after", 10 * 86400,
     "atomic absorption lines in the kilonova spectrum"),
    ("Galactic cosmic rays (confinement time)", 15e6 * data.SECONDS_PER_YEAR,
     "ultra-heavy nuclei tracks in space detectors / meteoritic olivine"),
    ("Earth's crust / meteorites (age of the Solar System)", 4.567e9 * data.SECONDS_PER_YEAR,
     "mass spectrometry of ores; fission tracks; fissiogenic Xe anomalies"),
]


def survival_fraction(half_life_s: float, age_s: float) -> float:
    """Fraction surviving after age_s, returned as log10 to avoid underflow."""
    if half_life_s <= 0:
        return -math.inf
    return -(age_s / half_life_s) * math.log10(2)


def natural_search_feasibility(half_life_s: float, detectable_log10_fraction: float = -15.0):
    """For each environment: log10(surviving fraction) and whether a search is
    conceivable (surviving fraction above ~1e-15 of the produced inventory)."""
    out = []
    for name, age, method in ENVIRONMENTS:
        lf = survival_fraction(half_life_s, age)
        out.append({
            "environment": name,
            "log10_surviving_fraction": lf,
            "feasible": lf > detectable_log10_fraction,
            "method": method,
        })
    return out


# ---------------------------------------------------------------------------
# A full confirmation plan for one candidate nucleus
# ---------------------------------------------------------------------------
def confirmation_plan(z: int, n: int, hypothesis: str = "Z=114",
                      det: DetectorConditions = DetectorConditions()) -> dict:
    chain, ending = decay_chain(z, n, hypothesis)
    s1, s2, kerr = kalpha_screening()
    zs = sorted({p.z for p in chain})
    kx = dict(zip(zs, map(float, k_alpha1_kev(zs))))
    n_rand = random_chain_expectation(chain, det)
    steps = []
    for p in chain:
        steps.append({
            "nucleus": p.name,
            "mode": p.dominant_mode,
            "branches": p.branches,
            "Q_alpha_MeV": p.q_alpha,
            "Q_alpha_sigma_MeV": p.q_alpha_sigma,
            "E_alpha_MeV": p.e_alpha,
            "log10_T_half_s": p.log_t_total,
            "log10_T_sigma": p.log_t_total_sigma,
            "known": data.is_known(p.z, p.a),
            "K_alpha1_keV": kx[p.z],
        })
    anchored = ending.startswith("reaches known")
    evidence = [
        "Genetic link to a known nucleus" if anchored
        else "Chain ends in fission before reaching known nuclei: needs cross-bombardment "
             "or direct Z identification",
        f"K X-rays of Z={z}: {kx[z]:.1f} keV (+/-{kerr * kx[z]:.1f}); "
        f"neighbour Z={z - 1} differs by {kx[z] - float(k_alpha1_kev(z - 1)[0]):.1f} keV",
        "A/q mass measurement of the recoil (FIONA-type) to fix A",
        "Excitation function: maximum near E* = 30-40 MeV for 3n/4n confirms the channel",
    ]
    t0 = chain[0].half_life
    if t0 < 1e-5:
        evidence.append(f"Predicted half-life {t0 * 1e6:.2g} us is comparable to the ~1 us separator "
                        "flight time: digital electronics are needed to catch decays within "
                        "microseconds of implantation, and many atoms will decay in flight")
    if t0 > 1.0:
        evidence.append("Half-life > 1 s: single-atom chemistry is possible "
                        "(group placement test for the new element)")
    return {
        "nucleus": chain[0].name,
        "hypothesis": hypothesis,
        "chain": steps,
        "chain_ending": ending,
        "expected_random_chains": n_rand,
        "evidence": evidence,
        "natural_occurrence": natural_search_feasibility(t0),
        "iupac_criteria": IUPAC_CRITERIA,
    }
