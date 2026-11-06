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
from .linear import LinearLaw
from .nuclear import MassModel, fissility, get_model

ALPHA_FINE = 1 / 137.035999
ME_C2 = 0.51099895  # MeV


# ---------------------------------------------------------------------------
# Alpha decay: the universal decay law (Qi, Xu, Liotta & Wyss, PRL 103, 072501,
# 2009), refitted here.  It is the WKB tunnelling exponent through the Coulomb
# barrier linearised in its two natural variables,
#   log10 T(s) = a chi + b rho + c,
#   chi = Z_a Z_d sqrt(A_red / Q),  rho = sqrt(A_red Z_a Z_d (A_d^1/3 + A_a^1/3)),
# with A_red = A_a A_d / (A_a + A_d).  Three coefficients, all with a physical
# meaning, versus six for Viola-Seaborg.  Out of sample it is the better law:
# fitted on Z < 104 and asked to predict Z >= 104 it misses by 0.5 decades,
# Viola-Seaborg by 1.4 (see validation.py).
# ---------------------------------------------------------------------------
def udl_features(z: int, n: int, q: float) -> list[float]:
    zd, ad = z - 2, z + n - 4
    mu = 4 * ad / (ad + 4)
    return [2 * zd * math.sqrt(mu / q), math.sqrt(mu * 2 * zd * (ad ** (1 / 3) + 4 ** (1 / 3))), 1.0]


@lru_cache(maxsize=1)
def alpha_law() -> LinearLaw:
    X = [udl_features(z, a - z, q) for z, a, q, _ in data.ALPHA_DECAYS]
    y = [math.log10(t) for *_, t in data.ALPHA_DECAYS]
    return LinearLaw.fit(X, y, names=("chi", "rho", "const"))


def log10_t_alpha(z: int, n: int, q: float) -> float:
    if q <= 0.5:
        return math.inf
    return alpha_law().mean(udl_features(z, n, q))


def log10_t_alpha_sd(z: int, n: int, q: float) -> float:
    """Predictive sd of the law itself at (z, n, q) (excludes the Q error)."""
    return alpha_law().sd(udl_features(z, n, max(q, 0.5)))


def dlogt_dq(z: int, q: float, n: int | None = None) -> float:
    n = int(1.5 * z) if n is None else n
    a = alpha_law().coef[0]
    return -0.5 * a * udl_features(z, n, q)[0] / q


# Kept for comparison: the six-parameter Viola-Seaborg form.
@lru_cache(maxsize=1)
def viola_seaborg_fit():
    """Returns (coefficients, rms).  log10 T = (aZ+b)/sqrt(Q) + cZ + d + h_odd."""
    X, y = zip(*(viola_seaborg_row(r) for r in data.ALPHA_DECAYS))
    law = LinearLaw.fit(X, y)
    return law.coef, law.rms()


def viola_seaborg_row(r):
    z, a, q, t = r
    return [z / math.sqrt(q), 1 / math.sqrt(q), z, 1.0, z % 2, (a - z) % 2], math.log10(t)


# ---------------------------------------------------------------------------
# Spontaneous fission: Swiatecki-type systematic.
#   log10 T_sf(s) = s0 + s1 u + s2 u^2 + k (-E_shell)   (+ odd hindrance)
#   u = Z^2/A - 37.5
# The fissility polynomial carries the liquid-drop trend and is held flat
# beyond its minimum: once the liquid-drop barrier has vanished (Z^2/A ~ 44)
# further charge no longer shortens the half-life -- only shell effects matter.
# k is the shell slope (orders of magnitude per MeV of shell energy).  258Fm (anomalous symmetric fission) is excluded from the fit.
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


# The shell slope k is the crux.  Fitted freely to these 15 even-even nuclei
# it comes out k = 0.2 +/- 1.0 (Z=114 shells): the data alone do NOT show that
# shell energy stabilises against fission, because the calibration nuclei have
# shell energies of only -5..0 MeV.  The island of stability (-7..-11 MeV)
# therefore rests on a physics prior.  We make that prior explicit,
#   k ~ N(K_PRIOR_MEAN, K_PRIOR_SD^2)   [orders of magnitude per MeV],
# matching the range [1, 3.5] the original grid search assumed, and let
# Bayesian linear regression report how much of each prediction is prior.
# The predictive sd then grows with |E_shell| beyond the data -- by many
# decades on the island, which is the honest state of knowledge.
K_PRIOR_MEAN, K_PRIOR_SD = 2.25, 0.75
ODD_SF_SD = 1.5  # uncertainty of the odd-nucleon hindrance, decades per odd type


def _sf_features(model: MassModel, z: int, n: int, u_cap: float = math.inf):
    u = min(z * z / (z + n) - 37.5, u_cap)
    return [1.0, u, u * u, -model.shell_energy(z, n)]


@lru_cache(maxsize=8)
def sf_law(hypothesis: str, prior: bool = True) -> tuple[LinearLaw, float]:
    """(law, u_cap).  u is capped where the fitted quadratic flattens (or at the
    edge of the data): once the liquid-drop barrier has vanished, extra charge
    no longer shortens the half-life.  prior=False gives the data-only fit."""
    m = get_model(hypothesis)
    pts = [(z, a, t) for z, a, t in data.SF_HALF_LIVES if (z, a) not in SF_EXCLUDED]
    X = [_sf_features(m, z, a - z) for z, a, _ in pts]
    y = [math.log10(t) for *_, t in pts]
    kw = dict(prior_mean=[0, 0, 0, K_PRIOR_MEAN], prior_sd=[None, None, None, K_PRIOR_SD]) if prior else {}
    kw = {k: (np.array([np.inf if v is None else v for v in vv], float) if k == "prior_sd" else vv)
          for k, vv in kw.items()}
    law = LinearLaw.fit(X, y, names=("s0", "s1", "s2", "k_shell"), **kw)
    s0, s1, s2, _ = law.coef
    u_max = max(x[1] for x in X)
    u_cap = min(-s1 / (2 * s2), u_max) if s2 > 0 else u_max
    return law, float(u_cap)


def sf_fit(hypothesis: str):
    """Back-compatible summary: ((s0, s1, s2, k, u_cap), rms)."""
    law, u_cap = sf_law(hypothesis)
    return (*map(float, law.coef), u_cap), law.rms()


def log10_t_sf(model: MassModel, z: int, n: int) -> float:
    return log10_t_sf_with_sd(model, z, n)[0]


def log10_t_sf_with_sd(model: MassModel, z: int, n: int) -> tuple[float, float]:
    law, u_cap = sf_law(model.hypothesis)
    m, sd = law.predict(_sf_features(model, z, n, u_cap))
    odd = (z % 2) + (n % 2)
    return m + ODD_SF_HINDRANCE * odd, math.hypot(sd, ODD_SF_SD * odd)


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
    qec = m.q_ec(z, n)
    qbm = m.q_beta_minus(z, n)
    p = DecayPrediction(
        z=z, n=n, hypothesis=hypothesis, q_alpha=q, q_alpha_sigma=qs,
        e_alpha=q * (z + n - 4) / (z + n), q_ec=qec,
        fission_barrier=fission_barrier(m, z, n),
    )
    p.log_t["alpha"] = log10_t_alpha(z, n, q)
    p.log_t_sigma["alpha"] = math.hypot(log10_t_alpha_sd(z, n, q), dlogt_dq(z, max(q, 1.0), n) * qs)
    lt_sf, sd_sf = log10_t_sf_with_sd(m, z, n)
    p.log_t["SF"] = lt_sf
    # The shell energy is uncertain by about as much as the local Q-alpha.
    k_sf = sf_law(hypothesis)[0].coef[3]
    p.log_t_sigma["SF"] = math.hypot(sd_sf, k_sf * qs)
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
    return {
        "alpha_halflife_rms_log10": alpha_law().rms(),
        "alpha_halflife_loo_log10": alpha_law().loo_rms(),
        "sf_halflife_rms_log10": sf_law(hypothesis)[0].rms(),
        "sf_halflife_loo_log10": sf_law(hypothesis)[0].loo_rms(),
        "weak_decay_assumed_sigma_log10": WEAK_SIGMA,
    }
