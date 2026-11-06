"""Macroscopic-microscopic nuclear mass model.

The binding energy is a liquid drop (volume, surface, Coulomb, volume and
surface symmetry, pairing) plus a Myers-Swiatecki style shell correction
built from a chosen set of *magic numbers*.  Because nobody yet knows where
the next proton shell closure lies (114, 120 or 126 are all predicted by
different theories), the magic numbers are a model *hypothesis*: the model
can be refitted under each hypothesis and the predictions compared.

All coefficients are obtained by a weighted linear least-squares fit to
measured binding energies AND measured alpha-decay Q values (the latter pin
down the shell structure in the superheavy region, where it matters most).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os

import numpy as np

from . import data

NEUTRON_MAGIC = (0, 2, 8, 14, 28, 50, 82, 126, 184, 258)

# Proton-shell hypotheses for the next closed shell beyond Z = 82.
PROTON_HYPOTHESES = {
    "Z=114": (0, 2, 8, 14, 28, 50, 82, 114, 164),
    "Z=120": (0, 2, 8, 14, 28, 50, 82, 120, 164),
    "Z=126": (0, 2, 8, 14, 28, 50, 82, 126, 184),
}

SHELL_C_SMALL = 0.325  # Myers-Swiatecki constant in the shell formula


def _shell_F(n: float, magic: tuple[int, ...]) -> float:
    """Myers-Swiatecki shell function F(n): zero at magic numbers, >0 between."""
    for lo, hi in zip(magic[:-1], magic[1:]):
        if lo <= n <= hi:
            q = 0.6 * (hi ** (5 / 3) - lo ** (5 / 3)) / (hi - lo)
            return q * (n - lo) - 0.6 * (n ** (5 / 3) - lo ** (5 / 3))
    return 0.0


def shell_function(z: int, n: int, proton_magic: tuple[int, ...]) -> float:
    """Dimensionless shell correction S/C.  Negative = extra binding."""
    a = z + n
    return (_shell_F(n, NEUTRON_MAGIC) + _shell_F(z, proton_magic)) / (a / 2) ** (
        2 / 3
    ) - SHELL_C_SMALL * a ** (1 / 3)


def _pairing(z: int, n: int) -> float:
    a = z + n
    if z % 2 == 0 and n % 2 == 0:
        return 1 / np.sqrt(a)
    if z % 2 == 1 and n % 2 == 1:
        return -1 / np.sqrt(a)
    return 0.0


TERM_NAMES = (
    "volume",
    "surface",
    "coulomb",
    "symmetry",
    "surface_symmetry",
    "pairing",
    "curvature",
    "coulomb_exchange",
    "coulomb_diffuseness",
    "wigner",
)

_U_GRID = np.linspace(0.0, 4.0, 401)  # theta^2 grid for the deformation minimisation


def basis(z: int, n: int) -> np.ndarray:
    """Liquid-drop feature vector phi such that B_LD(Z,N) = coeffs . phi."""
    a = z + n
    i2 = (n - z) ** 2
    return np.array(
        [
            a,
            -(a ** (2 / 3)),
            -z * (z - 1) / a ** (1 / 3),
            -i2 / a,
            i2 / a ** (4 / 3),
            _pairing(z, n),
            a ** (1 / 3),
            -(z ** (4 / 3)) / a ** (1 / 3),
            -z * z / a,
            -abs(n - z) / a,
        ]
    )


def fissility(z: int, n: int) -> float:
    a = z + n
    i = (n - z) / a
    return (z * z / a) / (50.883 * (1 - 1.7826 * i * i))


def shell_and_deformation(
    z: int, n: int, proton_magic: tuple[int, ...], c_shell: float, kappa: float
) -> tuple[float, float]:
    """Myers-Swiatecki ground-state microscopic energy (MeV, <0 = extra binding).

    E(theta) = C*S*(1 - 2 theta^2) exp(-theta^2) + K theta^2, minimised over
    theta.  Closed-shell nuclei (S < 0) stay spherical; mid-shell nuclei
    (S > 0) deform, and the deformation turns an anti-shell penalty into a
    net gain -- this is why the actinides are deformed and still bound.
    K = kappa (1 - x) A^{2/3} is the liquid-drop stiffness (x = fissility).
    Returns (energy, theta^2 at the minimum).
    """
    a = z + n
    s = c_shell * shell_function(z, n, proton_magic)
    k = kappa * max(1 - fissility(z, n), 0.02) * a ** (2 / 3)
    u = _U_GRID
    e = s * (1 - 2 * u) * np.exp(-u) + k * u
    i = int(np.argmin(e))
    return float(e[i]), float(u[i])


@dataclass
class MassModel:
    hypothesis: str = "Z=114"
    coeffs: np.ndarray = field(default=None, repr=False)
    c_shell: float = 5.8
    kappa: float = 1.0
    rms_binding: float = float("nan")
    rms_qalpha: float = float("nan")
    _gp: "ResidualGP" = field(default=None, repr=False)
    binding_data: list = field(default=None, repr=False)
    alpha_data: list = field(default=None, repr=False)
    # Growth of the Q-alpha error with distance from measured nuclei (MeV per
    # unit distance in the Z-N plane), calibrated by forward tests; None = off.
    extrap_k: float | None = None

    def __post_init__(self):
        self.binding_data = list(data.BINDING_ENERGIES if self.binding_data is None else self.binding_data)
        self.alpha_data = list(data.ALPHA_DECAYS if self.alpha_data is None else self.alpha_data)

    @property
    def proton_magic(self) -> tuple[int, ...]:
        return PROTON_HYPOTHESES[self.hypothesis]

    # -- fitting ----------------------------------------------------------
    def _linear_fit(self, c_shell, kappa, sigma_b, sigma_q):
        pm = self.proton_magic

        def esh(z, n):
            return shell_and_deformation(z, n, pm, c_shell, kappa)[0]

        rows, targets, weights = [], [], []
        for z, a, b in self.binding_data:
            n = a - z
            rows.append(basis(z, n))
            targets.append(b + esh(z, n))
            weights.append(1 / sigma_b)
        for z, a, q, _ in self.alpha_data:
            n = a - z
            rows.append(basis(z - 2, n - 2) - basis(z, n))
            targets.append(q - data.B_HE4 + esh(z - 2, n - 2) - esh(z, n))
            weights.append(1 / sigma_q)
        w = np.array(weights)
        X = np.array(rows) * w[:, None]
        y = np.array(targets) * w
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        chi2 = float(np.sum((X @ coeffs - y) ** 2))
        return coeffs, chi2

    def fit(self, sigma_b: float = 1.0, sigma_q: float = 0.15) -> "MassModel":
        """Grid search over the two non-linear shell parameters (C, kappa),
        linear least squares for the liquid-drop coefficients."""
        best = None
        for c in np.linspace(2.0, 10.0, 33):
            for kap in np.geomspace(0.05, 5.0, 25):
                coeffs, chi2 = self._linear_fit(c, kap, sigma_b, sigma_q)
                if best is None or chi2 < best[0]:
                    best = (chi2, c, kap, coeffs)
        _, self.c_shell, self.kappa, self.coeffs = best
        self._gp = None
        self.c_shell, self.kappa = float(self.c_shell), float(self.kappa)

        rb = [self.binding(z, a - z) - b for z, a, b in self.binding_data]
        rq = [self.q_alpha_global(z, a - z) - q for z, a, q, _ in self.alpha_data]
        self.rms_binding = float(np.sqrt(np.mean(np.square(rb))))
        self.rms_qalpha = float(np.sqrt(np.mean(np.square(rq))))
        return self

    def _require_fit(self):
        if self.coeffs is None:
            self.fit()

    # -- observables ------------------------------------------------------
    def shell_energy(self, z: int, n: int) -> float:
        """Microscopic energy (MeV). Negative = extra binding from shell effects."""
        self._require_fit()
        return shell_and_deformation(z, n, self.proton_magic, self.c_shell, self.kappa)[0]

    def deformation(self, z: int, n: int) -> float:
        """theta^2 at the ground-state minimum (0 = spherical)."""
        self._require_fit()
        return shell_and_deformation(z, n, self.proton_magic, self.c_shell, self.kappa)[1]

    def binding(self, z: int, n: int) -> float:
        """Total binding energy (MeV)."""
        self._require_fit()
        if z == 2 and n == 2:
            return data.B_HE4
        if z <= 0 or n < 0:
            return 0.0
        return float(self.coeffs @ basis(z, n)) - self.shell_energy(z, n)

    def q_alpha_global(self, z: int, n: int) -> float:
        """Q-alpha from the global macroscopic-microscopic model alone."""
        return self.binding(z - 2, n - 2) + data.B_HE4 - self.binding(z, n)

    def q_alpha(self, z: int, n: int) -> float:
        """Best-estimate Q-alpha (global model + local data-driven correction)."""
        return self.q_alpha_with_uncertainty(z, n)[0]

    def q_alpha_with_uncertainty(self, z: int, n: int) -> tuple[float, float]:
        """(Q-alpha, 1-sigma) in MeV.  Near measured nuclei the GP correction
        pulls the prediction onto the data and the error is small; far away it
        relaxes to the global model.

        A stationary GP's error saturates at its amplitude however far one
        extrapolates, while the global model's bias keeps growing.  Forward
        tests (fit Z < cut, predict Z >= cut) show the saturated GP error is
        2.5-3.5x too small.  So a distance term is added in quadrature,
            sd^2 = sd_GP^2 + (k d)^2,
        with d the distance in the (Z, N) plane to the nearest measured alpha
        emitter and k calibrated by maximum likelihood on the forward tests."""
        self._require_fit()
        if self._gp is None:
            self._gp = ResidualGP.train(self)
        corr, sig = self._gp.predict(z, n)
        k = self.extrap_k
        if k is None and self.alpha_data == list(data.ALPHA_DECAYS):
            k = extrapolation_rate(self.hypothesis)
        if k:
            sig = float(np.hypot(sig, k * self.distance_to_data(z, n)))
        return self.q_alpha_global(z, n) + corr, sig

    def distance_to_data(self, z: int, n: int) -> float:
        x = np.array([[zz, aa - zz] for zz, aa, *_ in self.alpha_data], float)
        return float(np.min(np.hypot(x[:, 0] - z, x[:, 1] - n)))

    def s_n(self, z: int, n: int) -> float:
        """One-neutron separation energy."""
        return self.binding(z, n) - self.binding(z, n - 1)

    def s_2n(self, z: int, n: int) -> float:
        return self.binding(z, n) - self.binding(z, n - 2)

    def q_ec(self, z: int, n: int) -> float:
        """Q value for electron capture (Z,N) -> (Z-1,N+1), MeV (atomic masses)."""
        dm = 0.782347  # m_n - m_H
        return self.binding(z - 1, n + 1) - self.binding(z, n) - dm

    def q_beta_minus(self, z: int, n: int) -> float:
        dm = 0.782347
        return self.binding(z + 1, n - 1) - self.binding(z, n) + dm

    def beta_stable_n(self, z: int) -> int:
        """Neutron number of the most bound nucleus for this Z that is stable
        against both beta- decay and electron capture."""
        cands = [
            n
            for n in range(int(1.2 * z), int(1.9 * z))
            if self.q_ec(z, n) <= 0 and self.q_beta_minus(z, n) <= 0
        ]
        if not cands:
            cands = list(range(int(1.2 * z), int(1.9 * z)))
            return min(cands, key=lambda k: abs(self.q_ec(z, k)) + abs(self.q_beta_minus(z, k)))
        return cands[len(cands) // 2]

    def describe(self) -> dict:
        self._require_fit()
        return {
            "hypothesis": self.hypothesis,
            "coefficients_MeV": dict(zip(TERM_NAMES, map(float, self.coeffs))),
            "shell_strength_C_MeV": self.c_shell,
            "deformation_stiffness_kappa": self.kappa,
            "rms_binding_MeV": self.rms_binding,
            "rms_qalpha_MeV": self.rms_qalpha,
            "gp_correction": None if self._gp is None else self._gp.describe(),
        }


class ResidualGP:
    """Gaussian process on Q-alpha residuals (data - global model) over (Z, N).

    Isotropic squared-exponential kernel; amplitude and length scale chosen by
    maximising the log marginal likelihood on a grid.  This is the standard
    "Bayesian model correction" used to extrapolate nuclear masses with honest
    error bars.
    """

    def __init__(self, x, r, amp, length, noise):
        self.x, self.r, self.amp, self.length, self.noise = x, r, amp, length, noise
        k = self._k(x, x) + noise**2 * np.eye(len(x))
        self.l_chol = np.linalg.cholesky(k)
        self.alpha = np.linalg.solve(self.l_chol.T, np.linalg.solve(self.l_chol, r))

    def _k(self, a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return self.amp**2 * np.exp(-0.5 * d2 / self.length**2)

    @staticmethod
    def _lml(x, r, amp, length, noise):
        d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
        k = amp**2 * np.exp(-0.5 * d2 / length**2) + noise**2 * np.eye(len(x))
        try:
            lc = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return -np.inf
        al = np.linalg.solve(lc.T, np.linalg.solve(lc, r))
        return float(-0.5 * r @ al - np.log(np.diag(lc)).sum())

    @classmethod
    def train(cls, model: "MassModel", noise: float = 0.08) -> "ResidualGP":
        x = np.array([[z, a - z] for z, a, _, _ in model.alpha_data], float)
        r = np.array(
            [q - model.q_alpha_global(z, a - z) for z, a, q, _ in model.alpha_data]
        )
        best = max(
            (
                (cls._lml(x, r, amp, ln, noise), amp, ln)
                for amp in np.linspace(0.2, 1.5, 27)
                for ln in np.linspace(1.5, 12.0, 43)
            ),
            key=lambda t: t[0],
        )
        return cls(x, r, best[1], best[2], noise)

    def predict(self, z: int, n: int) -> tuple[float, float]:
        xs = np.array([[z, n]], float)
        ks = self._k(xs, self.x)[0]
        mean = float(ks @ self.alpha)
        v = np.linalg.solve(self.l_chol, ks)
        var = max(self.amp**2 - float(v @ v), 0.0) + self.noise**2
        return mean, float(np.sqrt(var))

    def describe(self) -> dict:
        return {"amplitude_MeV": self.amp, "length_scale": self.length, "noise_MeV": self.noise}


_CACHE: dict[str, MassModel] = {}

FORWARD_CUTS = (104, 108, 110, 112, 114, 116)
_CACHE_DIR = Path(os.environ.get("ELEMENTAL_CACHE", Path.home() / ".cache" / "elemental_discovery"))


def forward_qalpha_errors(hypothesis: str, cuts=FORWARD_CUTS) -> list[tuple[float, float, float]]:
    """For each cut: refit on alpha emitters with Z < cut, predict those with
    Z >= cut.  Returns (distance to data, error, GP sd) for every prediction."""
    out = []
    for cut in cuts:
        train = [r for r in data.ALPHA_DECAYS if r[0] < cut]
        m = MassModel(hypothesis, alpha_data=train, extrap_k=0.0).fit()
        for z, a, q, _ in data.ALPHA_DECAYS:
            if z >= cut:
                qp, sd = m.q_alpha_with_uncertainty(z, a - z)
                out.append((m.distance_to_data(z, a - z), qp - q, sd))
    return out


def _calibrate(rows) -> dict:
    d, e, s = map(np.array, zip(*rows))
    ks = np.linspace(0.0, 0.5, 501)
    v = s[None, :] ** 2 + (ks[:, None] * d[None, :]) ** 2
    ll = -0.5 * (e[None, :] ** 2 / v + np.log(2 * np.pi * v)).sum(1)
    i = int(np.argmax(ll))
    k = float(ks[i])
    return {"k": k, "loglik": float(ll[i]), "n": int(len(e)),
            "rms_MeV": float(np.sqrt(np.mean(e**2))), "bias_MeV": float(np.mean(e)),
            "z_rms_gp_only": float(np.sqrt(np.mean((e / s) ** 2))),
            "z_rms_calibrated": float(np.sqrt(np.mean(e**2 / (s**2 + (k * d) ** 2))))}


def forward_calibration(hypothesis: str) -> dict:
    """Forward-test calibration of the Q-alpha error, cached on disk (it needs
    ~6 refits of the mass model).  k is the ML growth rate of the error (MeV
    per unit distance in the Z-N plane); loglik is the predictive
    log-likelihood of the forward tests -- how well this hypothesis *predicted*
    heavier nuclei before seeing them."""
    key = hashlib.sha1(json.dumps([hypothesis, data.ALPHA_DECAYS, data.BINDING_ENERGIES,
                                   FORWARD_CUTS, PROTON_HYPOTHESES[hypothesis], 2]).encode()).hexdigest()
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    f = _CACHE_DIR / "qalpha_extrapolation.json"
    try:
        cached = json.loads(f.read_text())
    except (OSError, ValueError):
        cached = {}
    if not isinstance(cached.get(key), dict):
        cached[key] = _calibrate(forward_qalpha_errors(hypothesis))
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(cached))
        except OSError:
            pass
    _RATE_CACHE[key] = cached[key]
    return cached[key]


def extrapolation_rate(hypothesis: str) -> float:
    return forward_calibration(hypothesis)["k"]


_RATE_CACHE: dict[str, dict] = {}


def get_model(hypothesis: str = "Z=114") -> MassModel:
    if hypothesis not in _CACHE:
        _CACHE[hypothesis] = MassModel(hypothesis).fit()
    return _CACHE[hypothesis]
