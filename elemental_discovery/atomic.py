"""Atomic physics of new elements.

* A radial Dirac-equation solver for one electron in the field of a finite
  (uniformly charged) nucleus.  Used for
    - the K-alpha X-ray energy of a new element (an *experimental fingerprint*
      of Z: characteristic X-rays in coincidence with alpha decay identify the
      atomic number directly), with screening constants calibrated on known
      elements;
    - the critical charge Z_cr at which the 1s level dives into the Dirac
      negative-energy continuum -- the physical end of the electronic
      periodic table as we know it.
* Ground-state electron configurations from the Madelung (n + l) rule,
  extended to the g-block, alongside relativistic literature predictions for
  where it is known to fail.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np

from . import data

ALPHA = 1 / 137.035999084
ME_KEV = 510.99895
LAMBDA_C_FM = 386.15926796  # reduced Compton wavelength of the electron, fm


def nuclear_radius_fm(a: float) -> float:
    """Equivalent uniform-sphere radius, fm."""
    return 1.2 * a ** (1 / 3)


def mass_number_guess(z: float) -> float:
    """Typical mass number near the valley of stability (A -> ~2.5 Z for heavy Z)."""
    z = np.asarray(z, float)
    return np.minimum(z * (2.0 + 0.0055 * z), 2.5 * z)


def _rhs(r, g, f, e, kappa, z, radius):
    inside = r < radius
    v = np.where(inside, -z * ALPHA * (3 - (r / radius) ** 2) / (2 * radius), -z * ALPHA / r)
    dg = -kappa / r * g + (e + 1 - v) * f
    df = kappa / r * f - (e - 1 - v) * g
    return dg * r, df * r  # derivatives with respect to x = ln r


def _count_nodes(e, kappa, z, radius, n_qn, steps=3000):
    """Integrate outward for every (z, e) pair, return number of nodes of G."""
    za = z * ALPHA
    r_max = 30.0 * n_qn**2 / za
    x0, x1 = math.log(1e-7), np.log(r_max)
    h = (x1 - x0) / steps
    ak = abs(kappa)
    r = np.exp(x0) * np.ones_like(e)
    v0 = -za * 1.5 / radius
    g = r**ak
    f = -(e - 1 - v0) / (2 * ak + 1) * r ** (ak + 1) if kappa < 0 else r ** (ak + 1)
    nodes = np.zeros(e.shape, int)
    x = np.full(e.shape, x0)
    for _ in range(steps):
        r = np.exp(x)
        k1g, k1f = _rhs(r, g, f, e, kappa, z, radius)
        rm = np.exp(x + h / 2)
        k2g, k2f = _rhs(rm, g + h / 2 * k1g, f + h / 2 * k1f, e, kappa, z, radius)
        k3g, k3f = _rhs(rm, g + h / 2 * k2g, f + h / 2 * k2f, e, kappa, z, radius)
        rp = np.exp(x + h)
        k4g, k4f = _rhs(rp, g + h * k3g, f + h * k3f, e, kappa, z, radius)
        gn = g + h / 6 * (k1g + 2 * k2g + 2 * k3g + k4g)
        fn = f + h / 6 * (k1f + 2 * k2f + 2 * k3f + k4f)
        nodes += (np.sign(gn) != np.sign(g)) & (g != 0)
        scale = np.maximum(np.abs(gn), 1e-300)
        big = scale > 1e100
        g = np.where(big, gn / scale, gn)
        f = np.where(big, fn / scale, fn)
        x = x + h
    return nodes, x1


def dirac_energy(z, kappa: int = -1, a=None, n_qn: int = 1, rounds: int = 5, width: int = 24):
    """Lowest bound-state energy (in units of m c^2, *including* rest mass, so
    -1 < E < 1) for each Z, for a uniformly charged finite nucleus.

    Lowest state of kappa = -1 is 1s1/2; lowest of kappa = -2 is 2p3/2.
    Vectorised: z can be an array.
    """
    z = np.atleast_1d(np.asarray(z, float))
    a = np.atleast_1d(mass_number_guess(z) if a is None else np.asarray(a, float))
    radius = nuclear_radius_fm(a) / LAMBDA_C_FM
    lo, hi = np.full(z.shape, -1.0), np.full(z.shape, 1.0 - 1e-12)
    for _ in range(rounds):
        grid = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, width)[None, :]
        zz = np.repeat(z[:, None], width, 1)
        rr = np.repeat(radius[:, None], width, 1)
        nodes, _ = _count_nodes(grid, kappa, zz, rr, n_qn)
        has = nodes >= 1
        idx = np.argmax(has, axis=1)  # first energy with a node
        idx = np.where(has.any(axis=1), idx, width - 1)
        idx = np.maximum(idx, 1)
        rows = np.arange(len(z))
        lo, hi = grid[rows, idx - 1], grid[rows, idx]
    return 0.5 * (lo + hi)


def binding_kev(z, kappa=-1, n_qn=1, a=None):
    """Binding energy (keV, positive) of the lowest state with this kappa."""
    return (1 - dirac_energy(z, kappa, a, n_qn)) * ME_KEV


# ---------------------------------------------------------------------------
# K-alpha1 X-ray fingerprint (2p3/2 -> 1s1/2) with fitted screening constants
# ---------------------------------------------------------------------------
_CACHE_FILE = Path(os.environ.get("ELEMENTAL_CACHE", Path.home() / ".cache" / "elemental_discovery")) / "kalpha.json"


@lru_cache(maxsize=1)
def kalpha_screening():
    """Screening constants, cached on disk (the Dirac fit takes ~20 s)."""
    key = hashlib.sha1(json.dumps(data.K_ALPHA1).encode()).hexdigest()
    try:
        cached = json.loads(_CACHE_FILE.read_text())
        if cached.get("key") == key:
            return tuple(cached["value"])
    except (OSError, ValueError):
        pass
    value = _fit_kalpha_screening()
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({"key": key, "value": value}))
    except OSError:
        pass
    return value


def _fit_kalpha_screening():
    """Fit screening constants (s1 for 1s, s2 for 2p3/2) to measured K-alpha1.

    Returns (s1, s2, relative rms error).  The two level tables are solved in
    one vectorised batch each, then every (s1, s2) pair is scored.
    """
    zs = np.array([z for z, _ in data.K_ALPHA1], float)
    obs = np.array([e for _, e in data.K_ALPHA1])
    s1_grid = np.linspace(0.0, 6.0, 13)
    s2_grid = np.linspace(2.0, 14.0, 25)
    a = mass_number_guess(zs)
    z1 = (zs[None, :] - s1_grid[:, None]).ravel()
    z2 = (zs[None, :] - s2_grid[:, None]).ravel()
    e1 = binding_kev(z1, -1, 1, a=np.tile(a, len(s1_grid))).reshape(len(s1_grid), -1)
    e2 = binding_kev(z2, -2, 2, a=np.tile(a, len(s2_grid))).reshape(len(s2_grid), -1)
    pred = e1[:, None, :] - e2[None, :, :]
    err = np.sqrt(np.mean(((pred - obs) / obs) ** 2, axis=-1))
    i, j = np.unravel_index(np.argmin(err), err.shape)
    return float(s1_grid[i]), float(s2_grid[j]), float(err[i, j])


def k_alpha1_kev(z):
    s1, s2, _ = kalpha_screening()
    z = np.atleast_1d(np.asarray(z, float))
    a = mass_number_guess(z)
    return binding_kev(z - s1, -1, 1, a) - binding_kev(z - s2, -2, 2, a)


def critical_charge(a_over_z: float = 2.5) -> dict:
    """Z at which the 1s1/2 level reaches E = -m c^2 (binding = 2 m c^2).

    Beyond Z_cr a bare nucleus would spontaneously create electron-positron
    pairs: the electronic structure of 'ordinary' atoms ends there.
    E(Z) is computed close to the dive point and extrapolated (quadratic).
    This is a bare one-electron ion; screening by the other electrons and the
    exact nuclear radius move the literature value within ~170-173.
    """
    zs = np.array([150, 154, 158, 161, 164, 166, 168, 169], float)
    e = dirac_energy(zs, -1, a=a_over_z * zs, rounds=6)
    ok = e > -0.99  # stay clear of the continuum edge, where shooting degrades
    c = np.polyfit(zs[ok], e[ok], 2)
    roots = np.roots([c[0], c[1], c[2] + 1.0])
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 160)
    z_cr = real[0] if real else float("nan")
    return {
        "Z_critical": float(z_cr),
        "point_nucleus_limit": 1 / ALPHA,
        "energies_mc2": dict(zip(map(int, zs), map(float, e))),
    }


# ---------------------------------------------------------------------------
# Electron configurations
# ---------------------------------------------------------------------------
L_LETTERS = "spdfghi"


@lru_cache(maxsize=1)
def madelung_order(max_n: int = 9):
    orbs = [(n, l) for n in range(1, max_n + 1) for l in range(0, min(n, 5))]
    return sorted(orbs, key=lambda nl: (nl[0] + nl[1], nl[0]))


def madelung_configuration(z: int) -> str:
    remaining, parts = z, []
    for n, l in madelung_order():
        if remaining <= 0:
            break
        k = min(remaining, 2 * (2 * l + 1))
        parts.append(f"{n}{L_LETTERS[l]}{k}")
        remaining -= k
    return " ".join(parts)


def madelung_block(z: int) -> str:
    n, l = None, None
    remaining = z
    for n, l in madelung_order():
        k = min(remaining, 2 * (2 * l + 1))
        remaining -= k
        if remaining <= 0:
            break
    return L_LETTERS[l]


# Relativistic Dirac-Fock / coupled-cluster predictions from the literature
# (Fricke et al. 1971; Pyykko 2011; Eliav et al.) for the first new elements.
RELATIVISTIC_PREDICTIONS = {
    119: {"config": "[Og] 8s1", "group": "1 (alkali metal)", "IE1_eV": 4.8,
          "note": "8s strongly contracted: IE higher than Fr/Cs, chemistry closer to K/Rb"},
    120: {"config": "[Og] 8s2", "group": "2 (alkaline earth)", "IE1_eV": 5.85,
          "note": "expected to resemble Ca/Sr more than Ra"},
    121: {"config": "[Og] 8s2 8p1/2^1", "group": "3 / start of superactinides",
          "note": "relativistically stabilised 8p1/2 fills before 5g (Madelung would say 5g1)"},
    122: {"config": "[Og] 7d1 8s2 8p1/2^1", "group": "superactinide",
          "note": "7d, 6f, 5g and 8p1/2 lie within ~1 eV: configuration mixing, ill-defined 'block'"},
}


def element_summary(z: int) -> dict:
    out = {
        "Z": z,
        "symbol": data.symbol(z),
        "period": madelung_order() and _period(z),
        "madelung_configuration": madelung_configuration(z),
        "madelung_block": madelung_block(z),
        "inner_electron_speed_v_over_c": z * ALPHA,
    }
    if z in RELATIVISTIC_PREDICTIONS:
        out["relativistic_prediction"] = RELATIVISTIC_PREDICTIONS[z]
    return out


def _period(z: int) -> int:
    edges = [2, 10, 18, 36, 54, 86, 118, 168, 218]
    return next(i + 1 for i, e in enumerate(edges) if z <= e)
