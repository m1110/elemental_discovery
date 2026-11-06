"""Evaluated nuclear data from NUBASE2020.

Source: F.G. Kondev, M. Wang, W.J. Huang, S. Naimi and G. Audi,
"The NUBASE2020 evaluation of nuclear physics properties",
Chin. Phys. C 45, 030001 (2021).  The file `nubase2020_subset.txt` is an
unmodified extract of `nubase_4.mas20.txt` (ground states with Z >= 80 and the
light calibration nuclei); regenerate it with `tools/extract_nubase.py`.

What is taken from the table
----------------------------
* Mass excesses -> binding energies and ground-state alpha Q values.
  Superheavy masses carry the flag '#' ("from systematics") because none has
  been weighed directly.  AME links them through the *measured* alpha-decay
  energies, so mass differences along a chain still reproduce measured
  Q-alpha (checked in the tests: even-even superheavies agree with the
  published decay energies to ~15 keV).  For odd nuclei the alpha decay feeds
  excited states whose energies AME estimates, so their ground-state Q-alpha
  is given a larger uncertainty (ODD_SYSTEMATIC_Q_SIGMA).
* Half-lives and branching ratios -> partial alpha and fission half-lives.
  Only measured half-lives (no '#', no limits) and numeric branches ('=' or
  '~') are used.
* Which nuclides are known: every Z >= 100 ground state with a measured
  half-life.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

FILE = Path(__file__).with_name("nubase2020_subset.txt")
CITATION = "Kondev et al., NUBASE2020, Chin. Phys. C 45, 030001 (2021)"

YEAR = 3.15576e7
UNITS = {"ys": 1e-24, "zs": 1e-21, "as": 1e-18, "fs": 1e-15, "ps": 1e-12, "ns": 1e-9,
         "us": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0,
         "y": YEAR, "ky": 1e3 * YEAR, "My": 1e6 * YEAR, "Gy": 1e9 * YEAR, "Ty": 1e12 * YEAR,
         "Py": 1e15 * YEAR, "Ey": 1e18 * YEAR, "Zy": 1e21 * YEAR, "Yy": 1e24 * YEAR}

M_NEUTRON_EXCESS = 8071.3181  # keV (NUBASE2020)
M_H1_EXCESS = 7288.971064  # keV
M_HE4_EXCESS = 2424.91587  # keV

# Extra 1-sigma on ground-state Q-alpha of odd nuclei with systematic masses
# (the observed decay feeds excited states of estimated energy).  Set from the
# 0.1-0.27 MeV offsets between AME g.s.-g.s. values and observed decay energies.
ODD_SYSTEMATIC_Q_SIGMA = 0.25  # MeV
BASE_Q_SIGMA = 0.0  # MeV, evaluated-data uncertainty is negligible otherwise

_BRANCH = re.compile(r"^\s*([A-Za-z0-9+\-]+?)\s*([=~<>])\s*([0-9.]+)")


@lru_cache(maxsize=1)
def table() -> dict:
    """{(Z, A): record} for every ground state in the subset file."""
    out = {}
    for line in FILE.read_text().splitlines():
        if line.startswith("#") or len(line) < 40:
            continue
        a, z = int(line[0:3]), int(line[4:7])
        mass = line[18:31].strip()
        t, unit = line[69:78].strip(), line[78:80].strip()
        rec = {"Z": z, "A": a, "name": line[11:16].strip(),
               "mass_excess_keV": float(mass.rstrip("#")) if mass else None,
               "mass_systematic": "#" in mass,
               "half_life_s": None, "half_life_measured": False,
               "branches": parse_branches(line[119:209] if len(line) > 119 else ""),
               "discovery_year": line[114:118].strip()}
        if t and t not in ("stbl", "p-unst") and unit in UNITS:
            rec["half_life_s"] = float(t.strip("#<>~")) * UNITS[unit]
            rec["half_life_measured"] = not any(c in t for c in "#<>~")
        rec["stable"] = t == "stbl"
        out[(z, a)] = rec
    return out


def parse_branches(br: str) -> dict:
    """'A=59 11;SF=41 11' -> {'A': ('=', 59.0), 'SF': ('=', 41.0)}.
    Entries without a number ('SF ?') are omitted."""
    out = {}
    for tok in br.split(";"):
        m = _BRANCH.match(tok)
        if m:
            out[m.group(1)] = (m.group(2), float(m.group(3)))
    return out


def _numeric_branch(rec, mode):
    b = rec["branches"].get(mode)
    if b and b[0] in "=~" and b[1] > 0:
        return b[1] / 100.0
    return None


def binding_energy_MeV(z: int, a: int) -> float:
    """Total (atomic) binding energy from the mass excess, as tabulated by AME."""
    r = table()[(z, a)]
    return (z * M_H1_EXCESS + (a - z) * M_NEUTRON_EXCESS - r["mass_excess_keV"]) / 1000.0


def q_alpha_MeV(z: int, a: int) -> float | None:
    t = table()
    p, d = t.get((z, a)), t.get((z - 2, a - 4))
    if not p or not d or p["mass_excess_keV"] is None or d["mass_excess_keV"] is None:
        return None
    return (p["mass_excess_keV"] - d["mass_excess_keV"] - M_HE4_EXCESS) / 1000.0


def alpha_decays(z_min: int = 84) -> list[tuple]:
    """(Z, A, Q_alpha [MeV], partial alpha half-life [s], extra Q sigma [MeV])
    for every ground state with a measured half-life and alpha branch."""
    out = []
    for (z, a), r in sorted(table().items()):
        if z < z_min or not r["half_life_measured"]:
            continue
        b = _numeric_branch(r, "A")
        q = q_alpha_MeV(z, a)
        if b is None or q is None or q <= 0:
            continue
        odd = (z % 2) or ((a - z) % 2)
        d = table().get((z - 2, a - 4))
        systematic = r["mass_systematic"] or (d is not None and d["mass_systematic"])
        sig = ODD_SYSTEMATIC_Q_SIGMA if (odd and systematic) else BASE_Q_SIGMA
        out.append((z, a, round(q, 5), r["half_life_s"] / b, sig))
    return out


def sf_half_lives(z_min: int = 90) -> list[tuple]:
    """(Z, A, partial SF half-life [s]) for ground states with a measured
    half-life and a numeric spontaneous-fission branch."""
    out = []
    for (z, a), r in sorted(table().items()):
        if z < z_min or not r["half_life_measured"]:
            continue
        b = _numeric_branch(r, "SF")
        if b is not None:
            out.append((z, a, r["half_life_s"] / b))
    return out


def known_nuclides(z_min: int = 100) -> frozenset:
    """Ground states with Z >= z_min whose half-life has been measured."""
    return frozenset(k for k, r in table().items() if k[0] >= z_min and r["half_life_measured"])
