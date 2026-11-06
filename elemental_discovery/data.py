"""Reference data used to calibrate and validate the models.

Nuclear masses, half-lives, branching ratios and the list of known nuclides
come from the evaluated NUBASE2020 table (see nubase.py for the source and
for how '#'-flagged superheavy masses are handled).  Fusion cross sections,
upper limits and K X-ray energies are not in NUBASE; they were compiled by
hand from the discovery papers and are rounded -- check them against the
primary sources before quoting any number.
"""

from __future__ import annotations

from . import nubase

SECONDS_PER_YEAR = 3.15576e7
B_HE4 = 28.2957  # MeV, binding energy of 4He

# ---------------------------------------------------------------------------
# Total nuclear binding energies (MeV), (Z, A, B)
# ---------------------------------------------------------------------------
_BINDING_NUCLEI = [
    (20, 40), (20, 48), (26, 56), (28, 58), (28, 62), (28, 64), (38, 88), (40, 90),
    (50, 116), (50, 120), (50, 132), (56, 138), (62, 144), (79, 197), (80, 200), (82, 208),
    (83, 209), (88, 226), (90, 232), (92, 235), (92, 238), (94, 239), (94, 244), (96, 248),
    (98, 252),
]
BINDING_ENERGIES = [(z, a, nubase.binding_energy_MeV(z, a)) for z, a in _BINDING_NUCLEI]

# ---------------------------------------------------------------------------
# Alpha decay: (Z, A, Q_alpha [MeV], partial alpha half-life [s]), Z >= 84,
# every NUBASE2020 ground state with a measured half-life and alpha branch.
# Partial half-life = total half-life / alpha branching ratio.
# ALPHA_Q_SIGMA holds the extra Q uncertainty of odd nuclei whose masses are
# systematic (their observed decays feed excited states).
# ---------------------------------------------------------------------------
_ALPHA = nubase.alpha_decays(84)
ALPHA_DECAYS = [r[:4] for r in _ALPHA]
ALPHA_Q_SIGMA = {(r[0], r[1]): r[4] for r in _ALPHA}

# ---------------------------------------------------------------------------
# Spontaneous fission: (Z, A, partial SF half-life [s]), Z >= 90, every
# NUBASE2020 ground state with a measured half-life and a numeric SF branch.
# ---------------------------------------------------------------------------
SF_HALF_LIVES = nubase.sf_half_lives(90)

# ---------------------------------------------------------------------------
# Measured maximum evaporation-residue cross sections for hot fusion (pb).
# (projectile Z, A, target Z, A, sigma_pb, label)
# ---------------------------------------------------------------------------
CROSS_SECTIONS = [
    (20, 48, 92, 238, 2.5, "48Ca+238U -> Cn"),
    (20, 48, 94, 242, 3.5, "48Ca+242Pu -> Fl"),
    (20, 48, 94, 244, 5.0, "48Ca+244Pu -> Fl"),
    (20, 48, 95, 243, 10.0, "48Ca+243Am -> Mc"),
    (20, 48, 96, 245, 3.7, "48Ca+245Cm -> Lv"),
    (20, 48, 96, 248, 3.4, "48Ca+248Cm -> Lv"),
    (20, 48, 97, 249, 1.1, "48Ca+249Bk -> Ts"),
    (20, 48, 98, 249, 0.5, "48Ca+249Cf -> Og"),
    (22, 50, 94, 244, 0.44, "50Ti+244Pu -> Lv (LBNL 2024)"),
    (24, 54, 92, 238, 0.6, "54Cr+238U -> Lv (Dubna 2023)"),
]

# Published non-observations (upper limits, pb) for element 119/120 searches.
CROSS_SECTION_LIMITS = [
    (22, 50, 97, 249, 0.065, "50Ti+249Bk -> 119 (GSI/TASCA)"),
    (22, 50, 98, 249, 0.20, "50Ti+249Cf -> 120 (GSI/TASCA)"),
    (24, 54, 96, 248, 0.56, "54Cr+248Cm -> 120 (GSI/SHIP)"),
    (26, 58, 94, 244, 0.40, "58Fe+244Pu -> 120 (Dubna)"),
    (28, 64, 92, 238, 0.094, "64Ni+238U -> 120 (GSI/SHIP)"),
]

# ---------------------------------------------------------------------------
# Characteristic K-alpha1 X-ray energies (keV), used to calibrate the
# screened-Dirac model that predicts the X-ray "fingerprint" of a new Z.
# ---------------------------------------------------------------------------
K_ALPHA1 = [
    (29, 8.048),
    (47, 22.163),
    (74, 59.318),
    (79, 68.804),
    (82, 74.969),
    (83, 77.108),
    (92, 98.439),
    (94, 103.734),
    (96, 109.271),
    (98, 115.032),
    (100, 121.030),
]

# ---------------------------------------------------------------------------
# Elements and the approximate known isotope range for Z >= 100 (used to
# decide whether a predicted decay chain lands on an already-known nucleus).
# ---------------------------------------------------------------------------
SYMBOLS = (
    "n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
    "Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()

# Nuclides with Z >= 100 whose half-life has been measured (NUBASE2020).
KNOWN_NUCLIDES = nubase.known_nuclides(100)
KNOWN_ISOTOPES = {  # Z: (A_min, A_max), for display only -- use is_known()
    z: (min(a for zz, a in KNOWN_NUCLIDES if zz == z), max(a for zz, a in KNOWN_NUCLIDES if zz == z))
    for z in sorted({z for z, _ in KNOWN_NUCLIDES})
}


def symbol(z: int) -> str:
    if 0 <= z < len(SYMBOLS):
        return SYMBOLS[z]
    # IUPAC systematic placeholder names: 119 -> Uue, 120 -> Ubn ...
    roots = ["n", "u", "b", "t", "q", "p", "h", "s", "o", "e"]
    s = "".join(roots[int(d)] for d in str(z))
    return s[0].upper() + s[1:]


def is_known(z: int, a: int) -> bool:
    if z < 100:
        return True  # treat the whole sub-fermium chart as charted territory
    return (z, a) in KNOWN_NUCLIDES
