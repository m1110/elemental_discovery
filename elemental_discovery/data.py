"""Curated reference data used to calibrate and validate the models.

All values were compiled by hand from the evaluated literature (AME2020 /
NUBASE2020 for masses and decay data; the original discovery papers for the
superheavy-element cross sections).  Values are rounded and some superheavy
entries carry real experimental uncertainties of ~0.05-0.1 MeV in Q-alpha and
tens of percent in half-life.  They are good enough to *calibrate a
phenomenological model*, not to replace the evaluated tables: check against
the primary source before quoting any number.
"""

from __future__ import annotations

SECONDS_PER_YEAR = 3.15576e7
B_HE4 = 28.2957  # MeV, binding energy of 4He

# ---------------------------------------------------------------------------
# Total nuclear binding energies (MeV), (Z, A, B)
# ---------------------------------------------------------------------------
BINDING_ENERGIES = [
    (20, 40, 342.052),
    (20, 48, 415.991),
    (26, 56, 492.254),
    (28, 58, 506.454),
    (28, 62, 545.259),
    (28, 64, 561.756),
    (38, 88, 768.467),
    (40, 90, 783.893),
    (50, 116, 988.680),
    (50, 120, 1020.546),
    (50, 132, 1102.851),
    (56, 138, 1158.293),
    (62, 144, 1195.730),
    (79, 197, 1559.400),
    (80, 200, 1581.180),
    (82, 208, 1636.430),
    (83, 209, 1640.230),
    (88, 226, 1731.610),
    (90, 232, 1766.690),
    (92, 235, 1783.870),
    (92, 238, 1801.690),
    (94, 239, 1806.920),
    (94, 244, 1836.060),
    (96, 248, 1859.200),
    (98, 252, 1881.270),
]

# ---------------------------------------------------------------------------
# Alpha decay: (Z, A, Q_alpha [MeV], partial alpha half-life [s])
# Partial half-life = total half-life / alpha branching ratio.
# ---------------------------------------------------------------------------
_y = SECONDS_PER_YEAR
ALPHA_DECAYS = [
    # even-even
    (84, 210, 5.407, 1.1956e7),
    (84, 212, 8.954, 2.99e-7),
    (84, 214, 7.833, 1.643e-4),
    (84, 216, 6.906, 0.145),
    (84, 218, 6.115, 186.0),
    (86, 220, 6.405, 55.6),
    (86, 222, 5.590, 3.3035e5),
    (88, 224, 5.789, 3.14e5),
    (88, 226, 4.871, 1600 * _y),
    (90, 226, 6.453, 1836.0),
    (90, 228, 5.520, 1.912 * _y),
    (90, 230, 4.770, 7.54e4 * _y),
    (90, 232, 4.082, 1.40e10 * _y),
    (92, 234, 4.858, 2.455e5 * _y),
    (92, 236, 4.573, 2.342e7 * _y),
    (92, 238, 4.270, 4.468e9 * _y),
    (94, 238, 5.593, 87.7 * _y),
    (94, 240, 5.256, 6561 * _y),
    (94, 242, 4.984, 3.75e5 * _y),
    (94, 244, 4.666, 8.0e7 * _y),
    (96, 244, 5.902, 18.1 * _y),
    (96, 246, 5.475, 4706 * _y),
    (96, 248, 5.162, 3.48e5 * _y / 0.9161),
    (98, 250, 6.128, 13.08 * _y / 0.9992),
    (98, 252, 6.217, 2.645 * _y / 0.969),
    (100, 250, 7.557, 1800.0),
    (100, 252, 7.154, 9.14e4),
    (100, 254, 7.307, 11664.0),
    (102, 252, 8.549, 2.44 / 0.73),
    (102, 254, 8.226, 51.0 / 0.90),
    (114, 286, 10.37, 0.12 / 0.5),
    (114, 288, 10.07, 0.66),
    (116, 290, 11.00, 8.3e-3),
    (116, 292, 10.77, 1.6e-2),
    (118, 294, 11.82, 6.9e-4),
    # odd-A / odd-odd
    (92, 235, 4.679, 7.04e8 * _y),
    (94, 239, 5.245, 24110 * _y),
    (95, 241, 5.638, 432.2 * _y),
    (112, 283, 9.67, 4.0),
    (112, 285, 9.32, 28.0),
    (114, 287, 10.17, 0.48),
    (114, 289, 9.97, 2.1),
    (115, 288, 10.61, 0.19),
    (116, 293, 10.68, 5.7e-2),
    (111, 280, 9.89, 4.6),
    (109, 276, 9.85, 0.6),
    (107, 272, 9.15, 9.8),
]

# ---------------------------------------------------------------------------
# Spontaneous fission: (Z, A, partial SF half-life [s]), even-even only
# ---------------------------------------------------------------------------
SF_HALF_LIVES = [
    (92, 238, 8.2e15 * _y),
    (94, 240, 1.14e11 * _y),
    (94, 242, 6.77e10 * _y),
    (96, 244, 1.32e7 * _y),
    (96, 248, 4.15e6 * _y),
    (98, 250, 1.7e4 * _y),
    (98, 252, 85.7 * _y),
    (98, 254, 5.23e6),
    (100, 256, 1.12e4),
    (100, 258, 3.7e-4),
    (102, 252, 9.0),
    (104, 256, 6.6e-3),
    (106, 260, 1.7e-2),
    (112, 282, 8.0e-4),
    (112, 284, 0.10),
    (114, 286, 0.24),
]

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

KNOWN_ISOTOPES = {  # Z: (A_min, A_max), approximate, gaps ignored
    100: (241, 260), 101: (244, 262), 102: (248, 264), 103: (251, 266),
    104: (252, 268), 105: (255, 270), 106: (257, 271), 107: (260, 274),
    108: (263, 277), 109: (265, 278), 110: (267, 281), 111: (272, 282),
    112: (276, 285), 113: (278, 286), 114: (284, 290), 115: (286, 290),
    116: (288, 293), 117: (293, 294), 118: (293, 294),
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
    rng = KNOWN_ISOTOPES.get(z)
    return bool(rng) and rng[0] <= a <= rng[1]
