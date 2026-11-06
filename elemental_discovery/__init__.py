"""Elemental Discovery: predicting, producing and confirming new elements.

Modules
-------
data          curated calibration data (masses, decays, cross sections, X-rays)
linear        linear laws with predictive variance, priors, closed-form LOO
nuclear       macroscopic-microscopic mass model + Gaussian-process correction
decay         alpha / fission / weak-decay half-lives and decay chains
atomic        finite-nucleus Dirac solver, X-ray fingerprints, critical charge
production    fusion-evaporation reaction planning and cross sections
confirmation  discovery statistics, IUPAC criteria, natural-occurrence searches
survey        island of stability, hypothesis scoring, candidate ranking
validation    out-of-sample (forward) tests of every law
calculus      probability of discovery: P(>= k anchored chains in D days)
report        Markdown report and figures
"""

__version__ = "0.2.0"
