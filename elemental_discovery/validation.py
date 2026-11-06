"""Out-of-sample validation: fit the past, predict the future.

In-sample scatter says how well a law *describes* the data it was fitted to.
What matters for new elements is how well it *predicts* nuclei it has not
seen.  Discovery history gives a natural test: fit on everything lighter than
a cut in Z, predict everything heavier, and compare the errors with the error
bars the model claimed.  A calibrated model has forward z-scores with rms ~1.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from . import data, decay, nuclear, production
from .linear import LinearLaw, forward_test

ALPHA_CUTS = (100, 104, 110, 112)


def _udl_row(r):
    z, a, q, t = r
    return decay.udl_features(z, a - z, q), math.log10(t)


@lru_cache(maxsize=1)
def alpha_laws() -> list[dict]:
    """Viola-Seaborg (6 parameters) vs universal decay law (3), forward and LOO."""
    out = []
    for name, f, npar in (("Viola–Seaborg", decay.viola_seaborg_row, 6),
                          ("Universal decay law (WKB) + odd hindrance", _udl_row, 4)):
        X, y = zip(*map(f, data.ALPHA_DECAYS))
        law = LinearLaw.fit(X, y)
        fw = [forward_test(f, data.ALPHA_DECAYS, lambda r: r[0], c) for c in ALPHA_CUTS]
        out.append({"law": name, "parameters": npar, "in_sample": law.rms(), "loo": law.loo_rms(),
                    "forward": {c: (t["rms"], t["z_rms"]) for c, t in zip(ALPHA_CUTS, fw)},
                    "forward_worst": max(t["rms"] for t in fw)})
    return out


@lru_cache(maxsize=1)
def fission_laws() -> list[dict]:
    """The shell slope k: data-only vs with the explicit physics prior."""
    out = []
    for h in nuclear.PROTON_HYPOTHESES:
        free, _ = decay.sf_law(h, prior=False)
        post, _ = decay.sf_law(h)
        m = nuclear.get_model(h)
        esh = [m.shell_energy(z, a - z) for z, a, _ in data.SF_HALF_LIVES]
        out.append({"hypothesis": h,
                    "k_data_only": (float(free.coef[3]), float(free.coef_sd()[3])),
                    "k_posterior": (float(post.coef[3]), float(post.coef_sd()[3])),
                    "loo": post.loo_rms(), "in_sample": post.rms(),
                    "shell_energy_range": (min(esh), max(esh))})
    return out


def sf_extrapolation(nuclei=((114, 184), (113, 183), (120, 176), (120, 178), (122, 180))) -> list[dict]:
    """Predictive sd of the fission law at representative unknown nuclei."""
    rows = []
    for z, n in nuclei:
        r = {"nucleus": f"{z + n}{data.symbol(z)}"}
        for h in nuclear.PROTON_HYPOTHESES:
            m = nuclear.get_model(h)
            r[h] = (*decay.log10_t_sf_with_sd(m, z, n), m.shell_energy(z, n))
        rows.append(r)
    return rows


def mass_model() -> list[dict]:
    """Forward tests of Q-alpha for every shell hypothesis (cached on disk)."""
    rows = []
    for h in nuclear.PROTON_HYPOTHESES:
        c = dict(nuclear.forward_calibration(h))
        m = nuclear.get_model(h)
        res = [m.q_alpha_global(z, a - z) - q for z, a, q, _ in data.ALPHA_DECAYS if z >= 104]
        c["hypothesis"] = h
        c["in_sample_rms_SHE"] = float(np.sqrt(np.mean(np.square(res))))
        rows.append(c)
    return rows


def cross_sections() -> dict:
    X = np.array([[1.0, production.effective_fissility(*r[:4])] for r in data.CROSS_SECTIONS])
    y = np.log10([r[4] for r in data.CROSS_SECTIONS])
    law = LinearLaw.fit(X, y)
    (c0, c1), s = production.cross_section_fit()
    return {"loo_measured": law.loo_rms(), "in_sample": law.rms(), "scatter": s, "slope": c1,
            "limits": production.validate_against_limits()}
