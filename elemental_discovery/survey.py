"""Survey the unknown territory of the chart of nuclides.

* Map predicted half-lives and decay modes over Z = 100-130 under each
  proton-shell hypothesis and locate the "island of stability".
* Score each hypothesis against what is already measured.
* Rank candidate new elements by how feasibly they can be made and how
  firmly they can be confirmed.
* Identify measurements whose outcome would discriminate between hypotheses.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from . import data
from .decay import predict, viola_seaborg_fit, log10_t_alpha
from .nuclear import PROTON_HYPOTHESES, get_model
from .production import beam_days, candidate_reactions
from .confirmation import random_chain_expectation
from .decay import decay_chain

SEPARATOR_FLIGHT_TIME_S = 1.0e-6  # recoil flight time through a gas-filled separator

Z_RANGE = range(100, 131)
N_RANGE = range(145, 216)


@lru_cache(maxsize=8)
def halflife_map(hypothesis: str):
    """Arrays (log10 T_half [s], dominant mode) on the Z x N grid.

    Nuclei with negative two-neutron separation energy, or far on the
    neutron-rich side of the beta-stability valley, are masked.
    """
    m = get_model(hypothesis)
    zs, ns = list(Z_RANGE), list(N_RANGE)
    logt = np.full((len(zs), len(ns)), np.nan)
    mode = np.full((len(zs), len(ns)), "", dtype=object)
    for i, z in enumerate(zs):
        n_hi = m.beta_stable_n(z) + 8  # stay near the valley; beyond it beta- dominates
        for j, n in enumerate(ns):
            if n < z * 1.3 or n > n_hi:
                continue
            if m.s_2n(z, n) < 0:
                continue
            p = predict(z, n, hypothesis)
            logt[i, j] = p.log_t_total
            mode[i, j] = p.dominant_mode
    return zs, ns, logt, mode


def island_of_stability(hypothesis: str, z_min: int = 110):
    """Longest-lived predicted nucleus per Z, and the overall maximum."""
    zs, ns, logt, mode = halflife_map(hypothesis)
    per_z = []
    for i, z in enumerate(zs):
        if z < z_min or np.all(np.isnan(logt[i])):
            continue
        j = int(np.nanargmax(logt[i]))
        per_z.append({"Z": z, "N": ns[j], "A": z + ns[j], "name": f"{z + ns[j]}{data.symbol(z)}",
                      "log10_T_s": float(logt[i, j]), "mode": mode[i, j]})
    peak = max(per_z, key=lambda r: r["log10_T_s"])
    return peak, per_z


def score_hypotheses(z_min: int = 104) -> list[dict]:
    """How well does each hypothesis' *global* model (before the local data-driven
    correction) describe measured superheavy nuclei?  The global model is what
    controls extrapolation into the unknown, so this is the fair comparison."""
    rows = []
    obs_modes = {(118, 294): "alpha", (116, 290): "alpha", (116, 292): "alpha",
                 (114, 288): "alpha", (114, 289): "alpha", (115, 288): "alpha",
                 (112, 284): "SF", (112, 282): "SF", (104, 256): "SF", (106, 260): "SF"}
    for h in PROTON_HYPOTHESES:
        m = get_model(h)
        res = [m.q_alpha_global(z, a - z) - q for z, a, q, _ in data.ALPHA_DECAYS if z >= z_min]
        correct = sum(predict(z, a - z, h).dominant_mode == md for (z, a), md in obs_modes.items())
        rows.append({
            "hypothesis": h,
            "global_rms_qalpha_SHE_MeV": float(np.sqrt(np.mean(np.square(res)))),
            "global_mean_bias_qalpha_SHE_MeV": float(np.mean(res)),
            "decay_modes_correct": f"{correct}/{len(obs_modes)}",
            "rms_binding_all_MeV": m.rms_binding,
        })
    return rows


def rank_new_elements(z_values=(119, 120, 121, 122), hypothesis: str = "Z=114",
                      beam_pua: float = 6.0, target_mg_cm2: float = 0.5) -> list[dict]:
    rows = []
    for z in z_values:
        for r in candidate_reactions(z, hypothesis)[:3]:
            zr, nr = r.residue()
            chain, ending = decay_chain(zr, nr, hypothesis)
            lg, sig = r.sigma_pb()
            survive = 2 ** (-SEPARATOR_FLIGHT_TIME_S / chain[0].half_life)
            epd = r.events_per_day(beam_pua, target_mg_cm2) * survive
            rows.append({
                "Z": z,
                "reaction": r.label,
                "residue": chain[0].name,
                "channel": f"{r.best_xn()}n",
                "E_star_MeV": r.e_star_at_barrier(),
                "sigma_fb": 10**lg * 1000,
                "sigma_factor": 10**sig,
                "survive_flight": survive,
                "days_for_3_chains_90pc": beam_days(epd, 3, 0.90),
                "first_step": chain[0].dominant_mode,
                "log10_T_s": chain[0].log_t_total,
                "log10_T_sigma": chain[0].log_t_total_sigma,
                "chain_ending": ending,
                "anchored": ending.startswith("reaches known"),
                "random_chains": random_chain_expectation(chain),
            })
    return rows


def reachable_nuclei(z_values=(119, 120, 121, 122)) -> list[tuple[int, int]]:
    """Evaporation residues of the candidate reactions plus their alpha daughters."""
    out = []
    for z in z_values:
        for r in candidate_reactions(z)[:3]:
            zr, nr = r.residue()
            for k in range(4):
                for dn in (0, 1):  # neighbouring xn channel too
                    key = (zr - 2 * k, nr - 2 * k - dn)
                    if key not in out and not data.is_known(key[0], sum(key)):
                        out.append(key)
    return out


def discriminating_measurements(nuclei=None) -> list[dict]:
    """Reachable nuclei whose Q-alpha differs most between hypotheses, relative
    to the model uncertainty: the measurements that would settle where the
    next proton shell lies."""
    if nuclei is None:
        nuclei = reachable_nuclei()
    rows = []
    for z, n in nuclei:
        vals = {h: get_model(h).q_alpha_with_uncertainty(z, n) for h in PROTON_HYPOTHESES}
        qs = [v[0] for v in vals.values()]
        sig = float(np.mean([v[1] for v in vals.values()]))
        spread = max(qs) - min(qs)
        rows.append({
            "nucleus": f"{z + n}{data.symbol(z)}", "Z": z, "N": n,
            "Q_alpha": {h: v[0] for h, v in vals.items()},
            "sigma": sig, "spread_MeV": spread, "power": spread / sig,
        })
    return sorted(rows, key=lambda r: -r["power"])
