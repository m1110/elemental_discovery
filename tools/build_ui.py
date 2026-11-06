"""Build ui/explorer.html: the interactive explorer, with model output embedded.

    python tools/build_ui.py            # recompute everything (~2 min)
    python tools/build_ui.py --reuse    # re-apply the template to existing data

The page is static: every nuclear prediction is precomputed here, and only the
discovery calculus (cheap: a 2-D quadrature) runs in the browser so its
controls respond instantly.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elemental_discovery import (atomic, calculus, data, decay, nubase, nuclear,  # noqa: E402
                                 production, validation)
from elemental_discovery.linear import _GH_W, _GH_X  # noqa: E402

HYPS = list(nuclear.PROTON_HYPOTHESES)
Z_RANGE = range(100, 131)
N_RANGE = range(145, 211)
MODE = {"alpha": "a", "SF": "f", "EC": "e", "beta-": "b"}


def r(x, d=2):
    return None if x is None or not math.isfinite(x) else round(float(x), d)


def grid():
    out = {}
    for h in HYPS:
        m = nuclear.get_model(h)
        cells = {}
        for z in Z_RANGE:
            n_hi = m.beta_stable_n(z) + 8
            for n in N_RANGE:
                if n < z * 1.3 or n > n_hi or m.s_2n(z, n) < 0:
                    continue
                p = decay.predict(z, n, h)
                cells[f"{z},{n}"] = [r(p.log_t_total), r(p.log_t_total_sigma), MODE[p.dominant_mode],
                                     r(p.q_alpha), r(p.q_alpha_sigma), r(calculus.p_alpha_fastest(p), 3),
                                     {MODE[k]: r(v, 3) for k, v in p.branches.items() if v > 1e-3}]
        out[h] = cells
    return out


def measured():
    out = {}
    for (z, a), rec in nubase.table().items():
        if z < 100 or not rec["half_life_measured"]:
            continue
        br = {k: v[1] for k, v in rec["branches"].items() if k in ("A", "SF", "B+", "EC", "B-", "e+")}
        q = nubase.q_alpha_MeV(z, a)
        out[f"{z},{a - z}"] = [r(math.log10(rec["half_life_s"])), br, r(q, 3), rec["discovery_year"]]
    return out


def reactions(setup):
    rows = []
    for z in (119, 120, 121, 122):
        for rx in production.candidate_reactions(z)[:6]:
            y, w = production.log10_sigma_samples(rx.zp, rx.ap, rx.zt, rx.at)
            # compress the cross-section distribution to 61 nodes
            edges = np.linspace(y[0], y[-1], 62)
            idx = np.clip(np.digitize(y, edges) - 1, 0, 60)
            wy = np.bincount(idx, w * y, 61)
            ww = np.bincount(idx, w, 61)
            keep = ww > 1e-9
            ys, ws = (wy[keep] / ww[keep]), ww[keep] / ww[keep].sum()
            per_h = {}
            for h in HYPS:
                dm = calculus.DiscoveryModel(production.Reaction(rx.zp, rx.ap, rx.zt, rx.at, rx.target_label, h), h, setup)
                per_h[h] = {"residue": dm.residue.name, "Zr": dm.residue.z, "Nr": dm.residue.n,
                            "lt": r(dm.residue.log_t_total, 3), "sd": r(dm.residue.log_t_total_sigma, 3),
                            "p_anchor": r(dm.p_anchor, 4), "path": dm.path}
            dm0 = calculus.DiscoveryModel(rx, "Z=114", setup)
            rows.append({
                "Z": z, "label": rx.label, "beam": f"{rx.ap}{data.symbol(rx.zp)}",
                "target": rx.target_label.split()[0],
                "target_mg_cm2": calculus.target_density(rx, setup),
                "target_limited": (rx.zt, rx.at) in calculus.TARGET_LIMIT_MG_CM2,
                "searched": calculus._has_null(rx),
                "limit_fb": next((x[4] * 1000 for x in data.CROSS_SECTION_LIMITS if x[:4] == (rx.zp, rx.ap, rx.zt, rx.at)), None),
                "E_star": r(rx.e_star_at_barrier(), 1), "xn": rx.best_xn(),
                # chains/day per pb at 1 pµA and 100 % efficiency
                "per_pb_day_unit": dm0.per_pb_day / (setup.beam_pua * setup.efficiency),
                "log_sigma": [round(float(v), 3) for v in ys], "w_sigma": [float(f"{v:.4g}") for v in ws],
                "hyp": per_h,
            })
    return rows


def validation_data():
    al = validation.alpha_laws()
    return {
        "alpha": [{"law": a["law"], "params": a["parameters"], "loo": r(a["loo"]),
                   "forward": {str(c): r(v[0]) for c, v in a["forward"].items()}} for a in al],
        "mass": [{"h": m["hypothesis"], "rms": r(m["rms_MeV"]), "z_gp": r(m["z_rms_gp_only"]),
                  "z_cal": r(m["z_rms_calibrated"]), "k": r(m["k"], 3)} for m in validation.mass_model()],
        "fission": [{"h": f["hypothesis"], "k_data": [r(v) for v in f["k_data_only"]],
                     "k_post": [r(v) for v in f["k_posterior"]], "loo": r(f["loo"])}
                    for f in validation.fission_laws()],
        "prior": [decay.K_PRIOR_MEAN, decay.K_PRIOR_SD],
        "limits": [{"label": x["reaction"], "pred_fb": r(x["predicted_pb"] * 1000, 0),
                    "limit_fb": r(x["upper_limit_pb"] * 1000, 0), "p0": r(x["p_zero_events"])}
                   for x in production.validate_against_limits()],
        "counts": {"alpha": len(data.ALPHA_DECAYS), "sf": len(data.SF_HALF_LIVES),
                   "known": len(data.KNOWN_NUCLIDES)},
    }


def main():
    out = ROOT / "ui" / "explorer.html"
    if "--reuse" in sys.argv:  # re-template only, keeping the exported data
        old = out.read_text()
        js = old[old.index("const D=") + 8:old.index(";</script>", old.index("const D="))]
        tpl = (ROOT / "ui" / "explorer.template.html").read_text()
        out.write_text(tpl.replace("/*__DATA__*/null", js))
        print("re-templated ui/explorer.html")
        return
    setup = calculus.Setup()
    wts = calculus.hypothesis_weights()
    payload = {
        "hyps": HYPS, "weights": {h: r(w, 4) for h, w in wts.items()},
        "Z": [Z_RANGE.start, Z_RANGE.stop - 1], "N": [N_RANGE.start, N_RANGE.stop - 1],
        "grid": grid(), "measured": measured(), "reactions": reactions(setup),
        "symbols": {z: data.symbol(z) for z in range(96, 131)},
        "kalpha": {z: r(float(atomic.k_alpha1_kev(z)[0]), 1) for z in range(100, 131)},
        "gh": {"x": [round(float(v), 6) for v in _GH_X], "w": [float(f"{v:.6g}") for v in _GH_W]},
        "validation": validation_data(),
        "setup": {"beam_pua": setup.beam_pua, "efficiency": setup.efficiency,
                  "flight_us": setup.flight_time_s * 1e6, "chains": setup.chains_needed},
        "citation": nubase.CITATION,
    }
    js = json.dumps(payload, separators=(",", ":"))
    tpl = (ROOT / "ui" / "explorer.template.html").read_text()
    (ROOT / "ui" / "explorer.html").write_text(tpl.replace("/*__DATA__*/null", js))
    print(f"wrote ui/explorer.html ({len(js) / 1024:.0f} kB of data)")


if __name__ == "__main__":
    main()
