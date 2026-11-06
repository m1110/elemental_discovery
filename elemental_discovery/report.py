"""Generate the full Markdown report and figures."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from . import atomic, calculus, data, decay, nuclear, production, survey, validation
from .confirmation import confirmation_plan, significance

# Reference palette (light mode) from the dataviz guidance.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"Z=114": "#2a78d6", "Z=120": "#eb6834", "Z=126": "#1baf7a"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    })


def fig_halflife_maps(out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    _style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, constrained_layout=True)
    bounds = [-6, -3, 0, 3, 6, 9]  # 5 bins + 2 open-ended = 7 ramp steps
    cmap = ListedColormap(SEQ)
    norm = BoundaryNorm(bounds, cmap.N, extend="both")
    for ax, h in zip(axes, nuclear.PROTON_HYPOTHESES):
        zs, ns, logt, _ = survey.halflife_map(h)
        im = ax.pcolormesh(np.array(ns) - 0.5, np.array(zs) - 0.5,
                           np.ma.masked_invalid(logt), cmap=cmap, norm=norm,
                           edgecolors=SURFACE, linewidth=0.3)
        # outline measured nuclei
        for z, (amin, amax) in data.KNOWN_ISOTOPES.items():
            ax.plot([amin - z - 0.5, amax - z + 0.5], [z, z], color=INK, lw=1.4,
                    solid_capstyle="butt", alpha=0.85)
        ax.axvline(184, color=INK2, lw=1, ls="--")
        magic_z = int(h.split("=")[1])
        ax.axhline(magic_z, color=INK2, lw=1, ls="--")
        peak, _ = survey.island_of_stability(h)
        ax.plot(peak["N"], peak["Z"], marker="*", ms=13, color=SERIES[h],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(f"Proton shell at {h}   (longest-lived: {peak['name']})", fontsize=10.5)
        ax.set_xlabel("Neutron number N")
        ax.grid(False)
    axes[0].set_ylabel("Proton number Z")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("Predicted log10 half-life (s)")
    fig.suptitle("Predicted half-lives of superheavy nuclei under three shell hypotheses "
                 "(black bars = nuclei already observed; dashed = magic numbers)", fontsize=11.5)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_cross_sections(out: Path):
    import matplotlib.pyplot as plt

    _style(plt)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    c0, c1, s_, wt = production.cross_section_posterior()
    xs = np.linspace(0.76, 0.92, 100)
    ys, sds = [], []
    for x in xs:
        m = c0 + c1 * x
        mu = float(wt @ m)
        ys.append(mu)
        sds.append(float(np.sqrt(wt @ (s_**2 + (m - mu) ** 2))))
    ys, sds = np.array(ys), np.array(sds)
    ax.fill_between(xs, 10 ** (ys - sds) * 1000, 10 ** (ys + sds) * 1000, color=SEQ[0], lw=0)
    ax.plot(xs, 10**ys * 1000, color=SEQ[4], lw=2,
            label="Poisson-exposure fit (±1σ predictive band)")
    for zp, ap, zt, at, s, lab in data.CROSS_SECTIONS:
        ax.plot(production.effective_fissility(zp, ap, zt, at), s * 1000, "o", ms=8,
                color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for zp, ap, zt, at, s, lab in data.CROSS_SECTION_LIMITS:
        ax.plot(production.effective_fissility(zp, ap, zt, at), s * 1000, "v", ms=9,
                color=INK2, markerfacecolor=SURFACE, markeredgewidth=1.5)
    ax.plot([], [], "o", color=INK, label="Measured (elements 112–118)")
    ax.plot([], [], "v", color=INK2, markerfacecolor=SURFACE, label="Upper limit (119/120 searches)")
    picks = [(production.candidate_reactions(119)[1], (-150, -45)),
             (production.candidate_reactions(120)[1], (30, -60))]
    for r, offset in picks:
        x = production.effective_fissility(r.zp, r.ap, r.zt, r.at)
        lg, _ = r.sigma_pb()
        ax.plot(x, 10**lg * 1000, "D", ms=7, color=SERIES["Z=120"], markeredgecolor=SURFACE,
                markeredgewidth=1.2)
        ax.annotate(f"{r.label} → Z={r.z_cn}\npredicted {10**lg * 1000:.0f} fb", (x, 10**lg * 1000),
                    xytext=offset, textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("Mean fissility x_m of the entrance channel")
    ax.set_ylabel("Evaporation-residue cross section (fb)")
    ax.set_title("Why each new element is ~10× harder: cross section vs. fissility", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_discriminators(out: Path, rows):
    import matplotlib.pyplot as plt

    _style(plt)
    rows = rows[:8]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(rows))
    for k, h in enumerate(nuclear.PROTON_HYPOTHESES):
        m = nuclear.get_model(h)
        q = [r["Q_alpha"][h] for r in rows]
        e = [m.q_alpha_with_uncertainty(r["Z"], r["N"])[1] for r in rows]
        ax.errorbar(x + (k - 1) * 0.22, q, yerr=e, fmt="o", ms=7, color=SERIES[h],
                    elinewidth=2, capsize=0, label=f"shell at {h}",
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xticks(x, [r["nucleus"] for r in rows], rotation=0)
    ax.set_ylabel("Predicted Q-alpha (MeV)")
    ax.set_title("Measurements that would locate the next proton shell "
                 "(reachable nuclei, largest disagreement first)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper right")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_xrays(out: Path):
    import matplotlib.pyplot as plt

    _style(plt)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    zs = np.arange(26, 123, 2)
    ax.plot(zs, atomic.k_alpha1_kev(zs), color=SEQ[4], lw=2, label="Screened Dirac model")
    ax.plot([z for z, _ in data.K_ALPHA1], [e for _, e in data.K_ALPHA1], "o", ms=8,
            color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, label="Measured K-alpha1")
    new = [119, 120, 121, 122]
    ev = atomic.k_alpha1_kev(new)
    ax.plot(new, ev, "D", ms=8, color=SERIES["Z=120"], markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Predicted, new elements")
    ax.annotate("  ".join(f"Z={z}: {e:.0f} keV" for z, e in zip(new, ev)), (new[0], ev[0]),
                xytext=(-330, -4), textcoords="offset points", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Atomic number Z")
    ax.set_ylabel("K-alpha1 energy (keV)")
    ax.set_title("X-ray fingerprint of the atomic number", fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fmt_t(logt: float) -> str:
    if not math.isfinite(logt):
        return "stable"
    t = 10**logt
    for unit, s in [("Gyr", 3.156e16), ("yr", 3.156e7), ("d", 86400), ("h", 3600), ("min", 60), ("s", 1),
                    ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)]:
        if t >= s:
            return f"{t / s:.3g} {unit}"
    return f"{t:.2g} s"


def _one_sided_sigma(p: float) -> float:
    from .confirmation import gaussian_sigma
    return gaussian_sigma(min(max(p, 1e-300), 0.5))


def _peak_sf_sd() -> float:
    sds = []
    for h in nuclear.PROTON_HYPOTHESES:
        peak, _ = survey.island_of_stability(h)
        sds.append(decay.predict(peak["Z"], peak["N"], h).log_t_sigma["SF"])
    return max(sds)


def _section_out_of_sample(w):
    w("### Out-of-sample checks: fit the past, predict the future\n")
    w("Discovery history provides the test: fit on everything lighter than a cut in Z, predict "
      "everything heavier, and compare the errors with the claimed error bars. A calibrated "
      "model has forward z-scores with rms ≈ 1.\n")
    w("| Alpha half-life law | Parameters | In sample | Leave-one-out | "
      + " | ".join(f"Forward, fit Z<{c}" for c in validation.ALPHA_CUTS) + " |")
    w("|---|---|---|---|" + "---|" * len(validation.ALPHA_CUTS))
    for r in validation.alpha_laws():
        w(f"| {r['law']} | {r['parameters']} | {r['in_sample']:.2f} | {r['loo']:.2f} | "
          + " | ".join(f"{r['forward'][c][0]:.2f}" for c in validation.ALPHA_CUTS) + " |")
    vs, udl = validation.alpha_laws()
    w(f"\nErrors are in decades, on {len(data.ALPHA_DECAYS)} NUBASE2020 alpha emitters. The "
      f"{udl['parameters']}-parameter WKB law (its odd-nucleon hindrance is fitted: "
      f"+{decay.alpha_law().coef[3]:.2f} decades) describes the data as well as the "
      f"{vs['parameters']}-parameter Viola–Seaborg form. Its worst forward error is "
      f"{udl['forward_worst']:.2f} decades against {vs['forward_worst']:.2f}, so it is the alpha "
      "law used throughout.\n")
    w("| Mass model | Forward Q-alpha rms (MeV) | Forward bias (MeV) | z-rms, GP error only | "
      "z-rms, calibrated | Error growth k (MeV per unit Z–N distance) |")
    w("|---|---|---|---|---|---|")
    for r in validation.mass_model():
        w(f"| {r['hypothesis']} | {r['rms_MeV']:.2f} | {r['bias_MeV']:+.2f} | {r['z_rms_gp_only']:.1f} | "
          f"{r['z_rms_calibrated']:.2f} | {r['k']:.3f} |")
    mm = validation.mass_model()
    lo, hi = min(r["z_rms_gp_only"] for r in mm), max(r["z_rms_gp_only"] for r in mm)
    w(f"\nThe GP's own error bars are {lo:.1f}–{hi:.1f}× too small out of sample. With the "
      "distance term they are calibrated (z-rms ≈ 1).\n")
    w("| Fission law | Shell slope k, data only | k with explicit prior N(2.25, 0.75²) | "
      "Shell energies in calibration set (MeV) | Leave-one-out (decades) |")
    w("|---|---|---|---|---|")
    for r in validation.fission_laws():
        w(f"| {r['hypothesis']} | {r['k_data_only'][0]:+.2f} ± {r['k_data_only'][1]:.2f} | "
          f"{r['k_posterior'][0]:.2f} ± {r['k_posterior'][1]:.2f} | "
          f"{r['shell_energy_range'][0]:.1f} to {r['shell_energy_range'][1]:.1f} | {r['loo']:.1f} |")
    fl = validation.fission_laws()
    lo = min(r["shell_energy_range"][0] for r in fl)
    w(f"\nWith the {len(data.SF_HALF_LIVES)} evaluated NUBASE2020 fission half-lives (odd nuclei "
      "included, with a fitted odd-nucleon hindrance of "
      f"{decay.sf_law('Z=114')[0].coef[4]:.1f} ± {decay.sf_law('Z=114')[0].coef_sd()[4]:.1f} decades), "
      "the data now *measure* the shell slope, and it comes out **below** the range [1, 3.5] that "
      "v0.1 assumed: shell energy stabilises against fission more weakly than assumed. The prior "
      "still matters, because the calibration nuclei reach shell energies of only "
      f"{lo:.1f} MeV while the island lies at −7 to −11 MeV. Its consequence is visible in the "
      "predictive uncertainty:\n")
    w("| Nucleus | " + " | ".join(f"log10 T_SF, {h} (±1σ)" for h in nuclear.PROTON_HYPOTHESES) + " |")
    w("|---|" + "---|" * len(nuclear.PROTON_HYPOTHESES))
    for r in validation.sf_extrapolation():
        w(f"| {r['nucleus']} | " + " | ".join(f"{r[h][0]:.1f} ± {r[h][1]:.1f}" for h in nuclear.PROTON_HYPOTHESES) + " |")
    w("\nFission half-lives on the island are uncertain by several decades: its lifetime is "
      "an extrapolation, not a measurement.\n")


def _route_insight(w, rows, hs):
    """Explain, from the numbers, when the best route is not the biggest cross section."""
    for z in sorted({r["Z"] for r in rows}):
        rz = [r for r in rows if r["Z"] == z]
        best = max(rz, key=lambda r: r["p_discovery_avg"])
        # compare with the most productive route whose target exists in quantity
        usable = [r for r in rz if r["target_mg_cm2"] >= 0.1]
        if not usable:
            continue
        big = max(usable, key=lambda r: r["sigma_median_fb"])
        mean_anchor = lambda r: sum(r["p_anchor"][h] for h in hs) / len(hs)  # noqa: E731
        if (best is big or best["sigma_median_fb"] > big["sigma_median_fb"] / 2
                or mean_anchor(best) <= mean_anchor(big)):
            continue
        unknown = len(best["path_Z114"]) - 1
        w(f"**Element {z}: the best route is not the most productive one.** {best['reaction']} has a "
          f"median cross section of only {best['sigma_median_fb']:.0f} fb, against "
          f"{big['sigma_median_fb']:.0f} fb for {big['reaction']}, yet it gives the higher discovery "
          f"probability ({best['p_discovery_avg']:.0%} vs {big['p_discovery_avg']:.0%}). Its residue "
          f"{best['residue']} is {unknown} alpha decay{'s' if unknown > 1 else ''} from known "
          f"{best['path_Z114'][-1]}, so its chain anchors with probability "
          + "/".join(f"{best['p_anchor'][h]:.2f}" for h in hs)
          + f" under shells at {'/'.join(h[2:] for h in hs)}, against "
          + "/".join(f"{big['p_anchor'][h]:.2f}" for h in hs)
          + f" for {big['reaction']}. "
          "A short chain into known territory is worth more than raw yield when every extra new "
          "link is another chance to fission first, and it holds up whichever shell hypothesis is "
          "true."
          + (" This reaction was searched before without events, but at an exposure too small "
             "to rule this out, and the probability already includes that null result."
             if best["past_null_exposure"] else "")
          + " The residue assumes a single evaporation channel; the neighbouring channel's residue "
          "should be checked before planning beam time.\n")


def _section_calculus(w):
    setup = calculus.Setup()
    days = 300
    rows = calculus.discovery_table(campaign_days=days, setup=setup, per_z=6)
    w("## 6. How likely is a discovery? (the discovery calculus)\n")
    w("Point estimates of \"beam days\" are biased in the optimistic direction. The cross section is "
      "log-normal, so its mean exceeds its median. The half-life is uncertain, so the fraction of "
      "atoms that survive the flight is lower than at the central value. And under the IUPAC "
      "criteria a chain counts only if it is anchored to known nuclei. The calculus carries all "
      "three into one number:\n")
    w("```")
    w("P_disc(D) = P_anchor · E_{σ,T}[ P_Poisson(≥ k ; λ(σ,T)·D) ]")
    w("λ        = σ · I_beam · n_target · ε · 2^(−t_flight/T)")
    w("P_anchor = Π_unknown links Φ( (log T_SF − log T_α) / √(sd_SF² + sd_α²) )")
    w("```\n")
    w("log σ has the posterior predictive distribution of the cross-section law, conditioned on "
      "any zero-event exposure already spent on that reaction. log T is normal with the decay "
      "laws' predictive sd. The expectation is evaluated by Gauss–Hermite quadrature.\n")
    w(f"Campaign: {days} beam days, k = {setup.chains_needed} anchored chains, "
      f"{setup.beam_pua:g} pµA, {setup.efficiency:.0%} efficiency, {setup.flight_time_s * 1e6:g} µs flight.\n")
    hs = list(nuclear.PROTON_HYPOTHESES)
    w("| Z | Reaction | Residue | σ median (fb) | Already searched? | Survive flight (Z=114) | "
      "P_anchor " + "/".join(h[2:] for h in hs) + " | P_disc " + "/".join(h[2:] for h in hs)
      + " | P_disc, weighted | Days to 50 % (Z=114) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        d50 = r["days_50pc_Z114"]
        w(f"| {r['Z']} | {r['reaction']} | {r['residue']} | {r['sigma_median_fb']:.0f} | "
          f"{'yes, 0 events' if r['past_null_exposure'] else 'no'} | {r['survival']['Z=114']:.0%} | "
          + "/".join(f"{r['p_anchor'][h]:.2f}" for h in hs) + " | "
          + "/".join(f"{r['p_discovery'][h]:.2f}" for h in hs)
          + f" | **{r['p_discovery_avg']:.2f}** | {'never' if not math.isfinite(d50) else f'{d50:.0f}'} |")
    best = {}
    for r in rows:
        if r["Z"] not in best or r["p_discovery_avg"] > best[r["Z"]]["p_discovery_avg"]:
            best[r["Z"]] = r
    w("\nBest route per element, by the weighted discovery probability in one campaign: "
      + "; ".join(f"Z={z}: {r['reaction']} ({r['p_discovery_avg']:.0%})" for z, r in best.items()) + ".\n")
    _route_insight(w, rows, hs)
    w("\"Never\" means that even unlimited beam time cannot reach 50 %: with the probability left "
      "after flight losses and the chance that the chain fails to anchor, discovery stays below "
      "even odds. Reactions already searched without success are penalised exactly once, through "
      "their own exposure. Einsteinium routes are limited by the target, not the physics.\n")


def build(out_dir: str | Path = "docs") -> Path:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    L = []
    w = L.append

    w("# Elemental Discovery: where new elements may exist, and how we would know\n")
    w("_Generated by `python -m elemental_discovery report`. Every number below comes "
      "from models calibrated in this package on measured data (see `elemental_discovery/data.py`). "
      "Uncertainties are 1σ; half-life uncertainties are in decades (log10 s)._\n")

    # 1. Calibration
    w("## 1. How good is the model where we can check it?\n")
    w("| Component | Calibrated on | Scatter |")
    w("|---|---|---|")
    for h in nuclear.PROTON_HYPOTHESES:
        d = nuclear.get_model(h).describe()
        w(f"| Mass model, shell at {h} | {len(data.BINDING_ENERGIES)} binding energies + "
          f"{len(data.ALPHA_DECAYS)} Q-alpha | B: {d['rms_binding_MeV']:.2f} MeV, "
          f"Q-alpha (global): {d['rms_qalpha_MeV']:.2f} MeV |")
    al = decay.alpha_law()
    w(f"| Alpha half-life (universal decay law, {len(al.coef)} parameters) | {len(data.ALPHA_DECAYS)} alpha emitters | "
      f"{al.rms():.2f} decades (leave-one-out {al.loo_rms():.2f}) |")
    for h in nuclear.PROTON_HYPOTHESES:
        law, _ = decay.sf_law(h)
        w(f"| Fission half-life, {h} | {len(data.SF_HALF_LIVES) - 1} SF emitters | "
          f"{law.rms():.1f} decades (leave-one-out {law.loo_rms():.1f}) |")
    (c0, c1), sig = production.cross_section_fit()
    w(f"| Fusion cross section | {len(data.CROSS_SECTIONS)} measured + "
      f"{len(data.CROSS_SECTION_LIMITS)} zero-event searches (Poisson likelihood) | ×{10**sig:.1f} |")
    s1, s2, kerr = atomic.kalpha_screening()
    w(f"| K-alpha1 X-ray energy (Dirac, finite nucleus) | {len(data.K_ALPHA1)} elements | "
      f"{100 * kerr:.2f} % |")
    w("")
    w("A Gaussian-process correction trained on the Q-alpha residuals pulls predictions onto the "
      "data near measured nuclei; far from data it relaxes to the global model. A stationary GP's "
      "error stops growing at its amplitude, so a term proportional to the distance from the "
      "nearest measured nucleus is added in quadrature, with its slope calibrated on forward tests "
      "(next section). Every other law reports its regression *predictive* standard deviation, "
      "which also grows as it extrapolates.\n")
    _section_out_of_sample(w)

    # 2. Hypotheses
    w("## 2. Where is the next magic proton number?\n")
    w("Theory disagrees: macroscopic–microscopic models favour **Z = 114**, many relativistic "
      "mean-field models **Z = 120**, some Skyrme models **Z = 126**. The model is fitted once per "
      "hypothesis and each version is scored on the superheavy data. The global model (before the "
      "data-driven correction) is what controls extrapolation, so that is what gets scored.\n")
    w("| Hypothesis | Q-alpha rms, Z ≥ 104, in sample (MeV) | Forward-test rms (MeV) | "
      "Forward log-likelihood | Weight | observed decay modes reproduced |")
    w("|---|---|---|---|---|---|")
    wts = calculus.hypothesis_weights()
    fw = {r["hypothesis"]: r for r in validation.mass_model()}
    for r in survey.score_hypotheses():
        f = fw[r["hypothesis"]]
        w(f"| {r['hypothesis']} | {r['global_rms_qalpha_SHE_MeV']:.2f} | {f['rms_MeV']:.2f} | "
          f"{f['loglik']:.1f} | {wts[r['hypothesis']]:.2f} | {r['decay_modes_correct']} |")
    best_in = min(survey.score_hypotheses(), key=lambda r: r["global_rms_qalpha_SHE_MeV"])["hypothesis"]
    best_fw = max(fw.values(), key=lambda r: r["loglik"])["hypothesis"]
    w(f"\nIn sample, {best_in} fits the superheavy data best. Asked to *predict* heavier nuclei "
      f"before seeing them, {best_fw} does best"
      + (" and the ranking reverses: in-sample fit is not evidence about extrapolation." if best_in != best_fw else ".")
      + " Read this as a statement about *this* macroscopic–microscopic model: the variant with its "
      "shell at 126 extrapolates best. The hypotheses also differ in the shapes of their crude "
      "shell functions, so this is not evidence that 126 is the true magic number."
      + " The weights come from the forward log-likelihood, tempered to one effective "
      "observation per tested element (nested cuts reuse nuclei, and isotopes of one element "
      "share the model's bias). They are used to average discovery probabilities (section 6); per-hypothesis "
      "results are always shown as well.\n")

    # 3. Island
    w("## 3. The island of stability\n")
    fig_halflife_maps(out / "figures" / "halflife_maps.png")
    w("![half-life maps](figures/halflife_maps.png)\n")
    w("| Hypothesis | Longest-lived predicted nucleus (Z ≥ 110) | Half-life | ±1σ (decades) | Mode |")
    w("|---|---|---|---|---|")
    peaks = {}
    for h in nuclear.PROTON_HYPOTHESES:
        peak, _ = survey.island_of_stability(h)
        pr = decay.predict(peak["Z"], peak["N"], h)
        peaks[h] = (peak, pr.log_t_total_sigma)
        w(f"| {h} | {peak['name']} (Z={peak['Z']}, N={peak['N']}) | ~{_fmt_t(peak['log10_T_s'])} | "
          f"{pr.log_t_total_sigma:.1f} | {peak['mode']} |")
    n_known = max(a - z for z, (_, a) in data.KNOWN_ISOTOPES.items() if z >= 110)
    n_peaks = [p["N"] for p, _ in peaks.values()]
    z_peaks = [p["Z"] for p, _ in peaks.values()]
    w(f"\nEvery hypothesis puts the longest lifetimes at **N = {min(n_peaks)}–{max(n_peaks)}**, "
      f"{min(n_peaks) - n_known}–{max(n_peaks) - n_known} neutrons beyond the most neutron-rich "
      f"superheavy nucleus made so far (N = {n_known}). Fusion of stable beams with actinide targets "
      "cannot reach that far: the compound nuclei are too neutron-poor. That neutron deficit, not the "
      "element count, is the real frontier. Even with the proton shell placed at 120 or 126, the "
      f"longest lifetimes sit at Z ≈ {min(z_peaks)}–{max(z_peaks)}, because Coulomb repulsion grows "
      "faster than the shell gain. The *location* (N ≈ 184) is robust across hypotheses. The "
      "*lifetimes* are not: they carry several decades of uncertainty, because the island's shell "
      "energies lie far beyond the fission calibration data (section 1).\n")

    # 4. Natural occurrence
    w("## 4. Could any of this exist in nature?\n")
    w("| Environment | Age scale | Half-life needed for 10⁻¹⁵ survival | Longest predicted half-life | "
      "Shortfall (decades) | P(long enough) |")
    w("|---|---|---|---|---|---|")
    from .confirmation import ENVIRONMENTS
    from .linear import normal_cdf
    best_h = max(peaks, key=lambda h: peaks[h][0]["log10_T_s"])
    best, best_sd = peaks[best_h][0]["log10_T_s"], peaks[best_h][1]
    probs = []
    for name, age, method in ENVIRONMENTS:
        need = math.log10(age * math.log10(2) / 15)
        p_ok = normal_cdf((best - need) / best_sd)
        probs.append((name.split(" (")[0], need - best, p_ok))
        w(f"| {name} | {_fmt_t(math.log10(age))} | ≳ {_fmt_t(need)} | {_fmt_t(best)} | "
          f"{max(need - best, 0):.1f} | {p_ok:.2f} |")
    w(f"\nThe longest-lived candidate is {peaks[best_h][0]['name']} (shell at {best_h}), "
      f"uncertain by ±{best_sd:.1f} decades. " + " ".join(
          f"{n}: {'passes on the central estimate' if gap <= 0 else f'falls {gap:.1f} decades short'}"
          f" (P = {p:.2f})." for n, gap, p in probs)
      + " Probabilities this far from the calibration data are only as good as the shell prior. "
      "With honest error bars the model **cannot rule out** long-lived superheavy nuclei on "
      "theoretical grounds alone: the case against primordial survival is the null result of "
      "decades of terrestrial and meteoritic searches, not nuclear theory. The island's own "
      "uncertainty is the largest gap in our knowledge. Kilonova spectra (r-process in "
      "neutron-star mergers) remain the most direct natural probe, although fission recycling "
      "near A ≈ 280–300 limits how far the r-process reaches.\n")

    # 5. New elements
    w("## 5. Making elements 119–122\n")
    fig_cross_sections(out / "figures" / "cross_sections.png")
    w("![cross sections](figures/cross_sections.png)\n")
    w("Assumptions: 6 pµA beam, 0.5 mg/cm² target (254Es: 0.001 mg/cm², since only micrograms "
      "exist), 50 % separator efficiency, "
      f"{survey.SEPARATOR_FLIGHT_TIME_S * 1e6:.0f} µs flight time through the separator. "
      f"Cross sections are uncertain by a factor of ~{10**sig:.1f}.\n")
    w("Cross sections here are the fit's central values *before* conditioning each reaction on its "
      "own null search, and beam days are point estimates. Section 6 replaces both with "
      "probabilities.\n")
    w("| Z | Reaction | Residue | E* (MeV) | σ (fb) | Half-life (1σ decades) | Reach detector | Beam days for 3 chains (90 %) | Chain ends |")
    w("|---|---|---|---|---|---|---|---|---|")
    ranked = survey.rank_new_elements()
    for r in ranked:
        w(f"| {r['Z']} | {r['reaction']} | {r['residue']} | {r['E_star_MeV']:.0f} | {r['sigma_fb']:.0f} | "
          f"{_fmt_t(r['log10_T_s'])} (±{r['log10_T_sigma']:.1f}) | {100 * r['survive_flight']:.0f} % | "
          f"{r['days_for_3_chains_90pc']:.0f} | {r['chain_ending']} |")
    w("\n**A hidden obstacle:** the predicted half-lives of element 120–122 isotopes are close to "
      "the ~1 µs it takes a recoil to cross the separator. If they are at the short end of the "
      "error band, most atoms decay in flight and are never seen. That is a separate route to a "
      "null result, unrelated to the cross section. Fast digital electronics, which catch decays "
      "within microseconds of implantation, and short separators help here.")
    w("\nCheck against published non-observations:\n")
    w("Each search is left out of the fit in turn, and the refitted model is asked how probable "
      "*zero* events were at that search's exposure (exposure = ln 10 / 90 %-CL limit).\n")
    w("| Search | Predicted (median) | Published upper limit | P(0 events) | Equivalent tension |")
    w("|---|---|---|---|---|")
    lims = production.validate_against_limits()
    for r in lims:
        z = _one_sided_sigma(r["p_zero_events"])
        w(f"| {r['reaction']} | {r['predicted_pb'] * 1000:.0f} fb | {r['upper_limit_pb'] * 1000:.0f} fb | "
          f"{r['p_zero_events']:.2f} | {f'{z:.1f}σ' if r['p_zero_events'] < 0.5 else 'none'} |")
    worst = min(lims, key=lambda r: r["p_zero_events"])
    w(f"\nThe most constraining null result is {worst['reaction']}: a model that has not seen it "
      f"gives zero events a probability of {worst['p_zero_events']:.2f} "
      f"({_one_sided_sigma(worst['p_zero_events']):.1f}σ). Either Ti-induced cross sections fall "
      "faster than the fissility trend or that search was unlucky; the fit and section 6 condition "
      "on it either way.\n")
    _section_calculus(w)

    # 6. Confirmation
    w("## 7. How a discovery would be confirmed\n")
    plan = confirmation_plan(120, 176)
    w(f"Worked example: **{plan['nucleus']}** from 50Ti + 249Cf (3n), shell at {plan['hypothesis']}.\n")
    w("| Step | Mode | E_alpha (MeV) | Half-life | K-alpha1 (keV) | Known? |")
    w("|---|---|---|---|---|---|")
    for s in plan["chain"]:
        w(f"| {s['nucleus']} | {s['mode']} | {s['E_alpha_MeV']:.2f} ± {s['Q_alpha_sigma_MeV']:.2f} | "
          f"{_fmt_t(s['log10_T_half_s'])} (±{s['log10_T_sigma']:.1f}) | {s['K_alpha1_keV']:.1f} | "
          f"{'yes' if s['known'] else 'new'} |")
    sig3 = significance(3, plan["expected_random_chains"])
    ce = plan["chain_ending"]
    w(f"\n* {ce[0].upper() + ce[1:]}.")
    w(f"* Expected random look-alike chains over a 100-day run: "
      f"{plan['expected_random_chains']:.1e}. Three real chains would be a "
      "result far beyond 5σ on counting statistics alone.")
    for e in plan["evidence"][1:]:
        w(f"* {e}")
    w("\nIUPAC/IUPAP discovery criteria (Joint Working Group, 2018) that a claim must meet:\n")
    for c in plan["iupac_criteria"]:
        w(f"1. {c}")
    w("")
    fig_xrays(out / "figures" / "xrays.png")
    w("![x-rays](figures/xrays.png)\n")
    disc = survey.discriminating_measurements()
    fig_discriminators(out / "figures" / "discriminators.png", disc)
    w("### Measurements that discriminate between shell hypotheses\n")
    w("![discriminators](figures/discriminators.png)\n")
    w("| Nucleus | Q-alpha, Z=114 | Q-alpha, Z=120 | Q-alpha, Z=126 | Model σ | Spread / σ |")
    w("|---|---|---|---|---|---|")
    for r in disc[:8]:
        q = r["Q_alpha"]
        w(f"| {r['nucleus']} | {q['Z=114']:.2f} | {q['Z=120']:.2f} | {q['Z=126']:.2f} | "
          f"{r['sigma']:.2f} | {r['power']:.1f} |")
    top = disc[0]
    w(f"\nThe most discriminating reachable nucleus is {top['nucleus']}: the hypotheses differ by "
      f"{top['spread_MeV']:.1f} MeV in Q-alpha, {top['power']:.1f}× the calibrated model uncertainty. "
      "With the error bars calibrated by forward tests (they grow with distance from data), one "
      "measured chain narrows the field but no single chain settles it.\n")

    # 7. Limit of the periodic table
    w("## 8. Where does the periodic table end?\n")
    zc = atomic.critical_charge()
    reach = []
    for h in nuclear.PROTON_HYPOTHESES:
        zs, ns, logt, _ = survey.halflife_map(h)
        sub = [(logt[i, j], z, n) for i, z in enumerate(zs) if z >= 125
               for j, n in enumerate(ns) if n <= 185 and np.isfinite(logt[i, j])]
        lt, z, n = max(sub)
        reach.append(f"{h}: {z + n}{data.symbol(z)}, {_fmt_t(lt)}")
    w("* **Nuclear limit.** For Z ≥ 125, the longest-lived isotopes with N ≤ 185 (the most "
      "fusion could conceivably reach) are " + "; ".join(reach) + ". Longer lifetimes appear "
      "only far out on the neutron-rich side (N > 200), which no known production route can reach. "
      "There the fission barrier is pure shell effect.")
    w(f"* **Atomic limit.** Solving the Dirac equation for a finite nucleus puts the 1s level "
      f"below −mc² at **Z_cr ≈ {zc['Z_critical']:.0f}** (bare ion; screening and nuclear-radius "
      "choices move this within ~170–173). Beyond it the vacuum itself would decay by emitting "
      f"positrons. The point-nucleus Dirac formula fails earlier, at Z = {zc['point_nucleus_limit']:.0f}.")
    w(f"* **Chemical limit.** Inner electrons move at v/c ≈ Zα (0.87 for Z = 119), so relativity "
      "reorders the orbitals and the Madelung rule breaks down from Z ≈ 121.\n")
    w("| Z | Symbol | Madelung configuration (tail) | Relativistic prediction | Note |")
    w("|---|---|---|---|---|")
    for z in (119, 120, 121, 122):
        e = atomic.element_summary(z)
        tail = " ".join(e["madelung_configuration"].split()[-3:])
        rp = e.get("relativistic_prediction", {})
        w(f"| {z} | {e['symbol']} | … {tail} | {rp.get('config', '')} | {rp.get('note', '')} |")
    w("")

    w("## 9. Limitations\n")
    w("* The mass model is spherical Myers–Swiatecki shell structure plus a deformation term. It "
      "does not resolve the deformed sub-shells at N = 152/162, which the Gaussian-process "
      "correction absorbs only near data.")
    w("* Fission half-lives are uncertain by about 2.5 decades even where calibrated (leave-one-out), "
      "and by far more on the island, where the shell slope rests on an explicit prior that the data "
      "cannot test. Weak-decay (EC/β) half-lives are order-of-magnitude estimates.")
    w("* The discovery calculus neglects correlations between the residue's half-life and the "
      "anchoring probability (both depend on the same Q-alpha), and picks one evaporation channel "
      "per reaction.")
    w("* Cross sections use one-parameter fissility systematics. Real reaction dynamics "
      "(quasi-fission, deformation of the target, beam energy choice) can shift them by "
      "factors of several.")
    w("* Reference data were compiled by hand and rounded. Check them against AME2020/NUBASE2020 "
      "before quoting any number.")
    report = out / "REPORT.md"
    report.write_text("\n".join(L) + "\n")
    return report
