"""Generate the full Markdown report and figures."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from . import atomic, data, decay, nuclear, production, survey
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
    (c0, c1), sig = production.cross_section_fit()
    xs = np.linspace(0.76, 0.92, 100)
    ys = c0 + c1 * xs
    ax.fill_between(xs, 10 ** (ys - sig) * 1000, 10 ** (ys + sig) * 1000, color=SEQ[0], lw=0)
    ax.plot(xs, 10**ys * 1000, color=SEQ[4], lw=2, label="Censored-likelihood fit (±1σ band)")
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
    _, rms_a = decay.viola_seaborg_fit()
    w(f"| Alpha half-life (Viola–Seaborg, refitted) | {len(data.ALPHA_DECAYS)} alpha emitters | "
      f"{rms_a:.2f} decades |")
    for h in nuclear.PROTON_HYPOTHESES:
        w(f"| Fission half-life, {h} | {len(data.SF_HALF_LIVES) - 1} SF emitters | "
          f"{decay.sf_fit(h)[1]:.1f} decades |")
    (c0, c1), sig = production.cross_section_fit()
    w(f"| Fusion cross section | {len(data.CROSS_SECTIONS)} measured + "
      f"{len(data.CROSS_SECTION_LIMITS)} upper limits (censored fit) | ×{10**sig:.1f} |")
    s1, s2, kerr = atomic.kalpha_screening()
    w(f"| K-alpha1 X-ray energy (Dirac, finite nucleus) | {len(data.K_ALPHA1)} elements | "
      f"{100 * kerr:.2f} % |")
    w("")
    w("A Gaussian-process correction trained on the Q-alpha residuals pulls predictions onto the "
      "data near measured nuclei; far from data it relaxes to the global model and the uncertainty "
      "grows. That growth is the point: it shows where we are extrapolating.\n")

    # 2. Hypotheses
    w("## 2. Where is the next magic proton number?\n")
    w("Theory disagrees: macroscopic–microscopic models favour **Z = 114**, many relativistic "
      "mean-field models **Z = 120**, some Skyrme models **Z = 126**. The model is fitted once per "
      "hypothesis and each version is scored on the superheavy data. The global model (before the "
      "data-driven correction) is what controls extrapolation, so that is what gets scored.\n")
    w("| Hypothesis | Q-alpha rms, Z ≥ 104 (MeV) | mean bias (MeV) | observed decay modes reproduced |")
    w("|---|---|---|---|")
    for r in survey.score_hypotheses():
        w(f"| {r['hypothesis']} | {r['global_rms_qalpha_SHE_MeV']:.2f} | "
          f"{r['global_mean_bias_qalpha_SHE_MeV']:+.2f} | {r['decay_modes_correct']} |")
    w("\nNo hypothesis wins on every test, so all three are carried forward.\n")

    # 3. Island
    w("## 3. The island of stability\n")
    fig_halflife_maps(out / "figures" / "halflife_maps.png")
    w("![half-life maps](figures/halflife_maps.png)\n")
    w("| Hypothesis | Longest-lived predicted nucleus (Z ≥ 110) | Half-life | Mode |")
    w("|---|---|---|---|")
    for h in nuclear.PROTON_HYPOTHESES:
        peak, _ = survey.island_of_stability(h)
        w(f"| {h} | {peak['name']} (Z={peak['Z']}, N={peak['N']}) | ~{_fmt_t(peak['log10_T_s'])} | {peak['mode']} |")
    n_known = max(a - z for z, (_, a) in data.KNOWN_ISOTOPES.items() if z >= 110)
    n_peaks = [survey.island_of_stability(h)[0]["N"] for h in nuclear.PROTON_HYPOTHESES]
    w(f"\nEvery hypothesis puts the longest lifetimes at **N = {min(n_peaks)}–{max(n_peaks)}**, "
      f"{min(n_peaks) - n_known}–{max(n_peaks) - n_known} neutrons beyond the most neutron-rich "
      f"superheavy nucleus made so far (N = {n_known}). Fusion of stable beams with actinide targets "
      "cannot reach that far: the compound nuclei are too neutron-poor. That neutron deficit, not the "
      "element count, is the real frontier. The island's centre in Z is also uncertain: even with "
      "the shell placed at 120 or 126, the longest lifetimes stay near Z ≈ 113–115, because "
      "Coulomb repulsion grows faster than the shell gain.\n")

    # 4. Natural occurrence
    w("## 4. Could any of this exist in nature?\n")
    w("| Environment | Age scale | Half-life needed for 10⁻¹⁵ survival | Longest predicted half-life |")
    w("|---|---|---|---|")
    from .confirmation import ENVIRONMENTS
    best = max(survey.island_of_stability(h)[0]["log10_T_s"] for h in nuclear.PROTON_HYPOTHESES)
    for name, age, method in ENVIRONMENTS:
        need = age * math.log10(2) / 15
        w(f"| {name} | {_fmt_t(math.log10(age))} | ≳ {_fmt_t(math.log10(need))} | {_fmt_t(best)} |")
    feas = [name.split(" (")[0] for name, age, _ in ENVIRONMENTS
            if best > math.log10(age * math.log10(2) / 15)]
    w(f"\nOnly **{', '.join(f.lower() for f in feas) if feas else 'none of these'}** passes on the model's best "
      "estimate. Primordial survival on Earth or in meteorites would need a half-life about "
      f"{math.log10(4.567e9 * data.SECONDS_PER_YEAR * math.log10(2) / 15) - best:.0f} decades longer "
      "than predicted, consistent with past claims of natural superheavy elements failing to hold up. "
      "Cosmic rays fall short by about "
      f"{math.log10(15e6 * data.SECONDS_PER_YEAR * math.log10(2) / 15) - best:.0f} decades, which is "
      "inside the ~2-decade fission uncertainty, so ultra-heavy cosmic-ray searches are a long shot "
      "rather than hopeless. Kilonova spectra (r-process in neutron-star mergers) are the most "
      "realistic natural channel, although fission recycling near A ≈ 280–300 limits how far the "
      "r-process reaches.\n")

    # 5. New elements
    w("## 5. Making elements 119–122\n")
    fig_cross_sections(out / "figures" / "cross_sections.png")
    w("![cross sections](figures/cross_sections.png)\n")
    w("Assumptions: 6 pµA beam, 0.5 mg/cm² target, 50 % separator efficiency, "
      f"{survey.SEPARATOR_FLIGHT_TIME_S * 1e6:.0f} µs flight time through the separator. "
      f"Cross sections are uncertain by a factor of ~{10**sig:.1f}.\n")
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
    w("| Search | Predicted | Published upper limit | Within 1σ? |")
    w("|---|---|---|---|")
    for r in production.validate_against_limits():
        w(f"| {r['reaction']} | {r['predicted_pb'] * 1000:.0f} fb | {r['upper_limit_pb'] * 1000:.0f} fb | "
          f"{'yes' if r['within_1_sigma'] else '**no** (tension)'} |")
    w("\nThe 50Ti + 249Bk limit sits about 1.4σ below the prediction. Either the cross section "
      "falls faster than the fissility trend for Ti-induced reactions, or that search was unlucky. "
      "This is the most informative constraint on element 119.\n")

    # 6. Confirmation
    w("## 6. How a discovery would be confirmed\n")
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
    w("\nIn element-122 chains the three hypotheses differ by about 3 MeV in Q-alpha, roughly 5× "
      "the model uncertainty, so a single measured chain would pick a winner. Element-120 chains "
      "separate Z=114 from the other two (about 1 MeV) but barely distinguish 120 from 126.\n")

    # 7. Limit of the periodic table
    w("## 7. Where does the periodic table end?\n")
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

    w("## 8. Limitations\n")
    w("* The mass model is spherical Myers–Swiatecki shell structure plus a deformation term. It "
      "does not resolve the deformed sub-shells at N = 152/162, which the Gaussian-process "
      "correction absorbs only near data.")
    w("* Fission half-lives are uncertain by about 2 decades even where calibrated. Weak-decay "
      "(EC/β) half-lives are order-of-magnitude estimates.")
    w("* Cross sections use one-parameter fissility systematics. Real reaction dynamics "
      "(quasi-fission, deformation of the target, beam energy choice) can shift them by "
      "factors of several.")
    w("* Reference data were compiled by hand and rounded. Check them against AME2020/NUBASE2020 "
      "before quoting any number.")
    report = out / "REPORT.md"
    report.write_text("\n".join(L) + "\n")
    return report
