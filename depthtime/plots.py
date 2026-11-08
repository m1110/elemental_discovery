"""Figures for the depth-of-thought time-perception model (matplotlib)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .model import (ARCHETYPES, DEPTH_ANCHORS, NEUTRAL, depth_sweep)  # noqa: E402
from .simulate import simulate_session  # noqa: E402

C_PRO, C_RET, C_TL, C_PASS = "#2a6fdb", "#d9822b", "#7a4fbf", "#2f9e6e"


def fig_depth_curve(path: str) -> None:
    sweep = depth_sweep(201, NEUTRAL)
    d = np.array([x for x, _ in sweep])
    pro = np.array([r.prospective for _, r in sweep])
    ret = np.array([r.retrospective for _, r in sweep])
    tl = np.array([r.timelessness for _, r in sweep])
    pas = np.array([r.passage for _, r in sweep])

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhline(1.0, color="#999", lw=0.8, ls=":")
    ax.plot(d, pro, color=C_PRO, lw=2.4, label="prospective felt duration (in the moment)")
    ax.plot(d, ret, color=C_RET, lw=2.0, label="retrospective (remembered) duration")
    ax.plot(d, pas * (1 - tl), color=C_PASS, lw=1.6, ls="--",
            label="felt passage speed (>1 = flies), faded out as time stops being represented")
    ax.fill_between(d, 0, tl, color=C_TL, alpha=0.18, label="timelessness (1 - self-model)")
    for x, *_, lab in DEPTH_ANCHORS:
        ax.axvline(x, color="#ccc", lw=0.8)
        ax.text(x, 2.02, lab, rotation=0, ha="center", va="bottom", fontsize=8,
                wrap=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 2.25)
    ax.set_xlabel("depth: deep sequential thought  ->  no-thought")
    ax.set_ylabel("ratio to ordinary waking mind (1 = ordinary)")
    ax.set_title("Depth-of-thought time-perception model: steady-state readouts", pad=28)
    ax.legend(loc="center left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 0.77))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_practice_matrix(rows: list[dict], path: str) -> None:
    """Prospective vs retrospective ratio, one point per practice x person."""
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"novice": "o", "intermediate": "s", "expert": "^", "anxious novice": "x"}
    for r in rows:
        x = r["felt_prospective_min"] / r["clock_min"]
        y = r["felt_retrospective_min"] / r["clock_min"]
        ax.scatter(x, y, s=40 + 160 * r["mean_timelessness"],
                   marker=markers.get(r["individual"], "o"),
                   color=plt.cm.viridis(r["mean_timelessness"] / 0.7), alpha=0.85)
        if r["individual"] in ("novice", "expert"):
            ax.annotate(f"{r['practice']} ({r['individual'][0]})", (x, y),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(1, color="#999", lw=0.8, ls=":")
    ax.axvline(1, color="#999", lw=0.8, ls=":")
    ax.text(0.42, 1.85, "flew by, remembered long\n(holiday paradox)", fontsize=8, color="#555")
    ax.text(1.2, 0.48, "dragged, remembered short\n(waiting)", fontsize=8, color="#555")
    ax.text(0.42, 0.35, "time vanished\n(absorption / no-thought)", fontsize=8, color="#555")
    ax.text(1.2, 1.85, "dense, slow & rich", fontsize=8, color="#555")
    ax.set_xlabel("prospective felt / clock duration")
    ax.set_ylabel("retrospective remembered / clock duration")
    ax.set_title("Where each practice puts time (o novice, ^ expert, size/colour = timelessness)")
    ax.set_xlim(0.3, 1.6)
    ax.set_ylim(0.2, 2.1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_session_traces(path: str, practices=("open_monitoring", "nondual", "bjj", "sequential"),
                       seed: int = 3) -> None:
    fig, axes = plt.subplots(len(practices), 1, figsize=(10, 2.3 * len(practices)), sharex=True)
    for ax, key in zip(axes, practices):
        for who, ls in (("novice", "-"), ("expert", "-")):
            res = simulate_session(key, ARCHETYPES[who], seed=seed, keep_trace=True)
            t = np.array(res.trace["t"])
            alpha = 0.55 if who == "novice" else 1.0
            ax.plot(t, res.trace["prospective"], color=C_PRO, lw=1, alpha=alpha, ls=ls,
                    label=f"prospective ({who})")
            ax.plot(t, res.trace["timelessness"], color=C_TL, lw=1, alpha=alpha,
                    ls="--" if who == "novice" else "-", label=f"timelessness ({who})")
        ax.axhline(1, color="#aaa", lw=0.7, ls=":")
        ax.set_ylabel(key, fontsize=9)
        ax.set_ylim(0, 2.0)
    axes[0].legend(fontsize=7, ncol=4, frameon=False, loc="upper right")
    axes[-1].set_xlabel("minutes into session")
    fig.suptitle("Moment-to-moment simulated time sense (one session each)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(rows: list[dict], outdir: str = "figures") -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, n) for n in
             ("depth_curve.png", "practice_matrix.png", "session_traces.png")]
    fig_depth_curve(paths[0])
    fig_practice_matrix(rows, paths[1])
    fig_session_traces(paths[2])
    return paths
