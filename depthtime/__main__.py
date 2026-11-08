"""Command-line entry point.

    python -m depthtime report  [--runs 40] [--out results]   # tables + figures
    python -m depthtime sweep                                  # depth curve table
    python -m depthtime session PRACTICE [--who novice]        # one practice
    python -m depthtime fit OBS.csv                            # n-of-1 calibration
"""

from __future__ import annotations

import argparse
import csv
import os

from .model import ARCHETYPES, depth_label, depth_sweep
from .practices import PRACTICES
from .simulate import monte_carlo


def _fmt_row(m: dict) -> str:
    return (f"| {PRACTICES[m['practice']].name} | {m['individual']} | {m['clock_min']:.0f} "
            f"| {m['felt_prospective_min']:.1f} | {m['felt_retrospective_min']:.1f} "
            f"| {m['mean_passage']:.2f} | {m['mean_timelessness']:.2f} "
            f"| {m['peak_timelessness']:.2f} | {m['time_wandering_frac']:.0%} |")


def cmd_report(args) -> None:
    from .plots import make_all

    os.makedirs(args.out, exist_ok=True)
    people = ["novice", "intermediate", "expert", "anxious novice"]
    rows = [monte_carlo(k, ARCHETYPES[w], n=args.runs) for k in PRACTICES for w in people]
    with open(os.path.join(args.out, "practice_results.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    lines = ["| practice | person | clock min | felt-in-moment min | remembered min "
             "| passage (>1 flies) | mean timelessness | peak timelessness | mind-wandering |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [_fmt_row(m) for m in rows]
    with open(os.path.join(args.out, "practice_results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    sweep_lines = ["| depth | state | prospective | passage | retrospective | timelessness |",
                   "|---|---|---|---|---|---|"]
    for d, r in depth_sweep(11):
        sweep_lines.append(f"| {d:.1f} | {depth_label(d)} | {r.prospective:.2f} | "
                           f"{r.passage:.2f} | {r.retrospective:.2f} | {r.timelessness:.2f} |")
    with open(os.path.join(args.out, "depth_sweep.md"), "w") as fh:
        fh.write("\n".join(sweep_lines) + "\n")
    figs = make_all(rows, args.figures)
    print("\n".join(sweep_lines))
    print()
    print("\n".join(lines))
    print("\nwrote", args.out, "and", ", ".join(figs))


def cmd_sweep(args) -> None:
    for d, r in depth_sweep(args.n):
        print(f"{d:5.2f}  {depth_label(d):32s} prospective={r.prospective:5.2f} "
              f"passage={r.passage:5.2f} retrospective={r.retrospective:5.2f} "
              f"timelessness={r.timelessness:4.2f}")


def cmd_session(args) -> None:
    m = monte_carlo(args.practice, ARCHETYPES[args.who], n=args.runs)
    for k, v in m.items():
        print(f"{k:28s} {v}")


def cmd_fit(args) -> None:
    from .calibrate import fit_individual, load_observations

    obs = load_observations(args.csv)
    ind, l = fit_individual(obs, n_random=args.random, n_refine=args.refine)
    print(f"fit loss {l:.4f} on {len(obs)} sessions")
    for k in ("experience", "interoception", "mind_wandering", "arousal_reactivity",
              "clock_rate"):
        print(f"  {k:20s} {getattr(ind, k):.3f}")
    print(f"  -> reachable depth {ind.max_depth:.2f} ({depth_label(ind.max_depth)})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="depthtime", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--runs", type=int, default=40)
    r.add_argument("--out", default="results")
    r.add_argument("--figures", default="figures")
    r.set_defaults(func=cmd_report)
    s = sub.add_parser("sweep")
    s.add_argument("--n", type=int, default=21)
    s.set_defaults(func=cmd_sweep)
    se = sub.add_parser("session")
    se.add_argument("practice", choices=sorted(PRACTICES))
    se.add_argument("--who", default="novice", choices=sorted(ARCHETYPES))
    se.add_argument("--runs", type=int, default=50)
    se.set_defaults(func=cmd_session)
    f = sub.add_parser("fit")
    f.add_argument("csv")
    f.add_argument("--random", type=int, default=40)
    f.add_argument("--refine", type=int, default=40)
    f.set_defaults(func=cmd_fit)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
