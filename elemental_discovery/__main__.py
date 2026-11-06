"""Command-line interface.

    python -m elemental_discovery report [--out docs]
    python -m elemental_discovery nucleus Z A [--hypothesis Z=120]
    python -m elemental_discovery element Z
    python -m elemental_discovery reactions Z
"""

from __future__ import annotations

import argparse
import json

from . import atomic, data, nuclear
from .confirmation import confirmation_plan
from .production import beam_days, candidate_reactions


def main(argv=None):
    ap = argparse.ArgumentParser(prog="elemental_discovery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="build docs/REPORT.md and figures")
    r.add_argument("--out", default="docs")
    n = sub.add_parser("nucleus", help="decay chain and confirmation plan for one nucleus")
    n.add_argument("z", type=int)
    n.add_argument("a", type=int)
    n.add_argument("--hypothesis", default="Z=114", choices=list(nuclear.PROTON_HYPOTHESES))
    e = sub.add_parser("element", help="atomic structure summary for element Z")
    e.add_argument("z", type=int)
    x = sub.add_parser("reactions", help="fusion reactions that could make element Z")
    x.add_argument("z", type=int)
    x.add_argument("--beam-pua", type=float, default=6.0)
    args = ap.parse_args(argv)

    if args.cmd == "report":
        from .report import build
        print(f"wrote {build(args.out)}")
    elif args.cmd == "nucleus":
        plan = confirmation_plan(args.z, args.a - args.z, args.hypothesis)
        print(json.dumps({k: v for k, v in plan.items() if k != "iupac_criteria"},
                         indent=2, default=float))
    elif args.cmd == "element":
        info = atomic.element_summary(args.z)
        info["K_alpha1_keV"] = float(atomic.k_alpha1_kev(args.z)[0])
        print(json.dumps(info, indent=2, default=float))
    elif args.cmd == "reactions":
        for rx in candidate_reactions(args.z):
            lg, sig = rx.sigma_pb()
            epd = rx.events_per_day(args.beam_pua)
            zr, nr = rx.residue()
            print(f"{rx.label:22s} -> {zr + nr}{data.symbol(zr)} ({rx.best_xn()}n)  "
                  f"E*={rx.e_star_at_barrier():5.1f} MeV  sigma={10**lg * 1000:7.1f} fb (x{10**sig:.1f})  "
                  f"days for 3 chains: {beam_days(epd, 3, 0.9):6.0f}")


if __name__ == "__main__":
    main()
