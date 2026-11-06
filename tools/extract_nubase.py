"""Regenerate elemental_discovery/nubase2020_subset.txt from the full NUBASE2020 file.

    python tools/extract_nubase.py path/to/nubase_4.mas20.txt

The full file is distributed by the Atomic Mass Data Center
(https://www-nds.iaea.org/amdc/) and cited as F.G. Kondev, M. Wang, W.J. Huang,
S. Naimi and G. Audi, Chin. Phys. C 45, 030001 (2021).  The subset keeps the
header, every ground state with Z >= 80, and the light nuclei used to
calibrate the mass model, unmodified.
"""

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "elemental_discovery" / "nubase2020_subset.txt"
LIGHT = {(0, 1), (1, 1), (2, 4), (20, 40), (20, 48), (22, 50), (23, 51), (24, 54), (26, 56),
         (26, 58), (28, 58), (28, 62), (28, 64), (38, 88), (40, 90), (50, 116), (50, 120),
         (50, 132), (56, 138), (62, 144), (79, 197)}


def main(src):
    keep = []
    for line in Path(src).read_text().splitlines():
        if line.startswith("#"):
            keep.append(line)
            continue
        a, z, i = int(line[0:3]), int(line[4:7]), int(line[7])
        if i == 0 and (z >= 80 or (z, a) in LIGHT):
            keep.append(line)
    OUT.write_text("\n".join(keep) + "\n")
    print(f"wrote {OUT} ({len(keep)} lines)")


if __name__ == "__main__":
    main(sys.argv[1])
