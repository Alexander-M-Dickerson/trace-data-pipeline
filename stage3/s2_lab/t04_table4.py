r"""t04_table4.py -- Table 4, "Look-ahead bias by factor".

The Section-4 anchor. 15 factors -- 12 whose result improves when the LEFT tail of the
return distribution is clipped, 3 the right -- x 6 printed columns, coefficient and
t-statistic each: 180 cells.

Each factor is compared against ITSELF under one ex-post filter: returns winsorized at
the 99.5th percentile of the full sample. The threshold is unknowable at formation, so
the Bias column is the part of the published result that comes from having seen the
data first.

    python s2_lab/run_lab.py          # produce the series
    python s2_lab/t04_table4.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_exhibit import run_lab_exhibit     # noqa: E402

LABEL = "tab:lab_ls_1"
# (leg, variant, stat) per printed column, left to right
COLUMNS = [("ls", "wins", "mu"), ("ls", "base", "mu"), ("ls", "bias", "mu"),
           ("ls", "wins", "alpha"), ("ls", "base", "alpha"), ("ls", "bias", "alpha")]
HEAD = [(r"Mean Returns (\%)", [r"$\tilde{\mu}$", r"$\mu$", "Bias"]),
        (r"CAPMB Alpha (\%)", [r"$\tilde{\alpha}$", r"$\alpha$", "Bias"])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    return run_lab_exhibit(exhibit="Table 4", label=LABEL, columns=COLUMNS,
                           stem="table04", head_groups=HEAD,
                           no_bench=args.no_bench)


if __name__ == "__main__":
    raise SystemExit(main())
