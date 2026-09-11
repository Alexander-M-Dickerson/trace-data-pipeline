r"""t01_table1.py -- Table 1, "Latent implementation bias in price-based factors".

Two panels x 7 factors x 10 columns, each printed position carrying a coefficient and
a t-statistic: 280 cells.

The three approaches, side by side, are the whole point of the table. Approach 1 forms
the portfolio on a noisy month-end signal and measures the month-end return; approach 2
gaps the signal back at least one business day; approach 3 keeps the noisy signal but
measures the return from the month's beginning. The two bias columns are the difference
between approach 1 and each of the others, tested on the difference series.

    python s1_lib/t01_table1.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import captions                                  # noqa: E402
import lib_engine as E                           # noqa: E402
from two_row import FACTORS, run_two_row_exhibit  # noqa: E402

LABEL = "tab:mmn_1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None,
                    help="override the sample end; T and the NW lag count then come "
                         "from the data rather than the paper's window")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    return run_two_row_exhibit(
        exhibit='Table 1', label=LABEL, caption=captions.caption(LABEL),
        factors=FACTORS, stem='table01', panel_specs=None,
        subtitles=None, end=args.end, no_bench=args.no_bench)


if __name__ == "__main__":
    raise SystemExit(main())
