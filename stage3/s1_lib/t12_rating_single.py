r"""t12_rating_single.py -- Table IA.XII: latent implementation bias, single sorts by rating.

Three panels of Table 1's shape -- all bonds (deciles), investment grade and
non-investment grade (quintiles) -- 3 x 7 factors x 10 columns x (coefficient + t) =
420 cells.

Panel A reproduces Table 1's Panel A by construction. It is computed independently
here rather than copied, so the two agreeing is a check rather than a tautology.

    python s1_lib/run_sorts.py --rating IG --sorts single
    python s1_lib/run_sorts.py --rating NIG --sorts single
    python s1_lib/t12_rating_single.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import captions                                  # noqa: E402
import drrlib as D                               # noqa: E402
import lib_engine as E                           # noqa: E402
from two_row import FACTORS, run_two_row_exhibit  # noqa: E402

LABEL = "tab:mmn_app_1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None,
                    help="override the sample end; T and the NW lag count then come "
                         "from the data rather than the paper's window")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    return run_two_row_exhibit(
        exhibit='Table IA.XII', label=LABEL, caption=captions.caption(LABEL),
        factors=FACTORS, stem='table12', panel_specs={"Panel A": E.LibSpec(sort="single", rating="all"),
                     "Panel B": E.LibSpec(sort="single", rating="ig"),
                     "Panel C": E.LibSpec(sort="single", rating="nig")},
        subtitles={"Panel A": "All Bonds", "Panel B": "Investment Grade",
                     "Panel C": "Non-Investment Grade"}, end=args.end, no_bench=args.no_bench)


if __name__ == "__main__":
    raise SystemExit(main())
