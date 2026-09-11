r"""t13_rating_wf.py -- Table IA.XIII: latent implementation bias, within-firm by rating.

The within-firm twin of Table IA.XII: three rating panels, each a high/low split
INSIDE the issuing firm, so a firm effect cannot drive the result. 420 cells.

    python s1_lib/run_sorts.py --rating IG --sorts wf
    python s1_lib/run_sorts.py --rating NIG --sorts wf
    python s1_lib/t13_rating_wf.py
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

LABEL = "tab:mmn_app_2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None,
                    help="override the sample end; T and the NW lag count then come "
                         "from the data rather than the paper's window")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    return run_two_row_exhibit(
        exhibit='Table IA.XIII', label=LABEL, caption=captions.caption(LABEL),
        factors=FACTORS, stem='table13', panel_specs={"Panel A": E.LibSpec(sort="wf", rating="all"),
                     "Panel B": E.LibSpec(sort="wf", rating="ig"),
                     "Panel C": E.LibSpec(sort="wf", rating="nig")},
        subtitles={"Panel A": "All Bonds", "Panel B": "Investment Grade",
                     "Panel C": "Non-Investment Grade"}, end=args.end, no_bench=args.no_bench)


if __name__ == "__main__":
    raise SystemExit(main())
