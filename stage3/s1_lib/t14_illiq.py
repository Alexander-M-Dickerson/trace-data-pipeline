r"""t14_illiq.py -- Table IA.XIV: latent implementation bias in ILLIQUIDITY factors.

Table 1's shape over the five illiquidity signals instead of the seven price-based
ones. These are the factors where the bias should be largest if it is a microstructure
artifact, because the signal is itself built from the same prices as the return.

    python s1_lib/t14_illiq.py
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

LABEL = "tab:illiq_1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default=None,
                    help="override the sample end; T and the NW lag count then come "
                         "from the data rather than the paper's window")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    return run_two_row_exhibit(
        exhibit='Table IA.XIV', label=LABEL, caption=captions.caption(LABEL),
        factors=D.ILLIQ_FACTORS, stem='table14', panel_specs=None,
        subtitles=None, end=args.end, no_bench=args.no_bench)


if __name__ == "__main__":
    raise SystemExit(main())
