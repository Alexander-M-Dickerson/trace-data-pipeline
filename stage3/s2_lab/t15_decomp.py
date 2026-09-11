r"""t15_decomp.py -- Tables IA.XV and IA.XVI: look-ahead bias decomposed by leg.

Table 4 reports the long-short result. These split it: 9 columns -- long, short and
long-short, each as winsorized / baseline / Bias -- of ONE statistic, mean returns
(IA.XV) or CAPM_B alphas (IA.XVI). 15 factors x 9 columns x 2 rows = 270 cells each.

The split is what identifies where the look-ahead acts. A filter that clips extreme
returns removes them from whichever leg holds the extreme bonds, so a bias that lives
almost entirely in one leg is a different claim from one spread across both.

    python s2_lab/t15_decomp.py --which mean
    python s2_lab/t15_decomp.py --which alpha
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_exhibit import run_lab_exhibit     # noqa: E402

CONFIG = {
    "mean": dict(exhibit="Table IA.XV", label="tab:lab_full_mean_8",
                 stem="table15", stat="mu"),
    "alpha": dict(exhibit="Table IA.XVI", label="tab:lab_full_alpha_9",
                  stem="table16", stat="alpha"),
}


def columns_for(stat: str) -> list[tuple]:
    return [(leg, var, stat) for leg in ("long", "short", "ls")
            for var in ("wins", "base", "bias")]


def head_for(stat: str) -> list[tuple[str, list[str]]]:
    sym = r"\mu" if stat == "mu" else r"\alpha"
    cols = [rf"$\tilde{{{sym}}}$", rf"${sym}$", "Bias"]
    return [(r"Long (\%)", cols), (r"Short (\%)", cols), (r"Long-Short (\%)", cols)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", required=True, choices=("mean", "alpha"))
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    cfg = CONFIG[args.which]
    return run_lab_exhibit(
        exhibit=cfg["exhibit"], label=cfg["label"],
        columns=columns_for(cfg["stat"]), stem=cfg["stem"],
        head_groups=head_for(cfg["stat"]), no_bench=args.no_bench)


if __name__ == "__main__":
    raise SystemExit(main())
