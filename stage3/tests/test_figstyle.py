"""test_figstyle.py -- the two figure looks (figstyle.py), without any data on disk.

  * no figure driver types a colour of its own: every colour comes from a style's roles, so
    a look is changed in one place;
  * Figure 4's decomposition says "Adjusted Return" (Referee 1, round 2, point B);
  * the NBER recessions every time-series figure shades are the paper's (peak to trough);
  * each driver draws the same data in both looks, to the paths the runner expects.

    python -m pytest tests/test_figstyle.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STAGE3 = Path(__file__).resolve().parents[1]
for p in (STAGE3, STAGE3 / "s1_lib", STAGE3 / "s2_lab", STAGE3 / "s3_nse"):
    sys.path.insert(0, str(p))

import figstyle  # noqa: E402

DRIVERS = ["s1_lib/f_lib_figures.py", "s2_lab/f_lab_figures.py", "s2_lab/f06_dua.py",
           "s3_nse/f_nse_figures.py"]


@pytest.mark.parametrize("rel", DRIVERS)
def test_no_driver_types_a_colour(rel):
    """A hex colour or a named colour in a driver is a look that --style cannot change."""
    src = (STAGE3 / rel).read_text(encoding="utf-8")
    hits = [f"{rel}:{i}: {line.strip()[:80]}"
            for i, line in enumerate(src.splitlines(), 1)
            if not line.lstrip().startswith("#")
            and (re.search(r"['\"]#[0-9a-fA-F]{6}['\"]", line)
                 or re.search(r"colou?r=['\"](gray|grey|red|black|blue|orange)['\"]", line))]
    assert not hits, "colours typed in a driver:\n" + "\n".join(hits)


def test_figure4_says_adjusted_return():
    src = (STAGE3 / "s1_lib" / "f_lib_figures.py").read_text(encoding="utf-8")
    assert '"Adjusted Return"' in src and "Actual Return" not in src


def test_recessions_are_the_papers():
    assert [(a[:7], b[:7]) for a, b, _ in figstyle.NBER] == [("2007-12", "2009-06"),
                                                              ("2020-02", "2020-04")]


def test_styles_have_the_same_roles():
    """A role the paper style knows and the house style does not would fail only at draw time."""
    assert set(figstyle.PAPER.colours) == set(figstyle.HOUSE.colours)
    assert len(figstyle.PAPER.clusters) == len(figstyle.HOUSE.clusters) == 9
    assert len(figstyle.PAPER.ramp) == len(figstyle.HOUSE.ramp) == 9


def test_house_colours_are_bond_schedules():
    c = figstyle.HOUSE.colours
    assert {c["unadjusted"], c["adj_signal"], c["adj_return"]} == {"#990F3D", "#0F5499", "#262A33"}


def test_out_paths(tmp_path):
    assert figstyle.out_path(tmp_path, "f.pdf", figstyle.PAPER) == tmp_path / "f.pdf"
    assert figstyle.out_path(tmp_path, "f.pdf", figstyle.HOUSE) == tmp_path / "house" / "f.pdf"
    assert [s.name for s in figstyle.styles("both")] == ["paper", "house"]
    with pytest.raises(SystemExit):
        figstyle.styles("ft")


def _fig4_data() -> dict:
    sig = ["ytm", "cs", "bbtm", "dcs6", "val_ipr", "val_hz", "str"]
    rng = np.random.default_rng(0)
    lib = rng.uniform(20, 80, 7)
    frame = pd.DataFrame({"signal": sig, "bias_1_2": rng.uniform(0.1, 0.9, 7),
                          "se_1_2": 0.05, "bias_1_3": rng.uniform(0.1, 0.9, 7), "se_1_3": 0.05,
                          "lib_pct": lib, "actual_pct": 100 - lib})
    return {"single": frame, "wf": frame.copy()}


@pytest.mark.parametrize("look", ["paper", "house"])
def test_each_driver_draws_in_both_looks(tmp_path, look):
    import f06_dua
    import f_lib_figures as L
    import f_nse_figures as N

    st = figstyle.BY_NAME[look]
    L.build_fig4(_fig4_data(), tmp_path / f"fig4_{look}.pdf", st)
    dates = pd.date_range("2005-01-31", periods=200, freq="ME")
    cum = pd.DataFrame({c: np.linspace(1, 2 + i, 200) for i, c in enumerate(["r1", "r2", "r3", "lib"])},
                       index=dates)
    L.build_fig3({k: {"sig": "str", "sort": "single", "cum": cum} for k in "ABCD"},
                 tmp_path / f"fig3_{look}.pdf", st)
    bars = pd.DataFrame({"alpha": np.linspace(0.5, 0.1, 9), "se": 0.1})
    f06_dua.build_figure({t: bars for _, _, t in f06_dua.PANELS}, tmp_path / f"fig6_{look}.pdf", st)
    sel = pd.DataFrame({"signal": [f"s{i}" for i in range(9)], "group": range(1, 10),
                        "median": 1.0, "q25": 0.5, "q75": 1.5, "min": -0.5, "max": 2.5})
    N.render(sel, list(sel["signal"]), {"s0": 1.2}, tmp_path / f"box_{look}.pdf", "$t$", st)
    for name in ("fig4", "fig3", "fig6", "box"):
        assert (tmp_path / f"{name}_{look}.pdf").stat().st_size > 1000
