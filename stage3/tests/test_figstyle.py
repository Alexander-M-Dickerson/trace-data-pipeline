"""test_figstyle.py -- the two figure looks (figstyle.py), without any data on disk.

  * no figure driver types a colour of its own: every colour comes from a style's roles, so
    a look is changed in one place;
  * Figure 4's decomposition says "Adjusted Return";
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
    """A hex colour, a named colour or a palette constant in a driver is a look that --style
    cannot change. White is the page, so it may be typed."""
    src = (STAGE3 / rel).read_text(encoding="utf-8")
    hits = [f"{rel}:{i}: {line.strip()[:80]}"
            for i, line in enumerate(src.splitlines(), 1)
            if not line.lstrip().startswith("#")
            and (re.search(r"['\"]#[0-9a-fA-F]{6}['\"]", line)
                 or re.search(r"['\"](gray|grey|red|black|blue|orange|green|purple)['\"]", line)
                 or re.search(r"figstyle\.(MAROON|BLUE|CHARCOAL|GREY|LIGHT\w*|RECESSION_GREY|FT)\b",
                              line))]
    assert not hits, "colours typed in a driver:\n" + "\n".join(hits)


@pytest.mark.parametrize("rel", DRIVERS)
def test_every_figure_is_saved_through_figstyle(rel):
    """figstyle.save clears the legends; a figure saved around it can hide a bar again."""
    src = (STAGE3 / rel).read_text(encoding="utf-8")
    assert "fig.savefig(" not in src and "figstyle.save(" in src


@pytest.mark.parametrize("look", ["house", "submitted"])
def test_no_legend_hides_data(look):
    """The published Figure IA.1(B) hid its tallest bar under the legend; now the panel grows."""
    st = figstyle.BY_NAME[look]
    with figstyle.use(st) as plt:
        fig, (ax, calm) = plt.subplots(1, 2, figsize=(8, 3))
        ax.bar([0, 1, 2], [0.3, 0.3, 1.0], yerr=0.05, label="tall")
        st.add_legend(ax, loc="upper right")
        calm.bar([0, 1, 2], [1.0, 0.2, 0.2], label="short")
        st.add_legend(calm, loc="upper right")
        fig.tight_layout()
        before = calm.get_ylim()
        moved = figstyle.clear_legends(fig)
        fig.canvas.draw()
        assert moved == [ax]
        assert not figstyle._covered(ax, ax.get_legend().get_window_extent(), 3.0)
        assert calm.get_ylim() == before          # nothing under its legend: left alone
        plt.close(fig)


def test_figure4_says_adjusted_return():
    src = (STAGE3 / "s1_lib" / "f_lib_figures.py").read_text(encoding="utf-8")
    assert '"Adjusted Return"' in src and "Actual Return" not in src


def test_recessions_are_the_papers():
    assert [(a[:7], b[:7]) for a, b, _ in figstyle.NBER] == [("2007-12", "2009-06"),
                                                              ("2020-02", "2020-04")]


def test_styles_have_the_same_roles():
    """A role the paper style knows and the house style does not would fail only at draw time."""
    assert set(figstyle.SUBMITTED.colours) == set(figstyle.HOUSE.colours)
    assert len(figstyle.SUBMITTED.clusters) == len(figstyle.HOUSE.clusters) == 9
    assert len(figstyle.SUBMITTED.ramp) == len(figstyle.HOUSE.ramp) == 9


def test_house_colours_are_bond_schedules():
    c = figstyle.HOUSE.colours
    assert {c["unadjusted"], c["adj_signal"], c["adj_return"]} == {"#990F3D", "#0F5499", "#262A33"}


def _lab(hex_colour: str) -> tuple[float, float, float]:
    """sRGB -> CIE Lab (D65)."""
    c = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c]
    xyz = ((0.4124 * c[0] + 0.3576 * c[1] + 0.1805 * c[2]) / 0.95047,
           0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2],
           (0.0193 * c[0] + 0.1192 * c[1] + 0.9505 * c[2]) / 1.08883)
    f = [t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in xyz]
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def test_house_clusters_can_be_told_apart():
    """The nine cluster colours of Figures IA.3-IA.6. The FT palette's teal and green are
    6 apart in CIE delta E, which the eye reads as one colour; the house look keeps every pair
    at least 15 apart (the submitted paper's own blue scale reached 10)."""
    import itertools
    labs = [_lab(c) for c in figstyle.HOUSE.clusters]
    worst = min(sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5
                for a, b in itertools.combinations(labs, 2))
    assert worst >= 15, f"two house cluster colours are only {worst:.1f} apart"


def test_out_paths(tmp_path):
    assert figstyle.out_path(tmp_path, "f.pdf", figstyle.HOUSE) == tmp_path / "f.pdf"
    assert figstyle.out_path(tmp_path, "f.pdf", figstyle.SUBMITTED) == tmp_path / "submitted" / "f.pdf"
    assert [s.name for s in figstyle.styles("both")] == ["house", "submitted"]
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


@pytest.mark.parametrize("look", ["house", "submitted"])
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
