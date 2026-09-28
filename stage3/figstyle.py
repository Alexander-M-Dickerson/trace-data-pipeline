r"""figstyle.py -- the two looks every stage 3 figure can be drawn in.

  paper   the look of the published paper's figures: serif type, the blue scale for the three
          approaches, orange for the latent implementation bias, framed legends, italic factor
          names, "(A) ..." panel titles. The default.
  house   the production-paper figure style of our bond_schedule project: Arial 8.5pt,
          charcoal axes without top or right spines, maroon #990F3D / blue #0F5499 /
          charcoal #262A33 with their light tints, the same family's FT palette where a
          figure needs more categories, "Panel A. ..." titles set left, and no title inside
          the image.

A driver computes its data once and draws it in each style asked for (`--style paper|house|
both`); a style never computes. The paper style writes to reports/figures/<stem>.pdf, the
house style to reports/figures/house/<stem>.pdf.

Every figure whose x-axis is time shades the NBER recessions it spans, in both styles, from
NBER below: the Great Recession and COVID-19, peak month to trough month.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

STYLES = ("paper", "house")

# NBER business-cycle peaks and troughs inside the sample, as the paper's captions state them
# (Great Recession 2007:12--2009:06, COVID-19 2020:02--2020:04); shaded from the first day of
# the peak month to the last day of the trough month.
NBER = (("2007-12-01", "2009-06-30", "Great Recession"),
        ("2020-02-01", "2020-04-30", "COVID-19"))

# the bond_schedule project's house colours and, where a figure needs more categories, the
# rest of the same family (its FT palette)
MAROON, BLUE, CHARCOAL = "#990F3D", "#0F5499", "#262A33"
GREY, LIGHTGREY = "#7F7F7F", "#D0D0D0"
LIGHT_MAROON, LIGHT_BLUE = "#E7C3CF", "#C3D4E6"
RECESSION_GREY = "#E8E8E8"
FT = {"teal": "#0F766E", "gold": "#F2B701", "purple": "#6F4E7C", "orange": "#D56F3E",
      "slate": "#4C78A8", "brown": "#8C6D31", "pink": "#E95D8E", "green": "#0D7680"}


def _tint(hex_colour: str, share: float) -> str:
    """A colour mixed with white: `share` of the colour, the rest white."""
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(255 - share * (255 - c)):02x}" for c in rgb)


def _ramp(dark: str, light: str, n: int) -> tuple[str, ...]:
    """n colours from `dark` to `light`, evenly spaced."""
    a = [int(dark.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(light.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return tuple("#" + "".join(f"{round(x + (y - x) * k / (n - 1)):02x}" for x, y in zip(a, b))
                 for k in range(n))


@dataclass(frozen=True)
class Style:
    name: str
    rc: dict
    colours: dict                     # a role -> a colour (see the two definitions below)
    clusters: tuple                   # the nine factor clusters, I..IX
    ramp: tuple                       # nine thresholds, tightest first (Figure 6)
    width: float | None               # inches; None keeps each figure's own size
    legend: dict = field(default_factory=dict)
    recession: dict = field(default_factory=dict)

    # ---- text
    def panel(self, ax, letter: str, text: str) -> None:
        if self.name == "paper":
            ax.set_title(f"({letter}) {text}")
        else:
            ax.set_title(f"Panel {letter}. {text}", loc="left")

    def factor(self, name: str) -> str:
        """A factor mnemonic as a label: italic in the paper style, plain in the house style."""
        if self.name == "paper":
            return "$\\mathit{" + name.replace("_", "\\_") + "}$"
        return name

    def suptitle(self, fig, text: str, **kw) -> None:
        """The paper style kept a few figure titles; the house style puts none in the image."""
        if self.name == "paper":
            fig.suptitle(text, **kw)

    # ---- marks
    def c(self, role: str) -> str:
        return self.colours[role]

    def add_legend(self, ax, *args, **kw):
        return ax.legend(*args, **{**self.legend, **kw})

    def legends(self, fig, per_axes: list, ncol: int | None = None) -> None:
        """Place a figure's legends, then lay it out.

        paper  each legend where the published figure had it: `per_axes` holds
               (ax, handles, labels, keyword arguments), in order.
        house  one legend for the whole figure, above the panels and outside them (the
               bond_schedule convention: nothing drawn over the data), every entry once.
        """
        if self.name == "paper":
            for ax, handles, labels, kw in per_axes:
                self.add_legend(ax, handles, labels, **kw)
            fig.tight_layout()
            return
        seen: dict[str, object] = {}
        for _, handles, labels, _ in per_axes:
            for h, lab in zip(handles, labels):
                seen.setdefault(lab, h)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.legend(list(seen.values()), list(seen), loc="upper center", frameon=False,
                   ncol=ncol or len(seen), bbox_to_anchor=(0.5, 1.0))

    def panel_legend(self, ax, handles, labels, loc: str = "upper left", **kw) -> None:
        """A legend that belongs to one panel (its entries differ from the next panel's).
        paper  inside the panel, where the published figure had it
        house  above the panel, under its title, so it never covers the data"""
        if self.name == "paper":
            self.add_legend(ax, handles, labels, loc=loc, **kw)
            return
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=15)
        ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
                  ncol=len(labels), borderaxespad=0.1, handlelength=1.6, columnspacing=1.0,
                  fontsize=7)

    def recessions(self, ax) -> None:
        """Shade the NBER recessions on a date axis, behind the data, with no legend entry."""
        import pandas as pd
        for start, end, _ in NBER:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), lw=0, zorder=0, **self.recession)

    def size(self, w: float, h: float) -> tuple[float, float]:
        """The figure size: the figure's own in the paper style; the house style's fixed width,
        at the same aspect ratio."""
        if self.width is None:
            return (w, h)
        return (self.width, self.width * h / w)

    def grid(self, ax, axis: str = "y") -> None:
        if self.name == "paper":
            ax.grid(True, alpha=0.25, lw=0.6, axis=axis)
        else:
            ax.set_axisbelow(True)
            ax.grid(True, axis=axis, color="#E6E6E6", lw=0.5)


def _font() -> str:
    import matplotlib.font_manager as fm
    return "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"


PAPER = Style(
    name="paper",
    # the published figures' PlotParams: serif, 10 / 10 / 9 / 9, no LaTeX
    rc={"text.usetex": False, "font.family": "serif", "font.size": 10, "axes.titlesize": 10,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.dpi": 150},
    colours={
        # the three approaches of Section 3, light to dark, and the cumulative LIB
        "unadjusted": "#a6cee3", "adj_signal": "#1f78b4", "adj_return": "#08306b",
        "lib_line": "gray", "lib_bar": "#F18F01", "implementable": "#08306b",
        "bias12": "#1f78b4", "bias13": "#08306b",
        # Section 4
        "wins": "#a6cee3", "base": "#08306b", "lab_line": "gray",
        "series1": "#08306b", "series2": "#2171b5", "series3": "#6baed6",
        "scatter": "#1f78b4", "fit": "#e31a1c",
        "long": "#08306b", "short": "#6baed6",
        "all": "#08306b", "ig": "#2171b5", "nig": "#6baed6",
        # Section 5
        "baseline": "red", "threshold": "red", "zero": "gray", "whisker": "black",
    },
    clusters=("#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
              "#a1d99b", "#fdae6b", "#bcbddc"),
    ramp=("#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef",
          "#deebf7", "#f7fbff"),
    width=None,
    legend={"frameon": True, "edgecolor": "gray"},
    recession={"color": "gray", "alpha": 0.15},
)

HOUSE = Style(
    name="house",
    rc={"font.family": _font(), "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 5,
        "axes.linewidth": 0.6, "axes.edgecolor": CHARCOAL, "axes.labelcolor": CHARCOAL,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.facecolor": "white", "figure.facecolor": "white", "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "lines.linewidth": 1.3, "lines.markersize": 3.0,
        "legend.frameon": False, "legend.fontsize": 7.5, "legend.handlelength": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.dpi": 150},
    colours={
        # the unadjusted (contaminated) series is the one the reader must see: maroon
        "unadjusted": MAROON, "adj_signal": BLUE, "adj_return": CHARCOAL,
        "lib_line": GREY, "lib_bar": MAROON, "implementable": LIGHT_BLUE,
        "bias12": BLUE, "bias13": CHARCOAL,
        # Section 4: the infeasible (winsorized) series in maroon, the feasible in charcoal
        "wins": MAROON, "base": CHARCOAL, "lab_line": GREY,
        "series1": CHARCOAL, "series2": BLUE, "series3": MAROON,
        "scatter": BLUE, "fit": MAROON,
        "long": CHARCOAL, "short": LIGHT_BLUE,
        "all": CHARCOAL, "ig": BLUE, "nig": MAROON,
        # Section 5
        "baseline": MAROON, "threshold": GREY, "zero": CHARCOAL, "whisker": "black",
    },
    # nine clusters: tints of the family, maroon left out so the maroon baseline mark shows
    clusters=tuple(_tint(c, 0.55) for c in (BLUE, FT["teal"], FT["gold"], FT["purple"],
                                            FT["orange"], FT["slate"], FT["brown"],
                                            FT["green"], FT["pink"])),
    ramp=_ramp(BLUE, LIGHT_BLUE, 9),
    width=6.5,
    legend={"frameon": False},
    recession={"color": RECESSION_GREY},
)

BY_NAME = {"paper": PAPER, "house": HOUSE}


def styles(arg: str) -> list[Style]:
    """The styles a `--style` argument asks for."""
    if arg == "both":
        return [PAPER, HOUSE]
    if arg not in BY_NAME:
        raise SystemExit(f"--style must be paper, house or both, not {arg!r}")
    return [BY_NAME[arg]]


def add_argument(ap) -> None:
    ap.add_argument("--style", default="paper", choices=("paper", "house", "both"),
                    help="the look to draw the figures in (default: the paper's)")


@contextlib.contextmanager
def use(style: Style):
    """Draw inside this block in `style` (matplotlib's settings are restored afterwards)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(style.rc):
        yield plt


def out_path(fig_dir: Path, name: str, style: Style) -> Path:
    """reports/figures/<name> for the paper style, reports/figures/house/<name> for the house
    style (created)."""
    d = fig_dir if style.name == "paper" else fig_dir / "house"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# ---------------------------------------------------------------- shared drawing helpers
import numpy as np  # noqa: E402


def fmt_final(v: float) -> str:
    """The growth of $1 as the figures print it: one decimal below 10, none from 10."""
    return f"${v:.1f}" if v < 10 else f"${v:.0f}"


def dollar_axes(ax, cum: pd.DataFrame, finals: dict, colours: list[str], st) -> None:
    """A log dollar axis with the paper's sparse 1-2-5 ticks on the left and each series'
    final value on the right, in that series' colour (Figures 3 and 8)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_yscale("log")
    vals_all = cum.to_numpy().ravel()
    ymin, ymax = float(np.min(vals_all)), float(np.max(vals_all))
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100) if ymin * 0.7 <= t <= ymax * 1.3] or [1]
    ax.set_ylim(ymin * 0.9, ymax * 1.1)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: f"${y:.0f}" if y >= 1 else f"${y:.1f}"))
    end_labels(ax, finals, colours, st)


def end_labels(ax, finals: list[float], colours: list[str], st) -> None:
    """Each series' final value beside the right edge, in its colour. Values that would print
    on top of each other are spread apart on the log scale (the label moves; the value it
    states does not)."""
    lo, hi = (np.log(v) for v in ax.get_ylim())
    gap = (0.06 if st.name == "paper" else 0.085) * (hi - lo)   # a label's height, about
    order = sorted(range(len(finals)), key=lambda i: finals[i])
    placed: dict[int, float] = {}
    last = -np.inf
    for i in order:
        y = max(np.log(finals[i]), last + gap)
        placed[i] = y
        last = y
    over = last - (hi - 0.2 * gap)             # the top label must stay inside the axes
    if over > 0:
        placed = {i: y - over for i, y in placed.items()}
    for i, v in enumerate(finals):
        ax.annotate(fmt_final(v), xy=(1.0, np.exp(placed[i])), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", va="center", ha="left",
                    color=colours[i], fontsize=(8 if st.name == "paper" else 7),
                    annotation_clip=False)


def date_axis(ax) -> None:
    from matplotlib.dates import AutoDateLocator, DateFormatter
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=8, interval_multiples=True))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
