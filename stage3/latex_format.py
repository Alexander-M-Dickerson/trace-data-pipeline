"""latex_format.py — number/text formatters obeying the house LaTeX style.

The house rules, applied everywhere so a number never renders two ways:
  * negatives use $-$ (math minus), NEVER a hyphen:           -0.95 -> "$-$0.95"
  * thousands separated with {,}:                             1247830 -> "1{,}247{,}830"
  * t-stats in parentheses on a sub-row:                      -2.14 -> "($-$2.14)"
  * 2--3 significant digits (Cochrane); booktabs; no vertical rules; right-align numbers.
"""
from __future__ import annotations

import math


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def num(x, dec: int = 2, grouping: bool = False) -> str:
    """A number in house style: $-$ for negatives, optional {,} thousands. NaN/None -> blank."""
    if _isnan(x):
        return ""
    neg = x < 0
    a = abs(float(x))
    if grouping and a == int(a):
        body = f"{int(a):,}".replace(",", "{,}")
    else:
        body = f"{a:.{dec}f}"
    return (r"$-$" if neg else "") + body


def paren(x, dec: int = 2) -> str:
    """A t-statistic (or any value) in parentheses, house-style negatives. NaN -> blank."""
    if _isnan(x):
        return ""
    return f"({num(x, dec)})"


def thousands(n) -> str:
    if _isnan(n):
        return ""
    return f"{int(n):,}".replace(",", "{,}")


def pct(x, dec: int = 1) -> str:
    """A percentage value (the number is already in percent). $-$ negatives; appends \\%."""
    if _isnan(x):
        return ""
    return num(x, dec) + r"\%"


def escape(s: str) -> str:
    """Escape LaTeX specials in data strings (company names, tickers, ...)."""
    s = str(s)
    for ch, esc in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(ch, esc)
    return s


def row(cells) -> str:
    """Join cells into a LaTeX table row."""
    return " & ".join(cells) + r" \\"
