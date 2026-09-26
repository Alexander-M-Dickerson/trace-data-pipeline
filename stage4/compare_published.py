r"""compare_published.py -- check Stage 4's factors against the ones openbondassetpricing.com serves.

    python compare_published.py                      # both sorts; downloads each archive once
    python compare_published.py --sort single
    python compare_published.py --published my.zip   # a local copy of one archive instead

Downloads the TRACE-only panel archive for this vintage (about 37 MB per sort, cached in
output/_published/), and compares it with what build_factors.py wrote, cell by cell:

  * the same (date, factor, leg, weighting, rating band, return type) rows, no more, no fewer;
  * the same columns and types;
  * `return`, `turnover` and `count`: the largest absolute difference and the number of
    cells where one side is missing and the other is not, per return type and band;
  * the same flip set.

Exit 0 when everything agrees within --tol (default 0: identical); 1 otherwise.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _stage4_settings as S   # noqa: E402
from factorlib import release         # noqa: E402

KEYS = ["date", "factor", "leg", "weighting", "rating_type", "return_type"]
VALUES = ["return", "turnover", "count"]


def fetch(sort: str, vintage: str, refresh: bool) -> Path:
    url = S.SPEC["published"][sort].format(vintage=vintage)
    cache = S.OUTPUT / "_published" / url.rsplit("/", 1)[1]
    if cache.exists() and not refresh:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}", flush=True)
    tmp = cache.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(cache)
    return cache


def read_published(path: Path, sort: str) -> tuple[pd.DataFrame, dict | None]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path), None
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        member = [n for n in names if n.endswith(".parquet") and n.startswith(f"{sort}_sort_")]
        if len(member) != 1:
            raise SystemExit(f"{path.name}: expected one {sort} parquet, found {member}")
        df = pd.read_parquet(io.BytesIO(z.read(member[0])))
        flips = (json.loads(z.read("flip_set.json").decode("utf-8"))
                 if "flip_set.json" in names else None)
    return df, flips


def compare(ours: pd.DataFrame, pub: pd.DataFrame, tol: float) -> tuple[bool, list[str]]:
    lines, ok = [], True
    if list(ours.columns) != list(pub.columns):
        ok = False
        lines.append(f"  columns differ: ours {list(ours.columns)}\n"
                     f"                 published {list(pub.columns)}")
    else:
        bad = [c for c in ours.columns if ours[c].dtype != pub[c].dtype]
        if bad:
            ok = False
            lines.append("  types differ: " + ", ".join(
                f"{c} {ours[c].dtype} vs {pub[c].dtype}" for c in bad))
    m = ours.merge(pub, on=KEYS, how="outer", suffixes=("_ours", "_pub"), indicator=True)
    only_o, only_p = int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum())
    if only_o or only_p:
        ok = False
        lines.append(f"  rows only in ours: {only_o:,}; only in the published file: {only_p:,}")
    m = m[m["_merge"] == "both"]
    lines.append(f"  {len(m):,} rows compared")
    lines.append(f"  {'return type':<12}{'band':<6}" + "".join(f"{v + ' max|d|':>18}{'NaN diff':>10}"
                                                          for v in VALUES))
    for (rt, band), g in m.groupby(["return_type", "rating_type"], sort=True):
        row = f"  {rt:<12}{band:<6}"
        for v in VALUES:
            a, b = g[f"{v}_ours"].to_numpy(float), g[f"{v}_pub"].to_numpy(float)
            nan_diff = int((np.isnan(a) != np.isnan(b)).sum())
            both = ~np.isnan(a) & ~np.isnan(b)
            d = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            if d > tol or nan_diff:
                ok = False
            row += f"{d:>18.3g}{nan_diff:>10,}"
        lines.append(row)
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sort", choices=("single", "within_firm", "all"), default="all")
    ap.add_argument("--published", type=Path, default=None,
                    help="a local copy of the published archive (.zip) or its parquet")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="largest absolute difference that still counts as agreement")
    ap.add_argument("--refresh", action="store_true", help="download again, ignoring the cache")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    vintage = S.vintage()
    sort_list = list(S.SORTS) if args.sort == "all" else [args.sort]
    if args.published and len(sort_list) != 1:
        ap.error("--published names one archive: give --sort as well")
    all_ok = True
    for sort in sort_list:
        mine = release.panel_file(sort, vintage)
        print(f"\n{sort}: {mine}")
        if not mine.exists():
            print("  not built yet: run `python build_factors.py` first")
            all_ok = False
            continue
        src = args.published or fetch(sort, vintage, args.refresh)
        print(f"  against {src}")
        ours = pd.read_parquet(mine)
        pub, pub_flips = read_published(src, sort)
        ok, lines = compare(ours, pub, args.tol)
        print("\n".join(lines))
        if pub_flips is not None:
            mine_flips = json.loads((mine.parent / "flip_set.json").read_text(encoding="utf-8"))
            same = mine_flips["flips"] == pub_flips["flips"]
            print(f"  flip set: {'identical' if same else 'DIFFERS'} "
                  f"({mine_flips['n_would_flip']} of {mine_flips['n_keys']} keys would flip; "
                  f"published {pub_flips['n_would_flip']} of {pub_flips['n_keys']})")
            ok = ok and same
        print(f"  => {'AGREES' if ok else 'DIFFERS'}" + (f" (tol {args.tol:g})" if args.tol else ""))
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
