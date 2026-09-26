"""release.py -- write one sort's factors in the layout openbondassetpricing.com publishes.

For each sort, two folders under `stage4/output/`, each holding what the matching published
archive holds (zip a folder and you have the archive, less its README):

  {sort}_sort_panel_trace_{vintage}/   {sort}_sort_trace_{vintage}.parquet  the long panel
                                       flip_set.json, MANIFEST.json
  {sort}_sort_trace_{vintage}_csv/     {sort}_sort_{return}_{band}_{weighting}.csv  24 files
                                       flip_set.json, MANIFEST.json

The CSVs are the long-short leg pivoted wide: one row per month, one column per factor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import _stage4_settings as S


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def folders(sort: str, vintage: str) -> tuple[Path, Path]:
    return (S.OUTPUT / f"{sort}_sort_panel_trace_{vintage}",
            S.OUTPUT / f"{sort}_sort_trace_{vintage}_csv")


def panel_file(sort: str, vintage: str) -> Path:
    return folders(sort, vintage)[0] / f"{sort}_sort_trace_{vintage}.parquet"


def wide_csvs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The long-short leg, one file per (return type, band, weighting), one column per factor."""
    out: dict[str, pd.DataFrame] = {}
    ls = df[df["leg"] == S.SPEC["csv"]["leg"]]
    for rt in sorted(ls["return_type"].unique()):
        for band in sorted(ls["rating_type"].unique()):
            for w in sorted(ls["weighting"].unique()):
                sub = ls[(ls["return_type"] == rt) & (ls["rating_type"] == band)
                         & (ls["weighting"] == w)]
                if sub.empty:
                    continue
                wide = sub.pivot(index="date", columns="factor", values="return").sort_index()
                wide.index.name = "date"
                out[f"{rt}_{band}_{w}.csv"] = wide
    # A pivot that drops or invents a cell still looks like a full matrix: count them.
    n_long = len(ls)
    n_wide = sum(w.shape[0] * w.shape[1] for w in out.values())
    if n_long != n_wide:
        raise AssertionError(f"the CSVs hold {n_wide:,} cells against {n_long:,} long-short rows")
    return out


def flip_blob(flips: dict) -> str:
    return json.dumps({
        "_doc": "What the full-sample sign rule WOULD choose, had it been applied: true where "
                "the long-short mean is negative. The series themselves are unflipped.",
        "keys": "|".join(["factor", "weighting", "return_type"]),
        "n_keys": len(flips),
        "n_would_flip": sum(flips.values()),
        "flips": flips,
    }, indent=2)


def write(sort: str, panel: pd.DataFrame, flips: dict, manifest: dict, vintage: str) -> dict:
    """Write both folders for one sort; return what was written, with hashes."""
    pdir, cdir = folders(sort, vintage)
    for d in (pdir, cdir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            if old.is_file():
                old.unlink()

    pfile = panel_file(sort, vintage)
    panel.to_parquet(pfile, index=False)
    blob = flip_blob(flips)
    (pdir / "flip_set.json").write_text(blob, encoding="utf-8", newline="\n")

    csvs = wide_csvs(panel)
    fmt = S.SPEC["csv"]["float_format"]
    for name, wide in csvs.items():
        wide.to_csv(cdir / f"{sort}_sort_{name}", float_format=fmt)
    (cdir / "flip_set.json").write_text(blob, encoding="utf-8", newline="\n")

    written = {}
    for d in (pdir, cdir):
        members = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                   for p in sorted(d.iterdir()) if p.is_file()}
        (d / "MANIFEST.json").write_text(
            json.dumps({**manifest, "folder": d.name, "members": members}, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        written[d.name] = members
    return written
