r"""compare_runs.py -- two stage 3 runs, cell by cell: what moved, what changed sign, and
which t-statistics crossed 1.96.

    python tools/compare_runs.py <run A> <run B> [--out DIR]

A run is a stage 3 data folder (`stage3/data`, or `stage3/variants/<type>/data` after
`run_stage3.sh --returns <type>`), or the folder that holds it. Every result CSV the section
folders hold in both runs is compared:

  * rows are joined on their text columns. The return-type column (`ret_type`,
    `return_type`) names each run's own type, so it is left out of the join. A table with
    no text key, or a duplicated one, is aligned row by row, and says so;
  * every number both runs report is compared. A t-statistic is a cell whose `kind` is `t`,
    or a column named `t` / `tstat`; one "crosses" when it is on different sides of 1.96 in
    the two runs. A sign change is counted on the other cells;
  * rows only one run has are counted (a table whose rows depend on the data, such as a list
    of the factors a filter affects, can differ in its rows, not only its numbers).

It writes `cells.csv` (every compared cell) and `summary.md` (per table, then every sign
change and every crossing) to --out, by default `comparison/` beside run B's data folder.
"""
# [tag:entry.stage3_compare] compares two stage 3 runs cell by cell: python tools/compare_runs.py <A> <B>
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SECTIONS = ("s0_data", "s1_lib", "s2_lab", "s3_nse", "s4_zoo")
IGNORE_KEYS = ("ret_type", "return_type")
Z = 1.96
LIST_CAP = 40          # rows of each list printed in summary.md; cells.csv holds them all


def data_dir(p: Path) -> Path:
    """The run's data folder, whether given it or the folder above it. (The stage 3 folder
    itself has code folders with the section names, so a `data/` beside them wins.)"""
    p = Path(p)
    if any((p / "data" / s).is_dir() for s in SECTIONS):
        p = p / "data"
    if not result_csvs(p):
        raise SystemExit(f"{p} holds no result CSV under {', '.join(SECTIONS)}: "
                         "not a stage 3 data folder")
    return p


def result_csvs(root: Path) -> dict[str, Path]:
    return {f"{s}/{f.name}": f for s in SECTIONS if (root / s).is_dir()
            for f in sorted((root / s).glob("*.csv"))}


def _is_t_measure(name: str) -> bool:
    n = name.lower()
    return n in ("t", "tstat", "t_stat") or n.endswith(("_t", "_tstat"))


def compare_table(a: pd.DataFrame, b: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    """The long table of compared cells for one CSV, and its counts."""
    text = lambda d, c: c in d and not pd.api.types.is_numeric_dtype(d[c])   # noqa: E731
    keys = [c for c in a.columns if c in b.columns and c not in IGNORE_KEYS
            and (text(a, c) or text(b, c))]
    nums = [c for c in a.columns if c in b.columns and c not in keys and c not in IGNORE_KEYS
            and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    info = {"table": name, "aligned": "by key", "only_a": 0, "only_b": 0}
    if not nums:
        return pd.DataFrame(), {**info, "aligned": "no numbers"}
    if not keys or a.duplicated(keys).any() or b.duplicated(keys).any():
        a, b = a.assign(_row=np.arange(len(a))), b.assign(_row=np.arange(len(b)))
        keys = keys + ["_row"]
        info["aligned"] = "by row"
    for k in keys:
        a[k], b[k] = a[k].astype(str), b[k].astype(str)
    m = a[keys + nums].merge(b[keys + nums], on=keys, how="outer", suffixes=("_a", "_b"),
                             indicator=True)
    info["only_a"] = int((m["_merge"] == "left_only").sum())
    info["only_b"] = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]
    key = both[keys].agg(", ".join, axis=1) if len(both) else pd.Series(dtype=str)
    kind_t = both["kind"].eq("t") if "kind" in both else pd.Series(False, index=both.index)
    parts = []
    for c in nums:
        va, vb = both[f"{c}_a"].astype(float), both[f"{c}_b"].astype(float)
        is_t = kind_t | _is_t_measure(c)
        parts.append(pd.DataFrame({
            "table": name, "key": key, "measure": c, "a": va, "b": vb, "diff": vb - va,
            "t": is_t,
            "sign_change": ~is_t & (np.sign(va) * np.sign(vb) < 0),
            "t_cross": is_t & ((va.abs() >= Z) != (vb.abs() >= Z)),
        }))
    cells = pd.concat(parts, ignore_index=True).dropna(subset=["a", "b"], how="all")
    both_num = cells.dropna(subset=["a", "b"])
    info.update(cells=len(cells),
                changed=int((both_num["diff"].abs() > 1e-12).sum()),
                mean_abs=float(both_num["diff"].abs().mean()) if len(both_num) else np.nan,
                sign_changes=int(cells["sign_change"].sum()),
                t_crossings=int(cells["t_cross"].sum()))
    if len(both_num):
        i = both_num["diff"].abs().idxmax()
        info.update(max_abs=float(abs(both_num.at[i, "diff"])),
                    max_at=f"{both_num.at[i, 'key']} / {both_num.at[i, 'measure']}")
    return cells, info


def _fmt(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4g}"


def _md(s) -> str:
    """Text for a markdown table cell."""
    return str(s).replace("|", "\\|")


def summary_md(a_root: Path, b_root: Path, infos: list[dict], cells: pd.DataFrame,
               only: dict[str, list[str]]) -> str:
    L = ["# Two stage 3 runs, cell by cell", "",
         f"- A: `{a_root.as_posix()}`", f"- B: `{b_root.as_posix()}`", "",
         f"t-statistics cross when they sit on different sides of {Z}. Differences are B "
         "minus A. `by row` marks a table without a text key, aligned by position.", "",
         "| table | cells | changed | mean abs diff | largest abs diff | where | sign changes "
         "| t crossings | rows only in A / B | aligned |",
         "|---|---:|---:|---:|---:|---|---:|---:|---|---|"]
    for i in infos:
        L.append(f"| {i['table']} | {i.get('cells', 0)} | {i.get('changed', 0)} | "
                 f"{_fmt(i.get('mean_abs'))} | {_fmt(i.get('max_abs'))} | "
                 f"{_md(i.get('max_at', ''))} | {i.get('sign_changes', 0)} | "
                 f"{i.get('t_crossings', 0)} | {i['only_a']} / {i['only_b']} | {i['aligned']} |")
    for what, col in (("t-statistics that cross 1.96", "t_cross"),
                      ("Numbers that change sign", "sign_change")):
        rows = cells[cells[col]] if len(cells) else cells
        L += ["", f"## {what} ({len(rows)})", ""]
        if not len(rows):
            L.append("None.")
            continue
        L += ["| table | cell | measure | A | B |", "|---|---|---|---:|---:|"]
        for r in rows.head(LIST_CAP).itertuples():
            L.append(f"| {r.table} | {_md(r.key)} | {_md(r.measure)} | {_fmt(r.a)} | "
                     f"{_fmt(r.b)} |")
        if len(rows) > LIST_CAP:
            L.append(f"\n...and {len(rows) - LIST_CAP} more, in cells.csv.")
    L += ["", "## Tables in one run only", ""]
    L += [f"- only in {k}: {', '.join(v)}" for k, v in only.items() if v] or ["None."]
    return "\n".join(L) + "\n"


def compare(a_root: Path, b_root: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    fa, fb = result_csvs(a_root), result_csvs(b_root)
    parts, infos = [], []
    for name in sorted(set(fa) & set(fb)):
        c, info = compare_table(pd.read_csv(fa[name]), pd.read_csv(fb[name]), name)
        infos.append(info)
        if len(c):
            parts.append(c)
    cells = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["table", "key", "measure", "a", "b", "diff", "t", "sign_change", "t_cross"])
    only = {"A": sorted(set(fa) - set(fb)), "B": sorted(set(fb) - set(fa))}
    return cells, infos, only


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path, help="run A: a stage 3 data folder (or the folder above)")
    ap.add_argument("b", type=Path, help="run B")
    ap.add_argument("--out", type=Path, help="where to write (default: comparison/ beside B)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    a_root, b_root = data_dir(args.a), data_dir(args.b)
    out = args.out or b_root.parent / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    cells, infos, only = compare(a_root, b_root)
    cells.to_csv(out / "cells.csv", index=False)
    (out / "summary.md").write_text(summary_md(a_root, b_root, infos, cells, only),
                                    encoding="utf-8")
    print(f"{len(infos)} tables, {len(cells):,} cells; {int(cells['t_cross'].sum())} t-statistics "
          f"cross {Z}, {int(cells['sign_change'].sum())} numbers change sign\n"
          f"wrote {out / 'cells.csv'} and {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
