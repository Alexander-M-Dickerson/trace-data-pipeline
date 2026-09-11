r"""check_inputs.py -- the input contract, checked before anything long starts.

Stage 3 reads five files. This says whether they are there, whether they are the right
shape, and whether they cover the sample the exhibits assume -- and it fails with the
whole list rather than one file at a time.

Run it first. A missing column surfaces here in a second, instead of forty minutes into
a grid.

    python tools/check_inputs.py
    python tools/check_inputs.py --verbose     # every declared column, present or not

The contract lives in `spec/inputs.json`, beside this file. Editing that file changes
what is required; editing this one changes how it is checked.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
sys.path.insert(0, str(STAGE3))

import _stage3_settings as S  # noqa: E402

SPEC = STAGE3 / "spec" / "inputs.json"


def _read_meta(path: Path) -> dict:
    """Rows, columns and the date span, without loading the file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = {"rows": pf.metadata.num_rows,
            "columns": list(pf.schema.names),
            "bytes": path.stat().st_size}
    date_col = next((c for c in ("date", "trd_exctn_dt") if c in meta["columns"]), None)
    if date_col:
        import duckdb
        p = path.as_posix()
        row = duckdb.sql(
            f'SELECT min("{date_col}") AS lo, max("{date_col}") AS hi '
            f"FROM read_parquet('{p}')").fetchone()
        meta["date_col"] = date_col
        meta["first"] = str(row[0])[:10]
        meta["last"] = str(row[1])[:10]
    return meta


def check(verbose: bool = False) -> tuple[list[dict], bool]:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    results, ok_all = [], True
    for name, decl in spec["inputs"].items():
        path = S.INPUTS.get(name)
        row = {"input": name, "what": decl["what"], "path": str(path) if path else None,
               "problems": []}
        if path is None or not Path(path).exists():
            row["problems"].append("not found")
            results.append(row)
            ok_all = False
            continue
        meta = _read_meta(Path(path))
        row.update({k: meta[k] for k in ("rows", "bytes") if k in meta})
        row["n_columns"] = len(meta["columns"])
        if "first" in meta:
            row["span"] = f"{meta['first']} .. {meta['last']}"

        if decl.get("min_rows") and meta["rows"] < decl["min_rows"]:
            row["problems"].append(
                f"{meta['rows']:,} rows, contract wants at least {decl['min_rows']:,}")
        missing = [c for c in decl.get("required_columns", [])
                   if c not in meta["columns"]]
        if missing:
            row["problems"].append(f"{len(missing)} missing column(s): "
                                   + ", ".join(missing[:8])
                                   + (" ..." if len(missing) > 8 else ""))
        if decl.get("min_end") and meta.get("last", "") < decl["min_end"]:
            row["problems"].append(
                f"ends {meta.get('last')}, contract wants at least {decl['min_end']}")
        if decl.get("max_start") and meta.get("first", "9999") > decl["max_start"]:
            row["problems"].append(
                f"starts {meta.get('first')}, contract wants no later than "
                f"{decl['max_start']}")
        if verbose:
            row["columns"] = meta["columns"]
        ok_all &= not row["problems"]
        results.append(row)
    return results, ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    results, ok = check(verbose=args.verbose)
    if args.json:
        print(json.dumps({"ok": ok, "inputs": results}, indent=2, default=str))
        return 0 if ok else 1

    for r in results:
        mark = "OK  " if not r["problems"] else "FAIL"
        print(f"{mark} {r['input']:15s} {r['what']}")
        print(f"       {r['path']}")
        if "rows" in r:
            print(f"       {r['rows']:,} rows x {r['n_columns']} cols"
                  + (f", {r['span']}" if "span" in r else "")
                  + f", {r['bytes'] / 1e6:.0f} MB")
        for p in r["problems"]:
            print(f"       -> {p}")
        if args.verbose and "columns" in r:
            print(f"       columns: {', '.join(r['columns'])}")
    n_ok = sum(not r["problems"] for r in results)
    print(f"\n{n_ok}/{len(results)} inputs satisfy the contract")

    # ❗A WARNING, not a failure. pdflatex is needed only by the last step, and
    # `make_report.py --no-compile` is a legitimate way to run the whole pipeline --
    # you still get every table and figure as a file, just not the assembled PDF.
    if shutil.which("pdflatex") is None:
        print("\nWARN pdflatex not found on PATH.")
        print("       Everything runs; the final step cannot compile reports/exhibits.pdf.")
        print("       Install TeX Live or MiKTeX, or run make_report.py --no-compile.")
    if not ok:
        print("\nStage 3 will not produce correct exhibits until these are fixed.\n"
              "  Run Stage 2 first, or point STAGE2_* at where its output lives.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
