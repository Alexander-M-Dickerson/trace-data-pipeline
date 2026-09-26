r"""build_factors.py -- Stage 4: the TRACE-only bond factors, from Stage 2's panel.

    python build_factors.py                     # both sorts, all four return types
    python build_factors.py --sort single       # one sort
    python build_factors.py --dry-run           # check the inputs, compute nothing
    python build_factors.py --signals cs mom6_1 # a quick run on two signals (-> output/_subset)

108 signals x 4 return types (exc, dur, dbns, dcls) x 3 rating bands (all, ig, nig), as
single sorts and within-firm sorts, through PyBondLab. The series are published unflipped;
flip_set.json records the flips the full-sample sign rule would make. Everything is
declared in spec/factors.json.

Each return type's panel is loaded once and sorted both ways. Writes, per sort, the folders
described in factorlib/release.py under stage4/output/, each with a MANIFEST.json naming its
inputs by sha256 and the PyBondLab release that produced it.

Then `python compare_published.py` checks the result against the files
openbondassetpricing.com serves.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _stage4_settings as S   # noqa: E402
import pybondlab_pin           # noqa: E402  (the root; _stage4_settings puts it on the path)
from factorlib import inputs, release, sorts   # noqa: E402


def _rel(p: Path) -> str:
    """A path as a manifest records it: relative to the repository, never a home directory."""
    try:
        return Path(p).resolve().relative_to(S.PIPELINE.resolve()).as_posix()
    except ValueError:
        return Path(p).name


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(S.PIPELINE), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:
        return None


def _input_record(return_types) -> dict:
    files = {"panel": S.PANEL, "risk_free": S.RISK_FREE}
    for rt in return_types:
        for b in S.RETURN_TYPES[rt]["blocks"] or []:
            files[b] = S.block_path(b)
    return {k: {"file": _rel(p), "sha256": release.sha256(p)} for k, p in files.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sort", choices=("single", "within_firm", "all"), default="all")
    ap.add_argument("--return-types", nargs="+", choices=list(S.RETURN_TYPES),
                    default=list(S.RETURN_TYPES))
    ap.add_argument("--signals", nargs="+", default=None,
                    help="a subset of the 108, for a quick run; writes to output/_subset")
    ap.add_argument("--dry-run", action="store_true", help="check the inputs, compute nothing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    sort_list = list(S.SORTS) if args.sort == "all" else [args.sort]
    rts = [rt for rt in S.RETURN_TYPES if rt in args.return_types]   # spec order
    signals = list(S.SIGNALS)
    if args.signals:
        unknown = sorted(set(args.signals) - set(S.SIGNALS))
        if unknown:
            ap.error(f"not among the 108 signals: {unknown}")
        signals = [s for s in S.SIGNALS if s in args.signals]
    subset = bool(args.signals) or rts != list(S.RETURN_TYPES)
    if subset:
        # Never over the full product: a partial grid is not what the release compares.
        S.OUTPUT = S.OUTPUT / "_subset"
    vintage = S.vintage()

    print("Stage 4: the TRACE-only bond factors")
    print(f"  panel          {S.PANEL}")
    print(f"  blocks         {S.BLOCKS}")
    print(f"  output         {S.OUTPUT}")
    print(f"  vintage        {vintage}")
    print(f"  sorts          {', '.join(sort_list)}")
    print(f"  return types   {', '.join(rts)}")
    print(f"  signals        {len(signals)}")

    problems = []
    engine = pybondlab_pin.check()
    if engine:
        problems.append(engine.rstrip())
    missing = inputs.missing_inputs(rts)
    if missing:
        problems.append("missing Stage 2 inputs:\n    " + "\n    ".join(missing))
    if problems:
        print("\nStage 4 cannot start:\n  - " + "\n  - ".join(problems))
        return 1
    fp = pybondlab_pin.fingerprint()
    print(f"  PyBondLab      v{fp['version']} tree={fp['tree_sha256']}")
    if args.dry_run:
        print("\ndry run: the inputs are all present; nothing computed.")
        return 0

    t0 = time.perf_counter()
    pieces: dict[str, list[pd.DataFrame]] = {s: [] for s in sort_list}
    for rt in rts:
        t = time.perf_counter()
        data = inputs.load(rt)
        print(f"\n  {rt:<4} panel {len(data):,} x {len(data.columns)}  "
              f"{data['date'].min():%Y-%m}..{data['date'].max():%Y-%m}  "
              f"[{time.perf_counter() - t:.0f}s]", flush=True)
        absent = [c for c in signals if c not in data.columns]
        if absent:
            raise SystemExit(f"{len(absent)} signals are not in the {rt} panel: {absent[:8]}")
        for sort in sort_list:
            pieces[sort] += sorts.run_bands(data, signals, sort=sort, return_type=rt)
        del data

    record = _input_record(rts)
    for sort in sort_list:
        panel = pd.concat(pieces[sort], ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])
        flips = sorts.flip_set(panel)
        manifest = {
            "product": {"dataset": "trace", "sort": sort,
                        "subset": subset},
            "vintage": vintage,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "orientation": "unflipped",
            "rows": int(len(panel)),
            "months": int(panel["date"].nunique()),
            "factors": int(panel["factor"].nunique()),
            "data_span": [f"{panel['date'].min():%Y-%m-%d}", f"{panel['date'].max():%Y-%m-%d}"],
            "grid": {"return_types": rts,
                     "rating_bands": [b["label"] for b in S.SPEC["rating_bands"]],
                     "weightings": sorted(panel["weighting"].unique().tolist()),
                     "legs": sorted(panel["leg"].unique().tolist())},
            "signals": len(signals),
            "inputs": record,
            "pybondlab": fp,
            "pipeline_commit": _git_commit(),
            "seconds": round(time.perf_counter() - t0, 1),
            "flip_set": {"n_keys": len(flips), "n_would_flip": sum(flips.values())},
        }
        written = release.write(sort, panel, flips, manifest, vintage)
        print(f"\n  {sort}: {len(panel):,} rows, {manifest['factors']} factors, "
              f"{manifest['months']} months; {manifest['flip_set']['n_would_flip']} of "
              f"{len(flips)} keys would flip")
        for folder, members in written.items():
            print(f"    -> {S.OUTPUT / folder}  ({len(members)} files)")

    print(f"\nStage 4 done in {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
