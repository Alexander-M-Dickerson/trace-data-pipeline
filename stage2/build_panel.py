"""build_panel.py -- the Stage 2 orchestrator: the monthly asset-pricing panel from the daily
input, driven by _stage2_settings. Users run `_run_stage2.py`, which calls `run_stage2` here;
running this file directly is for development.

    python build_panel.py                       # full build, all steps
    python build_panel.py --from-step 4         # resume (earlier blocks reused from disk)
    python build_panel.py --limit-cusips 200    # small-panel dev build
    python build_panel.py --validate            # coverage check on the finished panel

Steps (each writes its blocks under output/blocks/<mode>/ and is independently re-runnable):
  1 returns  2 illiquidity  3 bbw  4 betas  5 value  6 momentum  7 final merge -> panel/
A run manifest (manifests/monthly_<ts>_<mode>.json) records inputs, config, timings and -- with
--validate -- the full per-column validation reports.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _stage2_settings as cfg
import db as mdb
from lib import manifest as mf

STEPS: list[tuple[int, str, str, bool]] = [
    # (number, module, blurb, needs_con)
    (1, "step1_returns", "monthly returns + signals", True),
    (2, "step2_illiquidity", "illiquidity/risk signals + factors", True),
    (3, "step3_bbw", "BBW 4-factor sorts", False),
    (4, "step4_betas", "factor matrix + rolling betas", False),
    (5, "step5_value", "value signals + d-spreads", False),
    (6, "step6_momentum", "momentum/LTR + VaR", False),
    (7, "step7_final", "final merge -> main panel", False),
]

# After step 2 the step DAG forks into two independent chains -- {3: bbw -> 4: betas} and
# {5: value -> 6: momentum} share NO blocks (3/4 read step-1+2 blocks + quote; 5/6 read step-1
# blocks + quote; 7 joins everything) -- so a full build runs them as two concurrent subprocesses
#. Outputs are byte-identical either way; --sequential disables the overlap.
PARALLEL_CHAINS: tuple[tuple[int, int], ...] = ((3, 4), (5, 6))

# Fresh-process orchestration for FULL builds: DuckDB's parallelism collapses in a
# long-lived process that already ran a heavy unit. Measured: step 2's SQL phase is ~25 s in a fresh process but ~91 s in-process after
# step 1. So a full build runs each STAGE as child orchestrator processes: the DuckDB steps as
# sequential singletons, the {3,4}/{5,6} chains concurrently, the final merge last.
FULL_PLAN: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, 1),), ((2, 2),), PARALLEL_CHAINS, ((7, 7),))


# [tag:rule.fresh_process] each group of steps runs in a fresh process; a long-lived one makes DuckDB lose its parallelism
def _run_stage(ranges: tuple[tuple[int, int], ...], input_mode: str,
               limit_cusips: int | None, manifest) -> None:
    """Run one stage's step ranges as concurrent child orchestrator processes (fresh per range)."""
    import subprocess
    import sys
    t0 = time.time()
    procs = []
    for lo, hi in ranges:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--input-mode", input_mode, "--from-step", str(lo), "--to-step", str(hi),
               "--factor-source", cfg.FACTOR_SOURCE, "--sequential"]
        if limit_cusips:
            cmd += ["--limit-cusips", str(limit_cusips)]
        log = cfg.OUTPUT_DIR / "logs" / f"chain_{lo}{hi}_{input_mode}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[stage] steps {lo}-{hi} -> {log.name}", flush=True)
        fh = open(log, "w")
        procs.append((lo, hi, log, fh,
                      subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                       cwd=Path(__file__).resolve().parent)))
    failed = []
    for lo, hi, log, fh, p in procs:
        rc = p.wait()
        fh.close()
        if rc != 0:
            failed.append((lo, hi, log))
    wall = time.time() - t0
    if failed:
        for lo, hi, log in failed:
            print(f"[stage] steps {lo}-{hi} FAILED -- see {log}", flush=True)
        raise RuntimeError(f"stage failed: {failed}")
    tag = "+".join(f"{lo}-{hi}" for lo, hi in ranges)
    print(f"[stage] steps {tag} done in {wall:.0f}s", flush=True)
    manifest.add_gate(f"stage_steps{tag}", status="BUILT", wall_s=wall,
                      note="fresh child process per range; per-step metas under blocks/")


def run(input_mode: str, from_step: int, to_step: int, limit_cusips: int | None,
        validate: bool, factor_source: str | None = None,
        refresh_factors: bool = False, sequential: bool = False) -> None:
    cfg.INPUT_MODE = input_mode
    cfg.FACTOR_SOURCE = factor_source or cfg.factor_source_for(input_mode)
    cfg.ensure_dirs()
    # A full build's steps 3-4 and 5-6 start as two child processes in the same second, and
    # the run id was only that second-resolution stamp, so one child's manifest overwrote the
    # other's. A partial run's id now names its steps.
    manifest = mf.RunManifest(input_mode=input_mode,
                              tag=None if (from_step, to_step) == (1, 7)
                              else f"steps{from_step}-{to_step}")
    manifest.add_input("daily", cfg.daily_input(input_mode))
    for role in ("linker", "call", "fisd"):
        path = cfg.AUX.get(role)
        if path is not None:
            manifest.add_input(role, path)

    # materialize the factor-panel seam (blocks/<mode>/factors.parquet) BEFORE the steps that read
    # it (4 and 7): fetched fresh from the public sources by default, or the published file
    # with --factor-source pinned
    from steps import compute_factors
    factors_path = compute_factors.ensure(input_mode, force_fetch=refresh_factors)
    manifest.add_input(f"factors_{cfg.FACTOR_SOURCE}", factors_path)

    if not sequential and from_step == 1 and to_step == 7:
        # FULL build: staged fresh-process plan -- DuckDB steps in their own
        # processes, the {3,4}/{5,6} chains concurrent, final merge last
        for stage in FULL_PLAN:
            _run_stage(stage, input_mode, limit_cusips, manifest)
    else:
        # partial/sequential build: in-process; overlap the chains only if the range covers 3..6
        chain_lo = min(lo for lo, _ in PARALLEL_CHAINS)
        chain_hi = max(hi for _, hi in PARALLEL_CHAINS)
        use_chains = not sequential and from_step <= chain_lo and to_step >= chain_hi
        chains_done = False

        con = mdb.connect()
        import importlib
        for num, mod_name, blurb, needs_con in STEPS:
            if not (from_step <= num <= to_step):
                continue
            if use_chains and chain_lo <= num <= chain_hi:
                if not chains_done:
                    _run_stage(PARALLEL_CHAINS, input_mode, limit_cusips, manifest)
                    chains_done = True
                continue
            mod = importlib.import_module(f"steps.{mod_name}")
            t0 = time.time()
            print(f"[step {num}] {blurb} ...", flush=True)
            blocks = mod.build(con, mode=input_mode, limit_cusips=limit_cusips) if needs_con \
                else mod.build(mode=input_mode, limit_cusips=limit_cusips)
            wall = time.time() - t0
            print(f"[step {num}] done in {wall:.0f}s", flush=True)
            manifest.add_gate(f"step{num}_{mod_name}", status="BUILT", wall_s=wall,
                              output={n: str(p) for n, p in blocks.items()})

    # --validate. A public build has no reference panel to diff against, so it runs the check
    # that does apply: every column reaches the panel's last month. The "golden" branch diffs
    # against a reference build and runs only when one is configured (GOLDEN_OUTPUTS).
    if validate and input_mode == "golden":
        import validate_stage2
        all_pass = True
        for step_name in validate_stage2._specs():
            try:
                passed, report = validate_stage2.validate_step(step_name)
            except FileNotFoundError as exc:
                print(f"[validate:{step_name}] skipped: {exc}")
                continue
            all_pass &= passed
            manifest.add_gate(f"validate_{step_name}", status="PASS" if passed else "FAIL",
                              wall_s=0.0, validation=report)
        print("[validate] overall:", "PASS" if all_pass else "FAIL")
    elif validate:
        import validate_coverage
        built = cfg.PANEL_DIR / f"main_panel_{input_mode}.parquet"
        if built.exists():
            ok, report = validate_coverage.check_coverage(built)
            print(validate_coverage.format_report(report))
            manifest.add_gate("validate_coverage", status="PASS" if ok else "FAIL",
                              wall_s=0.0, validation=report)
            print("[validate] coverage:", "PASS" if ok else "FAIL")
        else:
            print(f"[validate] skipped: no panel at {built}")

    panel = cfg.PANEL_DIR / f"main_panel_{input_mode}.parquet"
    if panel.exists():
        import pyarrow.parquet as pq
        md = pq.ParquetFile(panel).metadata
        manifest.set_final(panel, rows=md.num_rows, cols=md.num_columns,
                           matches_golden=validate and input_mode == "golden")
    out = manifest.write()
    print(f"manifest: {out}")


def run_stage2(config: dict, *, from_step: int = 1, to_step: int = 7,
               limit_cusips: int | None = None, refresh_factors: bool = False,
               sequential: bool = False, validate: bool = False) -> None:
    """Entry point used by _run_stage2.py.

    Adapts the validated configuration dict to the orchestrator's argument shape.
    """
    run(config.get("input_mode", cfg.INPUT_MODE), from_step, to_step, limit_cusips,
        validate, factor_source=config.get("factor_source"),
        refresh_factors=refresh_factors, sequential=sequential)


def main() -> None:
    ap = argparse.ArgumentParser(description="daily -> monthly asset-pricing panel (stage2 port)")
    # The mode is a LABEL: it names the blocks/<mode>/ subdirectory. The reference used it
    # to pick between a frozen vintage and a live build; the public pipeline has one input.
    ap.add_argument("--input-mode", default=cfg.INPUT_MODE)
    ap.add_argument("--from-step", type=int, default=1, choices=range(1, 8))
    ap.add_argument("--to-step", type=int, default=7, choices=range(1, 8))
    ap.add_argument("--limit-cusips", type=int, default=None,
                    help="dev universe: first N cusips (deterministic)")
    ap.add_argument("--validate", action="store_true",
                    help="after the build, check that every column reaches the panel's last "
                         "month")
    ap.add_argument("--factor-source", choices=["pinned", "public"], default=None,
                    help="where the factor panel comes from. 'public' (the default) fetches "
                         "every source fresh; 'pinned' uses the file published for the vintage, "
                         "which reproduces a published panel exactly")
    ap.add_argument("--refresh-factors", action="store_true",
                    help="with --factor-source public: force re-fetch of every source")
    ap.add_argument("--sequential", action="store_true",
                    help="run the {3,4} and {5,6} step chains one after the other, not side by "
                         "side")
    args = ap.parse_args()
    run(args.input_mode, args.from_step, args.to_step, args.limit_cusips, args.validate,
        factor_source=args.factor_source, refresh_factors=args.refresh_factors,
        sequential=args.sequential)


if __name__ == "__main__":
    main()
