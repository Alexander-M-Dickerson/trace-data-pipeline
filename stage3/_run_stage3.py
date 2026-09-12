"""_run_stage3.py -- the Stage-3 entry point: produce, then render.

Mirrors `stage0/_run_enhanced_trace.py` and `stage1/_run_stage1.py`. Stage 3 has two
kinds of step and they cost very different amounts:

  PRODUCERS  run portfolio sorts through PyBondLab and save return series. Minutes to
             tens of minutes; the two uncertainty grids are the long ones.
  EXHIBITS   read those series and render a table or figure. Seconds.

So a producer is SKIPPED when its output already exists. Re-running this file after a
crash picks up where it stopped; `--force` recomputes from scratch.

    python _run_stage3.py --dry-run             # print what resolved, compute nothing
    python _run_stage3.py --section lib         # one section
    python _run_stage3.py --list                # the step list, with what each needs
    python _run_stage3.py                       # everything

PyBondLab's sort kernels are DETECTED, not assumed. The runner asks the installed
engine once and passes `--fast` to the producers only if they are there, so a plain
`pip install -r requirements.txt` runs the whole of Sections 3, 4 and the zoo on the
released PyBondLab -- slower, same numbers. `--no-fast` forces the slow path even when
the kernels are present.

❗The uncertainty grids (`--section nse`) are the exception: they cannot run without the
kernels and will say so rather than starting. See README_stage3.md.

Only the inputs a SECTION reads are required, so `--section nse` does not need Stage 1's
daily panel and `--section report` needs nothing at all.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _stage3_settings as S  # noqa: E402

SECTIONS = ("data", "lib", "lab", "nse", "zoo", "report")

# (section, kind, script, args, what it writes -- relative to data/ or reports/)
STEPS = [
    # -- the data appendix ---------------------------------------------------
    ("data", "exhibit", "s0_data/tA_filter_params.py", [], "reports/tables/tableA1.tex"),
    ("data", "exhibit", "s0_data/t_ia_daily.py", [], "reports/tables/table_ia1.tex"),
    ("data", "exhibit", "s0_data/t_ia_monthly.py", [], "reports/tables/table_ia3.tex"),

    # -- Section 3: latent implementation bias -------------------------------
    ("lib", "producer", "s1_lib/run_sorts.py", [],
     "data/sorts/exc_wf_all_mmn_bgn_p2.csv"),
    ("lib", "producer", "s1_lib/run_sorts.py", ["--rating", "IG"],
     "data/sorts/exc_wf_ig_mmn_bgn_p2.csv"),
    ("lib", "producer", "s1_lib/run_sorts.py", ["--rating", "NIG"],
     "data/sorts/exc_wf_nig_mmn_bgn_p2.csv"),
    ("lib", "producer", "s1_lib/run_lib_sorts.py",
     ["--sort", "single", "--timing", "end"],
     "data/sorts/lib/bond_single_sort_lib_all_end_p10_h1.csv"),
    ("lib", "producer", "s1_lib/run_lib_sorts.py",
     ["--sort", "single", "--timing", "bgn"],
     "data/sorts/lib/bond_single_sort_lib_all_bgn_p10_h1.csv"),
    ("lib", "producer", "s1_lib/run_lib_sorts.py",
     ["--sort", "wf", "--timing", "end"],
     "data/sorts/lib/bond_within_firm_lib_all_end_p2_h1.csv"),
    ("lib", "producer", "s1_lib/run_lib_sorts.py",
     ["--sort", "wf", "--timing", "bgn"],
     "data/sorts/lib/bond_within_firm_lib_all_bgn_p2_h1.csv"),
    ("lib", "exhibit", "s1_lib/t01_table1.py", [], "reports/tables/table01.tex"),
    ("lib", "exhibit", "s1_lib/t02_validation.py", [], "reports/tables/table02.tex"),
    ("lib", "exhibit", "s1_lib/t12_rating_single.py", [], "reports/tables/table12.tex"),
    ("lib", "exhibit", "s1_lib/t13_rating_wf.py", [], "reports/tables/table13.tex"),
    ("lib", "exhibit", "s1_lib/t14_illiq.py", [], "reports/tables/table14.tex"),
    ("lib", "exhibit", "s1_lib/tB1_lib_summary.py", [], "reports/tables/tableB1.tex"),
    ("lib", "exhibit", "s1_lib/f_lib_figures.py", [], "reports/figures/fig03_cumret.pdf"),

    # -- Section 4: look-ahead bias ------------------------------------------
    ("lab", "producer", "s2_lab/run_lab.py", [],
     "data/s2_lab/series/standard__NIG__right__ts_bias_ls.parquet"),
    ("lab", "exhibit", "s2_lab/t03_affected.py", [], "reports/tables/table03.tex"),
    ("lab", "exhibit", "s2_lab/t04_table4.py", [], "reports/tables/table04.tex"),
    ("lab", "exhibit", "s2_lab/t15_decomp.py", ["--which", "mean"],
     "reports/tables/table15.tex"),
    ("lab", "exhibit", "s2_lab/t15_decomp.py", ["--which", "alpha"],
     "reports/tables/table16.tex"),
    ("lab", "exhibit", "s2_lab/f06_dua.py", [],
     "reports/figures/fig06_momentum_trim.pdf"),
    ("lab", "exhibit", "s2_lab/f_lab_figures.py", [],
     "reports/figures/fig07_lab_bias_2x2.pdf"),

    # -- Section 5: non-standard errors --------------------------------------
    ("nse", "producer", "s3_nse/run_mua_grid.py", [],
     "data/grids/mua/_complete.json"),
    ("nse", "producer", "s3_nse/mua_summarize.py", [],
     "data/s3_nse/mua_summary/_complete.json"),
    ("nse", "producer", "s3_nse/run_dua_grid.py", [],
     "data/grids/dua/_complete.json"),
    ("nse", "producer", "s3_nse/run_dua_grid.py", ["--stats"],
     "data/grids/dua/_stats_complete.json"),
    ("nse", "exhibit", "s3_nse/t05_dua_nse.py", [],
     "reports/tables/table05_paper.tex"),
    ("nse", "exhibit", "s3_nse/t06_mua_nse.py", [],
     "reports/tables/table06_paper.tex"),
    ("nse", "exhibit", "s3_nse/t17_filter_paths.py", [],
     "reports/tables/table_ia17_paper.tex"),
    ("nse", "exhibit", "s3_nse/t18_portfolio_size.py", [],
     "reports/tables/table_ia18_paper.tex"),
    ("nse", "exhibit", "s3_nse/t19_mua_improvement.py", [],
     "reports/tables/table_ia19_paper.tex"),
    ("nse", "exhibit", "s3_nse/f_nse_figures.py", [],
     "reports/figures/figIA3_nse_alpha_tstat_dua_paper.pdf"),

    # -- the factor zoo -------------------------------------------------------
    ("zoo", "producer", "s4_zoo/run_zoo_sorts.py", [],
     "data/sorts/zoo/bond_within_firm_all_p2_h1.csv"),
    ("zoo", "exhibit", "s4_zoo/t_ia08.py", [], "reports/tables/table_ia08.tex"),
    ("zoo", "exhibit", "s4_zoo/t_ia09.py", [], "reports/tables/table_ia09.tex"),
    ("zoo", "exhibit", "s4_zoo/t_ia10_11.py", ["--which", "vw"],
     "reports/tables/table_ia10.tex"),
    ("zoo", "exhibit", "s4_zoo/t_ia10_11.py", ["--which", "ew"],
     "reports/tables/table_ia11.tex"),
    ("zoo", "exhibit", "s4_zoo/t_inline.py", [],
     "reports/tables/inline_counts_alpha.tex"),

    # -- the report ------------------------------------------------------------
    # Last, and deliberately part of the run: the exhibits are LaTeX fragments, and a
    # fragment that will not compile looks perfectly fine sitting on disk. Compiling
    # is what catches it.
    ("report", "exhibit", "make_report.py", [], "reports/exhibits.pdf"),
]


# Producers that accept `--fast` (PyBondLab's sort kernels) and `--force`. The runner
# appends these itself rather than baking them into STEPS: whether the kernels are
# available is a property of the installed engine, not of the step.
# How each section spells "end the sample here". One switch at the top; the drivers
# keep their own flags, so running one by hand is unchanged.
#
# ❗`--sample frontier` is the DEFAULT: if you built a panel reaching 2025-11, the
# exhibits use it. `--sample paper` reproduces the published window (2024-12, T=268) for
# anyone checking our output against the printed tables. Every caption states which one
# it was, so a PDF is never ambiguous about its own sample.
#
# ❗Three of the 108 signals stop before the frontier because their data does
# (`b_cptlt` 2025-05, `b_dcpi`/`b_cpi_vol6` 2025-10). That is coverage, not degeneracy:
# the status ledger judges each strategy inside its own signal's span, which is why
# extending the window does not invent degenerate strategies.
SAMPLE_FLAG = {
    "s0_data/t_ia_daily.py": "--end",
    "s0_data/t_ia_monthly.py": "--end",
    "s1_lib/run_sorts.py": None,          # producers save untruncated series;
    "s1_lib/run_lib_sorts.py": None,      # the window is applied at the statistics layer
    "s1_lib/t01_table1.py": "--end",
    "s1_lib/t02_validation.py": "--end",
    "s1_lib/t12_rating_single.py": "--end",
    "s1_lib/t13_rating_wf.py": "--end",
    "s1_lib/t14_illiq.py": "--end",
    "s1_lib/tB1_lib_summary.py": "--end",
    "s1_lib/f_lib_figures.py": "--end",
    "s2_lab/run_lab.py": "--date-end",    # ❗a PRODUCER argument here: the
                                          # winsorization threshold is a full-sample
                                          # quantile, so the window cannot be applied
                                          # afterwards
    "s2_lab/f06_dua.py": "--end",       # fits its OWN sweep, so the window is an
                                        # argument here too
    "s4_zoo/t_ia09.py": "--end",
    "s4_zoo/t_ia10_11.py": "--end",
    "s4_zoo/t_inline.py": "--end",
}
# Section 5 and the report name a WINDOW rather than a date.
WINDOW_FLAG = {"paper": "paper", "frontier": "full"}

# The flag each script spells it with. ❗Two things this table fixes:
#
#   * `run_dua_grid.py` is NOT here. It computes BOTH windows in one pass from the
#     saved series and has no window flag at all. It WAS listed, and every run passed
#     anyway because that producer was always skipped -- its `_complete.json` existed,
#     so the argv was never built. The first cold run that had to produce the grid
#     died on argparse exit 2 and stopped the whole chain behind it.
#   * `mua_summarize.py` spells it `--windows`, plural, because it takes a list. The
#     orchestrator passed `--window` and it worked -- by argparse PREFIX MATCHING,
#     which would stop the day a second `--window...` option is added. Named
#     explicitly now rather than left to luck.
TAKES_WINDOW = {"s3_nse/mua_summarize.py": "--windows",
                "s3_nse/t05_dua_nse.py": "--window",
                "s3_nse/t06_mua_nse.py": "--window",
                "s3_nse/t17_filter_paths.py": "--window",
                "s3_nse/t18_portfolio_size.py": "--window",
                "s3_nse/t19_mua_improvement.py": "--window",
                "s3_nse/f_nse_figures.py": "--window",
                "make_report.py": "--window"}

def window_is_stale(script: str, want_end: str) -> bool:
    """Was this producer's output built for a DIFFERENT sample window?

    ❗Almost every producer saves untruncated series and the window is applied later,
    so re-running one is unnecessary when the window changes. Section 4 is the exception
    and it matters: its winsorization threshold is a full-sample quantile BY
    CONSTRUCTION, so the window is a producer argument there and the saved series belong
    to one specific window.

    Without this check, `--sample frontier` would find the series present, skip the
    producer, and hand every Section-4 exhibit a 2024-12 sample while the user had asked
    for 2025-11 -- quietly, because each exhibit's caption reports the window of the
    series it was given rather than the one that was requested.
    """
    import json
    if script != "s2_lab/run_lab.py":
        return False
    man = _target("data/s2_lab/series/manifest.json")
    try:
        got = json.loads(man.read_text(encoding="utf-8"))["date_end"]
    except Exception:
        # Series with no readable manifest could have been built for any window. We
        # cannot tell, so we do not get to assume it is the right one: run the
        # producer, which re-reads the manifest itself and decides per cell.
        return True
    return str(got)[:10] != str(want_end)[:10]


ACCEPTS_FAST = {"s1_lib/run_sorts.py", "s1_lib/run_lib_sorts.py",
                "s4_zoo/run_zoo_sorts.py"}
ACCEPTS_FORCE = {"s1_lib/run_sorts.py", "s1_lib/run_lib_sorts.py", "s2_lab/run_lab.py",
                 "s3_nse/run_mua_grid.py", "s3_nse/run_dua_grid.py",
                 "s4_zoo/run_zoo_sorts.py"}

# Which inputs each section actually reads. Gating every section on all five would stop
# a user who has the monthly panel but not the 2.3 GB Stage-1 daily file from running
# anything at all -- and only the data appendix reads that one.
SECTION_INPUTS = {
    "data": ("STAGE2_PANEL", "STAGE1_DAILY"),
    "lib": ("STAGE2_PANEL", "STAGE2_MMN", "STAGE2_BBW", "STAGE2_FACTORS"),
    "lab": ("STAGE2_PANEL", "STAGE2_BBW", "STAGE2_FACTORS"),
    "nse": ("STAGE2_PANEL", "STAGE2_BBW"),
    "zoo": ("STAGE2_PANEL", "STAGE2_BBW", "STAGE2_FACTORS"),
    "report": (),
}


def _target(rel: str) -> Path:
    head, rest = rel.split("/", 1)
    return (S.DATA if head == "data" else S.REPORTS) / rest


def _label(script: str, args: list[str]) -> str:
    return script + (" " + " ".join(args) if args else "")


def engine_status() -> dict:
    """Is PyBondLab importable, and does it carry the fast sort kernels?

    Resolved ONCE, here, so every step gets the same answer and a missing engine is
    reported before 40 subprocesses each discover it separately.
    """
    try:
        import pblenv
        prov = pblenv.use(quiet=True)
    except ModuleNotFoundError:
        return {"ok": False, "fast": False,
                "why": "PyBondLab is not installed. `pip install -r requirements.txt`, "
                       "or see README_stage3.md."}
    except Exception as e:                       # noqa: BLE001 -- reported, not raised
        return {"ok": False, "fast": False, "why": str(e)}
    return {"ok": True, "fast": bool(prov.get("has_fast_kernels")),
            "version": prov.get("version"), "tree": prov.get("tree_sha256"), "why": ""}


def needed_inputs(steps) -> list[str]:
    """The inputs the SELECTED steps read -- not all five."""
    want: set[str] = set()
    for section, *_ in steps:
        want |= set(SECTION_INPUTS.get(section, S.INPUTS))
    return [k for k in S.INPUTS if k in want
            and (S.INPUTS[k] is None or not Path(S.INPUTS[k]).exists())]


def would_run(steps, force: bool, sample_end: str = "") -> tuple[list, list]:
    """Split the selected steps into (would run, would skip), applying the skip rule.

    ❗Applies the SAME rule as the run loop, window-staleness included. A preview that
    disagrees with the run it previews is worse than no preview.
    """
    run, skip = [], []
    for st in steps:
        stale = bool(sample_end) and window_is_stale(st[2], sample_end)
        (run if st[1] != "producer" or force or stale or not _target(st[4]).exists()
         else skip).append(st)
    return run, skip


def print_config(args, eng: dict, missing: list[str], sample_end: str) -> None:
    unset = "(unset: export it, or edit _stage3_settings.py)"
    print("Stage 3 configuration")
    print(f"  stage 0 dir    {S.STAGE0_DIR}")
    print(f"  stage 1 dir    {S.STAGE1_DIR}")
    print(f"  stage 2 dir    {S.STAGE2_DIR}  (mode {S.MODE})")
    print(f"  data dir       {S.DATA}")
    print(f"  reports dir    {S.REPORTS}")
    for name, v in S.INPUTS.items():
        mark = "  " if v and Path(v).exists() else "??"
        print(f"  {mark} {name:15s}{v if v else unset}")
    if eng["ok"]:
        where = S.PYBONDLAB_DIR or "(installed)"
        print(f"  PyBondLab      v{eng['version']} tree={eng['tree']}  {where}")
        print(f"  sort kernels   {'yes' if eng['fast'] else 'NO -- the slow path'}")
    else:
        print(f"  PyBondLab      UNAVAILABLE -- {eng['why']}")
    # the pinned S.SAMPLE is the PAPER window and is not necessarily this run's.
    # Printing it under `--sample frontier` told the operator 2024-12 while every
    # exhibit was in fact being built to the panel's own end.
    why = ("the panel's own frontier" if args.sample == "frontier"
           else "the published window")
    print(f"  sample         {S.SAMPLE['lib']['start'][:7]} .. {sample_end[:7]}"
          f"   (--sample {args.sample}: {why})")
    print(f"  portfolios     {S.N_PORTFOLIOS}")
    print(f"  columns        {S.COLUMNS}")
    if args.section:
        print(f"  section        {args.section}")
    if missing:
        print(f"\n  MISSING INPUTS for the selected steps: {', '.join(missing)}"
              "\n  Run `python tools/check_inputs.py` for detail.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", choices=SECTIONS, help="run one section's steps")
    ap.add_argument("--only", choices=("producer", "exhibit"),
                    help="run only the producers, or only the exhibits")
    ap.add_argument("--list", action="store_true", help="list the steps and stop")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the configuration, compute nothing")
    ap.add_argument("--force", action="store_true",
                    help="rerun producers whose output already exists")
    ap.add_argument("--no-fast", action="store_true",
                    help="never use PyBondLab's sort kernels, even when available. "
                         "Much slower; useful for checking the two paths agree")
    ap.add_argument("--sample", choices=("frontier", "paper"), default="frontier",
                    help="how far the exhibits run. `frontier` (default) uses whatever "
                         "the Stage-2 panel reaches; `paper` reproduces the published "
                         "window, 2002-09 to 2024-12, T=268. Every caption states which "
                         "one produced it.")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing step instead of stopping")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    steps = [s for s in STEPS
             if (args.section is None or s[0] == args.section)
             and (args.only is None or s[1] == args.only)]

    if args.list:
        for section, kind, script, sargs, target in steps:
            have = "have" if _target(target).exists() else "   -"
            print(f"  {section:6s} {kind:8s} [{have}] {_label(script, sargs)}")
        return 0

    eng = engine_status()
    use_fast = eng["fast"] and not args.no_fast
    import drrlib as _D
    sample_end = _D.resolve_sample_end(args.sample)
    missing = needed_inputs(steps)
    print_config(args, eng, missing, sample_end)

    will_run, will_skip = would_run(steps, args.force, sample_end)
    if args.dry_run:
        print(f"\n{len(will_run)} step(s) would run, {len(will_skip)} skipped "
              "(their output already exists).")
        if not eng["ok"]:
            print("  ...but PyBondLab is unavailable, so every producer would fail.")
        elif not eng["fast"]:
            # Only mention the grids if any of them is actually in this run.
            nse = [st for st in will_run
                   if st[0] == "nse" and st[1] == "producer"]
            print("  Sort kernels are absent: the producers would take the slow "
                  "path" + (",\n  and the two uncertainty grids would refuse to "
                            "start at all." if nse else " -- same numbers, longer."))
        return 0
    if missing:
        return 1
    if not eng["ok"] and any(k == "producer" for _, k, *_ in will_run):
        print(f"\nABORT: {eng['why']}")
        return 1
    if not eng["fast"] and not args.no_fast:
        print("\n❗PyBondLab's sort kernels are not available, so the producers will\n"
              "  take the slow path -- considerably longer. `--section nse` will refuse\n"
              "  to start: the uncertainty grids need them. See README_stage3.md.")

    print(f"\nrunning {len(steps)} step(s), "
          f"{'with' if use_fast else 'WITHOUT'} the sort kernels\n")
    t0 = time.perf_counter()
    ran, skipped, failed = [], [], []
    for section, kind, script, sargs, target in steps:
        label = _label(script, sargs)
        if (kind == "producer" and not args.force and _target(target).exists()
                and not window_is_stale(script, sample_end)):
            print(f"[skip] {label}  ({target} exists)")
            skipped.append(label)
            continue
        if kind == "producer" and window_is_stale(script, sample_end):
            print(f"[rebuild] {label}  (its series were built for a different window)")
        argv = [sys.executable, script, *sargs]
        if use_fast and script in ACCEPTS_FAST:
            argv.append("--fast")
        if args.force and script in ACCEPTS_FORCE:
            argv.append("--force")
        flag = SAMPLE_FLAG.get(script)
        if flag and not any(a == flag for a in sargs):
            argv += [flag, sample_end]
        elif script in TAKES_WINDOW and not any(a.startswith("--window")
                                                for a in sargs):
            argv += [TAKES_WINDOW[script], WINDOW_FLAG[args.sample]]
        extra = [a for a in argv[2 + len(sargs):]]
        print(f"[run ] {label}" + (f"  ({' '.join(extra)})" if extra else ""))
        t = time.perf_counter()
        r = subprocess.run(argv, cwd=HERE)
        dt = time.perf_counter() - t
        if r.returncode == 0:
            ran.append((label, dt))
            print(f"[ok  ] {label}  {dt:.1f}s\n")
        else:
            failed.append((label, r.returncode))
            print(f"[FAIL] {label}  exit {r.returncode}  {dt:.1f}s\n")
            # ❗A failed PRODUCER stops the chain: everything downstream reads what it
            # was supposed to write, so continuing would build exhibits on absent or
            # stale data. A failed EXHIBIT does not -- each one is independent, and one
            # red check is no reason to abandon the other thirty-nine steps and the
            # report. The run still exits non-zero either way.
            #
            # This is not hypothetical: `t06_mua_nse.py` is step 27 of 40 and its
            # twin-invariance check fails on every run while the engine's
            # restricted-universe cells keep flipping. Stopping there silently skipped
            # Tables IA.XVII-IA.XIX, the Section-5 figures, the whole zoo and the PDF --
            # from the command the README calls "everything".
            if kind == "producer" and not args.keep_going:
                print("  a producer failed, so everything downstream would read "
                      "missing or stale data.\n  Stopping. Use --keep-going to "
                      "override.")
                break

    wall = time.perf_counter() - t0
    print(f"\n{len(ran)} ran, {len(skipped)} skipped, {len(failed)} failed "
          f"in {wall / 60:.1f} min")
    if ran:
        for label, dt in sorted(ran, key=lambda x: -x[1])[:5]:
            print(f"  slowest: {dt:7.1f}s  {label}")
    for label, code in failed:
        print(f"  FAILED exit {code}: {label}")
    print(f"\ntables  -> {S.REPORTS / 'tables'}"
          f"\nfigures -> {S.REPORTS / 'figures'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
