r"""build_index.py -- generate INDEX.md, the map from a paper exhibit to the code.

    python tools/build_index.py            # write stage3/INDEX.md
    python tools/build_index.py --check    # fail if regenerating would change it

Why generated and not written by hand: a hand-kept list of which file makes Table 7 is
stale the first time a driver is renamed, and nothing notices because the run still
works. Every field here comes from something the code already holds --

  * `make_report.EXHIBITS`   the paper's exhibit numbers, in the order the PDF prints
  * `captions.CAPTIONS`      the caption title, keyed by LaTeX label
  * `_run_stage3.STEPS`      which driver the orchestrator runs, and with which flags
  * `data/<section>/*.json`  each result's own manifest: driver, label, sample, inputs

-- so the only way for a row to be wrong is for the run itself to be wrong.

The generator also closes a gap the test suite had: it checks that every exhibit stem
`make_report` expects is reachable from a step. Rename a figure and the suite used to
stay green; you found out from a "Not produced" stub at the end of a 14-minute run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
sys.path.insert(0, str(STAGE3))

import _run_stage3 as R        # noqa: E402
import captions as C           # noqa: E402
import make_report as MR       # noqa: E402

OUT = STAGE3 / "INDEX.md"

# The paper's section for each folder. The folder numbers are NOT the section numbers
# and never have been, which is the single most common way to misread this tree.
SECTION_OF = {
    "s0_data": ("Data appendix", "the daily and monthly panels as delivered"),
    "s1_lib": ("Section 3", "look-ahead bias in the published factor set"),
    "s2_lab": ("Section 4", "look-ahead bias measured under ex-post filtering"),
    "s3_nse": ("Section 5", "non-standard errors across the two uncertainty grids"),
    "s4_zoo": ("Factor zoo", "the 108-signal census and its cluster tables"),
}


def manifests() -> dict[str, dict]:
    """{result name: manifest block}, over every section's result JSONs."""
    out = {}
    for p in sorted((STAGE3 / "data").glob("*/*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        man = doc.get("manifest")
        if isinstance(man, dict) and man.get("name"):
            man = dict(man)
            man["_summary"] = doc.get("summary") if isinstance(
                doc.get("summary"), dict) else {}
            out[man["name"]] = man
    return out


def steps_by_driver() -> dict[str, list[tuple[str, list[str], str]]]:
    """{driver path: [(section, args, target), ...]} from the orchestrator's own table."""
    out: dict[str, list] = {}
    for section, kind, script, sargs, target in R.STEPS:
        out.setdefault(script, []).append((section, kind, sargs, target))
    return out


def driver_sources() -> dict[str, str]:
    """{driver path: source}, over the section drivers only.

    `make_report.py` names every stem because it assembles them, so including it would
    make every lookup ambiguous; the tests likewise.
    """
    return {p.relative_to(STAGE3).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(STAGE3.glob("s?_*/*.py"))}


def driver_by_scan(concrete: list[str], src: dict[str, str]) -> str:
    """Which driver file mentions this exhibit's stem.

    The fallback for exhibits whose driver writes several: one `write_result` records
    one name, so a manifest can only speak for the first. A stem is a literal in the
    code that writes it, which makes this exact -- with one boundary rule: `table_ia1`
    is a prefix of `table_ia10`, `table_ia11`, `table_ia17`..., so a match must not be
    followed by another digit.
    """
    hits = set()
    for c in concrete:
        base = c.replace("_paper", "").replace("_full", "")
        pat = re.compile(re.escape(base) + r"(?![0-9])")
        for name, s in src.items():
            if pat.search(s):
                hits.add(name)
    return sorted(hits)[0] if len(hits) == 1 else ""


def label_of(stem: str, man: dict | None) -> str:
    """The LaTeX label, from the manifest if it recorded one, else from the .tex."""
    if man and man.get("tex_label"):
        return man["tex_label"]
    tex = STAGE3 / "reports" / "tables" / f"{stem}.tex"
    if tex.exists():
        m = re.search(r"\\label\{([^}]+)\}", tex.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return ""


def sample_of(man: dict | None) -> str:
    """What the exhibit says about its own sample, from the block it recorded."""
    if not man:
        return ""
    blk = man.get("_summary", {}).get("sample")
    if not isinstance(blk, dict):
        return ""
    import drrlib as D
    return D.sample_sentence(blk).replace("Sample: ", "").rstrip(".")


def resolve(stem: str) -> tuple[str, list[str]]:
    """A `{window}` stem stands for two real ones. Return (display, [concrete stems]).

    `full` comes first because it is what `--sample frontier`, the default, produces --
    so the sample this index reports for such a row is the one a fresh run just wrote.
    """
    if "{window}" not in stem:
        return stem, [stem]
    return (stem.replace("_{window}", " (x2)"),
            [stem.format(window=w) for w in ("full", "paper")])


def rows() -> tuple[list[dict], list[str]]:
    """One row per exhibit the report expects, plus any stem nothing produces."""
    mans = manifests()
    src = driver_sources()
    # a manifest names its own driver; invert that so a stem can find its code
    driver_of = {name: m.get("driver") for name, m in mans.items()}

    # a figure driver writes one result for all of its figures, under its own tag
    # rather than a per-figure stem, so a figure's sample is found through its driver
    by_driver = {}
    for m in mans.values():
        if m.get("driver") and isinstance(m.get("_summary", {}).get("sample"), dict):
            by_driver.setdefault(m["driver"], m)

    out, orphans = [], []
    for group, group_title, items in MR.EXHIBITS:
        for number, kind, stem, note in items:
            display, concrete = resolve(stem)
            first = next((c for c in concrete if c in mans), concrete[0])
            man = mans.get(first)
            # the manifest is authoritative when there is one; a driver that writes
            # several exhibits records only the first, so the rest come from the scan
            driver = driver_of.get(first) or driver_by_scan(concrete, src)
            man = man or by_driver.get(driver)
            if not driver:
                orphans.append(f"{number} ({display})")
            sec = (man or {}).get("section") or (
                driver.split("/")[0] if "/" in driver else "")
            out.append({
                "group": group, "group_title": group_title,
                "number": number, "kind": kind, "stem": display,
                "label": label_of(first, man),
                "caption": C.CAPTIONS.get(label_of(first, man), ""),
                "driver": driver, "section": sec,
                "sample": sample_of(man),
                "note": note,
                "produced": any(c in mans for c in concrete)
                           or _artifact_exists(kind, concrete),
            })
    return out, orphans


def _artifact_exists(kind: str, stems: list[str]) -> bool:
    d = STAGE3 / "reports" / ("tables" if kind == "table" else "figures")
    ext = ".tex" if kind == "table" else ".pdf"
    return any((d / f"{s}{ext}").exists() for s in stems)


def render(rs: list[dict]) -> str:
    L = [
        "# Stage 3 exhibit index",
        "",
        "**Generated by `tools/build_index.py`. Do not edit by hand** -- "
        "`python tools/build_index.py --check` fails if this file and the code disagree.",
        "",
        "One row per exhibit the report prints: the paper's number, the LaTeX label, "
        "the file that produces it, and what that file says about its own sample. "
        "Every field is read from the code or from a result's manifest, so a renamed "
        "driver shows up here rather than as a gap in the PDF.",
        "",
        "## How the tree is laid out",
        "",
        "\u2757**The folder numbers are not the paper's section numbers.** `s1_lib` is "
        "Section 3, `s2_lab` is Section 4, `s3_nse` is Section 5. This has caught "
        "everyone who has read the tree.",
        "",
        "| folder | paper | what it covers |",
        "|---|---|---|",
    ]
    for folder, (sec, what) in SECTION_OF.items():
        L.append(f"| `{folder}` | {sec} | {what} |")
    L += [
        "",
        "Shared code sits at the top level: `drrlib.py` (statistics, sample "
        "provenance, result manifests), `captions.py` (every caption title), "
        "`bench.py` (the timing/check ledger), `paths.py`, `fastrun.py` (process "
        "parallelism), `pblenv.py` (which PyBondLab build is active), "
        "`_stage3_settings.py` (paths and constants) and `_run_stage3.py` (the "
        "orchestrator). `make_report.py` assembles the PDF.",
        "",
        "## Reading a row",
        "",
        "- **sample** is what the exhibit's own caption states, derived from the data "
        "it used. `T=279` means a single asserted length; `T 268-279 by series` means "
        "each series is its own length; a path count means the exhibit counts "
        "construction paths rather than months.",
        "- **(x2)** in a stem means the exhibit ships in two variants, one per sample "
        "window (`_paper` and `_full`).",
        "- an empty sample cell means the exhibit has none by nature -- a parameter "
        "table, or a classification list.",
        "",
    ]
    for group, group_title, _ in MR.EXHIBITS:
        L += [f"## {group_title}", "",
              "| exhibit | label | driver | sample | stem |",
              "|---|---|---|---|---|"]
        for r in (x for x in rs if x["group"] == group):
            mark = "" if r["produced"] else " \u26a0 not produced"
            lab = f"`{r['label']}`" if r["label"] else "--"
            L.append(f"| {r['number']}{mark} | {lab} | `{r['driver']}` | "
                     f"{r['sample'] or '--'} | `{r['stem']}` |")
        L.append("")
        cap = [x for x in rs if x["group"] == group and x["caption"]]
        if cap:
            L += ["<details><summary>caption titles</summary>", ""]
            for r in cap:
                L.append(f"- **{r['number']}** -- {r['caption']}")
            L += ["", "</details>", ""]

    # --- the same 44 rows the other way round: one entry per FILE ------------
    # An exhibit number is what a reader of the paper has; a filename is what a reader
    # of the repo has. Both need to reach the other in one step, and a driver that
    # makes four figures is invisible in the tables above.
    L += ["## By driver", "",
          "The same exhibits keyed on the file instead of the paper's numbering, in the "
          "order the run reaches them. An exhibit number is what a reader of the "
          "paper has; a filename is what a reader of the repo has.", "",
          "| driver | kind | produces |", "|---|---|---|"]
    seen: dict[str, list] = {}
    for r in rs:
        seen.setdefault(r["driver"], []).append(r["number"])
    order = {s: i for i, (_sec, _k, s, _a, _t) in enumerate(R.STEPS)}
    for drv in sorted(seen, key=lambda d: (order.get(d, 999), d)):
        kind = next((k for _s, k, s, _a, _t in R.STEPS if s == drv), "exhibit")
        L.append(f"| `{drv}` | {kind} | " + ", ".join(seen[drv]) + " |")
    L += ["",
          "Producers are the expensive half -- they run sorts through PyBondLab and "
          "save return series, and are skipped when their output already exists. "
          "Exhibits read those series and render in seconds. `python _run_stage3.py "
          "--list` prints all 41 steps with their arguments.",
          ""]

    # --- what each producer reads and writes ---------------------------------
    # Scanned from the source, not from a manifest: a manifest only exists after a
    # run, and this file is checked by a test that must pass on a fresh clone.
    INPUT_NAMES = {"PANEL": "the Stage-2 monthly panel",
                   "MMN": "the MMN price-based signals",
                   "BBW": "the BBW factor series",
                   "FACTORS": "the factor file",
                   "DAILY": "the Stage-1 daily panel"}
    L += ["## What the producers read and write", "",
          "Scanned from each producer's source. The five external files are pinned in "
          "`spec/inputs.json` and checked by `tools/check_inputs.py` before a run "
          "starts; everything else is another Stage-3 step's output.",
          "", "| producer | reads | writes |", "|---|---|---|"]
    sources = driver_sources()
    prod = {}
    for _sec, kind, script, _a, target in R.STEPS:
        if kind == "producer":
            prod.setdefault(script, set()).add(str(Path(target).parent).replace("\\", "/"))
    for script in sorted(prod, key=lambda d: order.get(d, 999)):
        body = sources.get(script, "")
        reads = [f"{INPUT_NAMES[k]} (`paths.{k}`)" for k in INPUT_NAMES
                 if f"paths.{k}" in body]
        if "paths.GRIDS" in body or "grids/" in body:
            reads.append("the grids under `data/grids/`")
        if "section_results" in body and not reads:
            reads.append("another step's `data/<section>/` output")
        L.append(f"| `{script}` | " + ("; ".join(reads) or "--") + " | "
                 + ", ".join(f"`{w}/`" for w in sorted(prod[script])) + " |")
    L.append("")

    L += [
        "## Figures the paper draws in LaTeX",
        "",
        f"{MR.NOT_DATA_FIGURES} are schematics -- a research framework, a return "
        "timeline and a look-ahead illustration. They have no data behind them, so "
        "Stage 3 does not produce them.",
        "",
        "## Running one exhibit",
        "",
        "Every driver runs on its own, from `stage3/`:",
        "",
        "```bash",
        "python s1_lib/t01_table1.py          # one exhibit",
        "python _run_stage3.py --section lib  # one section, producers included",
        "python _run_stage3.py                # everything, ending in the PDF",
        "```",
        "",
        "A driver reads what a producer wrote; if the producer has not run, the driver "
        "says so and names the command. `_run_stage3.py --dry-run` lists what would run.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if INDEX.md is not what this would write")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rs, orphans = rows()
    text = render(rs)

    if args.check:
        have = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if have != text:
            print("INDEX.md is out of date. Run `python tools/build_index.py`.")
            return 1
        print(f"INDEX.md is current ({len(rs)} exhibits).")
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}  ({len(rs)} exhibits)")

    missing = [r["number"] for r in rs if not r["produced"]]
    if missing:
        print(f"  not produced yet: {', '.join(missing)}")
    if orphans:
        print("  no driver found for: " + ", ".join(orphans))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
