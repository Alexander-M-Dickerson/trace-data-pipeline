# -*- coding: utf-8 -*-
"""
test_docs.py
============
Catch documentation that has drifted away from the code.

This exists because a single audit found, in docs that all read plausibly:

  * `run_all_trace.sh` referenced 23 times. It had not existed for some time.
  * Two repository trees listing `stage0/QUICKSTART_stage0.md` and
    `stage1/requirements.txt`, neither of which exists, while omitting `config.py`,
    `FAQ.md` and `stage1/stage1_pipeline.py`, which do.
  * "Edit `_trace_settings.py` and change `WRDS_USERNAME`" -- an instruction that
    silently does nothing, because that module imports the value from `config.py`.
  * `OSBAP_Linker_*.parquet`, renamed a release earlier.
  * Feature lists still saying "Fama-French 17 and 30" after FF12 shipped.

Every one of those reads fine. None can be caught by review, only by comparison. So:
numbers the docs quote are checked against the code that defines them, and files the
docs name are checked to exist.

    python3 tests/test_docs.py

CHANGELOG.md is deliberately EXEMPT from the file-existence check. Its old entries
record what was true at the time -- an entry naming a since-deleted script is correct
history, not drift.

WHAT THIS FILE DOES NOT DO
--------------------------
It cannot tell whether a sentence is TRUE. The 2.2.3 audit found nine worked examples
whose arithmetic was reasoned out rather than run; two taught the OPPOSITE of what the
code does, and one example's stated verdict flipped when the real function was executed
on its own input. Every check here passed throughout. Prose about behaviour is verified
by RUNNING the function on the documented input and reading the output -- there is no
substitute, and no check below attempts one.

The structural checks added in 2.2.3 cover the classes that came back twice: output
trees vs what the writer actually writes, `qsub <script>` paths, TOC completeness,
in-page anchors, and OUTPUT_FORMAT. They are deliberately narrow. Each was
mutation-tested -- break the thing it guards and the suite must go red.

Author: Open Source Bond Asset Pricing
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage0"))

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name:<58} {detail}")
    if not cond:
        FAILURES.append(name)


# Docs whose claims must track the code. CHANGELOG is loaded but exempt from the
# file-existence rule; see the module docstring.
DOC_FILES = [
    "README.md", "QUICKSTART.md", "FAQ.md", "CONTRIBUTING.md",
    "stage0/README_stage0.md", "stage0/quickstart.md",
    "stage1/README_stage1.md", "stage1/QUICKSTART_stage1.md",
]

# Named in prose but produced at runtime or shipped inside a download, so they are
# legitimately absent from a fresh clone.
RUNTIME_ARTIFACTS = {
    "fl_linker.parquet", "fl_verdicts.parquet", "firm_names.parquet",
    "liu_wu_yields.xlsx", "Siccodes12.txt", "Siccodes17.txt", "Siccodes30.txt",
    "smoke_test.out", "smoke_test.err",
}

# Tokens that must not reappear: things that were renamed or deleted.
BANNED = {
    "run_all_trace": "deleted; the orchestrator is run_pipeline.sh",
    "OSBAP_Linker": "renamed in 2.1.0 to bond_firm_linker_2026/",
    "QUICKSTART_stage0": "the file is stage0/quickstart.md",
}


def check_output_trees(docs):
    """Every basename stage 0 writes must appear in the docs' output trees.

    These trees were corrected in 2.2.0 and had drifted again within days; 2.2.3 found
    QUICKSTART advertising a `stage0/<member>/reports/` folder that does not exist and
    listing 2 of the 9 files a member actually produces. Reading cannot catch that --
    only comparing the tree against the dict the writer loops over.
    """
    import ast
    src = (ROOT / "stage0" / "create_daily_enhanced_trace.py").read_text(encoding="utf-8")
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "files":
            names += [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    check("found the stage-0 output filename table in the code", len(names) >= 8,
          f"{len(names)} basenames")

    # Checked PER FILE. Pooling the docs lets one tree cover another's omission --
    # which is how QUICKSTART kept passing while listing 2 of the 9 files.
    missing, trees = [], ""
    for f in ("QUICKSTART.md", "stage0/README_stage0.md"):
        blocks = [m for m in re.findall(r"```(?:\w+)?\n(.*?)```", docs_all[f], re.S)
                  if "\u251c" in m or "\u2514" in m]
        this = "\n".join(blocks)
        trees += this + "\n"
        # only files that show a stage-0 member panel are claiming to be output trees
        if "trace_enhanced_YYYYMMDD.parquet" not in this:
            continue
        missing += [f"{f}: {n}" for n in names if n not in this]
    check("output trees name every file stage 0 writes", not missing, str(missing))

    # A member folder never holds a reports/ subfolder -- reports live in data_reports/.
    bad = re.findall(r"^\W*(enhanced|standard|144a)/.*\n(?:^\W*\w[^\n]*\n){0,10}?^\W*reports/",
                     trees, re.M)
    check("no output tree puts reports/ under a member folder", not bad, str(bad))


def check_qsub_paths(docs):
    """A `qsub <script>` in the docs must name a path that exists from the repo root.

    2.2.3: every individual-submission example said `qsub run_enhanced_trace.sh`. The
    wrappers live in stage0/ and `cd stage0` themselves, so that command works from
    neither the repository root nor stage0/.
    """
    bad = []
    for fname, text in docs_all.items():
        for m in re.finditer(r"^\s*qsub\b([^\n]*)", text, re.M):
            script = next((tok for tok in m.group(1).split() if tok.endswith(".sh")), None)
            if script and not (ROOT / script).exists():
                bad.append(f"{fname}: qsub {script}")
    check("every 'qsub <script>' in the docs resolves from the repo root",
          not bad, str(bad))


def check_toc(docs):
    """Every `## ` section must appear in its file's table of contents, and every
    in-page anchor must resolve. Both drifted silently across three releases."""
    def slug(h):
        return re.sub(r"[^\w\s-]", "", h.strip().lower()).replace(" ", "-")
    SKIP = {"Table of Contents", "Contents"}
    missing_all, dead_all = [], []
    for fname, text in docs_all.items():
        toc = re.findall(r"^\s*(?:[-*]|\d+\.)\s*\[([^\]]+)\]\(#([^)]+)\)", text, re.M)
        heads = [(len(m.group(1)), m.group(2).strip())
                 for m in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, re.M)]
        allslugs = {slug(h) for _, h in heads}
        dead_all += [f"{fname}#{m.group(1)}"
                     for m in re.finditer(r"\[[^\]]*\]\(#([^)]+)\)", text)
                     if m.group(1) not in allslugs]
        if len(toc) < 4:
            continue
        have = {a for _, a in toc}
        missing_all += [f"{fname}: {h}" for lvl, h in heads
                        if lvl == 2 and slug(h) not in have and h not in SKIP]
    check("every H2 appears in its file's table of contents", not missing_all,
          str(missing_all))
    check("every in-page anchor resolves", not dead_all, str(dead_all))


def check_output_format(docs):
    """OUTPUT_FORMAT accepts only 'parquet'. No doc may offer csv as a working option.

    Stage 0 writes .csv.gzip happily; Stage 1 and the report builder read a hard-coded
    '*.parquet'. The run then failed hours in. _trace_settings.py refuses it at import
    since 2.2.3, and the docs must not contradict that.
    """
    ts = (ROOT / "stage0" / "_trace_settings.py").read_text(encoding="utf-8")
    check("_trace_settings.py rejects a non-parquet OUTPUT_FORMAT",
          'OUTPUT_FORMAT).lower() != "parquet"' in ts)
    bad = [f for f, t in docs_all.items()
           if re.search(r'OUTPUT_FORMAT\s*=\s*"csv"', t)
           and "not supported" not in t and "only supported value" not in t]
    check("no doc presents OUTPUT_FORMAT='csv' as usable", not bad, str(bad))


def check_fences(docs):
    """Code fences must balance, and a fence must not open while one is already open.

    2.2.3 found a stray closing ``` that silently swallowed the next three paragraphs
    of stage0/quickstart.md into a code block, and an empty ```python immediately
    followed by another ```python in stage1/README_stage1.md. Both render wrong and
    neither is visible in the source.
    """
    odd, nested = [], []
    for fname, text in docs_all.items():
        fences = [(i, l.strip()[3:].strip())
                  for i, l in enumerate(text.splitlines(), 1)
                  if l.strip().startswith("```")]
        if len(fences) % 2:
            odd.append(f"{fname} ({len(fences)} fences)")
        depth = 0
        for i, info in fences:
            if depth == 0:
                depth = 1
            elif info:          # a language tag while a fence is open: the first never closed
                nested.append(f"{fname}:{i} ```{info}")
            else:
                depth = 0
    check("every doc has an even number of code fences", not odd, str(odd))
    check("no code fence opens while another is open", not nested, str(nested))


def main():
    global docs_all
    docs = {f: (ROOT / f).read_text(encoding="utf-8") for f in DOC_FILES}
    # A wider set for the structural checks: they apply to every doc that
    # carries a tree, a TOC, a qsub line or a config snippet.
    docs_all = dict(docs)
    for extra in ("stage0/README_bounce_back_filter.md",
                  "stage0/README_decimal_shift_corrector.md",
                  "stage1/README_distressed_filter.md",
                  "CONTRIBUTING.md"):
        docs_all[extra] = (ROOT / extra).read_text(encoding="utf-8")
    prose = "\n".join(docs.values())

    # --- numbers quoted in prose must match the code ------------------
    from _trace_settings import (CONCURRENCY, MAX_WRDS_CONNECTIONS,
                                 TARGET_ROWS_PER_CHUNK, qsub_resources)
    from config import TRACE_MEMBERS

    s0 = docs["stage0/README_stage0.md"]
    check("stage0 README quotes the real connection ceiling",
          f"{MAX_WRDS_CONNECTIONS} connections held simultaneously" in s0,
          f"MAX_WRDS_CONNECTIONS={MAX_WRDS_CONNECTIONS}")

    for member in ("enhanced", "144a", "standard"):
        want = qsub_resources(member)
        check(f"stage0 README quotes the real qsub request for {member}",
              want in s0, want)
        check(f"stage0 README quotes the real connection count for {member}",
              f"| `{member}` | {CONCURRENCY[member]} |" in s0,
              f"CONCURRENCY={CONCURRENCY[member]}")

    check("row target in prose matches TARGET_ROWS_PER_CHUNK",
          f"{TARGET_ROWS_PER_CHUNK:,}" in prose, f"{TARGET_ROWS_PER_CHUNK:,}")
    check("docs describe the real default TRACE_MEMBERS",
          TRACE_MEMBERS == ["enhanced", "144a"], str(TRACE_MEMBERS))

    # --- WRDS_USERNAME lives in config.py, and docs must say so -------
    ts = (ROOT / "stage0" / "_trace_settings.py").read_text(encoding="utf-8")
    imports_it = "from config import" in ts and "WRDS_USERNAME" in ts.split("from config import")[1][:120]
    check("_trace_settings.py imports WRDS_USERNAME from config.py", imports_it)
    if imports_it:
        # Any verb, not just "edit": two real instances said "Open `_trace_settings.py`"
        # and "default fallback in `_trace_settings.py`" and slipped through a check
        # written around the word "Edit".
        near = re.compile(
            r"`?_trace_settings\.py`?.{0,80}?WRDS_USERNAME"
            r"|WRDS_USERNAME.{0,80}?`?_trace_settings\.py`?")
        bad = [f for f, t in docs.items() if near.search(t)]
        check("no doc tells you to set WRDS_USERNAME in _trace_settings.py",
              not bad, str(bad))

    # --- files named in prose must exist ------------------------------
    named = set(re.findall(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|sh|md|txt|xlsx))`", prose))
    missing = sorted(
        f for f in named
        if f not in RUNTIME_ARTIFACTS
        and "*" not in f
        and not (ROOT / f).exists()
        and not (ROOT / "stage0" / f).exists()
        and not (ROOT / "stage1" / f).exists()
        and not (ROOT / "tests" / f).exists()
    )
    check("every file named in the docs exists", not missing, str(missing))

    # --- renamed/deleted things must not come back --------------------
    for token, why in BANNED.items():
        hits = sorted(f for f, t in docs.items() if token in t)
        check(f"no reference to '{token}'", not hits, f"{hits} -- {why}" if hits else why)

    # --- the machinery must be documented somewhere real --------------
    for token in ("run_smoke_test.sh", "download_inputs.sh", "_chunk_runner",
                  "_wrds_pool", "CONCURRENCY", "target_rows_per_chunk"):
        check(f"'{token}' is documented outside the CHANGELOG", token in prose)

    check_output_trees(docs)
    check_qsub_paths(docs)
    check_toc(docs)
    check_output_format(docs)
    check_fences(docs)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all documentation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
