r"""t_ia08.py -- Table IA.VIII, Signal Definitions and Citations.

One row per column of the 140-column Stage-2 panel: the 108 sorted signals plus 32
identifiers, returns and characteristics, each with its definition and its citation.

This is the only exhibit in Stage 3 whose content is a SPEC rather than a computation.
`spec/signal_definitions.json` holds the rows; this file renders them as the paper's
longtable. That split is the point: the definitions can be checked against Stage 2's
dictionary and against the code by anyone, in a file that is not LaTeX.

    python s4_zoo/t_ia08.py                 # -> reports/tables/table_ia08.tex
    python s4_zoo/t_ia08.py --diffs         # print only the rows the paper prints
                                            #   differently, and why

Six rows differ from the printed table. Each carries `why_corrected` in the spec and is
listed in `RECONCILIATION_ia08.md`; each was settled by the code or by the paper
contradicting itself, never by preference. The table footnotes them, so a reader of the
PDF is told rather than left to diff two documents.

Reads:  spec/signal_definitions.json
Writes: reports/tables/table_ia08.tex, data/s4_zoo/table_ia08.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import captions             # noqa: E402
import drrlib as D          # noqa: E402
import paths                # noqa: E402
import zoo_engine as Z      # noqa: E402
from bench import Bench     # noqa: E402

LABEL = "tab:signal_definitions"
SPEC = Path(__file__).resolve().parents[1] / "spec" / "signal_definitions.json"


def load() -> list[dict]:
    if not SPEC.exists():
        raise SystemExit(f"Table IA.VIII needs its spec and it is not at {SPEC}.")
    return json.loads(SPEC.read_text(encoding="utf-8"))["rows"]


# The paper's Figure 2 is a LaTeX schematic with no data behind it, so Stage 3 does
# not produce it and this document has no such label. 11 rows point at it. Leaving the
# \ref would print "??" on eleven lines of a shipped table, so it is rendered as the
# words instead -- the spec keeps the paper's own markup.
CROSSREF = {
    r"Fig. \ref{fig:return_timeline}": "Figure 2 of the paper",
    r"\ref{fig:return_timeline}": "Figure 2 of the paper",
}


def tex_escape(x: str) -> str:
    """Escape underscores and ampersands in plain text, leaving LaTeX alone.

    The spec holds real LaTeX -- `$\\log(cs_{t-6})$`, `\\citet{bali2021long}` -- so a
    blanket escape would corrupt it. A macro and its braced arguments copy through
    untouched; so does anything between dollars. (An earlier version copied `\\ref` but
    not its argument, which turned `\\ref{fig:return_timeline}` into
    `\\ref{fig:return\\_timeline}` and broke the reference eleven times.)
    """
    for ref, words in CROSSREF.items():
        x = x.replace(ref, words)
    out, i, in_math = [], 0, False
    while i < len(x):
        c = x[i]
        if c == "$":
            in_math = not in_math
            out.append(c)
            i += 1
        elif c == "\\":
            j = i + 1
            while j < len(x) and x[j].isalpha():
                j += 1
            if j == i + 1 and j < len(x):     # an escaped character, e.g. \% \& \_
                j += 1
            while j < len(x) and x[j] == "{":  # ...and every braced argument
                depth = 0
                while j < len(x):
                    if x[j] == "{":
                        depth += 1
                    elif x[j] == "}":
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                    j += 1
            out.append(x[i:j])
            i = j
        elif in_math:
            out.append(c)
            i += 1
        else:
            out.append({"_": r"\_", "&": r"\&", "%": r"\%"}.get(c, c))
            i += 1
    return "".join(out)


def render_latex(rows: list[dict]) -> str:
    corrected = [r for r in rows if "why_corrected" in r]
    L = [
        r"\begin{longtable}{>{\ttfamily}p{2.2cm} p{3.8cm} p{7.5cm} p{2.8cm}}",
        r"\caption{" + captions.caption(LABEL) + r"} \label{" + LABEL + r"} \\",
        r"\toprule",
        r"\normalfont\textbf{Mnemonic} & \textbf{Name} & \textbf{Description} "
        r"& \textbf{Citation} \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\multicolumn{4}{c}{\textit{Table \ref{" + LABEL + r"} continued from "
        r"previous page}} \\",
        r"\toprule",
        r"\normalfont\textbf{Mnemonic} & \textbf{Name} & \textbf{Description} "
        r"& \textbf{Citation} \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{4}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        "",
        r"\bottomrule",
    ]
    if corrected:
        L += [
            r"\multicolumn{4}{p{16cm}}{\footnotesize $^{\dagger}$ This row differs "
            r"from the printed Table IA.VIII. In each case the code or the paper's own "
            r"other rows settle it; the spec records what was printed and why it was "
            r"changed.} \\",
        ]
    L += [r"\endlastfoot", ""]

    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            if L[-1] != "":
                L.append(r"\addlinespace[0.5em]")
            L += [r"\multicolumn{4}{l}{\textbf{" + tex_escape(group) + r"}} \\",
                  r"\midrule"]
        mark = r"$^{\dagger}$" if "why_corrected" in r else ""
        # ❗the citation column carries WORDS, not \citet. This document is a set of
        # tables with no bibliography, so a key renders as "[?]" -- and adding a
        # bibliography would mean shipping the paper's .bib with the pipeline. The
        # author-year text is resolved from that .bib once, into the spec.
        L.append(" & ".join([
            tex_escape(r["mnemonic"]) + mark,
            tex_escape(r["name"]),
            tex_escape(r["description"]),
            tex_escape(r.get("citation_text") or "--"),
        ]) + r" \\")
    L += [r"\addlinespace[0.5em]", r"\end{longtable}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diffs", action="store_true",
                    help="print the rows that differ from the printed table, and why")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = load()
    if args.diffs:
        for r in rows:
            if "why_corrected" not in r:
                continue
            print(f"\n{r['mnemonic']}  ({r['group']})")
            for k in ("paper_prints", "paper_prints_citation"):
                if k in r:
                    print(f"  printed : {r[k]}")
            print(f"  here    : "
                  f"{r['citation'] if 'paper_prints_citation' in r else r['description']}")
            print(f"  why     : {r['why_corrected']}")
        return 0

    t0 = time.perf_counter()
    with Bench("table_ia08", section="s4_zoo", sample=not args.no_bench,
               echo=True) as b:
        with b.phase("render"):
            out = paths.section_results("s4_zoo")
            tex = paths.TABLES / "table_ia08.tex"
            tex.write_text(render_latex(rows), encoding="utf-8")
            import pandas as pd
            pd.DataFrame(rows).to_csv(out / "table_ia08_cells.csv", index=False)
            n_fix = sum(1 for r in rows if "why_corrected" in r)
            D.write_result(
                "table_ia08",
                {"summary": {
                    "exhibit": "Table IA.VIII", "tex_label": LABEL,
                    "n_rows": len(rows),
                    "n_sorted_signals": sum(1 for r in rows
                                            if r["mnemonic"] in Z.CLUSTER_OF),
                    "n_corrected": n_fix,
                    "n_cited": sum(1 for r in rows if r["citation_keys"]),
                    # ❗no sample: this exhibit is a definition list. Its content
                    # does not depend on the window, and saying otherwise would be a
                    # sentence with nothing behind it.
                    "sample": None},
                 "rows": rows},
                section="s4_zoo", inputs=[SPEC], t0=t0,
                extra={"exhibit": "Table IA.VIII", "tex_label": LABEL})
        b.note(n_rows=len(rows), n_corrected=n_fix)

        # Every signal Stage 3 sorts must have a definition here, or the table
        # documents a different universe from the one the paper reports on.
        missing = sorted(set(Z.CLUSTER_OF) - {r["mnemonic"] for r in rows})
        ok = b.check(not missing,
                     f"{len(Z.CLUSTER_OF)} sorted signals all defined"
                     + (f"; MISSING {missing}" if missing else ""))

    print(f"\nTable IA.VIII: {len(rows)} rows, {n_fix} differing from the printed table")
    print(f"wrote {tex}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
