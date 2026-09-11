r"""make_report.py -- assemble the generated exhibits into one compiled PDF.

Stage 3 writes 32 table fragments and 11 figures. On their own they are 43 loose files;
this puts them in the paper's own order, under the paper's own exhibit numbers, with a
provenance page saying which panel and which engine produced them -- and compiles it.

    python make_report.py              # exhibits.tex + exhibits.pdf
    python make_report.py --no-compile # write the .tex only
    python make_report.py --clean      # remove the LaTeX intermediates

❗The exhibit NUMBERS here are the paper's (Table IA.XII, Figure IA.3), not LaTeX's
sequential ones. Automatic numbering is suppressed and each exhibit is announced by its
own heading, so a cross-reference to "Table 4" means the same table it does in the paper.

❗These are YOUR numbers, from whatever panel Stage 2 built. They are not the paper's
printed numbers and nothing here compares the two -- the title page says so, because a
PDF of tables with familiar captions is exactly the kind of artifact that gets mistaken
for the original.

Needs `pdflatex` on PATH (any TeX distribution) and these packages, all standard:
booktabs, longtable, colortbl, xcolor, graphicx, amsmath, geometry, caption, hyperref.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _stage3_settings as S  # noqa: E402
import paths                  # noqa: E402

# (paper's exhibit number, kind, file stem, one-line note or "")
# Order is the paper's. A stem that is not on disk is reported, never silently dropped.
EXHIBITS = [
    ("main", "Main text", [
        ("Table 1", "table", "table01", ""),
        ("Table 2", "table", "table02", ""),
        ("Table 3", "table", "table03", ""),
        ("Table 4", "table", "table04", ""),
        ("Table 5", "table", "table05_{window}", ""),
        ("Table 6", "table", "table06_{window}", ""),
        ("Figure 3", "figure", "fig03_cumret", ""),
        ("Figure 4", "figure", "fig04_bias", ""),
        ("Figure 6", "figure", "fig06_momentum_trim", ""),
        ("Figure 7", "figure", "fig07_lab_bias_2x2", ""),
        ("Figure 8", "figure", "fig08_lab_cumret_2x2", ""),
    ]),
    ("appA", "Appendix A -- cleaning-filter parameters", [
        ("Table A.1", "table", "tableA1", ""),
        ("Table A.2", "table", "tableA2", ""),
        ("Table A.3", "table", "tableA3", ""),
    ]),
    ("appB", "Appendix B -- the 108-factor census", [
        ("Table B.1", "table", "tableB1",
         "As published: the stored series are differenced WITHOUT undoing the "
         "extract-time sign corrections."),
        ("Table B.1 (corrected)", "table", "tableB1_corrected",
         "The same census with every sign flip undone first, so "
         "$\\Delta = \\text{End}-\\text{Bgn}$ throughout."),
    ]),
    ("ia", "Internet Appendix", [
        ("Table IA.I", "table", "table_ia1", ""),
        ("Table IA.II", "table", "table_ia2", ""),
        ("Table IA.III", "table", "table_ia3", ""),
        ("Table IA.IV", "table", "table_ia4", ""),
        ("Table IA.V", "table", "table_ia5", ""),
        ("Table IA.VI", "table", "table_ia6", ""),
        ("Table IA.VII", "table", "table_ia7", ""),
        ("Table IA.IX", "table", "table_ia09",
         "As published: \\texttt{b\\_rvol} is placed where the printed table puts it."),
        ("Table IA.IX (by the signal dictionary)", "table", "table_ia09_dictionary",
         "The same counts with \\texttt{b\\_rvol} placed where the paper's own signal "
         "dictionary puts it."),
        ("Table IA.X", "table", "table_ia10", ""),
        ("Table IA.XI", "table", "table_ia11", ""),
        ("Table IA.XII", "table", "table12", ""),
        ("Table IA.XIII", "table", "table13", ""),
        ("Table IA.XIV", "table", "table14", ""),
        ("Table IA.XV", "table", "table15", ""),
        ("Table IA.XVI", "table", "table16", ""),
        ("Table IA.XVII", "table", "table_ia17_{window}", ""),
        ("Table IA.XVIII", "table", "table_ia18_{window}", ""),
        ("Table IA.XIX", "table", "table_ia19_{window}", ""),
        ("Inline counts (alpha)", "table", "inline_counts_alpha",
         "The uncaptioned specification-count table in Section IA.3."),
        ("Inline counts (premium)", "table", "inline_counts_premium",
         "Its companion, for the mean premium."),
        ("Figure IA.1", "figure", "figIA1_bias_by_rating", ""),
        ("Figure IA.2", "figure", "figIA2_lab_decomposition_rating", ""),
        ("Figure IA.3", "figure", "figIA3_nse_alpha_tstat_dua_{window}", ""),
        ("Figure IA.4", "figure", "figIA4_nse_tstat_dua_{window}", ""),
        ("Figure IA.5", "figure", "figIA5_nse_alpha_tstat_mua_{window}", ""),
        ("Figure IA.6", "figure", "figIA6_nse_tstat_mua_{window}", ""),
    ]),
]

# Figures 1, 2 and 5 of the paper are schematics drawn in LaTeX -- a research framework,
# a return timeline, and a look-ahead illustration. They have no data behind them, so
# Stage 3 does not produce them and this document says so rather than leaving a gap.
NOT_DATA_FIGURES = "Figures 1, 2 and 5"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}

% The exhibit numbers in this document are the PAPER's, announced by each heading.
% LaTeX's own sequential numbering would disagree with them, so it is suppressed.
\captionsetup{labelformat=empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}

% Every exhibit is placed exactly where it appears, not floated away from its heading.
\makeatletter
\renewcommand{\fps@table}{H}
\renewcommand{\fps@figure}{H}
\makeatother
\usepackage{float}
"""


def _resolve(stem: str, window: str) -> str:
    return stem.replace("{window}", window)


def latex_escape(s: str) -> str:
    for ch, esc in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("_", r"\_"), ("#", r"\#"), ("$", r"\$"),
                    ("{", r"\{"), ("}", r"\}")):
        s = s.replace(ch, esc)
    return s


def provenance() -> dict:
    """What produced these exhibits: the inputs, the engine(s), when.

    ❗Collects EVERY distinct PyBondLab build the manifests record, not the first one.
    A producer is skipped when its output already exists, so a later run under a
    different build re-renders the exhibits while the grids keep the engine that made
    them -- and a title page naming one build would then be wrong about most of the
    document.
    """
    out = {"inputs": [], "builds": [], "written": None, "n_runs": 0}
    by_tree: dict = {}
    for f in sorted(paths.DATA.glob("*/*.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8")).get("manifest") or {}
        except Exception:            # noqa: BLE001 -- a half-written manifest is not fatal
            continue
        out["n_runs"] += 1
        pbl = m.get("pybondlab")
        if pbl:
            rec = by_tree.setdefault(pbl.get("tree_sha256"),
                                     {"build": pbl, "results": []})
            rec["results"].append(f.stem)
        if m.get("written_utc"):
            out["written"] = max(out["written"] or "", m["written_utc"])
        for i in m.get("inputs", []):
            if i.get("exists") and i not in out["inputs"]:
                out["inputs"].append(i)
    out["builds"] = sorted(by_tree.values(), key=lambda r: -len(r["results"]))
    # keep one row per distinct file
    seen, uniq = set(), []
    for i in out["inputs"]:
        if i["path"] not in seen:
            seen.add(i["path"])
            uniq.append(i)
    out["inputs"] = uniq
    return out


def title_page(window: str, prov: dict) -> str:
    L = [r"\begin{titlepage}", r"\vspace*{2cm}", r"\begin{center}",
         r"{\LARGE\bfseries The Corporate Bond Factor Replication Crisis}\\[0.4em]",
         r"{\Large Exhibits, rebuilt from your own data}\\[2em]",
         r"{\large Produced by Stage 3 of the TRACE Data Pipeline}\\[0.5em]",
         r"{\large \today}", r"\end{center}", r"\vspace{1.5cm}",
         r"\noindent\rule{\textwidth}{0.4pt}", r"\vspace{0.5em}", ""]
    L.append(r"\textbf{These are not the paper's printed numbers.} Every table and figure "
             r"here was computed from the monthly panel Stage 2 built on this machine, and "
             r"nothing in this document compares them to the published ones. Where they "
             r"differ, the data, the sample window or the sorting engine differs -- not "
             r"necessarily anything else." + "\n")
    L.append(r"\vspace{0.5em}\noindent\rule{\textwidth}{0.4pt}" + "\n")

    L.append(r"\subsection*{What produced these}")
    L.append(r"\begin{tabular}{ll}\toprule")
    L.append(r"Section 5 window & " + latex_escape(window) + r" \\")
    lib, lab = S.SAMPLE["lib"], S.SAMPLE["lab"]
    L.append(r"Section 3 and 5 sample & " + f"{lib['start']} to {lib['end']}" + r" \\")
    L.append(r"Section 4 sample & " + f"{lab['start']} to {lab['end']}" + r" \\")
    if prov.get("written"):
        L.append(r"Last exhibit written & " + latex_escape(prov["written"][:19]) + r" \\")
    L.append(r"Exhibit runs recorded & " + str(prov.get("n_runs", 0)) + r" \\")
    L.append(r"\bottomrule\end{tabular}")

    builds = prov.get("builds") or []
    L.append(r"\subsection*{Sorting engine}")
    if not builds:
        L.append(r"No PyBondLab provenance is recorded -- no producer has run yet.")
    else:
        if len(builds) > 1:
            L.append(r"\textbf{More than one PyBondLab build produced this document.} "
                     r"A producer is skipped when its output already exists, so results "
                     r"re-rendered later can carry a different engine from the one that "
                     r"made the sorts and grids. Each build is listed with what it "
                     r"produced." + "\n")
        L.append(r"{\footnotesize\begin{tabular}{llcl}\toprule")
        L.append(r"Version & Tree & Fast kernels & Produced \\ \midrule")
        for rec in builds:
            b = rec["build"]
            v = f"v{b.get('version', '?')}"
            if b.get("git_branch"):
                v += f" ({b['git_branch']}@{(b.get('git_sha') or '')[:7]})"
            names = ", ".join(sorted(rec["results"])[:4])
            if len(rec["results"]) > 4:
                names += f", +{len(rec['results']) - 4} more"
            L.append(latex_escape(v) + r" & \texttt{" + str(b.get("tree_sha256", "?"))
                     + r"} & " + ("yes" if b.get("has_fast_kernels") else "no")
                     + r" & " + latex_escape(names) + r" \\")
        L.append(r"\bottomrule\end{tabular}}")

    if prov["inputs"]:
        # The DATA inputs are what a reader needs to see. The intermediate sort CSVs an
        # exhibit happens to read are Stage 3's own output, listed in the manifests.
        # Show every data input; count the rest rather than silently truncating.
        data_in = [i for i in prov["inputs"]
                   if not str(i["path"]).replace("\\", "/").startswith("stage3/")]
        derived = len(prov["inputs"]) - len(data_in)
        L.append(r"\subsection*{Inputs}")
        L.append(r"{\footnotesize\begin{tabular}{lrl}\toprule")
        L.append(r"File & Size & sha256 \\ \midrule")
        for i in data_in:
            mb = i.get("bytes", 0) / 1e6
            L.append(f"{latex_escape(str(i['path']))} & {mb:,.0f} MB & "
                     f"\\texttt{{{i.get('sha256_16', '')}}} \\\\")
        L.append(r"\bottomrule\end{tabular}}")
        if derived:
            L.append(rf"{{\footnotesize A further {derived} intermediate file(s) that "
                     r"Stage 3 produced itself are recorded in the per-result "
                     r"manifests under \texttt{data/}.}")

    L.append(r"\vfill")
    L.append(r"{\footnotesize " + NOT_DATA_FIGURES + r" of the paper are schematics drawn "
             r"in \LaTeX{} -- a research framework, a return timeline and a look-ahead "
             r"illustration. They have no data behind them, so Stage 3 does not produce "
             r"them and they are absent here.}")
    L.append(r"\end{titlepage}")
    return "\n".join(L)


def build_tex(window: str) -> tuple[str, list[str], list[str]]:
    """The document source, plus what was found and what was missing."""
    prov = provenance()
    found, missing = [], []
    L = [PREAMBLE, r"\begin{document}", title_page(window, prov),
         r"\tableofcontents", r"\clearpage"]

    for _key, title, items in EXHIBITS:
        L.append(r"\section{" + title + "}")
        for number, kind, stem_t, note in items:
            stem = _resolve(stem_t, window)
            src = (paths.TABLES / f"{stem}.tex") if kind == "table" \
                else (paths.FIGURES / f"{stem}.pdf")
            if not src.exists():
                missing.append(f"{number} ({src.name})")
                L.append(r"\subsection{" + number + r"}")
                L.append(r"\emph{Not produced. Expected \texttt{"
                         + latex_escape(str(src.relative_to(paths.REPORTS)))
                         + r"}.}")
                continue
            found.append(number)
            L.append(r"\subsection{" + number + r"}")
            if note:
                L.append(r"\emph{" + note + r"}" + "\n")
            if kind == "table":
                rel = src.relative_to(paths.REPORTS).as_posix()
                L.append(r"\input{" + rel + "}")
            else:
                rel = src.relative_to(paths.REPORTS).as_posix()
                L.append(r"\begin{figure}[H]\centering")
                L.append(r"\includegraphics[width=\textwidth,height=0.8\textheight,"
                         r"keepaspectratio]{" + rel + "}")
                L.append(r"\end{figure}")
            L.append(r"\clearpage")

    L.append(r"\end{document}")
    return "\n".join(L) + "\n", found, missing


INTERMEDIATES = ("aux", "log", "out", "toc", "lof", "lot", "fls", "fdb_latexmk")


def clean() -> int:
    n = 0
    for ext in INTERMEDIATES:
        for f in paths.REPORTS.glob(f"exhibits.{ext}"):
            f.unlink()
            n += 1
    print(f"removed {n} LaTeX intermediate(s)")
    return 0


def compile_pdf(tex: Path) -> int:
    """Two passes: the first writes the table of contents, the second resolves it."""
    exe = shutil.which("pdflatex")
    if not exe:
        print("pdflatex is not on PATH -- wrote the .tex but did not compile it.\n"
              "  Install any TeX distribution (MiKTeX, TeX Live, MacTeX) and re-run,\n"
              "  or compile reports/exhibits.tex yourself.")
        return 2
    log = tex.with_suffix(".build.log")
    for i in (1, 2):
        print(f"[{i}/2] pdflatex")
        r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error",
                            tex.name], cwd=tex.parent,
                           capture_output=True, text=True)
        log.write_text(r.stdout + r.stderr, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"FAILED -- see {log}")
            tail = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("!")]
            for ln in tail[:10]:
                print("   " + ln)
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", default="paper", choices=("paper", "full"),
                    help="which Section-5 window's exhibits to include")
    ap.add_argument("--no-compile", action="store_true", help="write the .tex only")
    ap.add_argument("--clean", action="store_true",
                    help="remove the LaTeX intermediates and exit")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.clean:
        return clean()

    t0 = time.perf_counter()
    src, found, missing = build_tex(args.window)
    tex = paths.REPORTS / "exhibits.tex"
    tex.write_text(src, encoding="utf-8")
    n_total = sum(len(items) for _, _, items in EXHIBITS)
    print(f"wrote {tex}  ({len(found)}/{n_total} exhibits present)")
    if missing:
        print(f"  {len(missing)} not produced: " + ", ".join(missing[:8])
              + (" ..." if len(missing) > 8 else "")
              + "\n  Run `python _run_stage3.py` to produce them.")

    if args.no_compile:
        return 0
    rc = compile_pdf(tex)
    if rc:
        return rc

    pdf = tex.with_suffix(".pdf")
    pages = "?"
    aux = tex.with_suffix(".log")
    if aux.exists():
        import re
        m = re.search(r"Output written on .*?\((\d+) pages", aux.read_text(
            encoding="utf-8", errors="replace"))
        if m:
            pages = m.group(1)
    print(f"\nOK   {pdf}")
    print(f"     {pages} pages, {pdf.stat().st_size / 1e6:.1f} MB, "
          f"{time.perf_counter() - t0:.1f}s")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
