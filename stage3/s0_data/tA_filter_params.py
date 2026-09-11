r"""tA_filter_params.py -- Tables A.1, A.2 and A.3: the cleaning-filter parameters.

The three appendix tables that let a reader reimplement the cleaner: the decimal-shift
corrector, the bounce-back filter, and the distressed-bond filters.

Nothing here is estimated -- no sample, no t-statistics. But the tables are still a
claim, and the claim is that these are the numbers the pipeline actually ran with. So
the values are READ OUT OF THE LIVE CODE rather than transcribed: the Stage 0 filters'
parameters come from their function signatures, the distressed filter's from Stage 1's
settings dictionary.

❗Read by parsing the source, not by importing it. Importing Stage 0 would pull in its
whole dependency stack (a WRDS connection among them) to learn four default values, and
would fail on a machine set up only to run Stage 3.

If a parameter goes missing, the run says which one rather than printing a blank cell.

    python s0_data/tA_filter_params.py
"""
from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # stage3/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _stage3_settings as S    # noqa: E402
import captions                 # noqa: E402
import drrlib as D              # noqa: E402
import latex_format as F        # noqa: E402
import paths                    # noqa: E402
from bench import Bench         # noqa: E402

# (table, LaTeX label, printed symbol, source tag, key)
SPEC = [
    ("Table A.1", "tab:decimal_params", "$w$ (half-window; window 11)", "DS", "window"),
    ("Table A.1", "tab:decimal_params", r"$\tau_{pct}$", "DS", "tol_pct_good"),
    ("Table A.1", "tab:decimal_params", r"$\tau_{abs}$ (price points)", "DS", "tol_abs_good"),
    ("Table A.1", "tab:decimal_params", r"$\tau_{bad}$", "DS", "tol_pct_bad"),
    ("Table A.1", "tab:decimal_params", r"$\gamma$ (improvement frac.)", "DS", "improvement_frac"),
    ("Table A.1", "tab:decimal_params", "$P_{low}$", "DS", "low_pr"),
    ("Table A.1", "tab:decimal_params", "$P_{high}$", "DS", "high_pr"),
    ("Table A.1", "tab:decimal_params", "par-proximity band", "DS", "par_band"),

    ("Table A.2", "tab:bounce_params", r"$\tau$ (price points)", "BB", "threshold_abs"),
    ("Table A.2", "tab:bounce_params", "$L$ (forward rows)", "BB", "lookahead"),
    ("Table A.2", "tab:bounce_params", "$w$ (trailing anchor window)", "BB", "window"),
    ("Table A.2", "tab:bounce_params", r"$\alpha$ (return-to-anchor)", "BB", "back_to_anchor_tol"),
    ("Table A.2", "tab:bounce_params", r"$\ell_{min}$ (par run)", "BB", "par_min_run"),
    ("Table A.2", "tab:bounce_params", "cooldown (rows)", "BB", "par_cooldown_after_flag"),

    ("Table A.3", "tab:distressed_params", r"$\tau_{low}$", "UD", "ultra_low_threshold"),
    ("Table A.3", "tab:distressed_params", r"$\tau_{high}$", "UD", "high_spike_threshold"),
    ("Table A.3", "tab:distressed_params", r"$\tau_{plateau}$", "UD", "plateau_ultra_low_threshold"),
    ("Table A.3", "tab:distressed_params", r"$\rho_{anomaly}$", "UD", "min_normal_price_ratio"),
    ("Table A.3", "tab:distressed_params", r"$\rho_{spike}$", "UD", "min_spike_ratio"),
    ("Table A.3", "tab:distressed_params", r"$\rho_{recovery}$", "UD", "recovery_ratio"),
    ("Table A.3", "tab:distressed_params", r"$\gamma_{range}$", "UD", "intraday_range_threshold"),
    ("Table A.3", "tab:distressed_params", r"$\tau_{intraday}$ (\% of par)", "UD",
     "intraday_price_threshold"),
    ("Table A.3", "tab:distressed_params", r"$\ell_{min}$ (plateau days)", "UD",
     "min_plateau_days"),
]

# where each group of parameters actually lives in the pipeline
SOURCES = {
    "DS": ("function", "stage0", "create_daily_enhanced_trace.py", "decimal_shift_corrector"),
    "BB": ("function", "stage0", "create_daily_enhanced_trace.py", "flag_price_change_errors"),
    "UD": ("dict", "stage1", "_stage1_settings.py", "ULTRA_DISTRESSED_CONFIG"),
}
STAGE_DIR = {"stage0": S.STAGE0_DIR, "stage1": S.STAGE1_DIR}
CAPTION_OF = {"Table A.1": "tab:decimal_params", "Table A.2": "tab:bounce_params",
              "Table A.3": "tab:distressed_params"}


def _literal(node) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _function_defaults(tree, name: str) -> dict:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            a = node.args
            out = {}
            for kw, d in zip(a.kwonlyargs, a.kw_defaults):
                if d is not None:
                    out[kw.arg] = _literal(d)
            pos = a.posonlyargs + a.args
            for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                out[arg.arg] = _literal(d)
            return out
    raise KeyError(
        f"no function named {name!r} in the parsed source. Table A.1 reads the "
        "pipeline's filter parameters straight out of Stage 0/1 -- a rename upstream "
        "lands here, and the fix is to follow the rename, not to hard-code a value.")


def _module_dict(tree, name: str) -> dict:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            val = _literal(node.value)
            if isinstance(val, dict):
                return val
    raise KeyError(
        f"no module-level dict named {name!r} in the parsed source. Same cause as "
        "above: Table A.1 quotes the pipeline's own constants, so a rename upstream "
        "must be followed here.")


def load_params() -> dict[str, dict]:
    """Read the live parameter values out of the pipeline's source."""
    out = {}
    for tag, (kind, stage, fname, name) in SOURCES.items():
        f = STAGE_DIR[stage] / fname
        if not f.exists():
            raise SystemExit(
                f"Appendix A reads its parameters from {stage}/{fname}, which is not\n"
                f"  at {f}. Stage 3 expects to sit beside stage0/ and stage1/ in the\n"
                f"  pipeline tree; set {stage.upper()}_DIR if it lives elsewhere.")
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        out[tag] = (_function_defaults(tree, name) if kind == "function"
                    else _module_dict(tree, name))
    return out


def rows(params: dict[str, dict]) -> pd.DataFrame:
    recs = []
    for table, label, symbol, tag, key in SPEC:
        kind, stage, fname, name = SOURCES[tag]
        recs.append({"table": table, "label": label, "symbol": symbol,
                     "value": params[tag].get(key),
                     "parameter": key,
                     "source": f"{stage}/{fname}::{name}"})
    return pd.DataFrame(recs)


def render_latex(df: pd.DataFrame, table: str) -> str:
    label = CAPTION_OF[table]
    sub = df[df["table"] == table]
    src = sorted(set(sub["source"]))
    L = [r"\begin{table}[!ht]", r"\caption{" + captions.caption(label) + "}",
         r"\begin{center}", r"\label{" + label + r"}",
         r"\begin{tabular}{l r l}", r"\toprule",
         r"Parameter & Value & Name in code \\", r"\midrule"]
    for _, r in sub.iterrows():
        v = r["value"]
        val = (F.num(v, 2) if isinstance(v, float) and v != int(v)
               else F.thousands(v) if isinstance(v, (int, float)) else str(v))
        L.append(F.row([r["symbol"], val,
                        r"\texttt{" + str(r["parameter"]).replace("_", r"\_") + "}"]))
    L += [r"\bottomrule", r"\end{tabular}",
          r"\\[0.5em]\footnotesize Values read from " + ", ".join(
              r"\texttt{" + s.replace("_", r"\_") + "}" for s in src) + ".",
          r"\end{center}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()

    with Bench("tablesA-filter-params", section="s0_data",
               sample=not args.no_bench, echo=True) as b:
        with b.phase("load"):
            params = load_params()
        with b.phase("render"):
            df = rows(params)
            out = paths.section_results("s0_data")
            df.to_csv(out / "tablesA_filter_params.csv", index=False)
            for table, stem in (("Table A.1", "tableA1"), ("Table A.2", "tableA2"),
                                ("Table A.3", "tableA3")):
                (paths.TABLES / f"{stem}.tex").write_text(
                    render_latex(df, table), encoding="utf-8")
            D.write_result(
                "tablesA_filter_params",
                {"summary": {"exhibit": "Tables A.1-A.3",
                             "n_parameters": len(df),
                             "n_missing": int(df["value"].isna().sum())},
                 "parameters": df.to_dict("records")},
                section="s0_data",
                inputs=[STAGE_DIR[st] / fn for _, st, fn, _ in SOURCES.values()],
                t0=t0, extra={"exhibit": "Tables A.1-A.3"})
        b.note(n_parameters=len(df))
        missing = df[df["value"].isna()]
        ok = b.check(missing.empty,
                     f"{len(df) - len(missing)}/{len(df)} parameters read from the "
                     "live pipeline source"
                     + (f"; missing {sorted(missing['parameter'])}"
                        if not missing.empty else ""))

    print(f"\nTables A.1-A.3: {len(df)} parameters")
    print(df[["table", "parameter", "value"]].to_string(index=False))
    print(f"\nwrote {paths.TABLES / 'tableA1.tex'} (and A2, A3)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
