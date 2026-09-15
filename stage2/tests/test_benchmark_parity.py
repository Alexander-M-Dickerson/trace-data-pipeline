"""Every `tret`-derived quantity must have a benchmark twin, or a recorded reason why not.

WHY THIS TEST EXISTS. The panel ships five Treasury benchmarks. Making the duration-adjusted
machinery work for more than `tret` turned up the SAME defect three separate times, each found by
tripping over it while building the next thing rather than by looking:

  1. the published 1997-2002 quote panel carried `tret` and no alternatives, so a rolling beta on
     another benchmark had no pre-history and started a year late;
  2. the benchmarks were attached to the END return frame only, so 8.1% of `all_returns` -- the
     rows that come from the BEGIN frame -- had `tret` and nothing else, which would have estimated
     the alternative betas on a non-random 92% subsample of the rows the `tret` ones use;
  3. the published 1973-2023 extended BBW factor series carried the `tret` twins (MKTBx/DRFx/CRFx/
     TERM) and no others, so the factor side had the same hole the quote panel had on the returns
     side.

Three instances is the point to stop fixing instances. This is the gate: a FOURTH one fails here,
in seconds, instead of surfacing hours into whatever is being built on top of it.

HOW TO SATISFY IT. Either parameterise the site by benchmark -- the column name comes from a
variable, not a literal `"tret"` -- or add it to EXEMPT with a reason. A name in EXEMPT without a
reason is just a silenced alarm, which is the same rule `engine_drift`'s ALLOWED list follows.
"""
import ast
import re
from pathlib import Path

import pytest

import _stage2_settings as cfg

STAGE2 = Path(__file__).resolve().parents[1]

# Sites that read `tret` and are NOT expected to have a benchmark twin, and why.
EXEMPT = {
    "lib/treasury.py": (
        "the PRODUCER of `tret` -- it interpolates the CRSP key-rate curve at modified duration. "
        "The alternative benchmarks come from lib/duration_adjusted.py instead, which is a "
        "different construction, not a parameterisation of this one"),
    "lib/quote.py": (
        "the loader for the published quote panel. It names `tret` only in a docstring listing "
        "the file's columns; the benchmark columns are enforced by QUOTE_BENCHMARK_COLS"),
    "_build_data_report.py": (
        "reporting only -- it forms ret_vwx to describe the shipped panel, and the shipped panel's "
        "duration-adjusted columns are the tret ones by definition"),
    "lib/value.py": (
        "value signals are NOT duration-adjusted, by design: the dur_adj panel merges the SAME "
        "value_signals_std the std panel does -- the pre-TRACE engine says so in as many "
        "words, 'Both panels include value_signals_std' -- and no val_* is among the 68 "
        "swapped columns. Its internal ret_vwx "
        "is an intermediate for vol12_x, shared across return types on purpose"),
    "steps/step7_final.py": (
        "assembles the panel from blocks; it carries `tret` as a column and derives nothing"),
}

# The inputs a benchmark variant needs before any derivation site can work. Each was a bug.
REQUIRED_BENCHMARK_INPUTS = {
    "quote panel": ("QUOTE_BENCHMARK_COLS", "the 1997-2002 returns pre-history"),
    "all_returns": ("tret_bns", "the union of the END and BEGIN return frames"),
}

_DERIVES = re.compile(r"""\[["']tret["']\]|\btret\b\s*$|-\s*\w+\[["']tret["']\]""")


def _py_files():
    for sub in ("lib", "steps"):
        yield from sorted((STAGE2 / sub).glob("*.py"))
    yield STAGE2 / "_build_data_report.py"


def _rel(p: Path) -> str:
    return p.relative_to(STAGE2).as_posix()


def test_every_tret_derivation_is_parameterised_or_exempt():
    """A literal "tret" used to DERIVE something is the shape of all three past bugs."""
    offenders = {}
    for f in _py_files():
        if not f.exists():
            continue
        rel = _rel(f)
        if rel in EXEMPT:
            continue
        src = f.read_text(encoding="utf-8")
        hits = []
        for i, line in enumerate(src.split("\n"), 1):
            s = line.strip()
            if s.startswith("#") or '"""' in s:
                continue
            # a derivation looks like `x = a - b["tret"]` or `... - combined["tret"]`
            if re.search(r'-\s*\w+\[["\']tret["\']\]', line):
                hits.append((i, s))
        if hits:
            offenders[rel] = hits

    assert not offenders, (
        "these derive a quantity from a LITERAL `tret` and are neither parameterised nor "
        "EXEMPT:\n"
        + "\n".join(f"  {r}:{i}  {s}" for r, hs in offenders.items() for i, s in hs)
        + "\n\nParameterise the benchmark (take the column name as an argument, as "
          "step3_bbw.build(benchmark=) and duration_adjusted.attach(start_col=) do), or add the "
          "file to EXEMPT in this test WITH A REASON.")


def test_no_block_carries_tret_without_its_twins():
    """A block that PROPAGATES `tret` and not the alternatives is the real shape of this bug.

    All three original instances were propagation, not derivation: the quote panel, `all_returns`
    and the extended BBW series each CARRIED `tret` with no twin, and the derivation downstream was
    fine. A gate that only looks for `- frame["tret"]` misses every one of them -- which it did,
    until `all_returns_ext` turned up as a fourth.
    """
    import pyarrow.parquet as pq
    blocks = cfg.BLOCKS_DIR / cfg.INPUT_MODE
    if not blocks.exists():
        pytest.skip("no blocks built in this clone")

    # Blocks that legitimately carry `tret` alone, and why.
    CARRIES_TRET_ONLY = {
        "end_signals.parquet": "signal frame; tret arrives via end_returns",
        "bgn_returns.parquet": "a SHIPPED block with a frozen projection -- all_returns reads the "
                               "extended frame directly instead of widening this",
        "end_returns.parquet": "carries all five benchmarks already",
        "returns_alt.parquet": (
            "alternative RETURN DEFINITIONS (ret_vwp/ret_ew/ret_1st/ret_lst/ret_bid) with tret "
            "attached so a user can duration-adjust them. None of the 68 is built from it. "
            "Carrying the benchmarks here is a reasonable future extension -- it would let a user "
            "form ret_bid - tret_bns -- but it widens a PUBLISHED artifact for a use nobody has "
            "asked for yet"),
        "returns_alt_final.parquet": "the step-7 copy of returns_alt; same reason",
    }

    bad = {}
    for f in sorted(blocks.glob("*.parquet")):
        cols = set(pq.ParquetFile(f).schema.names)
        if "tret" not in cols:
            continue
        if f.name in CARRIES_TRET_ONLY:
            continue
        missing = [c for c in ("tret_bns", "tret_cls") if c not in cols]
        if missing:
            bad[f.name] = missing

    assert not bad, (
        "these blocks carry `tret` but not its alternatives:" + "\n"
        + "\n".join(f"  {k}: missing {v}" for k, v in bad.items())
        + "\n\nAnything downstream that reads one of these can only ever be "
          "duration-adjusted on `tret`. Either carry the benchmarks through, or add "
          "the block to CARRIES_TRET_ONLY in this test WITH A REASON.")


def test_exemptions_all_carry_a_reason():
    for name, reason in EXEMPT.items():
        assert reason and len(reason) > 40, f"{name} is exempt without a real reason"


def test_exemptions_are_not_stale():
    """An exemption for a file that no longer exists hides the next real one."""
    missing = [n for n in EXEMPT if not (STAGE2 / n).exists()]
    assert not missing, f"EXEMPT names files that are gone: {missing}"


def test_the_beta_engine_is_benchmark_parameterised():
    """compute_all_betas must not hardcode the pair it computes."""
    from lib import betas
    assert hasattr(betas, "factor_swap"), "the benchmark->factor-twin mapping is gone"
    assert betas.factor_swap(None) == {}
    assert betas.factor_swap("tret") == {"mktb": "mktbx", "drf": "drfx",
                                         "crf": "crfx", "lrf": "lrfx"}
    bns = betas.factor_swap("bns")
    assert bns["mktb"] == "mktb_bns"
    # TERM is MKTB_raw - MKTBx, so it moves with the benchmark; dcapm/psbm/amdm regress on it
    # ALONGSIDE mktbx, and leaving it behind puts two benchmarks on one right-hand side.
    assert bns["term"] == "term_bns", "term must be swapped for a non-tret benchmark"
    assert "term" not in betas.factor_swap("tret"), "there is no `termx`; tret's twin IS `term`"


def test_all_returns_carries_the_benchmarks():
    """The block the rolling betas read must cover every benchmark `tret` covers.

    This is bug 2. `all_returns` unions the END and BEGIN frames; the benchmarks were on the END
    frame only, so 8.1% of rows had `tret` and nothing else.
    """
    import pyarrow.parquet as pq
    p = cfg.BLOCKS_DIR / cfg.INPUT_MODE / "all_returns.parquet"
    if not p.exists():
        pytest.skip("all_returns not built in this clone")
    cols = set(pq.ParquetFile(p).schema.names)
    assert "tret" in cols
    for c in ("tret_bns", "tret_cls"):
        assert c in cols, (
            f"all_returns is missing {c}. The rolling betas read this block, so a benchmark "
            f"absent here is estimated on a different sample than `tret` -- which is a confound "
            f"in exactly the comparison the benchmarks exist for.")


def test_the_step_that_builds_factors_takes_a_benchmark():
    """A benchmark needs its OWN bond-market factor twins, not the tret ones."""
    import inspect
    from steps import step3_bbw
    sig = inspect.signature(step3_bbw.build)
    assert "benchmark" in sig.parameters, (
        "step3 must be able to build bbw_factors_<benchmark>.parquet; reusing the tret twins "
        "would make every b_* column plausible and wrong")
