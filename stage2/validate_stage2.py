"""validate_monthly.py -- the golden validation CLI: one step (or all), same rules as the gate sweep.

Usage:
    validate_monthly.py --step main             # the 140-col panel
    validate_monthly.py --step all              # every gate target
    validate_monthly.py --step betas --json-out report.json

Tolerances (context/golden_validation.md + the assumptions ledger):
  - default float tol 1e-6; RATE_TOL=1e-4 for the beta/ivol/iskew/sysmom/idimom families and ytm/cs
    (float32 winsorized rate-likes -- HANDOFF G5);
  - documented irreducible residuals: cs_sprd/ar_sprd 5e-3 (A15: numpy libm ulps at clip
    boundaries), b_dvixd 10.0 (A16: 23 near-singular numba-fastmath rows of 1.63M).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import _stage2_settings as cfg
from lib import validate_core

# rate-tol column families (applied where present on the golden side)
_RATE_PREFIXES = ("b_", "ivol_", "iskew", "sysmom", "idimom")
_RATE_EXTRA = ("ytm", "cs")
# A15/A16 documented residual tolerances (see assumptions.md)
_RESIDUAL_TOLS = {"cs_sprd": 5e-3, "ar_sprd": 5e-3, "b_dvixd": 10.0,
                  "cs_sprd_mmn": 5e-3, "ar_sprd_mmn": 5e-3, "b_dvixd_mmn": 10.0}


@dataclass(frozen=True)
class StepSpec:
    golden: Path
    ours: Path
    key: tuple[str, ...]
    apply_rate_families: bool = False
    col_tols: dict = field(default_factory=dict)


# step -> (key in cfg.GOLDEN_OUTPUTS, our output, join key, rate families, per-col tolerances)
_STEP_TABLE: dict[str, tuple] = {
    "returns":  ("returns",       "returns_alt.parquet",   ("cusip", "date"), False, {}),
    "illiq":    ("illiq_factors", "illiq_factors.parquet", ("date",),         False, {"ARS": 1e-4}),
    "bbw":      ("bbw_factors",   "bbw_factors.parquet",   ("date",),         False, {}),
    "betas":    ("betas",         "betas_x.parquet",       ("cusip", "date"), True,  {}),
    "momentum": ("momentum",      "mom_retx.parquet",      ("cusip", "date"), False, {}),
    "mmn":      ("price_signals", None,                    ("cusip", "date"), True,  _RESIDUAL_TOLS),
    "main":     ("main_panel",    None,                    ("cusip", "date"), True,  _RESIDUAL_TOLS),
}


def _specs() -> dict[str, StepSpec]:
    """Only the steps that actually have a golden configured.

    GOLDEN_OUTPUTS is populated from settings and is EMPTY in a public clone, which has no
    golden to compare against. Indexing it unguarded made `--help` itself raise KeyError,
    because argparse builds `choices` from this dict.
    """
    b = cfg.BLOCKS_DIR / "golden"
    ours_override = {
        "mmn": b / f"mmn_price_based_signals_{cfg.DATE_STAMP}.parquet",
        "main": cfg.PANEL_DIR / "main_panel_golden.parquet",
    }
    specs: dict[str, StepSpec] = {}
    for step, (gk, ours_name, key, rate, tols) in _STEP_TABLE.items():
        golden = cfg.GOLDEN_OUTPUTS.get(gk)
        if golden is None:
            continue
        ours = ours_override.get(step) or (b / ours_name)
        specs[step] = StepSpec(Path(golden), ours, key,
                               apply_rate_families=rate, col_tols=dict(tols))
    return specs


def _require_specs() -> dict[str, StepSpec]:
    specs = _specs()
    if not specs:
        raise SystemExit(
            "no golden outputs are configured, so there is nothing to validate against.\n"
            "GOLDEN_OUTPUTS in _stage2_settings.py is empty -- this is normal for a public\n"
            "clone. Golden validation is an internal reproduction check; a public build is\n"
            "checked by `pytest stage2/tests` and the gates inside run_stage2.py."
        )
    return specs


def _load_with_key(path: Path, key: tuple[str, ...]) -> pd.DataFrame:
    """Read a block; restore the key from the index if it was written as one (bbw_factors)."""
    df = pd.read_parquet(path)
    if any(k not in df.columns for k in key):
        df = df.reset_index()
    return df


def validate_step(step: str, verbose: bool = True) -> tuple[bool, dict]:
    specs = _require_specs()
    if step not in specs:
        raise SystemExit(f"no golden configured for step {step!r}; available: {sorted(specs)}")
    spec = specs[step]
    if not spec.ours.exists():
        raise FileNotFoundError(f"our {step} output not built yet: {spec.ours}")
    golden = _load_with_key(spec.golden, spec.key)
    ours = _load_with_key(spec.ours, spec.key)
    rate_cols = tuple(
        c for c in golden.columns
        if c.startswith(_RATE_PREFIXES) or c in _RATE_EXTRA
    ) if spec.apply_rate_families else ()
    passed, report = validate_core.diff_frames(
        ours, golden, key=spec.key, rate_cols=rate_cols, col_tols=spec.col_tols or None)
    if verbose:
        print(f"[validate:{step}] ours={spec.ours.name} golden={spec.golden.name}")
        print(validate_core.format_report(report))
    return passed, report


def main() -> None:
    ap = argparse.ArgumentParser(description="monthly golden validation")
    ap.add_argument("--step", required=True, choices=sorted(_specs()) + ["all"])
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    steps = list(_require_specs()) if args.step == "all" else [args.step]
    results = {}
    ok = True
    for s in steps:
        passed, report = validate_step(s)
        results[s] = report
        ok &= passed
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
