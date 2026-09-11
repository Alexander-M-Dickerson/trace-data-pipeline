# -*- coding: utf-8 -*-
"""
test_validate_cli.py
====================
`validate_stage2.py --help` used to raise `KeyError: 'returns'`.

The CLI compares each build block against a golden. A public clone has no golden, so
`_stage2_settings.GOLDEN_OUTPUTS` is empty -- but `_specs()` indexed it unguarded, and
argparse calls `_specs()` while building `choices`. The exception therefore fired before
argparse could print anything, including `--help`. It was shipped that way.

These tests pin both halves: the CLI is usable with no golden configured, and the specs it
builds when a golden IS configured are the ones the internal reproduction run expects.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STAGE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2))

import _stage2_settings as cfg  # noqa: E402
import validate_stage2 as v  # noqa: E402

# step -> (GOLDEN_OUTPUTS key, our output filename)
EXPECTED = {
    "returns": ("returns", "returns_alt.parquet"),
    "illiq": ("illiq_factors", "illiq_factors.parquet"),
    "bbw": ("bbw_factors", "bbw_factors.parquet"),
    "betas": ("betas", "betas_x.parquet"),
    "momentum": ("momentum", "mom_retx.parquet"),
    "mmn": ("price_signals", None),
    "main": ("main_panel", "main_panel_golden.parquet"),
}


@pytest.fixture()
def all_goldens(monkeypatch):
    """Pretend every golden is configured, as it is on the internal reproduction machine."""
    fake = {key: Path(f"/golden/{key}.parquet") for key, _ in EXPECTED.values()}
    monkeypatch.setattr(cfg, "GOLDEN_OUTPUTS", fake)
    return fake


def test_specs_is_empty_when_no_golden_is_configured(monkeypatch):
    monkeypatch.setattr(cfg, "GOLDEN_OUTPUTS", {})
    assert v._specs() == {}


def test_the_parser_builds_with_no_golden(monkeypatch, capsys):
    """The regression: --help must print, not raise."""
    monkeypatch.setattr(cfg, "GOLDEN_OUTPUTS", {})
    monkeypatch.setattr(sys, "argv", ["validate_stage2.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        v.main()
    assert exc.value.code == 0
    assert "--step" in capsys.readouterr().out


def test_all_explains_itself_instead_of_raising_keyerror(monkeypatch):
    monkeypatch.setattr(cfg, "GOLDEN_OUTPUTS", {})
    with pytest.raises(SystemExit) as exc:
        v._require_specs()
    assert "no golden outputs are configured" in str(exc.value)


def test_every_step_appears_when_the_goldens_are_configured(all_goldens):
    assert sorted(v._specs()) == sorted(EXPECTED)


def test_each_spec_points_at_the_right_pair(all_goldens):
    specs = v._specs()
    for step, (golden_key, ours_name) in EXPECTED.items():
        assert specs[step].golden == all_goldens[golden_key]
        if ours_name:
            assert specs[step].ours.name == ours_name


def test_the_two_overridden_outputs_are_not_in_the_blocks_golden_dir(all_goldens):
    """`main` lives in the panel dir and `mmn` carries the date stamp -- the two exceptions."""
    specs = v._specs()
    assert specs["main"].ours.parent == cfg.PANEL_DIR
    assert cfg.DATE_STAMP in specs["mmn"].ours.name


def test_rate_families_are_applied_to_exactly_the_four_steps(all_goldens):
    on = {s for s, spec in v._specs().items() if spec.apply_rate_families}
    assert on == {"betas", "mmn", "main"}


def test_the_documented_residual_tolerances_reach_the_panel_steps(all_goldens):
    specs = v._specs()
    for step in ("mmn", "main"):
        assert specs[step].col_tols["b_dvixd"] == 10.0
        assert specs[step].col_tols["cs_sprd"] == 5e-3
    assert specs["illiq"].col_tols == {"ARS": 1e-4}


def test_a_step_whose_golden_is_missing_is_dropped_not_crashed(monkeypatch):
    """A partial configuration is normal mid-build; it must narrow the CLI, not break it."""
    monkeypatch.setattr(cfg, "GOLDEN_OUTPUTS", {"betas": Path("/golden/betas.parquet")})
    assert sorted(v._specs()) == ["betas"]
    with pytest.raises(SystemExit, match="no golden configured for step"):
        v.validate_step("main")
