"""The manifest writer: fingerprints, full config snapshot, gate records, round-trip JSON."""
import json

import _stage2_settings as cfg
from lib import manifest as mf


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(mf, "_SHA_CACHE_PATH", tmp_path / "cache" / "sha_cache.json")
    (tmp_path / "manifests").mkdir(parents=True)
    (tmp_path / "cache").mkdir(parents=True)

    f = tmp_path / "input.bin"
    f.write_bytes(b"hello monthly panel")

    m = mf.RunManifest(input_mode="golden", run_id="test_run")
    m.add_input("daily", f)
    m.add_gate("G0_harness", status="PASS", wall_s=1.234,
               output={"path": str(f), "rows": 1, "cols": 2},
               validation={"pass": True})
    out = m.write()

    doc = json.loads(out.read_text())
    assert doc["run_id"] == "test_run"
    assert doc["input_mode"] == "golden"
    assert doc["config_snapshot"]["BUSINESS_DAY_GAP"] == 5      # the corrected upstream knob
    assert doc["config_snapshot"]["START_DATE"] == "2002-07-31"
    assert doc["inputs"][0]["sha256"] == mf.sha256_file(f)      # cached second call, same digest
    assert doc["gates"][0]["gate"] == "G0_harness" and doc["gates"][0]["status"] == "PASS"
    assert doc["git_commit"] != ""


def test_sha_cache_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(mf, "_SHA_CACHE_PATH", tmp_path / "sha_cache.json")
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    d1 = mf.sha256_file(f)
    cache = json.loads((tmp_path / "sha_cache.json").read_text())
    assert list(cache.values()) == [d1]
    assert mf.sha256_file(f) == d1


def test_concurrent_step_groups_get_different_manifests():
    """A full build starts steps 3-4 and 5-6 as two child processes in the same second. Until
    2026-09-23 both took the run id monthly_<stamp>_<mode>, so one manifest overwrote the other."""
    a = mf.RunManifest(input_mode="stage1", tag="steps3-4").doc["run_id"]
    b = mf.RunManifest(input_mode="stage1", tag="steps5-6").doc["run_id"]
    assert a != b and a.endswith("_stage1_steps3-4") and b.endswith("_stage1_steps5-6")
    assert mf.RunManifest(input_mode="stage1").doc["run_id"].endswith("_stage1")


def test_validate_imports_modules_that_exist():
    """--validate imported `validate_monthly`, which is not in this repository (2026-09-23)."""
    import ast
    from pathlib import Path
    stage2 = Path(cfg.__file__).resolve().parent
    tree = ast.parse((stage2 / "build_panel.py").read_text(encoding="utf-8"))
    local = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)
             and n.names[0].name.startswith("validate")}
    assert local and all((stage2 / f"{m}.py").exists() for m in local), local
