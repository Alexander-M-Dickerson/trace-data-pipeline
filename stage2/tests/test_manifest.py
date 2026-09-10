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
