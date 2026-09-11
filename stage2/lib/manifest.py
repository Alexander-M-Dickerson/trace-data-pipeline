"""manifest.py -- the run-manifest writer: the machine-readable re-runnability record.

Every full or partial build writes `manifests/monthly_<stamp>_<mode>.json` recording inputs +
fingerprints (sha256/bytes/mtime_ns), the FULL config snapshot, git SHA, per-gate timings and
validation reports, and the final output fingerprint. A cold machine with the repo + inputs can
reproduce the panel from the manifest alone.

sha256 of multi-GB inputs is cached by (path, bytes, mtime_ns) in output/_cache/sha_cache.json so
re-runs don't re-hash unchanged files.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _stage2_settings as cfg

_SHA_CACHE_PATH = cfg.CACHE_DIR / "sha_cache.json"
_HASH_CHUNK = 1 << 22  # 4 MiB read chunks


def sha256_file(path: Path) -> str:
    """sha256 hex digest, cached by (path, bytes, mtime_ns) -- multi-GB inputs hash once."""
    path = Path(path)
    st = path.stat()
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    cache: dict[str, str] = {}
    if _SHA_CACHE_PATH.exists():
        cache = json.loads(_SHA_CACHE_PATH.read_text())
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    digest = h.hexdigest()
    cache[key] = digest
    cfg.ensure_dirs()
    _SHA_CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return digest


def file_fingerprint(role: str, path: Path) -> dict[str, Any]:
    """The identity record for one input/output file: role, path, bytes, mtime_ns, sha256."""
    path = Path(path)
    st = path.stat()
    return {"role": role, "path": str(path), "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(path)}


def git_commit() -> str:
    """Current git SHA of the repo (or 'unknown' outside a checkout)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cfg.REPO, text=True).strip()
    except Exception:
        return "unknown"


def config_snapshot() -> dict[str, Any]:
    """The FULL knob set (nothing implicit). Every UPPER_CASE scalar/tuple in _stage2_settings."""
    snap = {}
    for name in dir(cfg):
        if name.isupper():
            val = getattr(cfg, name)
            if isinstance(val, (int, float, str, bool, tuple, list)):
                snap[name] = list(val) if isinstance(val, tuple) else val
    return snap


class RunManifest:
    """Accumulates gate results across a build, then writes one JSON record.

    Usage:
        m = RunManifest(input_mode='golden')
        m.add_input('daily', cfg.daily_input())
        m.add_gate('G1_returns', status='PASS', wall_s=12.3, output=..., validation=...)
        m.write()
    """

    def __init__(self, input_mode: str, run_id: str | None = None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.doc: dict[str, Any] = {
            "run_id": run_id or f"monthly_{stamp}_{input_mode}",
            "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_mode": input_mode,
            "duckdb_version": _duckdb_version(),
            "python": sys.executable,
            "hardware": {"cores": 24, "ram_gb": 128},
            "config_snapshot": config_snapshot(),
            "inputs": [],
            "gates": [],
            "final": None,
        }

    def add_input(self, role: str, path: Path) -> None:
        self.doc["inputs"].append(file_fingerprint(role, path))

    def add_gate(self, gate: str, status: str, wall_s: float,
                 output: dict[str, Any] | None = None,
                 validation: dict[str, Any] | None = None, **extra: Any) -> None:
        self.doc["gates"].append({"gate": gate, "status": status, "wall_s": round(wall_s, 3),
                                  "output": output, "validation": validation, **extra})

    def set_final(self, path: Path, rows: int, cols: int, matches_golden: bool) -> None:
        fp = file_fingerprint("final", path)
        self.doc["final"] = {"path": fp["path"], "rows": rows, "cols": cols,
                             "sha256": fp["sha256"], "matches_golden": matches_golden}

    def write(self) -> Path:
        cfg.ensure_dirs()
        out = cfg.MANIFEST_DIR / f"{self.doc['run_id']}.json"
        out.write_text(json.dumps(self.doc, indent=1, default=str))
        return out


def _duckdb_version() -> str:
    import duckdb
    return duckdb.__version__
