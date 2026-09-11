"""bench.py -- the run timer and the run's own completeness check.

Two jobs, deliberately in one object because they belong to the same run:

  * `phase()` splits the wall clock into named parts (load / sort / stats / render), so a
    slow step is a fact with a location rather than an impression. Every run appends one
    JSON line to `reports/timings.jsonl`.
  * `check()` records whether the run's own assertion held -- "every signal produced a
    grid file", "every cell is populated". `ok` is written BEFORE the timing fields,
    because a fast run that produced an incomplete artifact is not a result.

Usage:

    from bench import Bench
    with Bench("sorts", section="s1_lib") as b:
        with b.phase("load"):
            data = load()
        with b.phase("sort"):
            panel = run(data)
        b.note(n_signals=7, turnover=True)
        b.check(len(panel) == expected, f"{len(panel)} rows")

`check` failing does not raise -- the caller decides the exit code -- but it is printed
and it lands in the ledger.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import paths


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(paths.STAGE3), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


class Bench:
    """One timed run. Use as a context manager; the ledger line is written on exit."""

    def __init__(self, tag: str, *, section: str | None = None, echo: bool = True,
                 sample: bool = True, ledger: Path | None = None):
        self.tag = tag
        self.section = section
        self.echo = echo
        self.ledger = Path(ledger) if ledger else paths.TIMINGS
        self.phases: dict[str, float] = {}
        self.notes: dict = {}
        self.checks: list[dict] = []
        self.ok: bool | None = None
        self._t0 = 0.0
        self._sample = sample        # accepted for call compatibility; sampling is optional

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self) -> "Bench":
        self._t0 = time.perf_counter()
        if self.echo:
            print(f"[bench] {self.tag} start", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        wall = round(time.perf_counter() - self._t0, 3)
        rec = {
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tag": self.tag,
            "section": self.section,
            # ok first: a timing for a run that did not pass its own check is a warning
            "ok": self.ok,
            "checks": self.checks,
            "wall_s": wall,
            "phases": {k: round(v, 3) for k, v in self.phases.items()},
            "notes": self.notes,
            "failed": None if exc is None else f"{exc_type.__name__}: {exc}",
            "git": _git_sha(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        }
        try:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:          # a ledger write must never lose a computed result
            print(f"[bench] could not write {self.ledger}: {e}", flush=True)
        if self.echo:
            parts = "  ".join(f"{k} {v:.1f}s" for k, v in self.phases.items())
            print(f"[bench] {self.tag} done {wall:.1f}s   {parts}", flush=True)
        return False                    # never swallow an exception

    # -- recording ----------------------------------------------------------
    @contextmanager
    def phase(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t)

    def note(self, **kw) -> None:
        """Attach run parameters to the ledger line -- signal count, worker count, flags."""
        self.notes.update(kw)

    def check(self, passed: bool, detail: str = "") -> bool:
        """Record the run's own assertion. Several are ANDed into `ok`."""
        passed = bool(passed)
        self.checks.append({"passed": passed, "detail": detail})
        self.ok = passed if self.ok is None else (self.ok and passed)
        if self.echo:
            print(f"[bench] {'PASS' if passed else 'FAIL'} {self.tag}: {detail}", flush=True)
        return passed
