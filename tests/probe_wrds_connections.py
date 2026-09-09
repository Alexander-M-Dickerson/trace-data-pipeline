# -*- coding: utf-8 -*-
"""
probe_wrds_connections.py
=========================
Measure this account's WRDS connection limit, so the concurrency settings can be set
from evidence rather than folklore.

WRDS documents a limit of 5 CONCURRENT JOBS. It does not publish a per-user limit on
database CONNECTIONS -- but one exists, and hitting it is nasty: the wrds package's
connect-failure path calls input() to re-prompt for a username, so in a batch job a
connection-limit error surfaces as

    EOFError: EOF when reading a line

A rate limit reported as a keyboard error. This probe finds the real number and the
real error text.

Two separate questions, because they have different answers:

  HOLD    how many connections can be open at once? Opened one at a time, staggered,
          each proved live with SELECT 1, all held until the end. This is the number
          that bounds a worker pool where each worker owns a connection.

  STORM   how many can be opened SIMULTANEOUSLY? The known failure is arrival-order
          dependent -- several workers dialling at the same instant trip a limit that
          the same count opened sequentially does not. This is why a pool should
          serialise its handshake even when the hold count is fine.

Usage
-----
    python3 tests/probe_wrds_connections.py                 # both probes, up to 10
    python3 tests/probe_wrds_connections.py --max 8
    python3 tests/probe_wrds_connections.py --hold-only

Run it when no other WRDS jobs of yours are active: an in-flight pipeline holds
connections too, and this probe would be measuring the remainder rather than the cap.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback

try:
    import wrds
except ImportError:  # pragma: no cover
    print("[error] the wrds package is not installed")
    raise SystemExit(2)


def _classify(exc: BaseException) -> str:
    """Name what actually went wrong, including the disguised connection limit."""
    if isinstance(exc, EOFError):
        return ("CONNECTION LIMIT (disguised): the wrds package fell back to input() "
                "to re-prompt for a username, and stdin is closed")
    text = str(exc).lower()
    for needle, label in (
        ("too many connections", "CONNECTION LIMIT (server said so)"),
        ("remaining connection slots", "CONNECTION LIMIT (server said so)"),
        ("connection limit", "CONNECTION LIMIT (server said so)"),
        ("timeout", "TIMEOUT"),
        ("authentication", "AUTHENTICATION"),
        ("password", "AUTHENTICATION"),
    ):
        if needle in text:
            return label
    return f"{type(exc).__name__}"


def _connect(user: str):
    conn = wrds.Connection(wrds_username=user) if user else wrds.Connection()
    conn.raw_sql("SELECT 1")          # prove it is actually usable, not just built
    return conn


def probe_hold(user: str, max_n: int, stagger: float):
    """Open one at a time, hold them all, and report where it stops."""
    print()
    print("=" * 78)
    print(f"HOLD probe -- opening up to {max_n}, one at a time, {stagger}s apart")
    print("=" * 78)
    conns, failure = [], None
    try:
        for i in range(1, max_n + 1):
            t0 = time.time()
            try:
                conns.append(_connect(user))
                print(f"  [{i:2d}] connected in {time.time() - t0:5.1f}s  (holding {len(conns)})")
            except BaseException as exc:                     # noqa: BLE001 - we want EOFError too
                failure = (i, _classify(exc), str(exc).strip().splitlines()[:1])
                print(f"  [{i:2d}] FAILED after {time.time() - t0:5.1f}s")
                print(f"       classified: {failure[1]}")
                print(f"       raw       : {failure[2]}")
                break
            time.sleep(stagger)
    finally:
        held = len(conns)
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        print(f"  released {held} connection(s)")
    return held, failure


def probe_storm(user: str, counts, results: dict):
    """Open N at the same instant, which is the failure the serialised handshake avoids."""
    print()
    print("=" * 78)
    print("STORM probe -- opening N simultaneously (no stagger, no lock)")
    print("=" * 78)
    for n in counts:
        conns, errors = [], []
        lock = threading.Lock()
        start = threading.Barrier(n)

        def worker():
            try:
                start.wait(timeout=30)
                c = _connect(user)
                with lock:
                    conns.append(c)
            except BaseException as exc:                     # noqa: BLE001
                with lock:
                    errors.append(_classify(exc))

        threads = [threading.Thread(target=worker) for _ in range(n)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - t0

        ok = len(conns)
        for c in conns:
            try:
                c.close()
            except Exception:
                pass

        results[n] = (ok, errors)
        if errors:
            uniq = sorted(set(errors))
            print(f"  {n:2d} at once -> {ok}/{n} connected in {elapsed:5.1f}s   FAILURES: {uniq}")
        else:
            print(f"  {n:2d} at once -> {ok}/{n} connected in {elapsed:5.1f}s   all ok")
        time.sleep(2.0)          # let the server settle between rounds
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure the WRDS connection limit.")
    ap.add_argument("--user", default=os.environ.get("WRDS_USERNAME", ""),
                    help="WRDS username (default: $WRDS_USERNAME)")
    ap.add_argument("--max", type=int, default=10, help="highest count to attempt")
    ap.add_argument("--stagger", type=float, default=0.4,
                    help="seconds between sequential connects")
    ap.add_argument("--hold-only", action="store_true", help="skip the storm probe")
    args = ap.parse_args(argv)

    print(f"user      : {args.user or '(from ~/.pgpass)'}")
    print(f"max tried : {args.max}")
    print("NOTE: run this with no other WRDS jobs of yours active, or you are")
    print("      measuring what is left of the cap rather than the cap.")

    held, failure = probe_hold(args.user, args.max, args.stagger)

    storm = {}
    if not args.hold_only:
        counts = [c for c in (2, 4, 6, 8, 10) if c <= max(2, held)]
        if counts:
            probe_storm(args.user, counts, storm)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if failure is None:
        print(f"  Held {held} simultaneous connections with no failure "
              f"(the ceiling is at least {held}; raise --max to find it).")
    else:
        print(f"  Held {held} simultaneous connections; #{failure[0]} failed.")
        print(f"  Failure mode: {failure[1]}")
    if storm:
        worst = [n for n, (ok, errs) in sorted(storm.items()) if errs]
        if worst:
            print(f"  Simultaneous opens started failing at {min(worst)}.")
            print("  -> serialise the handshake; that is what makes the hold count usable.")
        else:
            print("  No simultaneous-open failures at the counts tried.")

    budget = max(1, held - 1)
    print()
    print(f"  Suggested total budget: {budget} "
          f"(one below the observed ceiling, leaving headroom for a retry).")
    print( "  Split it across the concurrent stage0 jobs, e.g. Enhanced N-1 + 144A 1.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
