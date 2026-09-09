# -*- coding: utf-8 -*-
"""
test_chunk_scheduler.py
=======================
The scheduler's job is to run every chunk and hand the results back in chunk order.
Both halves of that matter and neither is self-evident:

  * Out of order, the audit tables are corrupted rather than merely untidy -- the data
    report reconstructs each chunk's filter sequence from ROW ORDER
    (_error_plot_helpers.py: groupby("chunk").cumcount()).
  * A lost chunk is lost bonds, in a file that looks entirely normal and that stage 1
    will happily consume. It has to abort the run, loudly.

These run without WRDS: the pool tasks here are pure arithmetic. What they exercise is
the reassembly and the failure handling, which is the part that can silently drop data.

    python3 tests/test_chunk_scheduler.py
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage0"))
import _chunk_runner as CR                                        # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name:<58} {detail}")
    if not cond:
        FAILURES.append(name)


# Module-level so multiprocessing can pickle them by name (Windows spawns, and even
# under fork a pool task must be importable).
def _task_ok(task):
    chunk_id, cusips, n_chunks = task
    # Sleep in reverse order, so completions arrive scrambled and any reliance on
    # arrival order shows up as a failure rather than passing by luck.
    time.sleep(0.02 * (n_chunks - chunk_id))
    return CR.ChunkResult(chunk_id=chunk_id, data=list(cusips),
                          audit_rows=[{"chunk": chunk_id, "seq": k} for k in range(3)],
                          ct_audit_rows=[], bb_cusips=[], ds_cusips=[], ie_cusips=[],
                          n_rows=len(cusips))


def _task_raises(task):
    chunk_id, cusips, n_chunks = task
    if chunk_id == 3:
        raise RuntimeError("worker died on chunk 3")
    return CR.ChunkResult(chunk_id=chunk_id, data=list(cusips), audit_rows=[],
                          ct_audit_rows=[], bb_cusips=[], ds_cusips=[], ie_cusips=[],
                          n_rows=len(cusips))


def _serial(chunk_id, cusips, n_chunks):
    return CR.ChunkResult(chunk_id=chunk_id, data=list(cusips),
                          audit_rows=[{"chunk": chunk_id, "seq": k} for k in range(3)],
                          ct_audit_rows=[], bb_cusips=[], ds_cusips=[], ie_cusips=[],
                          n_rows=len(cusips))


def main():
    rng = random.Random(0)
    chunks = [[f"C{rng.randrange(10**8):08d}" for _ in range(rng.randrange(1, 6))]
              for _ in range(8)]

    # --- serial path ---------------------------------------------------
    res = CR.run_chunks(chunks, _serial, n_workers=1)
    check("serial: one result per chunk", len(res) == len(chunks), f"{len(res)}")
    check("serial: results in chunk order",
          [r.chunk_id for r in res] == list(range(len(chunks))))
    check("serial: every CUSIP survives, once",
          [c for r in res for c in r.data] == [c for ch in chunks for c in ch])

    check("empty universe -> no results", CR.run_chunks([], _serial, n_workers=1) == [])

    # A single chunk must not pay for a pool.
    one = CR.run_chunks(chunks[:1], _serial, n_workers=4, pool_task=_task_ok)
    check("one chunk stays serial even when workers are offered", len(one) == 1)

    # --- pool path -----------------------------------------------------
    res = CR.run_chunks(chunks, _serial, n_workers=4, pool_task=_task_ok)
    check("pool: one result per chunk", len(res) == len(chunks), f"{len(res)}")
    check("pool: results in chunk order, not completion order",
          [r.chunk_id for r in res] == list(range(len(chunks))))
    check("pool: every CUSIP survives, once",
          [c for r in res for c in r.data] == [c for ch in chunks for c in ch])

    # The reassembly the engines actually do, and the property the reports need.
    audit = [row for r in res for row in r.audit_rows]
    ok = all(audit[k]["chunk"] <= audit[k + 1]["chunk"] for k in range(len(audit) - 1))
    seqs = [row["seq"] for row in audit]
    check("pool: audit rows come back grouped and in sequence",
          ok and seqs == [0, 1, 2] * len(chunks), f"{len(audit)} rows")

    # Same input, same output -- twice.
    a = CR.run_chunks(chunks, _serial, n_workers=4, pool_task=_task_ok)
    b = CR.run_chunks(chunks, _serial, n_workers=4, pool_task=_task_ok)
    check("pool: deterministic across runs",
          [r.data for r in a] == [r.data for r in b])

    # Serial and concurrent must agree exactly.
    s = CR.run_chunks(chunks, _serial, n_workers=1)
    check("pool result == serial result", [r.data for r in a] == [r.data for r in s])

    # --- failure handling ----------------------------------------------
    try:
        CR.run_chunks(chunks, _serial, n_workers=4, pool_task=_task_raises)
        check("a dying worker aborts the run", False, "no exception raised")
    except RuntimeError as e:
        check("a dying worker aborts the run", True, type(e).__name__)

    try:
        CR.run_chunks(chunks, _serial, n_workers=4)
        check("n_workers > 1 without a pool_task is refused", False)
    except ValueError:
        check("n_workers > 1 without a pool_task is refused", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("all scheduler checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
