# -*- coding: utf-8 -*-
"""
_chunk_runner.py
================
How stage0 divides the CUSIP universe into work units, and how it runs them.

Two halves: plan_chunks decides what a work unit IS, run_chunks decides who executes
it. They sit together because the second only became worth having once the first made
chunks even enough in size to run several at once.

PART ONE -- THE PLAN

The original scheme was a fixed count of CUSIPs per chunk (250). That is simple but
badly unbalanced, because trading activity is enormously skewed: measured over the
Enhanced universe, 447 chunks of 250 CUSIPs ranged from 7,806 rows to 3,392,802, a
median of 646,240. The largest chunk is a ~2.7 GB frame before the cleaners make
their copies, and the peak inside decimal_shift_corrector is roughly 2.5x that.

Serially that is merely wasteful. Running several chunks at once it is the binding
constraint, because the memory a job must reserve is set by the WORST chunk, not the
average -- and reserving for the worst wastes most of the job's allocation.

Packing to a target ROW count instead bounds it: the same universe packs to chunks of
about the target, with the only hard floor being the busiest single CUSIP (513,372
rows in Enhanced), so the maximum falls from 3.39M to roughly the target.

This does NOT change the cleaned data. Every per-chunk filter in both engines groups
by cusip_id -- the decimal-shift anchor, the bounce-back scan, the initial-price-error
scan and the Dick-Nielsen reversal keys all lead with it -- and chunks are disjoint
CUSIP sets. So which chunk a bond lands in cannot affect its result. What DOES change
is the audit tables, whose chunk column and per-chunk counts follow the new grouping.

PART TWO -- THE SCHEDULER

run_chunks executes the plan, serially or across a pool of processes each holding its
own WRDS connection, and hands the results back IN CHUNK ORDER regardless of the order
they finished in. See its docstring for why the ordering is a correctness requirement
rather than a tidiness one.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence


def divide_chunks(seq: Sequence, n: int):
    """Fixed-size chunks -- the original scheme, kept as the fallback."""
    for i in range(0, len(seq), n):
        yield seq[i: i + n]


def plan_chunks(cusips: Sequence[str],
                row_counts: dict | None = None,
                target_rows: int | None = None,
                chunk_size: int = 250) -> list[list[str]]:
    """Partition `cusips` into work units.

    With `row_counts` and `target_rows`, packs greedily to the row target, preserving
    the input order so chunks stay contiguous slices of the universe and a run is
    reproducible. Without them, falls back to fixed-size chunks.

    A CUSIP whose own row count exceeds the target gets a chunk to itself -- it cannot
    be split, since every filter needs a bond's whole history in one place.

    Parameters
    ----------
    cusips : sequence of str
        The universe, in the order it should be walked.
    row_counts : dict, optional
        cusip -> number of trade rows. Missing entries count as 0.
    target_rows : int, optional
        Approximate rows per chunk. None disables row packing.
    chunk_size : int
        CUSIPs per chunk for the fallback.

    Returns
    -------
    list of list of str
        A true partition: every input CUSIP appears exactly once, order preserved.
    """
    cusips = list(cusips)
    if not cusips:
        return []

    if not row_counts or not target_rows or target_rows <= 0:
        return list(divide_chunks(cusips, chunk_size))

    chunks: list[list[str]] = []
    current: list[str] = []
    current_rows = 0

    for c in cusips:
        n = int(row_counts.get(c, 0) or 0)
        # Close the current chunk before adding a CUSIP that would push it past the
        # target -- unless the chunk is empty, in which case this CUSIP is oversized
        # and takes the chunk on its own.
        if current and current_rows + n > target_rows:
            chunks.append(current)
            current, current_rows = [], 0
        current.append(c)
        current_rows += n

    if current:
        chunks.append(current)
    return chunks


def summarize_plan(chunks: Iterable[Sequence[str]],
                   row_counts: dict | None,
                   logger: logging.Logger | None = None) -> dict:
    """Log what the plan looks like, so a bad one is visible before the run, not after."""
    chunks = list(chunks)
    log = (logger or logging.getLogger(__name__)).info
    if not chunks:
        log("Chunk plan: EMPTY")
        return {}

    sizes = [len(c) for c in chunks]
    stats = {"n_chunks": len(chunks), "n_cusips": sum(sizes),
             "min_cusips": min(sizes), "max_cusips": max(sizes)}

    if row_counts:
        rows = [sum(int(row_counts.get(c, 0) or 0) for c in ch) for ch in chunks]
        rows_sorted = sorted(rows)
        stats.update(total_rows=sum(rows), min_rows=rows_sorted[0],
                     max_rows=rows_sorted[-1],
                     median_rows=rows_sorted[len(rows_sorted) // 2])
        log("Chunk plan: %d chunks over %d CUSIPs | rows per chunk "
            "min %s / median %s / max %s (total %s)",
            stats["n_chunks"], stats["n_cusips"], f"{stats['min_rows']:,}",
            f"{stats['median_rows']:,}", f"{stats['max_rows']:,}",
            f"{stats['total_rows']:,}")
    else:
        log("Chunk plan: %d chunks over %d CUSIPs (fixed size; row counts unavailable)",
            stats["n_chunks"], stats["n_cusips"])
    return stats


# =========================================================================
# THE SCHEDULER
# =========================================================================
@dataclass
class ChunkResult:
    """One chunk's output, on its way back from wherever it was computed.

    `data` is None for a chunk whose WRDS query returned nothing -- distinct from an
    empty frame, and it carries no audit rows either, matching what the original
    inline loop did with its `continue`.
    """
    chunk_id: int
    data: object                 # pandas.DataFrame | None
    audit_rows: list
    ct_audit_rows: list
    bb_cusips: list
    ds_cusips: list
    ie_cusips: list
    n_rows: int = 0
    elapsed: float = 0.0


def run_chunks(chunks,
               run_serial,
               *,
               n_workers: int = 1,
               pool_task=None,
               pool_initializer=None,
               pool_initargs: tuple = (),
               logger: logging.Logger | None = None) -> list:
    """Run every chunk and return the results in ASCENDING CHUNK ORDER.

    `n_workers <= 1` calls `run_serial(chunk_id, cusips, n_chunks)` in this process --
    the original loop, on the caller's own connection. Above that, `pool_task` is run
    in a process pool, each worker holding its own WRDS connection opened by
    `pool_initializer`.

    Ordering is not cosmetic. The audit tables are read back by the reporting stage,
    which reconstructs each chunk's filter sequence from row order, so results must be
    reassembled by chunk id no matter what order they completed in.

    A chunk that produces no result ABORTS the run. Silently missing chunks are
    silently missing bonds in a file that otherwise looks entirely normal, and stage 1
    would happily consume it.
    """
    chunks = list(chunks)
    log = logger or logging.getLogger(__name__)
    n = len(chunks)
    if n == 0:
        return []

    if n_workers <= 1 or n == 1:
        return [run_serial(i, list(ch), n) for i, ch in enumerate(chunks)]

    if pool_task is None:
        raise ValueError("run_chunks needs a pool_task to run with n_workers > 1")

    import multiprocessing as mp

    n_workers = min(int(n_workers), n)
    tasks = [(i, list(ch), n) for i, ch in enumerate(chunks)]
    log.info("Running %d chunks across %d worker processes "
             "(one WRDS connection each)", n, n_workers)

    # NO maxtasksperchild. Workers stay alive for the whole run on purpose: recycling
    # one would mean a fresh ~5 s WRDS handshake per chunk -- 485 of them on Enhanced
    # -- and would hammer the per-user connection limit for nothing. (The house rule
    # in trace_duckdb about fresh processes exists for DuckDB's parallelism collapse
    # and does not transfer here.)
    pool = mp.Pool(n_workers, initializer=pool_initializer, initargs=pool_initargs)
    got: dict[int, ChunkResult] = {}
    try:
        for res in pool.imap_unordered(pool_task, tasks):
            got[res.chunk_id] = res
            log.info("chunk %d/%d complete (id %d, %s rows, %.1fs)",
                     len(got), n, res.chunk_id, f"{res.n_rows:,}", res.elapsed)
        pool.close()
    except BaseException:
        # A worker raised, or we were interrupted. Kill the pool rather than let it
        # linger holding connections.
        pool.terminate()
        raise
    finally:
        pool.join()

    missing = [i for i in range(n) if i not in got]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {n} chunks returned no result: "
            f"{missing[:20]}{' ...' if len(missing) > 20 else ''}. "
            "Refusing to export a partial tape.")

    return [got[i] for i in range(n)]
