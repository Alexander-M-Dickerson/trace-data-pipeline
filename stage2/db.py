# -*- coding: utf-8 -*-
"""
db.py
=====
DuckDB access for Stage 2.

Stage 2 uses DuckDB purely as a compute engine over parquet files -- it never opens a
database file and never queries a stored table. Every query reads through
``read_parquet(...)``, so the connection here is IN-MEMORY.

Threads and memory default to a fraction of the machine rather than fixed numbers, so
the same code is comfortable on a laptop and on a workstation. Override with the
``STAGE2_DUCKDB_THREADS`` / ``STAGE2_DUCKDB_MEMORY_LIMIT`` environment variables, or by
setting ``DUCKDB_THREADS`` / ``DUCKDB_MEMORY_LIMIT`` in ``_stage2_settings.py``.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

import duckdb

# Scratch space for DuckDB spills and for staged parquet writes.
DEFAULT_TMP = Path(os.environ.get("STAGE2_TMP")
                   or os.environ.get("TEMP")
                   or "/tmp") / "trace_stage2"


def _default_threads() -> int:
    """Most of the machine, leaving a little for the OS."""
    n = os.cpu_count() or 4
    return max(1, n - 2) if n > 4 else max(1, n)


def _default_memory_limit() -> str:
    """About 70% of physical RAM, or a safe floor when we cannot detect it."""
    try:
        import psutil  # optional
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        try:
            total_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
        except (ValueError, AttributeError, OSError):
            return "8GB"
    return f"{max(4, int(total_gb * 0.70))}GB"


def connect(
    *,
    threads: int | None = None,
    memory_limit: str | None = None,
    temp_directory: Path | str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a tuned in-memory DuckDB connection.

    Stage 2 stores nothing in DuckDB; results go to parquet via :func:`to_parquet`.
    """
    if threads is None:
        threads = int(os.environ.get("STAGE2_DUCKDB_THREADS", "0")) or _default_threads()
    if memory_limit is None:
        memory_limit = os.environ.get("STAGE2_DUCKDB_MEMORY_LIMIT") or _default_memory_limit()

    con = duckdb.connect()  # in-memory
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    tmp = Path(temp_directory) if temp_directory else DEFAULT_TMP
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")  # lets the writer parallelise
    return con


def to_parquet(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    out_path: Path | str,
    *,
    compression: str = "ZSTD",
    via_local_temp: bool = True,
) -> Path:
    """Run ``sql`` and write the result to ``out_path`` as parquet.

    ``via_local_temp`` writes to scratch first and moves the finished file into place,
    so a synced folder (Dropbox, OneDrive) sees one complete file instead of every
    partial flush.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if via_local_temp:
        DEFAULT_TMP.mkdir(parents=True, exist_ok=True)
        target = DEFAULT_TMP / f"_copy_{uuid.uuid4().hex}.parquet"
    else:
        target = out_path

    t0 = time.time()
    con.execute(
        f"COPY ({sql}) TO '{target.as_posix()}' "
        f"(FORMAT PARQUET, COMPRESSION {compression})"
    )
    elapsed = time.time() - t0

    if via_local_temp:
        shutil.move(str(target), str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"[to_parquet] wrote {out_path.name}: {size_mb:.1f} MB in {elapsed:.1f}s")
    return out_path


def daily_relation(path) -> str:
    """A ``read_parquet()`` relation string for the Stage 1 daily panel."""
    return f"read_parquet('{Path(path).as_posix()}')"


class timed:
    """Context manager that prints wall-clock time. Usage: ``with timed('label'):``"""

    def __init__(self, label: str = "block"):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"[timed] {self.label}: {time.time() - self.t0:.2f}s")
        return False
