"""fastrun.py -- process-parallel fan-out for the uncertainty grids.

Scale with PROCESSES, not threads. A long-lived Python process makes DuckDB collapse
towards one effective thread, and numba kernels do not share a process well either. So
each independent unit of work -- one signal's grid -- runs in a FRESH process (spawn,
maxtasksperchild=1) with a small thread budget, sized so `workers * threads <= cores`.

    results = pmap(my_top_level_task, list_of_specs)

`task_fn` must be a TOP-LEVEL (picklable) function. Each worker reads the panel columns
it needs itself; nothing large is passed through the pipe, because on Windows every
argument is re-pickled into every worker and a panel sent twelve times costs more than
the parallelism saves.
"""
from __future__ import annotations

import multiprocessing as mp
import os

CPU = os.cpu_count() or 8


def plan(n_items: int, total_cores: int = CPU, min_threads: int = 2):
    """Pick (workers, threads_per_worker) so workers*threads <= total_cores and we don't over-spawn."""
    usable = max(2, total_cores - 2)                 # leave headroom
    workers = max(1, min(n_items, usable // min_threads))
    threads = max(min_threads, usable // workers)
    return workers, threads


def pmap(task_fn, items, workers: int | None = None, threads: int | None = None,
         total_cores: int = CPU):
    """Process-parallel map. `task_fn` (TOP-LEVEL, picklable) gets one item and returns its result.
    Fresh process per task (spawn, maxtasksperchild=1). Returns results in input order.

    `threads` is published to the workers through the environment rather than the pipe;
    a task reads it with `worker_threads()`.
    """
    n = len(items)
    if n == 0:
        return []
    if workers is None or threads is None:
        w, t = plan(n, total_cores)
        workers = workers or w
        threads = threads or t
    if workers <= 1:
        return [task_fn(it) for it in items]         # serial fallback (e.g. n==1)

    ctx = mp.get_context("spawn")
    os.environ["STAGE3_WORKER_THREADS"] = str(threads)
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        return pool.map(task_fn, items)


def worker_threads(default: int = 4) -> int:
    return int(os.environ.get("STAGE3_WORKER_THREADS", default))
