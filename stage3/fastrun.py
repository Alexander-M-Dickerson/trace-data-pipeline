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


def sized(n_items: int, workers: int | None = None, threads: int | None = None,
          *, min_threads: int = 2, cap_workers: int | None = None) -> tuple[int, int]:
    """Resolve (workers, threads), filling in whichever the caller did not pin.

    ❗This exists because fixed defaults do not travel. 4 workers x 5 threads is sensible
    on 24 cores and 5x oversubscribed on 4 -- and oversubscribed numba kernels are slower
    than running the same work with fewer. Callers pass whatever the user asked for on
    the command line; this fills in the rest from the machine actually running it.

    `cap_workers` bounds the automatic answer where memory, not cores, is the limit.
    """
    auto_w, auto_t = plan(n_items, min_threads=min_threads)
    if cap_workers:
        auto_w = min(auto_w, cap_workers)
        auto_t = max(min_threads, max(2, CPU - 2) // max(1, auto_w))
    w = max(1, workers or auto_w)
    t = max(1, threads or auto_t)
    if w * t > CPU:
        print(f"[fastrun] workers x threads = {w * t} on {CPU} core(s) -- oversubscribed; "
              "lower --workers or --threads if this runs slowly", flush=True)
    return w, t


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
