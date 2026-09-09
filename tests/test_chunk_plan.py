# -*- coding: utf-8 -*-
"""
test_chunk_plan.py
==================
plan_chunks must produce a true PARTITION of the CUSIP universe.

That property is what makes row-balanced chunking safe. Every per-chunk filter in
stage0 groups by cusip_id and chunks are disjoint CUSIP sets, so if the partition is
exact -- every CUSIP present exactly once -- then regrouping cannot change a single
cleaned row. If it is not exact, bonds are silently duplicated or dropped.

    python3 tests/test_chunk_plan.py

Exits non-zero on failure.

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage0"))
from _chunk_runner import plan_chunks  # noqa: E402


FAILURES = []


def check(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")
    if not passed:
        FAILURES.append(name)


def is_partition(cusips, chunks):
    flat = [c for ch in chunks for c in ch]
    return flat == list(cusips)          # same members, same count, same order


def main():
    rng = random.Random(11)

    # 1. Fixed-size fallback is unchanged behaviour.
    universe = [f"{i:09d}" for i in range(1000)]
    chunks = plan_chunks(universe, chunk_size=250)
    check("fallback: 1000 CUSIPs / 250 -> 4 chunks", len(chunks) == 4, f"{len(chunks)}")
    check("fallback: exact partition", is_partition(universe, chunks))

    # 2. Row packing keeps every CUSIP exactly once, in order.
    counts = {c: rng.randint(0, 20_000) for c in universe}
    chunks = plan_chunks(universe, counts, target_rows=100_000)
    check("packed: exact partition", is_partition(universe, chunks),
          f"{len(chunks)} chunks")

    # 3. No chunk exceeds the target unless a single CUSIP does.
    over = []
    for ch in chunks:
        rows = sum(counts[c] for c in ch)
        if rows > 100_000 and len(ch) > 1:
            over.append((len(ch), rows))
    check("packed: no multi-CUSIP chunk exceeds the target", not over, f"{len(over)} over")

    # 4. An oversized CUSIP gets its own chunk rather than being dropped or split.
    counts["000000500"] = 5_000_000
    chunks = plan_chunks(universe, counts, target_rows=100_000)
    solo = [ch for ch in chunks if "000000500" in ch]
    check("packed: an oversized CUSIP is isolated", len(solo) == 1 and solo[0] == ["000000500"],
          f"chunk of {len(solo[0]) if solo else '?'}")
    check("packed: still an exact partition with an outlier", is_partition(universe, chunks))

    # 5. CUSIPs the count query never saw (zero rows) are still carried.
    partial = {c: 1000 for c in universe[:500]}          # second half missing entirely
    chunks = plan_chunks(universe, partial, target_rows=10_000)
    check("packed: CUSIPs missing from row_counts are not dropped",
          is_partition(universe, chunks), f"{len(chunks)} chunks")

    # 6. Degenerate inputs.
    check("empty universe -> no chunks", plan_chunks([], {}, 1000) == [])
    check("target_rows=None falls back to fixed size",
          plan_chunks(universe, counts, None, chunk_size=100) == plan_chunks(universe, chunk_size=100))
    check("empty row_counts falls back to fixed size",
          plan_chunks(universe, {}, 1000, chunk_size=100) == plan_chunks(universe, chunk_size=100))

    # 7. The real shape: the measured Enhanced distribution.
    #    447 chunks of 250 ranged 7,806 .. 3,392,802 rows; the busiest single CUSIP is
    #    513,372. Packing to 750k should bring the maximum near the target.
    big = [f"{i:09d}" for i in range(112_000)]
    real = {}
    for c in big:
        r = rng.random()
        real[c] = (rng.randint(0, 200) if r < 0.25 else
                   rng.randint(200, 20_000) if r < 0.90 else
                   rng.randint(20_000, 513_372))
    packed = plan_chunks(big, real, target_rows=750_000)
    check("real-shape: exact partition of 112k CUSIPs", is_partition(big, packed),
          f"{len(packed)} chunks")
    rows = [sum(real[c] for c in ch) for ch in packed]
    fixed = plan_chunks(big, chunk_size=250)
    fixed_rows = [sum(real[c] for c in ch) for ch in fixed]
    check("real-shape: packing lowers the worst chunk", max(rows) < max(fixed_rows),
          f"max {max(rows):,} packed vs {max(fixed_rows):,} fixed "
          f"({max(fixed_rows)/max(rows):.1f}x better)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all chunk-plan checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
