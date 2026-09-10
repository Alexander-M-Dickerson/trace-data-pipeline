# -*- coding: utf-8 -*-
"""
_run_stage2.py
==============
Runner for Stage 2 monthly panel construction. Called by run_stage2.sh.

Stage 2 runs on your own machine against the stage0/ and stage1/ folders downloaded
from WRDS. It needs no TRACE database access.

Usage
-----
    python3 _run_stage2.py                      # full build, steps 1-7
    python3 _run_stage2.py --dry-run            # resolve + validate config, build nothing
    python3 _run_stage2.py --from-step 4 --to-step 7
    python3 _run_stage2.py --limit-cusips 200   # fast development build
    python3 _run_stage2.py --factor-source pinned

Author: Open Source Bond Asset Pricing
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys

from _stage2_settings import (get_config, validate_config, print_config_summary,
                              ensure_dirs)


def _box(title: str, body: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}\n{body}\n{line}\n")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="_run_stage2.py",
        description="Build the monthly asset-pricing panel from Stage 1's daily panel.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and validate the configuration, then exit without building.")
    p.add_argument("--from-step", type=int, default=1, metavar="N",
                   help="First step to run (1-7, default 1).")
    p.add_argument("--to-step", type=int, default=7, metavar="N",
                   help="Last step to run (1-7, default 7).")
    p.add_argument("--limit-cusips", type=int, default=None, metavar="N",
                   help="Build on the first N CUSIPs only (development).")
    p.add_argument("--factor-source", choices=("public", "pinned"), default=None,
                   help="Override FACTOR_SOURCE for this run.")
    p.add_argument("--refresh-factors", action="store_true",
                   help="Force a re-fetch of every public factor source.")
    p.add_argument("--sequential", action="store_true",
                   help="Disable the parallel chain overlap (one step at a time).")
    p.add_argument("--validate", action="store_true",
                   help="Run the validation sweep after the build.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gc.collect()

    if not 1 <= args.from_step <= 7 or not 1 <= args.to_step <= 7:
        _box("CONFIGURATION ERROR", "--from-step and --to-step must be between 1 and 7.")
        return 1
    if args.from_step > args.to_step:
        _box("CONFIGURATION ERROR",
             f"--from-step ({args.from_step}) is after --to-step ({args.to_step}).")
        return 1

    config = get_config()
    if args.factor_source:
        config["factor_source"] = args.factor_source

    print_config_summary(config)

    try:
        validate_config(config)
    except (ValueError, FileNotFoundError) as e:
        _box("CONFIGURATION ERROR", str(e))
        return 1

    print("Configuration OK.")

    if args.dry_run:
        print("\n--dry-run: configuration resolved and validated. Nothing was built.")
        return 0

    ensure_dirs()

    try:
        from build_panel import run_stage2          # noqa: WPS433 - imported after validation
    except ImportError as e:
        _box("PIPELINE ERROR",
             f"Stage 2's build engine is not installed yet: {e}\n"
             f"Only the configuration layer is present in this tree.")
        return 1

    try:
        run_stage2(
            config,
            from_step=args.from_step,
            to_step=args.to_step,
            limit_cusips=args.limit_cusips,
            refresh_factors=args.refresh_factors,
            sequential=args.sequential,
            validate=args.validate,
        )
    except Exception as e:  # noqa: BLE001 - top-level reporter
        _box("PIPELINE ERROR", f"Error: {e}")
        logging.exception("Full traceback:")
        return 1

    print("\nStage 2 processing completed successfully!")
    print(f"  Panel   : {config['panel_dir']}")
    print(f"  Reports : run  ./run_build_data_reports.sh  to generate the data report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
